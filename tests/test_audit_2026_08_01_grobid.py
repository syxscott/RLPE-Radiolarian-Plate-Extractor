"""Regression tests for audit 2026-08-01 batch W2 — grobid M24/M26/D18.

Covers three audit findings against ``src/rlpe/grobid.py``:

* **M24** — ``GrobidClient.process_pdf`` retry loop double-slept on the
  cancel-aware backoff path. ``cancel_event.wait(timeout=delay)`` already
  blocks for the full ``delay`` (or until the cancel event fires); the
  trailing ``time.sleep(delay)`` doubled the documented backoff. The
  fix removes the redundant sleep so one retry's elapsed time matches
  the documented ``delay``.

* **M26** — ``infer_section_type`` used a substring check
  (``"reference" in t``) that misclassified legitimate section titles
  like "Cross-referenced Section" or "Biogeographic reference frame" as
  ``"references"``, then ``geology_extraction.py`` skipped them. The
  fix uses a word-boundary regex so the bare word ``reference`` /
  ``references`` matches but ``cross-referenced`` /
  ``reference frame`` does not.

* **D18** — ``tei_path.write_text(...)`` was non-atomic; a torn write
  would then be silently swallowed by ``parse_captions_from_tei`` /
  ``parse_fulltext_sections_from_tei`` (which caught ``ET.ParseError``
  and returned ``[]``). The fix replaces the write with an
  ``os.replace``-based atomic helper and re-raises ``GrobidParseError``
  so callers can see the failure.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.grobid import (  # noqa: E402
    GrobidClient,
    GrobidParseError,
    _write_tei_atomic,
    infer_section_type,
    parse_captions_from_tei,
)

# Minimal well-formed TEI used by retry-loop tests; ``<text>`` and
# ``<body>`` are stripped down so ``parse_captions_from_tei`` and
# ``parse_fulltext_sections_from_tei`` don't blow up.
_VALID_TEI = '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader/><text><body/></text></TEI>'


# ---------------------------------------------------------------------------
# M24 — retry-loop double-sleep
# ---------------------------------------------------------------------------


class TestGrobidRetry:
    """Audit 2026-08-01 W2 / M24: the cancel-aware backoff path used to
    sleep twice — once via ``cancel_event.wait(timeout=delay)`` and
    once via ``time.sleep(delay)``. These tests assert the post-fix
    behaviour: one retry's elapsed wall-clock is bounded by ``delay``
    (plus scheduling overhead), not ``2 * delay``."""

    def test_retry_backoff_does_not_double_sleep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Total elapsed for one retry must be ~``delay``, not ~``2 * delay``.

        We construct a client with ``retry_backoff=0.1`` and
        ``max_retries=2``, count both ``time.sleep`` and
        ``Event.wait`` calls, and assert that the backoff path uses
        exactly one blocking call of ~0.1s.
        """
        import requests

        call_log: list[tuple[str, float]] = []

        def fake_sleep(seconds: float) -> None:
            # NOTE: do NOT call ``time.sleep`` from inside here — that
            # would recurse into this very function. We only need to
            # log the call; the test verifies via call counts.
            call_log.append(("sleep", seconds))

        def fake_wait(self: threading.Event, timeout: float | None = None) -> bool:
            call_log.append(("wait", timeout if timeout is not None else 0.0))
            return False  # not cancelled

        def fake_post(*args, **kwargs):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr("time.sleep", fake_sleep)
        monkeypatch.setattr(threading.Event, "wait", fake_wait)
        monkeypatch.setattr(requests, "post", fake_post)

        client = GrobidClient(
            server_url="http://localhost:1",
            timeout=10,
            max_retries=2,
            retry_backoff=0.1,
            cancel_event=threading.Event(),
        )
        pdf = Path("/tmp/__audit_w2_m24_pdf__.pdf")
        if not pdf.exists():
            pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
        try:
            r = client.process_pdf(pdf, Path("/tmp/__audit_w2_m24_out__"))
            assert r.success is False
        finally:
            pdf.unlink(missing_ok=True)

        # Filter to only backoff-path calls (delay == 0.1).
        backoff_calls = [
            (kind, t) for kind, t in call_log if (kind == "sleep" or kind == "wait") and t == 0.1
        ]
        # Exactly ONE blocking call of 0.1s — not two.
        assert len(backoff_calls) == 1, (
            f"Expected exactly 1 blocking call of 0.1s; got {backoff_calls!r}. "
            "M24 not fixed: retry loop is double-sleeping."
        )
        # And it must be the Event.wait() (cancel-aware), not a bare
        # time.sleep(), so cancellation still short-circuits the
        # backoff (Phase 59 invariant).
        assert backoff_calls[0][0] == "wait", (
            f"Backoff path should use Event.wait() for cancel awareness; "
            f"got {backoff_calls[0][0]!r}"
        )

    def test_cancel_event_short_circuits_retry(self) -> None:
        """A pre-set cancel_event must abort the retry loop on the very
        first iteration — well before the backoff sleep would
        otherwise elapse. Without M24's wait() removal this still has
        to work; the fix just removes the trailing sleep."""
        import requests

        cancel = threading.Event()
        cancel.set()

        def fake_post(*args, **kwargs):
            time.sleep(60)
            raise requests.ConnectionError("refused")

        import rlpe.grobid as grobid_mod

        original_post = requests.post
        grobid_mod.requests.post = fake_post
        try:
            c = GrobidClient(
                server_url="http://localhost:1",
                timeout=120,
                max_retries=3,
                retry_backoff=0.1,  # would be ~0.1s (not 0.2s) post-M24
                cancel_event=cancel,
            )
            pdf = Path("/tmp/__audit_w2_m24_cancel_pdf__.pdf")
            if not pdf.exists():
                pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
            try:
                from rlpe.pipeline import PipelineCancelledError
            except ImportError:
                from rlpe.errors import PipelineCancelledError  # type: ignore

            start = time.monotonic()
            with pytest.raises(PipelineCancelledError):
                c.process_pdf(pdf, Path("/tmp/__audit_w2_m24_cancel_out__"))
            elapsed = time.monotonic() - start
            # The pre-set cancel aborts BEFORE the first request even
            # goes out, so elapsed should be sub-second.
            assert elapsed < 0.5, (
                f"Pre-set cancel_event must abort within 0.5s; took {elapsed:.3f}s"
            )
        finally:
            grobid_mod.requests.post = original_post
            Path("/tmp/__audit_w2_m24_cancel_pdf__.pdf").unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# M26 — infer_section_type word-boundary regex
# ---------------------------------------------------------------------------


class TestInferSectionType:
    """Audit 2026-08-01 W2 / M26: substring ``"reference" in t`` was too
    greedy. Word-boundary regex must match the bare heading
    ``References`` / ``Bibliography`` but NOT composite section titles
    like "Cross-referenced Section" or "Biogeographic reference
    frame"."""

    def test_normal_references(self) -> None:
        """The bare heading ``References`` (and common variants) must
        still classify as ``"references"``."""
        for title in ["References", "REFERENCES", "Bibliography", "Literature cited"]:
            got = infer_section_type(title)
            assert got == "references", (
                f"infer_section_type({title!r}) returned {got!r}; expected 'references'"
            )

    def test_reference_substring_not_section(self) -> None:
        """ "Biogeographic reference frame" must NOT be classified as
        ``"references"`` — the word-boundary regex keeps it in the
        normal body bucket."""
        got = infer_section_type("Biogeographic reference frame")
        assert got != "references", (
            f"infer_section_type('Biogeographic reference frame') returned {got!r}; "
            "M26 not fixed: substring match is still misclassifying"
        )

    def test_cross_referenced_not_references(self) -> None:
        """ "Cross-referenced Section" must NOT be classified as
        ``"references"``."""
        got = infer_section_type("Cross-referenced Section")
        assert got != "references", (
            f"infer_section_type('Cross-referenced Section') returned {got!r}; "
            "M26 not fixed: substring match is still misclassifying"
        )

    def test_reference_section_matches(self) -> None:
        """ "Reference Section" (a real GROBID heading style) must still
        match — the regex tolerates trailing non-word characters."""
        got = infer_section_type("Reference Section")
        assert got == "references", (
            f"infer_section_type('Reference Section') returned {got!r}; expected 'references'"
        )

    def test_reference_frames_classified_as_references(self) -> None:
        """Audit 2026-08-01 W2 / M26 explicitly notes that "Reference
        Frames" must be "handled properly" — i.e. classified as
        ``"references"`` so it gets skipped by ``geology_extraction``,
        just like a plain "References" heading. This freezes the
        leading-word semantics of the new regex."""
        got = infer_section_type("Reference Frames")
        assert got == "references", (
            f"infer_section_type('Reference Frames') returned {got!r}; "
            "expected 'references' (audit spec: handle 'Reference Section' "
            "/ 'Reference Frames' properly)"
        )


# ---------------------------------------------------------------------------
# D18 — atomic TEI write + no-longer-silent parse errors
# ---------------------------------------------------------------------------


class TestTEIAtomic:
    """Audit 2026-08-01 W2 / D18: the ``tei_path.write_text(...)`` call
    in ``process_pdf`` was non-atomic, and parse errors were silently
    swallowed. These tests assert the post-fix invariants:

    * ``_write_tei_atomic`` writes via a sibling ``.tmp`` then
      ``os.replace``; if ``os.replace`` fails, the tmp file is cleaned
      up and the original ``tei_path`` is not corrupted.
    * ``parse_captions_from_tei`` raises ``GrobidParseError`` on
      malformed XML instead of returning ``[]``.
    """

    def test_atomic_tei_write(self, tmp_path: Path) -> None:
        """Normal path: the atomic write produces a file with the
        expected content and leaves no ``.tmp`` scratch behind."""
        tei_path = tmp_path / "paper.tei.xml"
        text = _VALID_TEI
        _write_tei_atomic(tei_path, text)
        assert tei_path.exists()
        assert tei_path.read_text(encoding="utf-8") == text
        scratch = list(tmp_path.glob("*.tmp"))
        assert not scratch, (
            f"Atomic write should not leave .tmp scratch files behind; got {scratch!r}"
        )

    def test_atomic_tei_write_replace_failure_cleans_tmp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If ``os.replace`` raises mid-write, the tmp scratch file
        must be cleaned up and the original ``tei_path`` must NOT be
        left in a corrupted (truncated / partial) state."""
        tei_path = tmp_path / "paper.tei.xml"
        # Pre-existing file with good content — must remain intact.
        good_content = "GOOD ORIGINAL CONTENT"
        tei_path.write_text(good_content, encoding="utf-8")

        original_replace = __import__("os").replace

        def boom_replace(src: str, dst: str) -> None:
            raise OSError("simulated replace failure")

        monkeypatch.setattr("os.replace", boom_replace)

        with pytest.raises(OSError):
            _write_tei_atomic(tei_path, _VALID_TEI)

        # Restore os.replace for the cleanup check below (monkeypatch
        # restores automatically at teardown).
        # The original file must be unchanged.
        assert tei_path.read_text(encoding="utf-8") == good_content, (
            "Original tei_path must remain intact when os.replace fails"
        )
        # No scratch .tmp files should linger.
        scratch = list(tmp_path.glob("*.tmp"))
        assert not scratch, f"Atomic write should clean up .tmp on replace failure; got {scratch!r}"

        # And calling original_replace still works (sanity).
        assert original_replace is not None

    def test_parse_error_no_longer_silent(self) -> None:
        """Malformed XML must raise ``GrobidParseError`` — the previous
        behaviour was to return ``[]`` and silently swallow the
        failure."""
        bad_xml = "<TEI><body><unclosed>"
        with pytest.raises(GrobidParseError):
            parse_captions_from_tei(bad_xml, paper_id="test")

    def test_parse_error_classified_as_parse_error(self) -> None:
        """``GrobidParseError`` must classify as ``"parse_error"`` via
        ``_classify_exception`` so the retry loop and pipeline fallback
        logic treat it consistently with raw ``ET.ParseError``."""
        try:
            parse_captions_from_tei("<bad", paper_id="test")
        except GrobidParseError as exc:
            cls = GrobidClient._classify_exception(exc)
            assert cls == "parse_error", (
                f"_classify_exception(GrobidParseError) returned {cls!r}; expected 'parse_error'"
            )

    def test_empty_xml_still_returns_empty(self) -> None:
        """Empty / whitespace-only XML is not a parse error — it must
        still return ``[]`` without raising (backward-compat for
        callers that pass ``tei_xml=''``)."""
        assert parse_captions_from_tei("", paper_id="test") == []
        assert parse_captions_from_tei("   \n\n  ", paper_id="test") == []


# ---------------------------------------------------------------------------
# Source-guard: helpers + exception are reachable from the package
# ---------------------------------------------------------------------------


def test_grobid_parse_error_is_importable() -> None:
    """``GrobidParseError`` must be importable from ``rlpe.grobid`` so
    external callers (pipeline.py, the web app) can catch it."""
    from rlpe.grobid import GrobidParseError as _Gpe  # noqa: F401

    assert issubclass(_Gpe, Exception)


def test_write_tei_atomic_callable_with_path_and_str() -> None:
    """Source guard: ``_write_tei_atomic`` accepts ``(Path, str)``."""
    import inspect

    sig = inspect.signature(_write_tei_atomic)
    params = list(sig.parameters.keys())
    assert params[:2] == ["tei_path", "text"]
