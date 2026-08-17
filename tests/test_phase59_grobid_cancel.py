"""Phase 59 — Pipeline correctness, Bug 2.2.

GROBID's retry loop (``process_pdf`` in ``src/rlpe/grobid.py``) did
not check the caller's ``cancel_event`` between retries. A 200-page
paper with ``max_retries=3`` + ``timeout=300s`` could run for 15
minutes after the user clicked Cancel — they thought the GUI was
hung.

The fix threads ``cancel_event`` through ``GrobidClient.__init__``
and checks it at the top of every retry iteration. When set, the
client raises ``PipelineCancelledError`` so the pipeline can return
the rows processed so far.

This test asserts:

  1. ``GrobidClient.__init__`` accepts ``cancel_event``.
  2. Setting the event before ``process_pdf`` is called raises
     ``PipelineCancelledError`` within 1 second (well under the
     300s default timeout).
  3. The exception is the well-known ``PipelineCancelledError`` (or
     its importable alias) so downstream catch handlers still work.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.grobid import GrobidClient  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]


def test_grobid_client_accepts_cancel_event_kwarg() -> None:
    """Bug 2.2 fix: ``GrobidClient.__init__`` accepts ``cancel_event``."""
    import inspect

    params = list(inspect.signature(GrobidClient.__init__).parameters.keys())
    assert "cancel_event" in params, (
        f"GrobidClient.__init__ must accept cancel_event; got {list(params)}"
    )


def test_grobid_cancel_breaks_retry_loop() -> None:
    """Bug 2.2 fix: pre-set cancel_event short-circuits the retry loop.

    Without the fix, ``process_pdf`` sleeps ``max_retries * timeout``
    seconds (default 900s) before returning a failure. With the fix,
    the first retry iteration sees the event and raises within ~1s.
    """
    import requests

    cancel = threading.Event()
    cancel.set()  # user already clicked Cancel before we call

    def fake_post(*args, **kwargs):
        # Long artificial sleep — the cancel check must short-circuit
        # *before* requests.post completes.
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
            retry_backoff=0.0,
            cancel_event=cancel,
        )
        pdf = Path("/tmp/__grobid_cancel_pdf__.pdf")
        if not pdf.exists():
            pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
        try:
            from rlpe.pipeline import PipelineCancelledError
        except ImportError:
            from rlpe.errors import PipelineCancelledError  # type: ignore

        start = time.monotonic()
        with pytest.raises(PipelineCancelledError):
            c.process_pdf(pdf, Path("/tmp/__grobid_cancel_out__"))
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"cancel_event must short-circuit within 1s; took {elapsed:.2f}s"
    finally:
        grobid_mod.requests.post = original_post
        Path("/tmp/__grobid_cancel_pdf__.pdf").unlink(missing_ok=True)


def test_grobid_cancel_during_retry_loop_aborts() -> None:
    """Bug 2.2 fix: cancel set mid-loop aborts on the next iteration.

    The first attempt fails normally; the cancel event is set during
    the backoff sleep; the second iteration sees the event and aborts.
    """
    from unittest.mock import MagicMock

    import requests

    cancel = threading.Event()
    call_count = {"n": 0}

    def fake_post(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise requests.ConnectionError("transient")
        # Second call shouldn't happen because cancel kicks in.
        resp = MagicMock()
        resp.text = (
            '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader/><text><body/></text></TEI>'
        )
        resp.raise_for_status = lambda: None
        return resp

    import rlpe.grobid as grobid_mod

    original_post = requests.post
    grobid_mod.requests.post = fake_post
    try:
        c = GrobidClient(
            server_url="http://localhost:1",
            timeout=10,
            max_retries=3,
            retry_backoff=0.5,  # non-zero so the backoff sleep path is exercised
            cancel_event=cancel,
        )
        pdf = Path("/tmp/__grobid_cancel_during__.pdf")
        if not pdf.exists():
            pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
        try:
            from rlpe.pipeline import PipelineCancelledError
        except ImportError:
            from rlpe.errors import PipelineCancelledError  # type: ignore

        # Set cancel during the (zero-length) backoff between attempts.
        def set_cancel_later():
            time.sleep(0.05)
            cancel.set()

        threading.Thread(target=set_cancel_later, daemon=True).start()
        with pytest.raises(PipelineCancelledError):
            c.process_pdf(pdf, Path("/tmp/__grobid_cancel_during_out__"))
        # At most 2 attempts (first fails, second sees cancel).
        assert call_count["n"] <= 2, (
            f"Expected <=2 attempts (cancel after first); got {call_count['n']}"
        )
    finally:
        grobid_mod.requests.post = original_post
        Path("/tmp/__grobid_cancel_during__.pdf").unlink(missing_ok=True)


def test_grobid_pipeline_cancelled_error_exists() -> None:
    """Bug 2.2 fix: ``PipelineCancelledError`` is importable from the
    rlpe package so catch handlers can reference it.
    """
    try:
        from rlpe.pipeline import PipelineCancelledError  # type: ignore
    except ImportError:
        from rlpe.errors import PipelineCancelledError  # type: ignore
    assert PipelineCancelledError is not None
    assert issubclass(PipelineCancelledError, BaseException)


def test_grobid_no_cancel_runs_to_completion() -> None:
    """Bug 2.2 backward-compat: with cancel_event=None, retry loop runs
    to completion as before (no regression).
    """
    import requests

    call_count = {"n": 0}

    def fake_post(*args, **kwargs):
        call_count["n"] += 1
        raise requests.ConnectionError("refused")

    import rlpe.grobid as grobid_mod

    original_post = requests.post
    grobid_mod.requests.post = fake_post
    try:
        c = GrobidClient(
            server_url="http://localhost:1",
            timeout=10,
            max_retries=3,
            retry_backoff=0.0,
            # cancel_event left as default (None)
        )
        pdf = Path("/tmp/__grobid_no_cancel__.pdf")
        if not pdf.exists():
            pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
        try:
            r = c.process_pdf(pdf, Path("/tmp/__grobid_no_cancel_out__"))
            assert r.success is False
            assert r.retry_count == 3
            assert call_count["n"] == 3
        finally:
            pdf.unlink(missing_ok=True)
    finally:
        grobid_mod.requests.post = original_post
