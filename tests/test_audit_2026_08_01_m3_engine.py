"""Regression tests for audit 2026-08-01 batch W5 — m3_engine.py 4 bugs (M8/M9/M10/D2)."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from rlpe.m3_engine import M3Engine, _redact_enrichment_caption  # noqa: E402
from tests.fakes.fake_m3_backend import FakeM3Backend  # noqa: E402


def _engine(raw_text, **config) -> M3Engine:
    """Engine wired to a fake backend that always answers with ``raw_text``."""
    backend = FakeM3Backend(canned_responses=[{"raw_text": raw_text, "fallback_used": False}])
    return M3Engine(backend=backend, config=config)


def _plate() -> Image.Image:
    """A plate image large enough to clear the <32px short-circuit."""
    return Image.new("RGB", (256, 256))


def _enrich(engine: M3Engine) -> list[dict]:
    return engine.enrich_plate_panels(
        image=_plate(),
        page_caption="Plate 7. Radiolarians from the Upper Jurassic.",
        paper_id="audit0801",
        figure_id="fig_plate7",
        expected_plate_label="Plate 7",
    )


class TestEnrichPlatePanelsErrorPath:
    """M8: unparseable M3 output must degrade to [], not raise.

    The four sibling methods (``extract_geology``, ``extract_schematic``,
    ``cross_figure_visual_inference``, ``infer_species_age_formation``)
    all wrap ``_safe_json_loads`` in try/except and return an empty
    result. ``enrich_plate_panels`` didn't, so a malformed response
    escaped to the pipeline and was logged as an opaque
    "multi_plate_enrich failed".
    """

    def test_malformed_json_returns_empty(self):
        engine = _engine("not JSON at all")
        out = _enrich(engine)
        assert out == [], f"malformed JSON must yield [], got {out!r}"

    def test_attribute_error_returns_empty(self):
        """A non-string ``raw_text`` hits ``text.strip()`` → AttributeError.

        ``m3_retry_without_thinking`` is off so the value reaches the
        JSON-parse site instead of the retry condition's own ``.strip()``.
        """
        engine = _engine(12345, m3_retry_without_thinking=False)
        out = _enrich(engine)
        assert out == [], f"non-string raw_text must yield [], got {out!r}"

    def test_none_raw_text_returns_empty(self):
        engine = _engine(None)
        assert _enrich(engine) == []

    def test_malformed_json_does_not_raise(self):
        """Explicitly pin the no-exception contract callers rely on."""
        engine = _engine("```json\n{panels: [oops}\n```")
        try:
            out = _enrich(engine)
        except Exception as exc:  # pragma: no cover - the bug being fixed
            raise AssertionError(
                f"enrich_plate_panels must not propagate parse errors: {exc!r}"
            ) from exc
        assert out == []


class TestEnrichPlatePanelsListWrapped:
    """M10: ``_safe_json_loads`` can return a list-wrapped object."""

    def test_list_wrapped_response_unwrapped(self):
        engine = _engine(
            '[{"panels": [{"label": "1", "species": "Archaeodictyomitra apiarium",'
            ' "confidence": 0.9}, {"label": "2", "species": "Pseudodictyomitra'
            ' carpatica", "confidence": 0.8}]}]'
        )
        out = _enrich(engine)
        assert [p["label"] for p in out] == ["1", "2"], (
            f"list-wrapped {{'panels': [...]}} must be unwrapped, got {out!r}"
        )
        assert out[0]["species"] == "Archaeodictyomitra apiarium"
        assert out[1]["confidence"] == 0.8

    def test_bare_list_of_panel_dicts(self):
        engine = _engine('[{"label": "1", "species": "X"}]')
        out = _enrich(engine)
        assert len(out) == 1
        assert out[0]["label"] == "1"
        assert out[0]["species"] == "X"
        assert out[0]["confidence"] == 0.7, "missing confidence defaults to 0.7"

    def test_list_wrapped_with_empty_panels(self):
        engine = _engine('[{"panels": []}]')
        assert _enrich(engine) == []

    def test_list_wrapped_with_non_list_panels(self):
        """``{"panels": "oops"}`` must not be iterated as a panel array."""
        engine = _engine('[{"panels": "oops"}]')
        assert _enrich(engine) == []

    def test_plain_object_still_works(self):
        """The happy path (model contract) must be untouched by the unwrap."""
        engine = _engine('{"panels": [{"label": "3", "species": "Y", "confidence": 0.5}]}')
        out = _enrich(engine)
        assert out == [{"label": "3", "species": "Y", "confidence": 0.5}]


class TestRedactEnrichmentCaption:
    """M9: the helper must do what its docstring claims — keep the current
    plate's caption verbatim and cap the *other* plates' text."""

    PLATE1 = (
        "Plate 1. Radiolarians from the Upper Jurassic of Costa Rica. "
        "1, Archaeodictyomitra apiarium (Ruest); "
        "2, Pseudodictyomitra carpatica (Lozyniak)."
    )
    # Long enough that the 200-char unrelated budget must truncate it.
    PLATE5 = (
        "Plate 5. Radiolarians from the Lower Cretaceous of Nicoya. "
        + "".join(f"{i}, Genus species number {i} (Author); " for i in range(1, 21))
        + "ZZZTAILMARKER."
    )

    def _page(self) -> str:
        return f"{self.PLATE1}\n\n{self.PLATE5}"

    def test_redaction_removes_other_plates_captions(self):
        out = _redact_enrichment_caption(self._page(), self.PLATE1)
        # The current plate's caption survives verbatim — species labels
        # are exactly what the enrichment pass needs.
        assert self.PLATE1 in out, (
            f"the current plate's caption must be preserved verbatim; got {out!r}"
        )
        assert "Archaeodictyomitra apiarium" in out
        # The other plate on the page is cut down to the unrelated budget.
        assert "ZZZTAILMARKER" not in out, (
            "plate 5's caption must be truncated to the unrelated budget"
        )
        assert "redacted" in out, "truncation must be marked in the payload"
        assert len(out) < len(self._page())

    def test_current_plate_caption_survives_when_it_is_last(self):
        """Order must not matter: keep plate 1, cap plate 5's bulk.

        Note the budget is spent on the text *adjacent* to the kept
        section, so the tail of plate 5 (which abuts plate 1 here)
        legitimately survives — what must not survive is the bulk.
        """
        page = f"{self.PLATE5}\n\n{self.PLATE1}"
        out = _redact_enrichment_caption(page, self.PLATE1)
        assert self.PLATE1 in out
        assert "Plate 5. Radiolarians from the Lower Cretaceous" not in out, (
            "the other plate's caption head must be redacted away"
        )
        assert "Genus species number 1 (Author)" not in out
        assert len(out) < len(self.PLATE1) + 300

    def test_whitespace_drift_still_matches(self):
        """OD captions often re-wrap lines; the normalised path must hit."""
        page = self._page().replace(" ", "  ").replace("\n\n", "\n")
        out = _redact_enrichment_caption(page, self.PLATE1)
        assert "Archaeodictyomitra" in out
        assert "ZZZTAILMARKER" not in out

    def test_caption_not_found_hard_truncates(self):
        out = _redact_enrichment_caption(self._page(), "Plate 99. Nothing like this.")
        assert len(out) <= 200
        assert out == self._page()[:200]

    def test_empty_inputs(self):
        assert _redact_enrichment_caption("", "Plate 1.") == ""
        assert _redact_enrichment_caption(None, "Plate 1.") == ""
        page = self._page()
        assert _redact_enrichment_caption(page, None) == page[:200]
        assert _redact_enrichment_caption(page, "   ") == page[:200]

    def test_non_string_page_caption_does_not_raise(self):
        """M9 robustness: OD sometimes hands us a list of caption blocks."""
        assert _redact_enrichment_caption(["Plate 1."], "Plate 1.") == ""
        assert _redact_enrichment_caption(self._page(), 1234) == self._page()[:200]


class TestEnableThinkingThreadSafety:
    """D2: a worker's paid FIRST call must never run with thinking silently
    disabled just because another worker is inside its retry window."""

    class _RaceBackend:
        """Retry path flips ``enable_thinking`` off and holds it there;
        the ``fast`` workers record what the flag looked like when their
        own first call started."""

        backend_name = "fake"

        def __init__(self) -> None:
            self.enable_thinking = True
            self.first_call_thinking: list[bool] = []
            self.retry_started = threading.Event()
            self.retry_saw_thinking_off = False
            self._lock = threading.Lock()

        def infer_panel(self, *, user_prompt: str = "", **_kw):
            if user_prompt == "slow":
                if self.enable_thinking:
                    # First attempt: empty text → engine enters the retry path.
                    return {"raw_text": "", "fallback_used": False}
                # Retry attempt: the flag is flipped off for this window.
                self.retry_saw_thinking_off = True
                self.retry_started.set()
                time.sleep(0.2)
                return {"raw_text": "ok", "fallback_used": False}
            with self._lock:
                self.first_call_thinking.append(self.enable_thinking)
            return {"raw_text": "ok", "fallback_used": False}

    def test_first_call_reads_thinking_under_lock(self):
        backend = self._RaceBackend()
        engine = M3Engine(backend=backend, config={"m3_retry_without_thinking": True})

        def retry_worker():
            engine._infer_vision("sys", "slow", None)

        def fast_worker():
            # Only call once the retry window is definitely open — this is
            # exactly the interleaving that used to leak thinking=False
            # into another worker's first (paid) call.
            backend.retry_started.wait(timeout=5)
            engine._infer_vision("sys", "fast", None)

        threads = [threading.Thread(target=retry_worker)]
        threads += [threading.Thread(target=fast_worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        assert not any(t.is_alive() for t in threads), "workers deadlocked"

        # Sanity: the race window really was exercised.
        assert backend.retry_saw_thinking_off, "retry path never ran"
        assert backend.retry_started.is_set()
        assert len(backend.first_call_thinking) == 6

        assert all(backend.first_call_thinking), (
            "a first call observed enable_thinking=False while another "
            "worker was retrying — the paid call silently ran without "
            f"thinking: {backend.first_call_thinking}"
        )
        # And the flag is restored for everyone afterwards.
        assert backend.enable_thinking is True

    def test_infer_text_first_call_also_guarded(self):
        """``_infer_text`` mutates the same shared flag — same guarantee."""

        class _TextBackend(TestEnableThinkingThreadSafety._RaceBackend):
            def infer_text(self, *, user_prompt: str = "", **_kw):
                return self.infer_panel(user_prompt=user_prompt)

        backend = _TextBackend()
        engine = M3Engine(backend=backend, config={"m3_retry_without_thinking": True})

        threads = [threading.Thread(target=lambda: engine._infer_text("sys", "slow"))]
        threads += [
            threading.Thread(
                target=lambda: (
                    backend.retry_started.wait(timeout=5),
                    engine._infer_text("sys", "fast"),
                )
            )
            for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        assert not any(t.is_alive() for t in threads), "workers deadlocked"
        assert backend.retry_saw_thinking_off
        assert all(backend.first_call_thinking), backend.first_call_thinking
        assert backend.enable_thinking is True

    def test_gate_is_reentrant_for_the_writer(self):
        """A backend that re-enters the engine from inside its own retry
        handler must not deadlock (same rationale as the RLock choice)."""
        engine = M3Engine(backend=None)
        with engine._thinking_gate.write():
            with engine._thinking_gate.write():
                with engine._thinking_gate.read():
                    pass

    def test_concurrent_reads_are_not_serialised(self):
        """The gate must keep worker concurrency: many first calls at once."""
        barrier = threading.Barrier(4, timeout=5)
        engine = M3Engine(backend=None)
        errors: list[BaseException] = []

        def worker():
            try:
                with engine._thinking_gate.read():
                    barrier.wait()  # deadlocks if reads are exclusive
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert errors == [], f"read side is not shared: {errors}"
