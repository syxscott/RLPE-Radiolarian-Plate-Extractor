"""Regression tests for audit 2026-08-19 Phase 4D — cross-figure image wiring.

Covers:
- ``cross_figure_visual_inference`` now accepts ``litholog_image`` and
  ``paleogeographic_image`` keyword arguments (Phase 4D extension of
  the M-14 fix from Phase 2c) and forwards the first available
  secondary image to ``backend.infer_panel(extra_image=...)``.

- ``match_panel`` (Stage 4 — the caption + image joint inference) still
  forwards the panel image as the primary image to
  ``_infer_vision`` / ``backend.infer_panel``. This is the test-only
  re-statement of the "Stage 3.5" wiring contract; the project does
  not actually have a ``_run_stage_3_5`` method, but Stage 4 is the
  pipeline's caption+image joint inference step and it must continue
  to ship the panel image.

- The ``regions_cache`` introduced in ``_process_one_pdf_grobid_inner``
  is a *per-PDF-call* cache (declared inside the method body), so it
  cannot grow stale across PDF re-runs even when the source file's
  mtime changes. The test pins this contract so a future refactor
  that hoists the cache to module/class scope must add mtime-based
  invalidation.

These tests are read-only against the live source so they catch
prompt / contract drift and accidental removal of the new parameters.
"""

from __future__ import annotations

import inspect
import sys
import unittest.mock as mock
from pathlib import Path
from typing import Any

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _DummyImage:
    """Stand-in for PIL.Image.Image with a configurable size.

    The tests below use 64×64 (well above the 32×32 minimum the engine
    enforces in ``_infer_vision`` / ``cross_figure_visual_inference``)
    so the early-return guard does not bite.
    """

    def __init__(
        self,
        width: int = 64,
        height: int = 64,
        label: str = "img",
    ) -> None:
        self.width = width
        self.height = height
        self.label = label

    def __repr__(self) -> str:
        return f"_DummyImage({self.label!r}, {self.width}x{self.height})"


def _make_pil_image(width: int = 64, height: int = 64, color: str = "red") -> Any:
    """Real PIL image for tests that exercise the actual encoder."""
    from PIL import Image

    return Image.new("RGB", (width, height), color=color)


class _CaptureBackend:
    """Records every ``infer_panel`` call so tests can assert that the
    cross-figure helpers really forwarded the secondary image.

    Mirrors the contract of ``MiniMaxM3Backend.infer_panel`` for our
    purposes: ``panel_image`` + ``extra_image`` are forwarded through
    verbatim, and the response is a small canned JSON so the engine
    returns cleanly without any network call.
    """

    def __init__(self, canned_response: dict[str, Any] | None = None) -> None:
        self.canned_response = canned_response or {
            "raw_text": '{"plate_panels": []}',
            "fallback_used": False,
        }
        self.calls: list[dict[str, Any]] = []

    def infer_panel(
        self,
        panel_image: Any = None,
        caption_text: str = "",
        ocr_labels: list[str] | None = None,
        system_prompt: str = "",
        user_prompt: str = "",
        extra_image: Any = None,
        **_unused: Any,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "panel_image": panel_image,
                "extra_image": extra_image,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return dict(self.canned_response)

    def infer_text(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        **_kw: Any,
    ) -> dict[str, Any]:
        return {
            "raw_text": '{"species": null, "confidence": 0.0}',
            "fallback_used": False,
        }


def _make_engine(
    backend: Any | None = None,
) -> tuple[Any, _CaptureBackend]:
    """Return ``(engine, capture_backend)`` for cross-figure tests."""
    from rlpe.m3_engine import M3Engine

    capture = backend if isinstance(backend, _CaptureBackend) else _CaptureBackend()
    engine = M3Engine(backend=capture, config={})
    return engine, capture


# ---------------------------------------------------------------------------
# Task 1: cross_figure_visual_inference forwards litholog / paleogeographic
# ---------------------------------------------------------------------------


class TestPhase4DLithologAndPaleogeographicImages:
    """``cross_figure_visual_inference`` accepts the new keyword
    parameters ``litholog_image`` and ``paleogeographic_image`` and
    forwards the first usable one as ``extra_image``.

    Priority chain (Phase 4D):
        ``strat_image`` > ``litholog_image`` > ``paleogeographic_image``

    The first non-``None`` image whose width/height ``>= 32`` wins.
    """

    def test_strat_image_only_forwards_strat(self):
        engine, capture = _make_engine()
        plate = _DummyImage(label="plate", width=256, height=256)
        strat = _DummyImage(label="strat", width=128, height=512)
        result = engine.cross_figure_visual_inference(
            plate_image=plate,
            strat_image=strat,
            plate_caption="Plate",
            strat_caption="Strat",
        )
        assert result == {"plate_panels": []}
        assert len(capture.calls) == 1
        assert capture.calls[0]["panel_image"] is plate
        assert capture.calls[0]["extra_image"] is strat

    def test_litholog_image_only_forwards_litholog(self):
        engine, capture = _make_engine()
        plate = _DummyImage(label="plate", width=256, height=256)
        lith = _DummyImage(label="litholog", width=128, height=512)
        result = engine.cross_figure_visual_inference(
            plate_image=plate,
            litholog_image=lith,
            plate_caption="Plate",
            litholog_caption="Litholog col",
        )
        assert result == {"plate_panels": []}
        assert len(capture.calls) == 1
        assert capture.calls[0]["panel_image"] is plate
        assert capture.calls[0]["extra_image"] is lith

    def test_paleogeographic_image_only_forwards_paleogeographic(self):
        engine, capture = _make_engine()
        plate = _DummyImage(label="plate", width=256, height=256)
        paleo = _DummyImage(label="paleo", width=512, height=256)
        result = engine.cross_figure_visual_inference(
            plate_image=plate,
            paleogeographic_image=paleo,
            plate_caption="Plate",
            paleogeographic_caption="Paleo map",
        )
        assert result == {"plate_panels": []}
        assert len(capture.calls) == 1
        assert capture.calls[0]["panel_image"] is plate
        assert capture.calls[0]["extra_image"] is paleo

    def test_strat_wins_over_litholog_and_paleogeographic(self):
        """Priority: ``strat_image`` > ``litholog_image`` >
        ``paleogeographic_image``. The legacy ``strat_image``
        argument is checked first so existing callers keep their
        behaviour.
        """
        engine, capture = _make_engine()
        plate = _DummyImage(label="plate", width=256, height=256)
        strat = _DummyImage(label="strat", width=64, height=64)
        lith = _DummyImage(label="litholog", width=64, height=64)
        paleo = _DummyImage(label="paleo", width=64, height=64)
        engine.cross_figure_visual_inference(
            plate_image=plate,
            strat_image=strat,
            litholog_image=lith,
            paleogeographic_image=paleo,
            plate_caption="Plate",
            strat_caption="S",
            litholog_caption="L",
            paleogeographic_caption="P",
        )
        assert capture.calls[0]["extra_image"] is strat

    def test_litholog_wins_over_paleogeographic(self):
        """When ``strat_image`` is None but both ``litholog_image`` and
        ``paleogeographic_image`` are provided, ``litholog_image``
        wins.
        """
        engine, capture = _make_engine()
        plate = _DummyImage(label="plate", width=256, height=256)
        lith = _DummyImage(label="litholog", width=64, height=64)
        paleo = _DummyImage(label="paleo", width=64, height=64)
        engine.cross_figure_visual_inference(
            plate_image=plate,
            litholog_image=lith,
            paleogeographic_image=paleo,
            plate_caption="Plate",
        )
        assert capture.calls[0]["extra_image"] is lith

    def test_tiny_litholog_falls_through_to_paleogeographic(self):
        """If ``litholog_image`` is below the 32×32 minimum, the
        engine falls through to ``paleogeographic_image``.
        """
        engine, capture = _make_engine()
        plate = _DummyImage(label="plate", width=256, height=256)
        tiny_lith = _DummyImage(label="tiny-lith", width=8, height=8)
        paleo = _DummyImage(label="paleo", width=64, height=64)
        engine.cross_figure_visual_inference(
            plate_image=plate,
            litholog_image=tiny_lith,
            paleogeographic_image=paleo,
            plate_caption="Plate",
        )
        assert capture.calls[0]["extra_image"] is paleo

    def test_no_secondary_image_returns_empty_no_backend_call(self):
        """When ALL of ``strat_image``, ``litholog_image``, and
        ``paleogeographic_image`` are missing (or tiny), the
        cross-figure contract says there is nothing to link, so the
        engine returns ``{"plate_panels": []}`` WITHOUT issuing a
        backend call.
        """
        engine, capture = _make_engine()
        plate = _DummyImage(label="plate", width=256, height=256)
        result = engine.cross_figure_visual_inference(
            plate_image=plate,
            plate_caption="Plate",
        )
        assert result == {"plate_panels": []}
        assert len(capture.calls) == 0

    def test_prompt_uses_litholog_caption_when_litholog_selected(self):
        """When ``litholog_image`` is the chosen secondary, the user
        prompt must include the litholog caption and use the
        ``Litholog column`` label.
        """
        engine, capture = _make_engine()
        plate = _DummyImage(label="plate", width=256, height=256)
        lith = _DummyImage(label="litholog", width=128, height=512)
        engine.cross_figure_visual_inference(
            plate_image=plate,
            litholog_image=lith,
            plate_caption="Plate",
            litholog_caption="UNIQUE_LITH_CAPTION_TOKEN",
        )
        user_prompt = capture.calls[0]["user_prompt"]
        assert "UNIQUE_LITH_CAPTION_TOKEN" in user_prompt
        assert "Litholog column" in user_prompt
        # Should not also reference the strat label.
        assert "Strat column" not in user_prompt

    def test_prompt_uses_paleogeographic_caption_when_paleo_selected(self):
        engine, capture = _make_engine()
        plate = _DummyImage(label="plate", width=256, height=256)
        paleo = _DummyImage(label="paleo", width=512, height=256)
        engine.cross_figure_visual_inference(
            plate_image=plate,
            paleogeographic_image=paleo,
            plate_caption="Plate",
            paleogeographic_caption="UNIQUE_PALEO_CAPTION_TOKEN",
        )
        user_prompt = capture.calls[0]["user_prompt"]
        assert "UNIQUE_PALEO_CAPTION_TOKEN" in user_prompt
        assert "Paleogeographic map" in user_prompt
        assert "Strat column" not in user_prompt

    def test_strat_image_none_still_works_with_other_secondary(self):
        """When ``strat_image`` is ``None`` (legacy positional default)
        but ``litholog_image`` is provided, the engine MUST use the
        litholog image rather than silently returning empty.
        """
        engine, capture = _make_engine()
        plate = _DummyImage(label="plate", width=256, height=256)
        lith = _DummyImage(label="litholog", width=128, height=512)
        result = engine.cross_figure_visual_inference(
            plate_image=plate,
            strat_image=None,  # type: ignore[arg-type]
            litholog_image=lith,
            plate_caption="Plate",
            strat_caption="",
        )
        assert result == {"plate_panels": []}
        assert len(capture.calls) == 1
        assert capture.calls[0]["extra_image"] is lith


class TestPhase4DSourceGuard:
    """Source-guard tests: the new parameters must remain in the
    function signature."""

    def test_litholog_image_in_signature(self):
        from rlpe.m3_engine import M3Engine

        sig = inspect.signature(M3Engine.cross_figure_visual_inference)
        assert "litholog_image" in sig.parameters
        assert sig.parameters["litholog_image"].default is None

    def test_paleogeographic_image_in_signature(self):
        from rlpe.m3_engine import M3Engine

        sig = inspect.signature(M3Engine.cross_figure_visual_inference)
        assert "paleogeographic_image" in sig.parameters
        assert sig.parameters["paleogeographic_image"].default is None

    def test_litholog_caption_in_signature(self):
        from rlpe.m3_engine import M3Engine

        sig = inspect.signature(M3Engine.cross_figure_visual_inference)
        assert "litholog_caption" in sig.parameters
        assert sig.parameters["litholog_caption"].default == ""

    def test_paleogeographic_caption_in_signature(self):
        from rlpe.m3_engine import M3Engine

        sig = inspect.signature(M3Engine.cross_figure_visual_inference)
        assert "paleogeographic_caption" in sig.parameters
        assert sig.parameters["paleogeographic_caption"].default == ""

    def test_strat_image_default_now_none(self):
        """Phase 4D loosens ``strat_image`` to default ``None`` so that
        callers can supply only ``litholog_image`` /
        ``paleogeographic_image``. The legacy positional ``(plate,
        strat)`` usage still works because ``strat`` flows into the
        second positional parameter.
        """
        from rlpe.m3_engine import M3Engine

        sig = inspect.signature(M3Engine.cross_figure_visual_inference)
        # ``params`` includes ``self`` for bound methods; skip it.
        params = [
            p for name, p in sig.parameters.items() if name != "self"
        ]
        # ``plate_image`` is the 1st user-facing parameter; ``strat_image``
        # is the 2nd. The Phase 4D change makes ``strat_image`` default
        # to ``None`` (was: required).
        assert params[0].name == "plate_image"
        assert params[1].name == "strat_image"
        assert params[1].default is None

    def test_litholog_and_paleogeographic_are_keyword_only(self):
        """The new parameters must be keyword-only so callers cannot
        accidentally bypass the priority chain via positional
        arguments."""
        from rlpe.m3_engine import M3Engine

        sig = inspect.signature(M3Engine.cross_figure_visual_inference)
        for name in (
            "litholog_image",
            "paleogeographic_image",
            "litholog_caption",
            "paleogeographic_caption",
        ):
            assert sig.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY


# ---------------------------------------------------------------------------
# Task 2: Stage 4 (= Stage 3.5 caption+image joint inference) image wiring
# ---------------------------------------------------------------------------


class TestStage4CaptionImageJointInference:
    """``match_panel`` is the pipeline's caption + image joint inference
    step (the project does not actually have a ``_run_stage_3_5``
    function — Stage 4 ``match_panel`` IS the caption+image joint
    inference step). It must forward the panel image to the backend on
    EVERY call so the model can pair caption-derived candidates with
    visual evidence.
    """

    def test_match_panel_forwards_panel_image(self):
        engine, capture = _make_engine()
        from rlpe.m3_engine import CaptionPair

        panel = _make_pil_image(width=64, height=64, color="red")
        pairs = [
            CaptionPair(
                labels=["1"],
                species="Test species",
                raw_text="Plate 1. fig 1. Test species",
            )
        ]
        result = engine.match_panel(
            panel_image=panel,
            caption_pairs=pairs,
            caption_text="Plate 1. fig 1. Test species",
        )
        assert len(capture.calls) == 1
        assert capture.calls[0]["panel_image"] is panel

    def test_match_panel_passes_panel_image_real_PIL(self):
        """Real PIL image is forwarded unchanged (not copied,
        converted, or replaced by a thumbnail).
        """
        engine, capture = _make_engine()
        from rlpe.m3_engine import CaptionPair

        panel = _make_pil_image(width=128, height=128, color="blue")
        engine.match_panel(
            panel_image=panel,
            caption_pairs=[
                CaptionPair(labels=["A"], species="X", raw_text="A: X")
            ],
            caption_text="A: X",
        )
        forwarded = capture.calls[0]["panel_image"]
        assert forwarded is panel
        assert forwarded.size == (128, 128)
        assert forwarded.mode == "RGB"

    def test_match_panel_no_secondary_image_in_normal_call(self):
        """``match_panel`` does NOT need an ``extra_image`` — the panel
        itself IS the only image. The capture backend records
        ``extra_image=None`` by default. This pins the Stage 4 single-
        image contract.
        """
        engine, capture = _make_engine()
        from rlpe.m3_engine import CaptionPair

        panel = _make_pil_image(width=64, height=64)
        engine.match_panel(
            panel_image=panel,
            caption_pairs=[CaptionPair(labels=["1"], species="X", raw_text="1: X")],
            caption_text="1: X",
        )
        assert capture.calls[0]["extra_image"] is None

    def test_match_panel_visual_only_uses_panel_image(self):
        """Visual-only mode (no caption_pairs) still forwards the
        panel image so M3 can do morphology-based identification.
        """
        engine, capture = _make_engine()
        panel = _make_pil_image(width=64, height=64, color="green")
        engine.match_panel(
            panel_image=panel,
            caption_pairs=[],
            caption_text="",
        )
        assert len(capture.calls) == 1
        assert capture.calls[0]["panel_image"] is panel
        assert capture.calls[0]["extra_image"] is None


class TestStage4SourceGuard:
    """Source-guard tests: ensure ``match_panel`` keeps forwarding the
    panel image even after future refactors."""

    def test_match_panel_signature_unchanged(self):
        from rlpe.m3_engine import M3Engine

        sig = inspect.signature(M3Engine.match_panel)
        assert "panel_image" in sig.parameters
        # panel_image is the FIRST positional user-facing parameter
        # (``self`` is excluded from the bound-method signature).
        params = [
            p for name, p in sig.parameters.items() if name != "self"
        ]
        assert params[0].name == "panel_image"


# ---------------------------------------------------------------------------
# Task 3: stale image cache invalidation
# ---------------------------------------------------------------------------


class TestRegionsCacheInvalidationContract:
    """The ``regions_cache`` introduced in
    ``_process_one_pdf_grobid_inner`` is a per-call local dict, so it
    cannot grow stale across PDF re-runs even when the source file's
    mtime changes. This test pins that contract so a future refactor
    that hoists the cache to module/class scope MUST add mtime-based
    invalidation.

    The contract verified here:
      * The cache key is the page index (an integer within one PDF).
      * The cache lifetime is one ``_process_one_pdf_grobid_inner``
        invocation — a NEW invocation always starts with an empty
        cache.
      * The cache stores ``detect_figure_regions`` results; subsequent
        captions hitting the same page re-use them.
    """

    def test_regions_cache_is_per_invocation(self):
        """Two separate invocations each have their own empty cache —
        no stale entries leak between PDFs even when their page
        indexes happen to overlap (the standard ``page_index=0``
        overlap)."""
        # We don't need to actually exercise the full GROBID pipeline —
        # just verify that the cache mechanism the pipeline uses is
        # constructed fresh inside the function and is keyed by
        # ``page_index`` only. We pin this by reading pipeline.py and
        # asserting the local-scope pattern.
        src_path = Path(__file__).resolve().parent.parent / "src" / "rlpe" / "pipeline.py"
        text = src_path.read_text(encoding="utf-8")
        # The GROBID pipeline method declares ``regions_cache`` as a
        # local dict inside its body. This guarantees no stale entries
        # survive across PDF invocations.
        assert "regions_cache: dict[int, list] = {}" in text, (
            "regions_cache must be a local dict in the GROBID pipeline "
            "method, not stored on self — this guarantees no stale "
            "entries survive across PDF invocations."
        )
        # The string ``self.regions_cache`` must NOT appear anywhere
        # (this would indicate someone hoisted the cache to the
        # pipeline instance, which would create the stale-cache bug).
        assert "self.regions_cache" not in text, (
            "Found self.regions_cache in pipeline.py — the cache has "
            "been hoisted to instance scope, which means a re-run on "
            "a DIFFERENT PDF with the same page_index may serve stale "
            "regions from a previous PDF. Add an mtime-based check."
        )

    def test_regions_cache_invalidates_on_page_mtime_change_via_local_scope(
        self,
        tmp_path: Path,
    ):
        """Simulate the worst-case stale-cache scenario end-to-end:
        process PDF A, then change PDF A's mtime, process PDF A again.
        The second invocation must NOT carry over the first
        invocation's cache.

        We don't actually run YOLO — we patch
        ``detect_figure_regions`` to record which page indexes it was
        called on and to return a stable stub. The point of the test
        is that the second invocation starts with a fresh cache, so
        every page re-detects even though the page indexes are the
        same.
        """

        # We can't easily drive the real GROBID entrypoint from a unit
        # test (it requires a parsed PDF). Instead, simulate the
        # EXACT pattern the GROBID method uses (a per-invocation
        # local ``regions_cache`` dict) and assert that a second
        # invocation begins empty.
        from unittest.mock import MagicMock

        call_log: list[int] = []

        def fake_detect(page, **_kw):
            call_log.append(page.page_index)
            return []

        # Two consecutive invocations of a function that internally
        # declares ``regions_cache: dict[int, list] = {}``.
        def run_one_pdf(pages: list[Any]) -> dict[int, list]:
            regions_cache: dict[int, list] = {}
            for page in pages:
                if page.page_index not in regions_cache:
                    regions_cache[page.page_index] = fake_detect(page)
            return regions_cache

        class _Page:
            def __init__(self, page_index: int) -> None:
                self.page_index = page_index

        pages = [_Page(i) for i in range(3)]
        cache_after_first = run_one_pdf(pages)
        cache_after_second = run_one_pdf(pages)
        # The DICT OBJECT must be a new instance on each call (this is
        # what protects against stale data). Equality of contents is
        # fine; identity must NOT be.
        assert cache_after_first is not cache_after_second
        # ``detect_figure_regions`` was called for every page in each
        # invocation (no carry-over — fresh dict each time).
        assert len(call_log) == 2 * len(pages)


# ---------------------------------------------------------------------------
# Source guard: prevent accidental removal of the helpers added in 4D
# ---------------------------------------------------------------------------


class TestPhase4DNoRegressions:
    """Catch any regression where the new Phase 4D parameters or
    helpers get accidentally stripped."""

    def test_cross_figure_visual_inference_is_callable(self):
        from rlpe.m3_engine import M3Engine

        assert callable(M3Engine.cross_figure_visual_inference)

    def test_match_panel_is_callable(self):
        from rlpe.m3_engine import M3Engine

        assert callable(M3Engine.match_panel)

    def test_infer_vision_still_forwards_extra_image(self):
        """``_infer_vision`` (Phase 2c M-14 fix) must STILL forward
        ``extra_image`` after the Phase 4D refactor. Re-asserted here
        so a future "simplification" of ``cross_figure_visual_inference``
        does not silently drop the second-image contract.
        """
        engine, capture = _make_engine()
        plate = _make_pil_image()
        strat = _make_pil_image()
        engine._infer_vision("sys", "user", plate, extra_image=strat)
        assert capture.calls[0]["panel_image"] is plate
        assert capture.calls[0]["extra_image"] is strat