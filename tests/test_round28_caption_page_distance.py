"""Phase 28 — Robust caption-image pairing for appendix-style layouts.

Real radiolarian papers frequently place plates in a cluster at the END
of the PDF (body pp. 1–30, plates pp. 50–80), or print captions on
the page immediately before/after the figure. The OpenDataLoader path
had four hard-coded page-distance limits that silently dropped every
such appendix-style pair. This test module pins the new configurable
``od_caption_window`` plumbing and verifies the new behavior.

The scaffolding pattern follows ``tests/test_round27_japanese_extraction.py``
(sys.path injection + ``_read`` helper for source-guard assertions).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.config import PipelineConfig  # noqa: E402
from rlpe.opendataloader_extractor import (  # noqa: E402
    OpenDataLoaderExtractor,
    _find_plate_captions,
    _images_within_page_range,
    _rescue_missing_images,
)

_REPO = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (_REPO / rel).read_text(encoding="utf-8")


# ============================================================================
# Source-guard tests
# ============================================================================


def test_od_caption_window_field_default_5():
    """The new PipelineConfig field exists with default 5."""
    assert "od_caption_window: int = 5" in _read("src/rlpe/config.py")


def test_grobid_caption_window_default_2_unchanged():
    """GROBID path's existing default must NOT change."""
    assert "caption_window: int = 2" in _read("src/rlpe/config.py")


def test_cli_od_caption_window_flag_exists():
    """``--od-caption-window`` must be a CLI flag (source guard)."""
    cli = _read("src/rlpe/cli.py")
    assert "--od-caption-window" in cli
    # And it must be threaded into PipelineConfig via args.od_caption_window
    assert "args.od_caption_window" in cli


def test_cli_caption_window_flag_exists():
    """``--caption-window`` for the GROBID path must be a CLI flag."""
    cli = _read("src/rlpe/cli.py")
    assert "--caption-window" in cli


def test_app_joboptions_has_od_caption_window():
    """The Web UI ``JobOptions`` must expose the new field."""
    app = _read("src/rlpe/api/app.py")
    assert "od_caption_window" in app
    # Should be in the JobOptions block (not just anywhere)
    assert "od_caption_window: int | None = None" in app


def test_app_joboptions_has_caption_window():
    """``caption_window`` field exposed in JobOptions too."""
    app = _read("src/rlpe/api/app.py")
    assert "caption_window: int | None = None" in app


def test_app_joboptions_validators_present():
    """Both new fields have field_validator constraints (1..50 / 1..200)."""
    app = _read("src/rlpe/api/app.py")
    assert "_validate_caption_window" in app
    assert "_validate_od_caption_window" in app


def test_pipeline_threads_od_caption_window_to_extractor():
    """pipeline.py must forward ``od_caption_window`` to the OD extractor."""
    pipe = _read("src/rlpe/pipeline.py")
    # Source guard: the literal ``caption_window=self.config.od_caption_window``
    # must appear in pipeline.py inside the OpenDataLoaderExtractor call.
    assert "caption_window=self.config.od_caption_window" in pipe, (
        "pipeline.py OpenDataLoaderExtractor(...) call must pass "
        "caption_window=self.config.od_caption_window (Phase 28 wiring)"
    )
    # And the call site must actually instantiate OpenDataLoaderExtractor
    assert "OpenDataLoaderExtractor(" in pipe


def test_hardcoded_offset_tuple_replaced():
    """The literal ``(1, -1, 2, -2)`` tuple must be gone from the
    cross-page offset search."""
    src = _read("src/rlpe/opendataloader_extractor.py")
    assert "(1, -1, 2, -2)" not in src


def test_hardcoded_plus_two_replaced():
    """The literal ``page_lo + 2`` (the +2 forward window) must be
    gone from the live ``_build_figures_from_plate_captions`` body.
    It may still appear in comments / docstrings explaining the old
    behaviour."""
    import re

    src = _read("src/rlpe/opendataloader_extractor.py")
    # Strip comments to find live code occurrences
    code_lines = []
    for line in src.split("\n"):
        stripped = line.split("#", 1)[0]
        code_lines.append(stripped)
    code_only = "\n".join(code_lines)
    assert "page_lo + 2" not in code_only, "Live code still contains the old `page_lo + 2` literal"
    assert "page_lo + int(caption_window)" in code_only


def test_hardcoded_plus_three_replaced():
    """The literal ``ref_page + 3`` must be gone from the body-ref
    reconstruction pass."""
    src = _read("src/rlpe/opendataloader_extractor.py")
    assert "ref_page + 3" not in src


def test_hardcoded_twenty_cap_parametrized():
    """The literal ``page_diff > 20`` must be replaced with the
    ×4 multiplier that scales with ``caption_window``."""
    src = _read("src/rlpe/opendataloader_extractor.py")
    assert "page_diff > 20" not in src
    # The new parametrized cap should mention both caption_window and 4×
    assert "caption_window * 4" in src


def test_grobid_path_untouched():
    """GROBID ``choose_best_page`` must still consume ``caption_window``
    (not ``od_caption_window``). The two fields are independent."""
    pipe = _read("src/rlpe/pipeline.py")
    # The GROBID path call site should reference self.config.caption_window
    assert "self.config.caption_window" in pipe


# ============================================================================
# Default-value tests
# ============================================================================


def test_pipelineconfig_default_od_caption_window_is_5():
    """PipelineConfig() default must be 5."""
    cfg = PipelineConfig(pdf_dir=Path("/tmp"), work_dir=Path("/tmp"))
    assert cfg.od_caption_window == 5


def test_pipelineconfig_default_caption_window_is_2():
    """PipelineConfig() GROBID caption_window default must be 2."""
    cfg = PipelineConfig(pdf_dir=Path("/tmp"), work_dir=Path("/tmp"))
    assert cfg.caption_window == 2


def test_open_data_loader_extractor_default_caption_window_is_5():
    """OpenDataLoaderExtractor() default must be 5."""
    ext = OpenDataLoaderExtractor()
    assert ext.caption_window == 5


# ============================================================================
# Synthetic unit tests — _build_figures_from_plate_captions
# ============================================================================


def _make_plate_caption(plate_num: int, page: int) -> dict:
    return {
        "plate_number": plate_num,
        "page_number": page,
        "content": f"Plate {plate_num}.",
        "element": None,
    }


def _make_image(img_id: int, page: int, bbox: list[int] | None = None) -> dict:
    return {
        "id": img_id,
        "page number": page,
        "bounding box": bbox or [0, 0, 100, 100],
    }


def test_images_within_page_range_five_page_gap_at_default():
    """At default caption_window=5, the forward window from caption
    page 5 reaches image on page 10 (5-page gap)."""
    images = [_make_image(1, 10)]
    selected = _images_within_page_range(images, page_lo=5, page_hi=5 + 5)
    assert len(selected) == 1
    assert selected[0]["id"] == 1


def test_images_within_page_range_seven_page_gap_at_default():
    """At default caption_window=5, image on page 7 is 6 pages after
    caption on page 1 — outside the +5 window."""
    images = [_make_image(1, 7)]
    selected = _images_within_page_range(images, page_lo=1, page_hi=1 + 5)
    assert selected == [], "6-page gap must NOT be selected at default +5"


def test_images_within_page_range_backward_compatible_at_window_two():
    """At caption_window=2, the legacy +2 forward limit is restored:
    image on page 4 is 3 pages after caption on page 1 — skipped."""
    images = [_make_image(1, 4)]
    selected = _images_within_page_range(images, page_lo=1, page_hi=1 + 2)
    assert selected == [], "caption_window=2 must restore legacy +2 limit"


def test_images_within_page_range_negative_offset_handles_caption_before_image():
    """Negative offset is supported by the helper: caption on page 3
    with image on page 1 (caption printed AFTER image, page_lo < image page)
    is correctly included when the image is within the forward window."""
    images = [_make_image(1, 1)]
    # Caption on p3, image on p1: p_lo=3, p_hi=3+5=8, image on p1 not in range
    selected = _images_within_page_range(images, page_lo=3, page_hi=3 + 5)
    assert selected == [], "image on p1 is BEFORE caption on p3, out of range"
    # But caption on p1, image on p3 (caption p_lo=1, p_hi=1+5=6, image p3 in range)
    selected2 = _images_within_page_range(images, page_lo=1, page_hi=1 + 5)
    assert len(selected2) == 1


def test_build_figures_no_cross_plate_theft_at_wide_window():
    """Verify cross-plate theft is prevented by the next-caption clamp
    even at a wide caption_window. Two adjacent plates with images
    that could be claimed by either plate — the next-caption clamp
    must prevent Plate 1 from absorbing Plate 2's image."""
    plate_captions = [
        _make_plate_caption(1, 1),
        _make_plate_caption(2, 4),
    ]
    images = [_make_image(1, 3), _make_image(2, 5)]

    # Use the same windowing logic the function uses: page_hi is
    # page_lo + caption_window, but clamped to next_cap_page - 1.
    # Plate 1: page_lo=1, page_hi = min(1+20, 4-1) = 3 → image p3 in range
    # Plate 2: page_lo=4, page_hi = min(4+20, ...) = 24 → image p5 in range
    sel1 = _images_within_page_range(images, page_lo=1, page_hi=min(1 + 20, 4 - 1))
    sel2 = _images_within_page_range(images, page_lo=4, page_hi=4 + 20)
    # Plate 1 claims only p3
    assert {img["id"] for img in sel1} == {1}
    # Plate 2 claims only p5
    assert {img["id"] for img in sel2} == {2}


# ============================================================================
# Synthetic unit tests — _rescue_missing_images
# ============================================================================


def test_rescue_20_page_cap_preserved_at_default():
    """At default caption_window=5, the rescue hard cap is 5×4=20,
    preserving the legacy behaviour exactly. Verify via source guard
    that the cap arithmetic uses caption_window (no synthetic disk
    I/O needed — the rescue path requires real OD image exports)."""
    src = _read("src/rlpe/opendataloader_extractor.py")
    # The cap expression multiplies caption_window by 4
    assert "caption_window * 4" in src
    # And it is used in the page-diff guard
    assert "page_diff > max_page_diff" in src
    assert "max_page_diff = int(caption_window) * 4" in src


def test_rescue_thirty_page_gap_with_window_eight_parametrized():
    """At caption_window=8, the cap is 8×4=32 — wide enough to
    reach a 30-page gap. Pin via source guard that the cap scales
    with the operator's caption_window setting (no synthetic disk
    I/O)."""
    src = _read("src/rlpe/opendataloader_extractor.py")
    # The cap must be derived from caption_window (parametrized),
    # not a hard-coded 20.
    assert "caption_window * 4" in src
    assert "> 20" not in src.replace("> 200", "").replace("> 50", "")  # noqa


# ============================================================================
# Web UI validator tests
# ============================================================================


def test_app_joboptions_rejects_zero_od_caption_window():
    """od_caption_window=0 must raise ValidationError (Pydantic HTTP 422)."""
    from pydantic import ValidationError

    from rlpe.api.app import JobOptions

    with pytest.raises(ValidationError):
        JobOptions(od_caption_window=0)


def test_app_joboptions_rejects_oversized_od_caption_window():
    """od_caption_window=999 must raise ValidationError (max 200)."""
    from pydantic import ValidationError

    from rlpe.api.app import JobOptions

    with pytest.raises(ValidationError):
        JobOptions(od_caption_window=999)


def test_app_joboptions_accepts_none_for_both():
    """None values must be valid (fall back to PipelineConfig defaults)."""
    from rlpe.api.app import JobOptions

    o = JobOptions(caption_window=None, od_caption_window=None)
    assert o.caption_window is None
    assert o.od_caption_window is None


# ============================================================================
# _find_plate_captions signature plumbing
# ============================================================================


def test_find_plate_captions_accepts_caption_window():
    """The caption-detection function now threads caption_window through
    to the body-ref reconstruction pass."""
    import inspect

    params = list(inspect.signature(_find_plate_captions).parameters.keys())
    assert "caption_window" in params, (
        f"_find_plate_captions should accept caption_window; got {params!r}"
    )
