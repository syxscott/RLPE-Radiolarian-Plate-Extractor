"""Phase 61 Plan 4 (Bug 4.5): Stage 3 bbox DPI bookkeeping.

Previously M3 Stage 3 returned ``bbox`` in pixels at the extraction DPI
(the DPI used to render the plate for the model). The crop helper
re-rendered / re-saved the crop at a different DPI (the visual storage
DPI, often the same but not guaranteed — operators sometimes override
render_dpi after running Stage 3). When the cropped image was fed back
to the LLM as a smaller-resolution panel, the bbox would not match the
pixels in the new image.

The fix adds ``stage3_rescale_bbox`` that takes (bbox, source_dpi,
crop_dpi) and returns a bbox scaled to the crop's pixel space, so the
caller can use it directly without manual arithmetic.
"""

from __future__ import annotations

import pytest

from rlpe.pipeline import stage3_rescale_bbox


def test_stage3_bbox_rescaled_on_crop():
    """When source_dpi != crop_dpi, bbox must scale proportionally."""
    # 200 dpi bbox at source_dpi=200, crop saved at crop_dpi=100
    # (half-resolution). Pixels should scale by 0.5.
    bbox = (100, 200, 50, 80)  # x, y, w, h
    out = stage3_rescale_bbox(bbox, source_dpi=200, crop_dpi=100)
    assert out == (50, 100, 25, 40)


def test_stage3_bbox_identity_when_dpi_match():
    """Same DPI → identity transform (no rescaling)."""
    bbox = (100, 200, 50, 80)
    out = stage3_rescale_bbox(bbox, source_dpi=200, crop_dpi=200)
    assert out == bbox


def test_stage3_bbox_invalid_dpi_safe():
    """Zero / negative DPI must not divide-by-zero."""
    bbox = (10, 20, 30, 40)
    # source_dpi=0 → falls back to bbox unchanged.
    out = stage3_rescale_bbox(bbox, source_dpi=0, crop_dpi=200)
    assert out == bbox


def test_stage3_bbox_rounded_to_int():
    """Output bbox is always a 4-tuple of ints (pixel coords)."""
    bbox = (100, 100, 33, 33)
    out = stage3_rescale_bbox(bbox, source_dpi=300, crop_dpi=200)
    assert all(isinstance(v, int) for v in out)
    assert len(out) == 4
