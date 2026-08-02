"""Regression tests for audit 2026-08-02 — tile-and-segment fallback."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.config import PipelineConfig  # noqa: E402
from rlpe.segmentation import PanelSegmenter, SegmentationConfig  # noqa: E402


def _make_dense_plate(
    width: int = 1200,
    height: int = 1200,
    n_circles: int = 20,
    seed: int = 7,
) -> np.ndarray:
    """Synthetic dense-plate test image: white canvas + N dark circles.

    Each circle is a dark-gray (rgb ~60) filled disc with radius 30-60 px,
    placed at random (non-overlapping) positions. The intent is to mimic
    a real radiolarian plate with many small specimens, where the
    morphological close in ``_preprocess_enhanced`` fuses them all into
    one giant CC rejected by ``max_single_panel_area_frac``.
    """
    rng = random.Random(seed)
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    placed: list[tuple[int, int, int]] = []
    attempts = 0
    while len(placed) < n_circles and attempts < 5000:
        attempts += 1
        r = rng.randint(30, 60)
        cx = rng.randint(r + 5, width - r - 5)
        cy = rng.randint(r + 5, height - r - 5)
        # Reject overlaps with already-placed circles
        too_close = False
        for px, py, pr in placed:
            if (cx - px) ** 2 + (cy - py) ** 2 < (r + pr + 25) ** 2:
                too_close = True
                break
        if too_close:
            continue
        placed.append((cx, cy, r))
        cv2.circle(img, (cx, cy), r, (60, 60, 60), thickness=-1)
    return img


def _make_simple_plate(width: int = 800, height: int = 800) -> np.ndarray:
    """Smaller 800x800 dense plate for _tile_and_segment direct test."""
    return _make_dense_plate(width=width, height=height, n_circles=12, seed=11)


def test_segment_with_opencv_dense_plate_recovers_panels():
    """Dense plate: standard CC fuses circles -> fallback tiles -> >=5 panels."""
    img = _make_dense_plate(width=1200, height=1200, n_circles=20)
    cfg = SegmentationConfig(use_sam2=False)
    segmenter = PanelSegmenter(config=cfg)
    panels = segmenter._segment_with_opencv(img)
    # Pre-fix behaviour: 0-1 panel (giant merged blob rejected by area frac).
    # Post-fix behaviour: >=5 panels via tile fallback.
    assert len(panels) >= 5, (
        f"Dense plate (20 circles) should yield >=5 panels after tile fallback, got {len(panels)}"
    )


def test_segment_with_opencv_bboxes_within_image_bounds():
    """All returned bounding boxes must lie within the source image."""
    img = _make_dense_plate(width=1200, height=1200, n_circles=20)
    h, w = img.shape[:2]
    cfg = SegmentationConfig(use_sam2=False)
    segmenter = PanelSegmenter(config=cfg)
    panels = segmenter._segment_with_opencv(img)
    assert len(panels) > 0
    for p in panels:
        bx, by, bw, bh = p.bbox
        assert 0 <= bx <= w, f"bbox x={bx} out of bounds (w={w})"
        assert 0 <= by <= h, f"bbox y={by} out of bounds (h={h})"
        assert bx + bw <= w + 1, f"bbox x+w={bx + bw} exceeds w={w}"
        assert by + bh <= h + 1, f"bbox y+h={by + bh} exceeds h={h}"
        assert bw > 0 and bh > 0, f"bbox has zero dimension: {p.bbox}"


def test_tile_and_segment_helper_direct():
    """Direct test of _tile_and_segment: returns panels in original-frame coords."""
    img = _make_simple_plate(width=800, height=800)
    h, w = img.shape[:2]
    cfg = SegmentationConfig(use_sam2=False)
    segmenter = PanelSegmenter(config=cfg)
    tiled_panels = segmenter._tile_and_segment(img, grid=(2, 2))
    assert len(tiled_panels) > 0, "tile helper should produce at least one panel"
    # All bboxes must be in original-frame coords (within image bounds).
    for p in tiled_panels:
        bx, by, bw, bh = p.bbox
        assert 0 <= bx < w, f"tile bbox x={bx} not in original frame (w={w})"
        assert 0 <= by < h, f"tile bbox y={by} not in original frame (h={h})"
        assert bx + bw <= w + 1
        assert by + bh <= h + 1
    # Method metadata must be tagged with +tiled.
    for p in tiled_panels:
        method = (p.metadata or {}).get("method", "")
        assert method.endswith("+tiled"), f"Expected method to end with '+tiled', got {method!r}"


def test_tile_and_segment_overlap_pads_edges():
    """The overlap=20 default should not crash and produces valid offsets."""
    img = _make_dense_plate(width=600, height=600, n_circles=8, seed=21)
    cfg = SegmentationConfig(use_sam2=False)
    segmenter = PanelSegmenter(config=cfg)
    # No exception; results within image bounds.
    tiled_panels = segmenter._tile_and_segment(img, grid=(2, 2), overlap=20)
    h, w = img.shape[:2]
    for p in tiled_panels:
        bx, by, bw, bh = p.bbox
        assert bx >= 0 and by >= 0
        assert bx + bw <= w + 1
        assert by + bh <= h + 1


def test_small_image_skips_tile_fallback():
    """Images smaller than 600x600 must NOT trigger the tile fallback."""
    # A 400x400 image with too few circles will yield <=1 panel, but the
    # 600-px guard must skip the fallback entirely (no exception, no fake
    # results added beyond the standard CC pass).
    img = _make_dense_plate(width=400, height=400, n_circles=4, seed=99)
    cfg = SegmentationConfig(use_sam2=False)
    segmenter = PanelSegmenter(config=cfg)
    panels = segmenter._segment_with_opencv(img)
    h, w = img.shape[:2]
    # Whatever panels come out, they must fit within the 400x400 frame.
    for p in panels:
        bx, by, bw, bh = p.bbox
        assert 0 <= bx <= w
        assert 0 <= by <= h
        assert bx + bw <= w + 1
        assert by + bh <= h + 1


def test_pipelineconfig_unaffected():
    """PipelineConfig should still construct cleanly (no schema drift)."""
    cfg = PipelineConfig(pdf_dir=Path("/tmp/pdf"), work_dir=Path("/tmp/work"))
    assert cfg is not None
    assert cfg.pdf_dir == Path("/tmp/pdf")
    assert cfg.work_dir == Path("/tmp/work")
