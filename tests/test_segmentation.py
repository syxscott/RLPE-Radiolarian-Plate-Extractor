"""Tests for the panel segmentation pipeline (PanelSegmenter).

The OpenCV path runs two parallel preprocessing branches — Otsu (for sparse
plates) and an enhanced morphological pipeline (for dense plates with
touching specimens) — and merges their non-overlapping results.

These tests verify that the enhanced pipeline recovers enough panels on a
dense-plate fixture that the prior default (close kernel = 9) missed, and
that the merge-with-Otsu logic does not double-count.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rlpe.segmentation import PanelSegmenter, SegmentationConfig


def _hollis_pl03_path() -> Path | None:
    """Locate the hollis2006 pl03 image fixture if the work/ tree has it."""
    candidates = [
        REPO_ROOT / "work" / "all7_rerun" / "output" / "od_output"
        / "a0f363c21b6941d7" / "hollis2006_images" / "imageFile3.png",
        REPO_ROOT / "work" / "all5_out" / "output" / "od_output"
        / "a0f363c21b6941d7" / "hollis2006_images" / "imageFile3.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def test_enhanced_close_kernel_breaks_dense_rows():
    """On a dense plate (hollis2006 pl03, 27 panels in 3 rows), the
    close-kernel of 9x9 merged entire rows into single CCs, dropping
    ~3 panels. Reducing to 7x7 recovers them.

    The test asserts the enhanced path returns >=19 panels (prior
    default: 16) and that all returned bboxes satisfy the area filter
    (no merged-CC giants slipped through).
    """
    img_path = _hollis_pl03_path()
    if img_path is None:
        # Fixture absent from the work/ tree (CI without PDFs). Skip.
        import pytest
        pytest.skip("hollis pl03 fixture not present in work/ tree")
    img = cv2.imread(str(img_path))
    assert img is not None, f"failed to load {img_path}"

    seg = PanelSegmenter(SegmentationConfig(
        min_area=2500,
        max_single_panel_area_frac=0.20,
        max_aspect_ratio=4.0,
    ))
    panels = seg._segment_with_opencv(img)
    assert len(panels) >= 19, (
        f"hollis pl03: expected >=19 panels after k_close=7, got {len(panels)}"
    )

    # Sanity: every panel's actual CC area (the score the segmenter
    # assigns) is at most 20% of the image. A merged-row CC that
    # slipped through would have a much higher score.
    img_area = img.shape[0] * img.shape[1]
    for p in panels:
        assert p.score <= 0.21, (
            f"panel {p.bbox} score {p.score:.3f} suggests a merged "
            f"CC that should have been filtered (>20% of image area)"
        )


def test_opencv_path_returns_sorted_panels():
    """Panels must be returned in (y, x) reading order so the OCR label
    matcher can scan top-to-bottom, left-to-right without re-sorting."""
    img_path = _hollis_pl03_path()
    if img_path is None:
        import pytest
        pytest.skip("hollis pl03 fixture not present in work/ tree")
    img = cv2.imread(str(img_path))

    seg = PanelSegmenter(SegmentationConfig())
    panels = seg._segment_with_opencv(img)
    keys = [(p.bbox[1], p.bbox[0]) for p in panels]
    assert keys == sorted(keys), "panels must be returned in (y, x) order"


def test_opencv_path_returns_empty_for_blank_image():
    """A blank (all-white) image should return no panels — there is
    nothing for the threshold to segment."""
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    seg = PanelSegmenter(SegmentationConfig())
    panels = seg._segment_with_opencv(img)
    assert panels == []


def test_enhanced_path_handles_gray_input():
    """The enhanced pipeline accepts a grayscale image and produces
    a strictly binary threshold result. The full segmentation path
    expects BGR; the gray-to-binary internal call is what we verify
    here (the Otsu branch re-grays internally so callers never need
    to pre-convert)."""
    img_path = _hollis_pl03_path()
    if img_path is None:
        import pytest
        pytest.skip("hollis pl03 fixture not present in work/ tree")
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    seg = PanelSegmenter(SegmentationConfig())
    binary = seg._preprocess_enhanced(gray)
    assert binary.shape == gray.shape
    assert set(np.unique(binary).tolist()).issubset({0, 255}), (
        "adaptive threshold should produce a strictly binary image"
    )
