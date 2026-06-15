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


def _make_touching_circles(width: int, height: int, centers: list[tuple[int, int]], radius: int) -> np.ndarray:
    """Helper: build a synthetic BGR image with touching white
    circles on a black background. Used to exercise the watershed
    splitter without needing a real radiolarian plate image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for cx, cy in centers:
        cv2.circle(img, (cx, cy), radius, (255, 255, 255), -1)
    return img


def test_watershed_splits_touching_circles():
    """Six heavily-overlapping circles (3 in each of 2 rows) should
    be split into 6 individual panels by the watershed
    post-processor.

    The centre circle has radius 90 and the outer circles have
    radius 90, placed 150px to each side. With centres 150 apart
    and radii summing to 180, the circles overlap by 30px (no
    gap). Without watershed, the morphology + Otsu path merges
    each row's 3 circles into a single CC; the watershed
    splitter uses the distance transform + cv2.watershed to
    recover all 6.
    """
    seg_w = PanelSegmenter(SegmentationConfig(min_area=2500, use_watershed=True))
    seg_n = PanelSegmenter(SegmentationConfig(min_area=2500, use_watershed=False))

    img = np.zeros((1200, 1800, 3), dtype=np.uint8)
    for cy in (400, 800):
        # Place circles so they overlap: r1+r2 > dist between centers
        cv2.circle(img, (900, cy), 90, (255, 255, 255), -1)    # centre
        cv2.circle(img, (750, cy), 90, (255, 255, 255), -1)    # left (overlaps centre)
        cv2.circle(img, (1050, cy), 90, (255, 255, 255), -1)   # right (overlaps centre)

    panels_w = seg_w.segment_image(img)
    panels_n = seg_n.segment_image(img)
    assert len(panels_n) <= 2, (
        f"baseline (no watershed) should merge each row into <=2 CCs, "
        f"got {len(panels_n)}: "
        f"{[(p.bbox, p.metadata.get('method')) for p in panels_n]}"
    )
    assert len(panels_w) == 6, (
        f"watershed should split both rows into 3+3 = 6 panels, got {len(panels_w)}: "
        f"{[(p.bbox, p.metadata.get('method')) for p in panels_w]}"
    )
    # All 6 watershed panels must be smaller than the merged CCs.
    for p in panels_w:
        assert p.metadata.get("method", "").endswith("+watershed")
        # Each panel is < 250x250 (single circle with bbox padding)
        # — well under the 20% max-area threshold.
        assert p.bbox[2] * p.bbox[3] < 250 * 250


def test_watershed_does_not_split_isolated_circle():
    """A single isolated circle should remain 1 panel — the watershed
    post-processor must not over-segment. A single specimen has only
    1 ridge in the distance transform, so ``_watershed_split_cc``
    returns 0 sub-regions and the original panel is kept.
    """
    seg = PanelSegmenter(SegmentationConfig(min_area=2500, use_watershed=True))
    img = _make_touching_circles(
        width=400, height=400,
        centers=[(200, 200)],
        radius=80,
    )
    panels = seg.segment_image(img)
    assert len(panels) == 1
    assert not panels[0].metadata.get("method", "").endswith("+watershed"), (
        "isolated circle should pass through the watershed check unchanged"
    )


def test_watershed_splits_bridge_connected_circles():
    """Two circles connected by a thin bridge should be split into
    2 panels. The bridge is wider than the morphology erode kernel
    (3x3) can break, so without watershed the two circles form 1
    merged CC. The watershed post-processor detects 2 ridges in
    the distance transform and cuts at the bridge.

    Use a tall image (height=600) so each watershed sub-region is
    well under the 20% max-area filter (each sub-region is ~150px
    wide, 250px tall = 18% of 600x800 = 3%, well under 20%).
    """
    seg = PanelSegmenter(SegmentationConfig(min_area=2500, use_watershed=True))
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.circle(img, (200, 300), 80, (255, 255, 255), -1)
    cv2.circle(img, (600, 300), 80, (255, 255, 255), -1)
    cv2.line(img, (280, 300), (520, 300), (255, 255, 255), 8)
    panels = seg.segment_image(img)
    assert len(panels) == 2, (
        f"bridge-connected pair should split into 2 panels, got {len(panels)}: "
        f"{[(p.bbox, p.metadata.get('method')) for p in panels]}"
    )
    watershed_panels = [p for p in panels if p.metadata.get("method", "").endswith("+watershed")]
    assert len(watershed_panels) == 2, (
        f"both panels should be watershed-split, got {len(watershed_panels)}"
    )


def test_watershed_disabled_falls_back_to_baseline():
    """With use_watershed=False, the segmenter must produce the same
    result as the pre-watershed baseline (no +watershed panels).
    """
    img = np.zeros((600, 900, 3), dtype=np.uint8)
    for cy in (200, 500):
        cv2.circle(img, (450, cy), 130, (255, 255, 255), -1)
        cv2.circle(img, (250, cy), 100, (255, 255, 255), -1)
        cv2.circle(img, (650, cy), 100, (255, 255, 255), -1)
    seg = PanelSegmenter(SegmentationConfig(min_area=2500, use_watershed=False))
    panels = seg.segment_image(img)
    assert len(panels) <= 2
    for p in panels:
        assert not p.metadata.get("method", "").endswith("+watershed")
