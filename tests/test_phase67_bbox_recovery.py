"""Phase 67 — real-bbox recovery for LLM-first placeholder rows.

The 2026-08-18 audit found that 35.2% of panels in the v18 9-paper eval
had placeholder bboxes (``None`` / ``(0, 0, 0, 0)`` / ``(0, 0, W, H)``).
Image-verified F1 was 8.3% because EasyOCR reads the top-left label on
the WHOLE plate image instead of the named panel.

Phase 67 fix: ``_recover_bboxes_via_segmentation`` runs
``self.segmenter._segment_with_opencv(region_img)`` whenever every row
for a figure shares a placeholder bbox, then assigns real bboxes by
reading order.

These tests lock down the four expected behaviours + a source guard.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.pipeline import RadiolarianPipeline
from rlpe.types import PanelCandidate


def _make_pipeline(tmp_path: Path) -> RadiolarianPipeline:
    """Build a minimal RadiolarianPipeline with a mockable segmenter."""
    from rlpe.config import PipelineConfig

    cfg = PipelineConfig(
        pdf_dir=tmp_path,
        work_dir=tmp_path,
        save_intermediate=False,
        min_panel_score=0.0,
    )
    return RadiolarianPipeline(cfg)


def _build_synthetic_plate(n: int = 9, panel_w: int = 256, panel_h: int = 256) -> np.ndarray:
    """Build a grid of fake radiolarian panels on a white plate.

    Default panel size is 256x256 (large enough that bbox clip math
    doesn't bite). For specific bbox-coord tests, callers can override.
    """
    import math

    grid = int(math.ceil(math.sqrt(n)))
    plate_w = grid * panel_w
    plate_h = grid * panel_h
    plate = np.full((plate_h, plate_w, 3), 240, dtype=np.uint8)  # off-white bg
    # Draw a dark square in the centre of each panel position
    for i in range(n):
        row = i // grid
        col = i % grid
        y0 = row * panel_h + 30
        x0 = col * panel_w + 30
        plate[y0 : y0 + 80, x0 : x0 + 60] = 30  # dark blob
    return plate


def _stub_segmenter_with(panels: list[PanelCandidate]) -> MagicMock:
    """Return a mock segmenter that returns ``panels`` from
    ``_segment_with_opencv``."""
    seg = MagicMock()
    seg._segment_with_opencv = MagicMock(return_value=panels)
    return seg


class TestRecoverBboxes:
    def test_placeholder_bbox_triggers_segmentation(self, tmp_path):
        """When all rows have placeholder bboxes, segmentation runs and
        real bboxes get stamped on each row."""
        pipe = _make_pipeline(tmp_path)
        pipe.segmenter = _stub_segmenter_with(
            [
                PanelCandidate(panel_id="P1", bbox=(10, 10, 60, 80), score=0.9),
                PanelCandidate(panel_id="P2", bbox=(200, 10, 60, 80), score=0.9),
                PanelCandidate(panel_id="P3", bbox=(400, 10, 60, 80), score=0.9),
            ]
        )

        plate = _build_synthetic_plate(3)
        results = [
            {"panel_id": "1", "bbox": None, "panel_path": None, "metadata": {}},
            {"panel_id": "2", "bbox": [0, 0, 0, 0], "panel_path": None, "metadata": {}},
            {
                "panel_id": "3",
                "bbox": [0, 0, plate.shape[1], plate.shape[0]],
                "panel_path": None,
                "metadata": {},
            },
        ]

        out = pipe._recover_bboxes_via_segmentation(
            results, plate, paper_id="p1", figure_id="od_p1_p1_pl01"
        )

        # Every row now has a real subregion bbox (not None and not full-image).
        for r in out:
            bbox = r["bbox"]
            assert bbox is not None, f"row {r['panel_id']} still has bbox=None"
            assert bbox != [0, 0, 0, 0], f"row {r['panel_id']} still has (0,0,0,0)"
            assert bbox != [0, 0, plate.shape[1], plate.shape[0]], (
                f"row {r['panel_id']} still has full-image bbox"
            )
            assert r["metadata"]["panel_id_source"] == "phase67_segmentation_recovery"

        # Segmentation was called exactly once.
        pipe.segmenter._segment_with_opencv.assert_called_once()

    def test_real_bbox_skips_segmentation(self, tmp_path):
        """If any row already has a real bbox, the fallback is a no-op."""
        pipe = _make_pipeline(tmp_path)
        pipe.segmenter = _stub_segmenter_with(
            [PanelCandidate(panel_id="PX", bbox=(10, 10, 60, 80), score=0.9)]
        )

        plate = _build_synthetic_plate(3)
        # First row already has a real bbox — fallback must not fire.
        results = [
            {"panel_id": "1", "bbox": [5, 5, 100, 100], "panel_path": None, "metadata": {}},
            {"panel_id": "2", "bbox": None, "panel_path": None, "metadata": {}},
        ]

        out = pipe._recover_bboxes_via_segmentation(
            results, plate, paper_id="p1", figure_id="od_p1_p1_pl01"
        )

        # The real-bbox row is untouched.
        assert out[0]["bbox"] == [5, 5, 100, 100]
        # The None-bbox row stays None (we didn't fire).
        assert out[1]["bbox"] is None
        # Segmentation was NOT called.
        pipe.segmenter._segment_with_opencv.assert_not_called()

    def test_reading_order_assignment(self, tmp_path):
        """LLM rows are paired with segmented bboxes by reading-order
        index, not insertion order.

        Segmented bboxes (sorted y-then-x by the segmenter):
          idx 0 → bbox at y=10 (top row)
          idx 1 → bbox at y=10 (top row, x=200)
          idx 2 → bbox at y=200 (bottom row)
        LLM rows: panel_id "1", "2", "3" in that numeric order, so:
          panel_id "1" → idx 0 (top-left)
          panel_id "2" → idx 1 (top-right)
          panel_id "3" → idx 2 (bottom)
        """
        pipe = _make_pipeline(tmp_path)
        pipe.segmenter = _stub_segmenter_with(
            [
                PanelCandidate(panel_id="seg0", bbox=(10, 10, 60, 80), score=0.9),
                PanelCandidate(panel_id="seg1", bbox=(200, 10, 60, 80), score=0.9),
                PanelCandidate(panel_id="seg2", bbox=(10, 200, 60, 80), score=0.9),
            ]
        )

        plate = _build_synthetic_plate(3)
        # Insert out of numeric order to verify sorting
        results = [
            {"panel_id": "3", "bbox": None, "panel_path": None, "metadata": {}},
            {"panel_id": "1", "bbox": None, "panel_path": None, "metadata": {}},
            {"panel_id": "2", "bbox": None, "panel_path": None, "metadata": {}},
        ]

        out = pipe._recover_bboxes_via_segmentation(
            results, plate, paper_id="p1", figure_id="od_p1_p1_pl01"
        )

        # Build a map from panel_id -> assigned bbox
        bbox_by_pid = {r["panel_id"]: r["bbox"] for r in out}
        assert bbox_by_pid["1"] == [10, 10, 60, 80], (
            f"panel 1 should get the top-left bbox, got {bbox_by_pid['1']}"
        )
        assert bbox_by_pid["2"] == [200, 10, 60, 80], (
            f"panel 2 should get the top-right bbox, got {bbox_by_pid['2']}"
        )
        assert bbox_by_pid["3"] == [10, 200, 60, 80], (
            f"panel 3 should get the bottom bbox, got {bbox_by_pid['3']}"
        )

    def test_mismatched_panel_count_uses_min(self, tmp_path):
        """When LLM declares 5 panels but segmentation returns 3, the
        first 3 (by reading order) get real bboxes; the rest keep
        placeholder."""
        pipe = _make_pipeline(tmp_path)
        pipe.segmenter = _stub_segmenter_with(
            [
                PanelCandidate(panel_id="seg0", bbox=(10, 10, 60, 80), score=0.9),
                PanelCandidate(panel_id="seg1", bbox=(200, 10, 60, 80), score=0.9),
                PanelCandidate(panel_id="seg2", bbox=(10, 200, 60, 80), score=0.9),
            ]
        )

        plate = _build_synthetic_plate(3)
        results = [
            {"panel_id": "1", "bbox": None, "panel_path": None, "metadata": {}},
            {"panel_id": "2", "bbox": None, "panel_path": None, "metadata": {}},
            {"panel_id": "3", "bbox": None, "panel_path": None, "metadata": {}},
            {"panel_id": "4", "bbox": None, "panel_path": None, "metadata": {}},
            {"panel_id": "5", "bbox": None, "panel_path": None, "metadata": {}},
        ]

        out = pipe._recover_bboxes_via_segmentation(
            results, plate, paper_id="p1", figure_id="od_p1_p1_pl01"
        )

        # First 3 (sorted by panel_id) get real bboxes; last 2 stay None.
        recovered = sum(1 for r in out if r["bbox"] is not None)
        placeholder = sum(1 for r in out if r["bbox"] is None)
        assert recovered == 3, f"expected 3 recovered, got {recovered}"
        assert placeholder == 2, f"expected 2 placeholder, got {placeholder}"

    def test_writes_panel_crops_to_standard_path(self, tmp_path):
        """Real PNG files are written to panels/<paper>/<fig>/panel_NN.png
        with dimensions matching the assigned bbox."""
        pipe = _make_pipeline(tmp_path)
        pipe.segmenter = _stub_segmenter_with(
            [
                PanelCandidate(panel_id="seg0", bbox=(10, 10, 60, 80), score=0.9),
                PanelCandidate(panel_id="seg1", bbox=(200, 10, 60, 80), score=0.9),
            ]
        )

        plate = _build_synthetic_plate(2)
        results = [
            {"panel_id": "1", "bbox": None, "panel_path": None, "metadata": {}},
            {"panel_id": "2", "bbox": None, "panel_path": None, "metadata": {}},
        ]

        out = pipe._recover_bboxes_via_segmentation(
            results, plate, paper_id="p1", figure_id="od_p1_p1_pl01"
        )

        for r in out:
            path = r["panel_path"]
            assert path is not None, f"panel {r['panel_id']} has no panel_path"
            assert Path(path).is_file(), f"crop not written: {path}"
            # Dimensions match the assigned bbox (w, h).
            from PIL import Image

            with Image.open(path) as img:
                w, h = img.size
            bbox = r["bbox"]
            assert (w, h) == (bbox[2], bbox[3]), (
                f"panel {r['panel_id']}: crop is {w}x{h} but bbox is {bbox}"
            )

    def test_no_op_when_results_empty(self, tmp_path):
        """Empty results list is a no-op (no crash)."""
        pipe = _make_pipeline(tmp_path)
        pipe.segmenter = _stub_segmenter_with([])
        plate = _build_synthetic_plate(3)
        out = pipe._recover_bboxes_via_segmentation(
            [], plate, paper_id="p1", figure_id="od_p1_p1_pl01"
        )
        assert out == []

    def test_no_op_when_region_img_is_none(self, tmp_path):
        """None region_img is a no-op (defensive)."""
        pipe = _make_pipeline(tmp_path)
        pipe.segmenter = _stub_segmenter_with(
            [PanelCandidate(panel_id="X", bbox=(10, 10, 60, 80), score=0.9)]
        )
        results = [{"panel_id": "1", "bbox": None, "panel_path": None, "metadata": {}}]
        out = pipe._recover_bboxes_via_segmentation(
            results, None, paper_id="p1", figure_id="od_p1_p1_pl01"
        )
        # Untouched.
        assert out[0]["bbox"] is None
        pipe.segmenter._segment_with_opencv.assert_not_called()

    def test_segmentation_failure_returns_results_unchanged(self, tmp_path):
        """If segmentation raises, the helper must swallow it and return
        the original results unchanged (defensive)."""
        pipe = _make_pipeline(tmp_path)
        pipe.segmenter = MagicMock()
        pipe.segmenter._segment_with_opencv = MagicMock(
            side_effect=RuntimeError("opencv kernel died")
        )
        plate = _build_synthetic_plate(3)
        results = [{"panel_id": "1", "bbox": None, "panel_path": None, "metadata": {}}]
        out = pipe._recover_bboxes_via_segmentation(
            results, plate, paper_id="p1", figure_id="od_p1_p1_pl01"
        )
        assert out[0]["bbox"] is None  # unchanged


# --- Source guard ---------------------------------------------------------
#
# The Phase 67 fix relies on `_segment_with_opencv` being invoked inside
# `_recover_bboxes_via_segmentation`. A future refactor that swaps the
# segmenter for a stub, removes the call, or short-circuits on a different
# condition would silently re-introduce the 35.2% placeholder-bbox class.
# This guard pins the design.


class TestPhase67SourceGuard:
    def test_helper_invokes_segment_with_opencv(self, tmp_path):
        """Source guard: the helper MUST call ``_segment_with_opencv``.

        Any refactor that removes / renames / no-ops this call would
        silently regress image-verified F1. The test pins the structural
        relationship between ``_recover_bboxes_via_segmentation`` and
        ``self.segmenter._segment_with_opencv``.
        """
        src = (Path(__file__).resolve().parents[1] / "src" / "rlpe" / "pipeline.py").read_text(
            encoding="utf-8"
        )
        # Both pieces of the design must be present.
        assert "def _recover_bboxes_via_segmentation" in src, (
            "_recover_bboxes_via_segmentation removed — Phase 67 fix reverted"
        )
        # The call site must exist inside the helper. Search for
        # ``self.segmenter._segment_with_opencv`` to avoid matching the
        # unrelated call at pipeline.py:5034 (classical-path).
        assert "self.segmenter._segment_with_opencv(region_img)" in src, (
            "_segment_with_opencv no longer invoked from the bbox-recovery "
            "helper. Image-verified F1 will silently regress to ~8%."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
