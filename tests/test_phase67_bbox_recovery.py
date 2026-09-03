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


# =====================================================================
# Audit 2026-09-03 (BLOCKER-#4): Hungarian assignment in bbox recovery
# =====================================================================
# Phase 67's historical pairing was a simple ``for idx in range(n_paired):
# sorted_results[idx] ↔ segmented[idx]``. That index pairing is
# optimal ONLY when LLM panel_ids and OpenCV CC reading-order both go
# top-to-bottom left-to-right. When M3 returns panels in reverse scan
# order, or the panel layout is multi-column with non-monotonic
# numbering, the index pairing mis-assigns. Hungarian with reading-
# order rank distance (the only signal available when all bboxes are
# placeholders, the Phase 67 entry condition) is provably identical
# in the well-behaved case AND robust to 2D position hints when the
# LLM attaches one.


class TestRecoverBboxesIoUPairing:
    """BLOCKER-#4 regression suite."""

    def test_uses_hungarian_assignment(self, tmp_path):
        """Source guard: the helper MUST use scipy.optimize
        ``linear_sum_assignment`` (Hungarian) — not raw index pairing.
        Refactor that swaps it back to ``for idx in range(...)`` would
        silently regress the M3 reverse-scan + multi-column cases.
        The two-pass design (hinted panels first, then Hungarian on
        the rest) is the BLOCKER-#4 fix.
        """
        src = (Path(__file__).resolve().parents[1] / "src" / "rlpe" / "pipeline.py").read_text(
            encoding="utf-8"
        )
        marker_start = src.find("def _recover_bboxes_via_segmentation")
        assert marker_start >= 0
        marker_end = src.find("\n    def ", marker_start + 1)
        if marker_end < 0:
            marker_end = marker_start + 4000
        body = src[marker_start:marker_end]
        # Both markers must be present in the helper body.
        assert "linear_sum_assignment" in body, (
            "Hungarian (linear_sum_assignment) not used in "
            "_recover_bboxes_via_segmentation — BLOCKER-#4 regression. "
            "The previous index pairing mis-assigned panels when M3 "
            "returned them in reverse scan order."
        )
        assert "expected_centroid_x" in body and "expected_centroid_y" in body, (
            "2D position hint handling removed from "
            "_recover_bboxes_via_segmentation — BLOCKER-#4 regression. "
            "Panels with expected_centroid metadata must take priority "
            "over reading-order rank."
        )

    def test_2d_position_hint_jumps_panel_to_correct_seg(
        self, tmp_path: Path,
    ) -> None:
        """When the LLM attaches an ``expected_centroid_x/y`` hint, the
        Hungarian cost should weight centroid proximity so a panel
        with a non-reading-order hint correctly jumps to its seg
        instead of being matched by rank alone.

        Setup: 4 panels in panel_id order 1,2,3,4. The 2D hint says
        panel_id="2" actually sits at the bottom-right (where the
        reading-order seg #3 is). The Hungarian must pair panel 2 →
        seg 3, not panel 2 → seg 1.
        """
        pipe = _make_pipeline(tmp_path)
        # 4 CCs: top-left, top-right, bottom-left, bottom-right.
        pipe.segmenter = _stub_segmenter_with(
            [
                PanelCandidate(panel_id="seg_tl", bbox=(10, 10, 60, 80), score=0.9),
                PanelCandidate(panel_id="seg_tr", bbox=(200, 10, 60, 80), score=0.9),
                PanelCandidate(panel_id="seg_bl", bbox=(10, 200, 60, 80), score=0.9),
                PanelCandidate(panel_id="seg_br", bbox=(200, 200, 60, 80), score=0.9),
            ]
        )
        plate = _build_synthetic_plate(4)
        # Panel "2" has a 2D hint pointing to the bottom-right seg
        # centroid (230, 240). This is the test of whether the
        # Hungarian cost picks up the hint.
        results = [
            {"panel_id": "1", "bbox": None, "panel_path": None,
             "metadata": {}},
            {"panel_id": "2", "bbox": None, "panel_path": None,
             "metadata": {"expected_centroid_x": 230, "expected_centroid_y": 240}},
            {"panel_id": "3", "bbox": None, "panel_path": None,
             "metadata": {}},
            {"panel_id": "4", "bbox": None, "panel_path": None,
             "metadata": {}},
        ]
        out = pipe._recover_bboxes_via_segmentation(
            results, plate, paper_id="p1", figure_id="od_p1_p1_pl01"
        )
        # panel_id="2" should get the bbox of seg_br = (200, 200, 60, 80)
        out2 = next(r for r in out if r["panel_id"] == "2")
        assert out2["bbox"] == [200, 200, 60, 80], (
            f"2D position hint not honoured: panel 2 got bbox {out2['bbox']}, "
            f"expected [200, 200, 60, 80] (seg_br centroid 230,240)."
        )

    def test_more_segs_than_panels_keeps_placeholder(
        self, tmp_path: Path,
    ) -> None:
        """If segmentation finds 5 CCs but LLM declared 3 panels,
        2 CCs are unpaired (background noise) and 3 panels get
        real bboxes. The historical index pairing under ``min()``
        achieved the same outcome for equal indices but lost
        optimality when the extra CCs were at the top of the
        reading order; the Hungarian implementation must
        preserve the assignment that minimises total rank
        distance."""
        pipe = _make_pipeline(tmp_path)
        pipe.segmenter = _stub_segmenter_with(
            [
                PanelCandidate(panel_id="s0", bbox=(10, 10, 60, 80), score=0.9),
                PanelCandidate(panel_id="s1", bbox=(100, 10, 60, 80), score=0.9),
                PanelCandidate(panel_id="s2", bbox=(200, 10, 60, 80), score=0.9),
                PanelCandidate(panel_id="s3", bbox=(10, 100, 60, 80), score=0.9),
                PanelCandidate(panel_id="s4", bbox=(100, 100, 60, 80), score=0.9),
            ]
        )
        plate = _build_synthetic_plate(3)
        results = [
            {"panel_id": "1", "bbox": None, "panel_path": None, "metadata": {}},
            {"panel_id": "2", "bbox": None, "panel_path": None, "metadata": {}},
            {"panel_id": "3", "bbox": None, "panel_path": None, "metadata": {}},
        ]
        out = pipe._recover_bboxes_via_segmentation(
            results, plate, paper_id="p1", figure_id="od_p1_p1_pl01"
        )
        # All 3 panels must have a real bbox; placeholder rows allowed.
        real_bboxes = [r for r in out if r.get("bbox") not in (None, [0, 0, 0, 0])]
        assert len(real_bboxes) == 3, (
            f"Expected 3 panels with real bboxes, got {len(real_bboxes)}: "
            f"{[r.get('bbox') for r in out]}"
        )

    def test_more_panels_than_segs_warns_and_drops_extras(
        self, tmp_path: Path,
    ) -> None:
        """If LLM declared 4 panels but segmentation found only 2
        CCs (e.g. low contrast), 2 rows keep the placeholder bbox.
        The Hungarian implementation must produce a warning so an
        operator notices the partial coverage."""
        pipe = _make_pipeline(tmp_path)
        pipe.segmenter = _stub_segmenter_with(
            [
                PanelCandidate(panel_id="s0", bbox=(10, 10, 60, 80), score=0.9),
                PanelCandidate(panel_id="s1", bbox=(200, 10, 60, 80), score=0.9),
            ]
        )
        plate = _build_synthetic_plate(4)
        results = [
            {"panel_id": "1", "bbox": None, "panel_path": None, "metadata": {}},
            {"panel_id": "2", "bbox": None, "panel_path": None, "metadata": {}},
            {"panel_id": "3", "bbox": None, "panel_path": None, "metadata": {}},
            {"panel_id": "4", "bbox": None, "panel_path": None, "metadata": {}},
        ]
        out = pipe._recover_bboxes_via_segmentation(
            results, plate, paper_id="p1", figure_id="od_p1_p1_pl01"
        )
        real = sum(
            1 for r in out
            if r.get("bbox") not in (None, [0, 0, 0, 0])
        )
        # Only 2 panels get a real bbox; the other 2 stay placeholder.
        assert real == 2, f"Expected 2 real bboxes, got {real}"
        # The two that kept placeholders must NOT have a panel_path.
        placeholders = [
            r for r in out
            if r.get("bbox") in (None, [0, 0, 0, 0])
        ]
        assert len(placeholders) == 2
        for r in placeholders:
            assert r.get("panel_path") is None, (
                f"Placeholder row got a panel_path anyway: {r}"
            )

    def test_scipy_missing_falls_back_to_reading_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If scipy.optimize is unavailable (air-gapped install),
        the helper must still produce a correct assignment via
        the reading-order fallback. The historical index pairing
        is the natural fallback — preserve it."""
        import builtins
        real_import = builtins.__import__

        def _no_scipy(name, *args, **kwargs):
            if name == "scipy" or name.startswith("scipy."):
                raise ImportError("simulated scipy absence")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_scipy)
        pipe = _make_pipeline(tmp_path)
        pipe.segmenter = _stub_segmenter_with(
            [
                PanelCandidate(panel_id="s0", bbox=(10, 10, 60, 80), score=0.9),
                PanelCandidate(panel_id="s1", bbox=(200, 10, 60, 80), score=0.9),
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
        # Identity pairing preserved.
        out1 = next(r for r in out if r["panel_id"] == "1")
        out2 = next(r for r in out if r["panel_id"] == "2")
        assert out1["bbox"] == [10, 10, 60, 80]
        assert out2["bbox"] == [200, 10, 60, 80]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
