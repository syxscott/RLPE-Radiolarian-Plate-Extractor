"""Tests for the panel-NMS dedup helper and the cross-figure panel
reassignment that runs at the end of the per-paper pipeline.

Background
----------
The segmenter internally dedups candidates from a single method (SAM2
or OpenCV) using IoU >= 0.7, but two panels from *different* methods
(SAM2's full specimen + OpenCV's split-into-two boxes) can still slip
through with IoU in the 0.5-0.7 band. ``deduplicate_panels_nms`` is the
second-pass dedup that runs after the segmenter.

``Pipeline._cross_figure_reassign`` moves panels from "orphan" figures
(those with no species and a placeholder/empty caption) sitting within
3 pages of a real plate figure. Without it, OpenDataLoader's habit of
extracting a real plate as one figure and a sub-image of the same plate
as another figure leaves 20-30 panels silently unmatched (Bandini 2011
was the trigger).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.association import deduplicate_panels_nms  # noqa: E402
from rlpe.types import PanelCandidate  # noqa: E402

# ---------------------------------------------------------------------------
# deduplicate_panels_nms
# ---------------------------------------------------------------------------


def _p(pid, bbox, score=0.5):
    return PanelCandidate(panel_id=pid, bbox=bbox, score=score)


def test_nms_merges_same_label_high_overlap():
    panels = [
        _p("1", (0, 0, 100, 100), 0.9),
        _p("1", (5, 5, 100, 100), 0.7),  # IoU ≈ 0.95 with #0
        _p("2", (200, 0, 100, 100), 0.8),
    ]
    out = deduplicate_panels_nms(panels, iou_threshold=0.6, label_match=True)
    assert len(out) == 2
    kept_labels = [p.panel_id for p in out]
    assert "1" in kept_labels
    assert "2" in kept_labels
    # Higher score wins
    one_panel = next(p for p in out if p.panel_id == "1")
    assert one_panel.score == 0.9


def test_nms_dedups_unlabeled_overlap():
    panels = [
        _p(None, (0, 0, 100, 100), 0.9),
        _p(None, (10, 10, 100, 100), 0.6),  # IoU ≈ 0.81
    ]
    out = deduplicate_panels_nms(panels, iou_threshold=0.6, label_match=True)
    assert len(out) == 1


def test_nms_keeps_different_label_overlap():
    """When two near-duplicate panels carry *different* labels (e.g.
    one OCR read "1" and the other read "7"), don't merge — they
    might be the same specimen read wrong twice. Let the higher-scored
    one win by score order, but keep both for downstream audit."""
    panels = [
        _p("1", (0, 0, 100, 100), 0.9),
        _p("7", (5, 5, 100, 100), 0.7),
    ]
    out = deduplicate_panels_nms(panels, iou_threshold=0.6, label_match=True)
    assert len(out) == 2


def test_nms_no_false_merge_below_threshold():
    panels = [
        _p("1", (0, 0, 100, 100), 0.9),
        _p("1", (200, 0, 100, 100), 0.7),  # IoU = 0
    ]
    out = deduplicate_panels_nms(panels, iou_threshold=0.6, label_match=True)
    assert len(out) == 2


def test_nms_sorts_by_y_then_x():
    panels = [
        _p("3", (200, 0, 100, 100), 0.5),
        _p("1", (0, 100, 100, 100), 0.9),
        _p("2", (0, 0, 100, 100), 0.8),
    ]
    out = deduplicate_panels_nms(panels, iou_threshold=0.6, label_match=True)
    ys = [p.bbox[1] for p in out]
    assert ys == sorted(ys)


def test_nms_empty():
    assert deduplicate_panels_nms([]) == []


# ---------------------------------------------------------------------------
# Pipeline._cross_figure_reassign
# ---------------------------------------------------------------------------


def _make_result(fid, page, species=None, caption="", label_text=None, panel_id=None):
    return {
        "paper_id": "p1",
        "figure_id": fid,
        "panel_id": panel_id,
        "species": species,
        "label_text": label_text or panel_id,
        "panel_path": None,
        "bbox": [0, 0, 100, 100],
        "confidence": 0.5,
        "caption_snippet": caption,
        "ocr_text": None,
        "metadata": {"page_index": page, "panel_score": 0.5},
    }


def test_cross_figure_reassign_moves_orphan_to_nearest_plate():
    """An orphan figure (no species, placeholder caption) on page 13
    sitting between two real plates on pages 12 and 15 should be
    absorbed by the nearest real plate."""
    from rlpe.pipeline import RadiolarianPipeline
    pipe = RadiolarianPipeline.__new__(RadiolarianPipeline)
    # real plate on p12, orphan on p13, real plate on p15
    results = []
    # Plate 1: 2 panels, 1 with species
    for i, sp in enumerate([None, "SpeciesA"]):
        results.append(_make_result(
            "pl01", 12, species=sp, caption="Plate 1. figs 1-2. SpeciesA: 1, 2",
            label_text=str(i+1), panel_id=str(i+1),
        ))
    # Orphan: 3 panels, no species, placeholder caption
    for i in range(3):
        results.append(_make_result(
            "fig_orphan", 13, species=None, caption="Auto-generated figure for page 13",
            label_text=None, panel_id=None,
        ))
    # Plate 2: 2 panels, 1 with species
    for i, sp in enumerate([None, "SpeciesB"]):
        results.append(_make_result(
            "pl02", 15, species=sp, caption="Plate 2. figs 1-2. SpeciesB: 1, 2",
            label_text=str(i+1), panel_id=str(i+1),
        ))

    out = pipe._cross_figure_reassign(results)
    figure_ids = [r["figure_id"] for r in out]
    # The 3 orphan panels should have been moved to pl01 (page 12)
    # since that's the nearest real plate.
    orphan_moved = [r for r in out if r.get("metadata", {}).get("reassigned_from_figure") == "fig_orphan"]
    assert len(orphan_moved) == 3
    for r in orphan_moved:
        assert r["figure_id"] == "pl01"
    # Originals should be gone (replaced)
    assert "fig_orphan" not in figure_ids


def test_cross_figure_reassign_keeps_orphan_far_from_plate():
    """An orphan figure on page 1 with no real plate anywhere within 3
    pages is left alone (we don't move it to a far-away plate)."""
    from rlpe.pipeline import RadiolarianPipeline
    pipe = RadiolarianPipeline.__new__(RadiolarianPipeline)
    results = []
    # Real plate on page 20
    for i, sp in enumerate([None, "SpeciesX"]):
        results.append(_make_result(
            "pl01", 20, species=sp, caption="Plate 1. figs 1-2. SpeciesX: 1, 2",
            label_text=str(i+1), panel_id=str(i+1),
        ))
    # Orphan on page 1
    for i in range(2):
        results.append(_make_result(
            "fig_orphan", 1, species=None, caption="Auto-generated figure for page 1",
            label_text=None, panel_id=None,
        ))
    out = pipe._cross_figure_reassign(results)
    moved = [r for r in out if r.get("metadata", {}).get("reassigned_from_figure") == "fig_orphan"]
    assert len(moved) == 0  # far away, don't touch


def test_cross_figure_reassign_keeps_figure_with_real_caption_even_if_no_species():
    """A figure with a real (non-placeholder) caption but no species
    matched should NOT be treated as orphan — it might be a plate whose
    caption parser missed the species. Leave it alone."""
    from rlpe.pipeline import RadiolarianPipeline
    pipe = RadiolarianPipeline.__new__(RadiolarianPipeline)
    results = []
    for i, sp in enumerate([None, "SpeciesY"]):
        results.append(_make_result(
            "pl01", 12, species=sp, caption="Plate 1. figs 1-2. SpeciesY: 1, 2",
            label_text=str(i+1), panel_id=str(i+1),
        ))
    # Real caption, but parser missed the species (so 0 species matched)
    for i in range(2):
        results.append(_make_result(
            "fig_real", 13, species=None, caption="Plate 2. Some complicated caption without extracted species.",
            label_text=str(i+1), panel_id=str(i+1),
        ))
    out = pipe._cross_figure_reassign(results)
    moved = [r for r in out if r.get("metadata", {}).get("reassigned_from_figure") == "fig_real"]
    assert len(moved) == 0  # real caption → keep as a real figure


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
