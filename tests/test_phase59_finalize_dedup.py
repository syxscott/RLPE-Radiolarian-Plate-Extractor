"""Phase 59 — Pipeline correctness, Bug 2.4.

``_finalize_rows`` keyed dedup on ``(figure_id, panel_id)``. Rows
with ``panel_id=None`` for the same ``figure_id`` collapsed into one
because ``None == None`` matched under the key. Three distinct
no-panel rows for the same figure (e.g. one from caption parser,
one from layout-only fallback, one from OD's unpaired-figure stub)
should remain three rows after dedup; previously they collapsed to
one and the others were silently dropped, reducing panel coverage.

The fix: when ``panel_id is None``, dedup by ``(figure_id, bbox_tuple,
species, panel_index)`` instead so distinct rows survive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _make_pipeline():
    """Build a pipeline with __new__ so we can call ``_finalize_rows``
    without running the heavy ``__init__`` chain."""
    from rlpe.pipeline import RadiolarianPipeline

    pipe = RadiolarianPipeline.__new__(RadiolarianPipeline)
    return pipe


def test_finalize_keeps_distinct_panel_id_none_rows() -> None:
    """Bug 2.4 fix: 3 rows, all ``panel_id=None``, distinct bbox →
    expect 3 rows kept (was 1 before the fix)."""
    pipe = _make_pipeline()
    rows = [
        {
            "paper_id": "p1",
            "figure_id": "fig1",
            "panel_id": None,
            "species": "A",
            "panel_path": None,
            "bbox": (10, 20, 30, 40),
            "confidence": 0.5,
            "label_text": None,
            "caption_snippet": "cap1",
            "ocr_text": None,
            "paper_metadata": None,
            "metadata": {},
        },
        {
            "paper_id": "p1",
            "figure_id": "fig1",
            "panel_id": None,
            "species": "B",
            "panel_path": None,
            "bbox": (50, 60, 70, 80),
            "confidence": 0.5,
            "label_text": None,
            "caption_snippet": "cap1",
            "ocr_text": None,
            "paper_metadata": None,
            "metadata": {},
        },
        {
            "paper_id": "p1",
            "figure_id": "fig1",
            "panel_id": None,
            "species": "C",
            "panel_path": None,
            "bbox": (90, 100, 110, 120),
            "confidence": 0.5,
            "label_text": None,
            "caption_snippet": "cap1",
            "ocr_text": None,
            "paper_metadata": None,
            "metadata": {},
        },
    ]
    result = pipe._finalize_rows(rows)
    # Expect all 3 to survive dedup. Note: rows with no species AND
    # no panel_path are dropped by the empty-row filter, so we make
    # sure each row has at least a species OR panel_path. Here all
    # have species but no panel_path, so all should be kept.
    assert len(result) == 3, (
        f"Expected 3 distinct panel_id=None rows kept; got {len(result)}: {result}"
    )
    species_kept = sorted(r.get("species") for r in result)
    assert species_kept == ["A", "B", "C"], (
        f"All 3 species should survive dedup; got {species_kept}"
    )


def test_finalize_dedups_identical_panel_id_none() -> None:
    """Bug 2.4 fix: rows with ``panel_id=None`` AND identical bbox+species
    collapse to one (dedup still works for true duplicates)."""
    pipe = _make_pipeline()
    rows = [
        {
            "paper_id": "p1",
            "figure_id": "fig1",
            "panel_id": None,
            "species": "A",
            "panel_path": None,
            "bbox": (10, 20, 30, 40),
            "confidence": 0.5,
            "label_text": None,
            "caption_snippet": "cap1",
            "ocr_text": None,
            "paper_metadata": None,
            "metadata": {},
        },
        {
            "paper_id": "p1",
            "figure_id": "fig1",
            "panel_id": None,
            "species": "A",
            "panel_path": None,
            "bbox": (10, 20, 30, 40),
            "confidence": 0.6,  # higher confidence
            "label_text": None,
            "caption_snippet": "cap1",
            "ocr_text": None,
            "paper_metadata": None,
            "metadata": {},
        },
    ]
    result = pipe._finalize_rows(rows)
    assert len(result) == 1, (
        f"True duplicates should still dedup to one; got {len(result)}"
    )
    # Higher confidence wins.
    assert result[0].get("confidence") == 0.6, (
        f"Higher confidence should win; got {result[0].get('confidence')}"
    )


def test_finalize_dedups_panel_id_correctly() -> None:
    """Bug 2.4 backward-compat: rows with real ``panel_id`` still dedup
    on ``(figure_id, panel_id)`` as before.
    """
    pipe = _make_pipeline()
    rows = [
        {
            "paper_id": "p1",
            "figure_id": "fig1",
            "panel_id": "1",
            "species": "A",
            "panel_path": "/p.png",
            "bbox": None,
            "confidence": 0.5,
            "label_text": None,
            "caption_snippet": "cap1",
            "ocr_text": None,
            "paper_metadata": None,
            "metadata": {},
        },
        {
            "paper_id": "p1",
            "figure_id": "fig1",
            "panel_id": "1",
            "species": "A",
            "panel_path": "/p.png",
            "bbox": None,
            "confidence": 0.6,
            "label_text": None,
            "caption_snippet": "cap1",
            "ocr_text": None,
            "paper_metadata": None,
            "metadata": {},
        },
        {
            "paper_id": "p1",
            "figure_id": "fig1",
            "panel_id": "2",
            "species": "B",
            "panel_path": "/p2.png",
            "bbox": None,
            "confidence": 0.7,
            "label_text": None,
            "caption_snippet": "cap1",
            "ocr_text": None,
            "paper_metadata": None,
            "metadata": {},
        },
    ]
    result = pipe._finalize_rows(rows)
    assert len(result) == 2, (
        f"Expected 2 deduped rows; got {len(result)}"
    )
