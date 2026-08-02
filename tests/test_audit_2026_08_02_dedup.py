"""Regression tests for audit 2026-08-02 — _finalize_rows (figure_id, panel_id) dedup.

The dedup at pipeline.py:3047-3130 was already implemented as part of
Round 11 Bug 1 fix and refined by Phase 59 Bug 2.4 (panel_id=None rows
now dedup by ``(figure_id, bbox, species, panel_index)`` instead of
collapsing to one row). This module locks the contract that callers
rely on:

  1. Highest-confidence row wins on tied (figure_id, panel_id).
  2. panel_id=None rows are NOT collapsed when their alternate keys
     (bbox / species / panel_index) differ.
  3. Stub panel_ids (e.g. ``_map`` / ``MAP_CONTEXT``) are deduped in
     their own bucket and never shadow a real panel row.
  4. Different figure_ids always stay separate, even when their
     panel_ids happen to collide.

The 4 tests instantiate ``RadiolarianPipeline`` via ``__new__`` (bypasses
the heavy ``__init__`` which would try to build a GROBID client, OCR
backend, segmenter, etc. and fail in a cv2-less test environment).
``_finalize_rows`` only touches ``self._STUB_PANEL_IDS`` (a class-level
frozenset) and ``logger``, both of which are reachable via ``__new__``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from rlpe.pipeline import RadiolarianPipeline

    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


def _make_pipeline() -> RadiolarianPipeline:
    """Build a ``RadiolarianPipeline`` without running ``__init__``.

    The dedup helper only reads ``self._STUB_PANEL_IDS`` (a class
    attribute) and writes to ``logger``. Bypassing ``__init__`` keeps
    these tests free of GROBID / OCR / SAM2 / MiniMax dependencies.
    """
    return RadiolarianPipeline.__new__(RadiolarianPipeline)


class TestFinalizeRowsDedup:
    """Regression suite for the (figure_id, panel_id) dedup contract."""

    def test_dedup_keeps_highest_confidence(self):
        """Two rows with the same ``(figure_id, panel_id)`` but
        different confidences collapse to one; the higher-confidence
        row wins (Feng-2007-style over-segmented panel scenario)."""
        if not _HAS_CV2:
            pytest.skip("cv2 not available")
        p = _make_pipeline()
        rows = [
            {
                "paper_id": "feng2007",
                "figure_id": "f1",
                "panel_id": "1",
                "species": "Species A",
                "panel_path": "/segments/low_conf.png",
                "confidence": 0.45,
                "metadata": {"panel_score": 0.50},
            },
            {
                "paper_id": "feng2007",
                "figure_id": "f1",
                "panel_id": "1",
                "species": "Species A",
                "panel_path": "/segments/high_conf.png",
                "confidence": 0.92,  # higher confidence
                "metadata": {"panel_score": 0.50},
            },
        ]
        out = p._finalize_rows(rows)
        assert len(out) == 1, f"Expected dedup to one row, got {len(out)}: {out}"
        # The higher-confidence row survives.
        assert out[0]["panel_path"] == "/segments/high_conf.png"
        assert float(out[0]["confidence"]) == pytest.approx(0.92)

    def test_dedup_handles_none_panel_id_separately(self):
        """Three ``(figure_id, panel_id=None)`` rows with distinct
        alternate keys (bbox / species) must NOT be collapsed — they
        represent three distinct no-panel observations (caption-parser,
        layout-only fallback, OD-unpaired stub) that downstream
        consumers depend on.

        This is the Phase 59 (Bug 2.4) contract: panel_id=None is no
        longer a single-row bucket.
        """
        if not _HAS_CV2:
            pytest.skip("cv2 not available")
        p = _make_pipeline()
        rows = [
            {
                "paper_id": "feng2007",
                "figure_id": "f1",
                "panel_id": None,
                "species": "Species A",
                "panel_path": "/a.png",
                "confidence": 0.85,
                "bbox": (10, 20, 30, 40),
                "metadata": {},
            },
            {
                "paper_id": "feng2007",
                "figure_id": "f1",
                "panel_id": None,
                "species": "Species B",  # different species
                "panel_path": "/b.png",
                "confidence": 0.85,
                "bbox": (10, 20, 30, 40),  # same bbox
                "metadata": {},
            },
            {
                "paper_id": "feng2007",
                "figure_id": "f1",
                "panel_id": None,
                "species": "Species C",
                "panel_path": "/c.png",
                "confidence": 0.85,
                "bbox": (50, 60, 70, 80),  # different bbox
                "metadata": {},
            },
        ]
        out = p._finalize_rows(rows)
        assert len(out) == 3, f"Distinct None-panel rows must NOT collapse; got {len(out)}: {out}"
        species = sorted(r["species"] for r in out)
        assert species == ["Species A", "Species B", "Species C"]
        # A true duplicate (same key) should still dedup to one row.
        rows_with_dup = rows + [dict(rows[0])]  # exact copy of row A
        out2 = p._finalize_rows(rows_with_dup)
        assert len(out2) == 3, "Exact-duplicate None-panel row should collapse"

    def test_dedup_stub_panel_ids_dont_shadow_real(self):
        """Stub rows live in a SEPARATE bucket from real rows and are
        unconditionally dropped at the end of ``_finalize_rows`` (their
        content has already been cross-linked to real rows upstream by
        the range-chart / map linking pass). The "don't shadow" contract
        is therefore: a stub row present in the input does NOT interfere
        with the dedup of real rows at the same ``figure_id``.

        Stub panel_ids are the four in ``RadiolarianPipeline._STUB_PANEL_IDS``:
        ``MAP_CONTEXT``, ``RANGE_CHART``, ``_ingestion_od_failed``,
        ``_ingestion_grobid_failed``. We use ``MAP_CONTEXT`` here — the
        most representative stub from ``_process_map``.
        """
        if not _HAS_CV2:
            pytest.skip("cv2 not available")
        p = _make_pipeline()
        rows = [
            {
                "paper_id": "feng2007",
                "figure_id": "f1",
                "panel_id": "MAP_CONTEXT",  # stub from _process_map
                "species": None,
                "panel_path": "/map.png",
                "confidence": 0.0,
                "metadata": {"extraction_method": "map_caption_heuristic"},
            },
            {
                "paper_id": "feng2007",
                "figure_id": "f1",  # same figure_id as stub
                "panel_id": "1",  # real panel — must survive dedup
                "species": "Species A",
                "panel_path": "/real.png",
                "confidence": 0.9,
                "metadata": {"panel_score": 0.85},
            },
        ]
        out = p._finalize_rows(rows)
        # Real row survives; the stub is silently dropped (stubs are
        # not in the kept list because they go to stub_rows which never
        # enters the dedup→filter pipeline).
        panel_ids = [r["panel_id"] for r in out]
        assert panel_ids == ["1"], f"Only the real row should survive; got {panel_ids}"
        assert out[0]["species"] == "Species A"

    def test_dedup_does_not_merge_different_figures(self):
        """Two rows with the same ``panel_id`` but different
        ``figure_id``s are NOT duplicates — they belong to different
        plates and must both survive the dedup."""
        if not _HAS_CV2:
            pytest.skip("cv2 not available")
        p = _make_pipeline()
        rows = [
            {
                "paper_id": "feng2007",
                "figure_id": "plate1",
                "panel_id": "1",
                "species": "Species A",
                "panel_path": "/plate1/1.png",
                "confidence": 0.9,
                "metadata": {},
            },
            {
                "paper_id": "feng2007",
                "figure_id": "plate2",  # different figure, same panel_id
                "panel_id": "1",
                "species": "Species B",
                "panel_path": "/plate2/1.png",
                "confidence": 0.9,
                "metadata": {},
            },
        ]
        out = p._finalize_rows(rows)
        assert len(out) == 2, f"Different figure_ids must not merge; got {len(out)}: {out}"
        figs = sorted(r["figure_id"] for r in out)
        assert figs == ["plate1", "plate2"]
