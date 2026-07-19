"""Round 11 source-guard tests.

Locks in the four pipeline post-processing bug fixes discovered
during the live MiniMax M3 smoke tests on real PDFs:

  Bug 1  panel_id duplicates with different panel_path
  Bug 2  LLM-first hallucinated panel_ids not in caption set
  Bug 3  MAP_CONTEXT / RANGE_CHART / _ingestion_* stub rows leak
         into matches.jsonl
  Bug 4  empty species + no panel_path rows leak through
  Bug 6  invalid panel_id format (e.g. "10, 11") leaks through

The tests verify (a) the source shape in pipeline.py after the fix
and (b) the runtime behaviour on small in-memory synthetic row lists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The new helper imports cv2 transitively; skip the runtime tests
# in cv2-less environments.
try:
    from rlpe.pipeline import RadiolarianPipeline

    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


# ---------------------------------------------------------------------------
# Bug 2 — LLM-first hallucination filter
# ---------------------------------------------------------------------------


def test_hallucination_filter_present():
    """The hallucination filter must live in ``_process_region`` after
    the caption parser has built ``pair_lookup`` (round 11 placement
    fix). Without this, M3-returned panels with labels NOT in the
    caption set pollute eval — e.g. Pouille 2014 returns 35 panels
    with pid=2,4,7,9,10,11,13,14b all invented (caption only lists 6).
    """
    src = Path(__file__).resolve().parents[1] / "src" / "rlpe" / "pipeline.py"
    src_text = src.read_text(encoding="utf-8")
    # Find the hallucination filter block.
    assert "Hallucination filter" in src_text or "hallucination filter" in src_text, (
        "Missing hallucination filter block in pipeline.py"
    )
    # The filter must reference _label_in_caption helper.
    assert "_label_in_caption" in src_text
    # The filter must use pair_lookup (caption-derived labels).
    assert "pair_lookup" in src_text
    # The filter must be INSIDE _process_region, AFTER the pair_lookup
    # is built. pair_lookup is built around line 2493; the filter
    # must come later. Verify the order: search for "def _process_region"
    # then check pair_lookup assignment comes before hallucination filter.
    pr_idx = src_text.find("def _process_region(")
    assert pr_idx > 0
    region = src_text[pr_idx : pr_idx + 25000]
    pl_idx = region.find("pair_lookup: dict[str, str] = {}")
    hf_idx = region.find("Hallucination filter")
    assert pl_idx > 0 and hf_idx > 0, (
        "Both pair_lookup assignment and hallucination filter must be inside _process_region"
    )
    assert pl_idx < hf_idx, "pair_lookup must be assigned BEFORE the hallucination filter runs"


# ---------------------------------------------------------------------------
# Bug 1 / 3 / 4 / 6 — _finalize_rows helper
# ---------------------------------------------------------------------------


def test_finalize_rows_helper_exists():
    src = Path(__file__).resolve().parents[1] / "src" / "rlpe" / "pipeline.py"
    src_text = src.read_text(encoding="utf-8")
    # Helper definition
    assert "def _finalize_rows(" in src_text, (
        "_finalize_rows helper missing — needed for Round 11 dedup + filter"
    )
    # Stub panel_ids set
    assert "_STUB_PANEL_IDS = frozenset(" in src_text, "_STUB_PANEL_IDS constant missing"
    assert '"MAP_CONTEXT"' in src_text
    assert '"RANGE_CHART"' in src_text
    # Wired into both _process_one_pdf_od and _process_one_pdf_grobid
    assert src_text.count("self._finalize_rows(results)") >= 2, (
        "_finalize_rows must be called at the end of BOTH "
        "_process_one_pdf_od and _process_one_pdf_grobid"
    )


def test_finalize_rows_dedups_by_figure_panel_id():
    """Two rows with the same (figure_id, panel_id) but different
    panel_paths must collapse to one. The kept row is the one with
    the higher confidence (and ties broken by panel_score)."""
    if not _HAS_CV2:
        pytest.skip("cv2 not available")
    # Construct a RadiolarianPipeline via __new__ to avoid __init__ deps.
    p = RadiolarianPipeline.__new__(RadiolarianPipeline)
    rows = [
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "1",
            "species": "A",
            "panel_path": "/a.png",
            "confidence": 0.5,
            "metadata": {"panel_score": 0.4},
        },
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "1",
            "species": "A",
            "panel_path": "/b.png",
            "confidence": 0.9,  # ← higher
            "metadata": {"panel_score": 0.7},
        },
    ]
    out = p._finalize_rows(rows)
    assert len(out) == 1, f"Expected dedup to one row, got {len(out)}: {out}"
    assert out[0]["panel_path"] == "/b.png", (
        f"Expected higher-confidence row to win, got {out[0]['panel_path']}"
    )


def test_finalize_rows_dedups_tiebreak_by_panel_score():
    """Two rows with same (fig, pid) and same confidence — keep the
    one with higher panel_score."""
    if not _HAS_CV2:
        pytest.skip("cv2 not available")
    p = RadiolarianPipeline.__new__(RadiolarianPipeline)
    rows = [
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "2",
            "species": "X",
            "panel_path": "/low.png",
            "confidence": 0.5,
            "metadata": {"panel_score": 0.3},
        },
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "2",
            "species": "X",
            "panel_path": "/high.png",
            "confidence": 0.5,
            "metadata": {"panel_score": 0.9},
        },
    ]
    out = p._finalize_rows(rows)
    assert len(out) == 1
    assert out[0]["panel_path"] == "/high.png"


def test_finalize_rows_keeps_distinct_panel_ids():
    """Rows with DIFFERENT (fig, pid) must not be merged — only true
    duplicates collapse."""
    if not _HAS_CV2:
        pytest.skip("cv2 not available")
    p = RadiolarianPipeline.__new__(RadiolarianPipeline)
    rows = [
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "1",
            "species": "A",
            "panel_path": "/a.png",
            "confidence": 0.9,
            "metadata": {"panel_score": 0.5},
        },
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "2",
            "species": "B",
            "panel_path": "/b.png",
            "confidence": 0.9,
            "metadata": {"panel_score": 0.5},
        },
    ]
    out = p._finalize_rows(rows)
    assert len(out) == 2


def test_finalize_rows_drops_map_context_stub():
    """MAP_CONTEXT rows are stubs from _process_map — their
    location data has already been propagated to other rows via
    _cross_link_map_and_range_chart. They must be dropped."""
    if not _HAS_CV2:
        pytest.skip("cv2 not available")
    p = RadiolarianPipeline.__new__(RadiolarianPipeline)
    rows = [
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "MAP_CONTEXT",
            "species": None,
            "panel_path": "/map.png",
            "confidence": 0.0,
            "metadata": {"extraction_method": "map_caption_heuristic"},
        },
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "1",
            "species": "A",
            "panel_path": "/a.png",
            "confidence": 0.9,
            "metadata": {},
        },
    ]
    out = p._finalize_rows(rows)
    assert len(out) == 1
    assert out[0]["panel_id"] == "1"


def test_finalize_rows_drops_range_chart_stub():
    """Same as MAP_CONTEXT — RANGE_CHART stubs from range-chart
    vision are dropped at finalize."""
    if not _HAS_CV2:
        pytest.skip("cv2 not available")
    p = RadiolarianPipeline.__new__(RadiolarianPipeline)
    rows = [
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "RANGE_CHART",
            "species": None,
            "panel_path": None,
            "confidence": 0.0,
            "metadata": {"extraction_method": "range_chart_vision"},
        },
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "1",
            "species": "A",
            "panel_path": "/a.png",
            "confidence": 0.9,
            "metadata": {},
        },
    ]
    out = p._finalize_rows(rows)
    assert len(out) == 1
    assert out[0]["panel_id"] == "1"


def test_finalize_rows_drops_empty_signal_row():
    """A row with no species AND no panel_path carries no signal —
    drop it. (Stub rows are caught earlier; this catches the
    LLM-first 'not a radiolarian' rows where caption-parser
    couldn't fill in a species either.)"""
    if not _HAS_CV2:
        pytest.skip("cv2 not available")
    p = RadiolarianPipeline.__new__(RadiolarianPipeline)
    rows = [
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "5",
            "species": None,
            "panel_path": None,
            "confidence": 0.0,
            "metadata": {},
        },
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "1",
            "species": "A",
            "panel_path": "/a.png",
            "confidence": 0.9,
            "metadata": {},
        },
    ]
    out = p._finalize_rows(rows)
    assert len(out) == 1
    assert out[0]["panel_id"] == "1"


def test_finalize_rows_drops_invalid_panel_id_format():
    """Invalid panel_id formats ('10, 11', empty string, etc.) drop.
    Round 9 added the shape regex; Round 11 enforces it at emission.

    Phase 59 (Bug 2.4): panel_id=None is now a valid category and is
    kept (the dedup preserves distinct None-rows by bbox+key). The
    "invalid format" assertion therefore applies only to non-None
    strings that fail the SHAPE regex.
    """
    if not _HAS_CV2:
        pytest.skip("cv2 not available")
    p = RadiolarianPipeline.__new__(RadiolarianPipeline)
    rows = [
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "10, 11",
            "species": "A",
            "panel_path": "/a.png",
            "confidence": 0.9,
            "metadata": {},
        },
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "",
            "species": "A",
            "panel_path": "/b.png",
            "confidence": 0.9,
            "metadata": {},
        },
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "1",
            "species": "A",
            "panel_path": "/d.png",
            "confidence": 0.9,
            "metadata": {},
        },
    ]
    out = p._finalize_rows(rows)
    # Both invalid-format panel_id rows drop; only "1" remains.
    assert len(out) == 1
    assert out[0]["panel_id"] == "1"


def test_finalize_rows_preserves_distinct_panel_id_none_rows():
    """Phase 59 (Bug 2.4): distinct panel_id=None rows are preserved.

    Previously the dedup keyed on (figure_id, panel_id) and collapsed
    all None-rows into one. The fix keeps distinct None-rows by
    (figure_id, bbox, species, panel_index). This is the canonical
    regression test for the fix.
    """
    if not _HAS_CV2:
        pytest.skip("cv2 not available")
    p = RadiolarianPipeline.__new__(RadiolarianPipeline)
    rows = [
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": None,
            "species": "A",
            "panel_path": "/a.png",
            "confidence": 0.9,
            "bbox": (10, 20, 30, 40),
            "metadata": {},
        },
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": None,
            "species": "B",
            "panel_path": "/b.png",
            "confidence": 0.9,
            "bbox": (50, 60, 70, 80),
            "metadata": {},
        },
    ]
    out = p._finalize_rows(rows)
    assert len(out) == 2
    species = sorted(r["species"] for r in out)
    assert species == ["A", "B"]


def test_finalize_rows_keeps_real_panel_with_no_species_but_has_path():
    """A row with no species but a real panel_path is a valid
    'image-OCR'd panel, label known, species unknown' entry. The
    web UI shows it as ⚠ positional. Don't drop it just because
    species is None."""
    if not _HAS_CV2:
        pytest.skip("cv2 not available")
    p = RadiolarianPipeline.__new__(RadiolarianPipeline)
    rows = [
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "3",
            "species": None,
            "panel_path": "/real.png",
            "confidence": 0.6,
            "metadata": {},
        },
    ]
    out = p._finalize_rows(rows)
    assert len(out) == 1
    assert out[0]["panel_path"] == "/real.png"
