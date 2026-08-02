"""Regression tests for audit 2026-08-02 - _finalize_rows heuristic noise filter."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The pipeline imports cv2 transitively; skip the runtime tests in
# cv2-less environments (CI runners without system OpenCV).
try:
    from rlpe.pipeline import RadiolarianPipeline

    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _new_pipeline():
    """Construct a RadiolarianPipeline via __new__ to avoid __init__ deps.

    _finalize_rows only touches ``self._STUB_PANEL_IDS`` and
    ``self._apply_review_corrections`` (which no-ops when ``self.config``
    is missing) so we never need a full config object.
    """
    return RadiolarianPipeline.__new__(RadiolarianPipeline)


# ---------------------------------------------------------------------------
# Phase 1.5 - heuristic noise filter
# ---------------------------------------------------------------------------


class TestHeuristicNoiseFilter:
    """Phase 1.5 of ``_finalize_rows``: drop low-confidence heuristic
    fallback rows that pollute F1 denominators with no signal.

    Filter rule (from audit 2026-08-02, W-NOISE-1):
      drop row iff
        confidence < 0.30
        AND matcher_type == "heuristic"
        AND metadata.gemma_used is False
    """

    def test_filters_low_conf_heuristic(self):
        """A row with conf=0.20, matcher_type=heuristic, gemma_used=False
        is rule-pipeline noise — drop it."""
        if not _HAS_CV2:
            pytest.skip("cv2 not available")
        p = _new_pipeline()
        rows = [
            {
                "paper_id": "p1",
                "figure_id": "f1",
                "panel_id": "1",
                "species": "Williriedellum sp. cf. W. sp. S",
                "panel_path": "/a.png",
                "confidence": 0.20,
                "matcher_type": "heuristic",
                "metadata": {"gemma_used": False},
            },
        ]
        out = p._finalize_rows(rows)
        assert out == [], "Expected low-conf heuristic row (gemma_used=False) to be dropped"

    def test_keeps_high_conf_heuristic(self):
        """A row with conf=0.85, matcher_type=heuristic is a real match.
        Do not drop it."""
        if not _HAS_CV2:
            pytest.skip("cv2 not available")
        p = _new_pipeline()
        rows = [
            {
                "paper_id": "p1",
                "figure_id": "f1",
                "panel_id": "1",
                "species": "Actinomma leptodermum",
                "panel_path": "/a.png",
                "confidence": 0.85,
                "matcher_type": "heuristic",
                "metadata": {},
            },
        ]
        out = p._finalize_rows(rows)
        assert len(out) == 1, "High-conf heuristic row must NOT be filtered"
        assert out[0]["species"] == "Actinomma leptodermum"

    def test_keeps_low_conf_with_gemma(self):
        """When gemma_used=True, the LLM was tried and the rule pipeline
        was the fallback. Low confidence here is meaningful (genuinely
        hard case); do not drop."""
        if not _HAS_CV2:
            pytest.skip("cv2 not available")
        p = _new_pipeline()
        rows = [
            {
                "paper_id": "p1",
                "figure_id": "f1",
                "panel_id": "2",
                "species": "Praeconocaryomma universa",
                "panel_path": "/a.png",
                "confidence": 0.20,
                "matcher_type": "heuristic",
                "metadata": {"gemma_used": True, "gemma_error": "rate_limited"},
            },
        ]
        out = p._finalize_rows(rows)
        assert len(out) == 1, (
            "Low-conf row WITH gemma_used=True must NOT be filtered "
            "(LLM was tried; rule was fallback)"
        )
        assert out[0]["species"] == "Praeconocaryomma universa"

    def test_keeps_non_heuristic_low_conf(self):
        """A row with matcher_type='neural' (e.g. neural-graph) and
        low confidence is a real ML emission; do not drop."""
        if not _HAS_CV2:
            pytest.skip("cv2 not available")
        p = _new_pipeline()
        rows = [
            {
                "paper_id": "p1",
                "figure_id": "f1",
                "panel_id": "3",
                "species": "Spongurus sp.",
                "panel_path": "/a.png",
                "confidence": 0.20,
                "matcher_type": "neural-graph",
                "metadata": {"gemma_used": False},
            },
        ]
        out = p._finalize_rows(rows)
        assert len(out) == 1, (
            "Low-conf non-heuristic row must NOT be filtered "
            "(only heuristic matches are noise candidates)"
        )
        assert out[0]["species"] == "Spongurus sp."

    def test_filters_only_noise_keeps_real_in_same_paper(self):
        """Mixed batch: drop noise rows, keep real matches."""
        if not _HAS_CV2:
            pytest.skip("cv2 not available")
        p = _new_pipeline()
        rows = [
            # Noise: rule fallback, no LLM attempt, very low conf
            {
                "paper_id": "p1",
                "figure_id": "f1",
                "panel_id": "1",
                "species": "Williriedellum sp. cf. W. sp. S",
                "panel_path": "/noise1.png",
                "confidence": 0.22,
                "matcher_type": "heuristic",
                "metadata": {"gemma_used": False},
            },
            # Real heuristic match
            {
                "paper_id": "p1",
                "figure_id": "f1",
                "panel_id": "2",
                "species": "Actinomma leptodermum",
                "panel_path": "/real1.png",
                "confidence": 0.91,
                "matcher_type": "heuristic",
                "metadata": {"gemma_used": False},
            },
            # Noise: rule fallback on a different species
            {
                "paper_id": "p1",
                "figure_id": "f1",
                "panel_id": "3",
                "species": "Spongurus sp.",
                "panel_path": "/noise2.png",
                "confidence": 0.21,
                "matcher_type": "heuristic",
                "metadata": {"gemma_used": False},
            },
            # Real LLM-first match
            {
                "paper_id": "p1",
                "figure_id": "f1",
                "panel_id": "4",
                "species": "Praeconocaryomma universa",
                "panel_path": "/real2.png",
                "confidence": 0.95,
                "matcher_type": "heuristic",
                "metadata": {"gemma_used": True},
            },
        ]
        out = p._finalize_rows(rows)
        assert len(out) == 2, f"Expected 2 real rows kept, got {len(out)}: {out}"
        species = sorted(r["species"] for r in out)
        assert species == ["Actinomma leptodermum", "Praeconocaryomma universa"]

    def test_filter_uses_metadata_matcher_type(self):
        """The canonical schema stores ``matcher_type`` inside
        ``metadata``. The filter must work when only the canonical
        location is populated."""
        if not _HAS_CV2:
            pytest.skip("cv2 not available")
        p = _new_pipeline()
        rows = [
            {
                "paper_id": "p1",
                "figure_id": "f1",
                "panel_id": "5",
                "species": "Williriedellum sp.",
                "panel_path": "/a.png",
                "confidence": 0.20,
                "metadata": {
                    "matcher_type": "heuristic",
                    "gemma_used": False,
                },
            },
        ]
        out = p._finalize_rows(rows)
        assert out == [], (
            "Filter must also work when matcher_type is in metadata (the canonical schema location)"
        )

    def test_filter_respects_confidence_threshold(self):
        """Confidence exactly at 0.30 is the boundary — must be kept."""
        if not _HAS_CV2:
            pytest.skip("cv2 not available")
        p = _new_pipeline()
        rows = [
            {
                "paper_id": "p1",
                "figure_id": "f1",
                "panel_id": "6",
                "species": "Borderline sp.",
                "panel_path": "/a.png",
                "confidence": 0.30,
                "matcher_type": "heuristic",
                "metadata": {"gemma_used": False},
            },
        ]
        out = p._finalize_rows(rows)
        assert len(out) == 1, (
            "Confidence exactly at 0.30 is the threshold; row must be kept (strict-less-than 0.30)"
        )


def test_heuristic_noise_filter_source_guard():
    """Source guard: the Phase 1.5 filter block must exist in pipeline.py
    with the documented comment markers. Prevents future refactors from
    silently removing the filter."""
    src = Path(__file__).resolve().parents[1] / "src" / "rlpe" / "pipeline.py"
    text = src.read_text(encoding="utf-8")
    # Comment marker from the audit
    assert "Phase 1.5" in text, (
        "Missing Phase 1.5 comment in pipeline.py — heuristic noise filter was removed"
    )
    # The filter threshold
    assert "0.30" in text, "Missing 0.30 confidence threshold in filter"
    # The matcher_type + gemma_used triple check
    assert "matcher_type" in text and "gemma_used" in text, (
        "Filter must check both matcher_type and gemma_used"
    )
