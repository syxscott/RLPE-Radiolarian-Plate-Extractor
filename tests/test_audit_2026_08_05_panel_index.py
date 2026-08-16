"""Regression tests for audit 2026-08-05 (Fill Gaps) — Fix 5.

Fix 5: ``PanelRecord.pipeline_panel_index`` 1-based population.
  - audit 2026-08-05 verified on Beccaro 2006 that
    ``pipeline_panel_index`` was always ``None`` even though the
    schema declares it as ``int | None``.
  - Root cause: Phase 55's CRITICAL-2 fix (commit ``6defce2``)
    hard-coded ``panel_index=None`` in both LLM-first MatchResult
    construction sites in ``src/rlpe/pipeline.py`` on the grounds
    that no ``PanelCandidate`` exists for LLM-first rows. With
    schema v1.1.0+ declaring the field as a recoverable integer,
    the natural 1-based list position is the right value.

Tests:

1. ``panel_record_from_match`` propagates ``match.panel_index`` to
   ``PanelRecord.pipeline_panel_index`` (already worked but locked
   here so a future refactor doesn't regress).
2. Source guard: ``pipeline.py`` MUST use ``enumerate(..., start=1)``
   in both LLM-first sites so the field is actually populated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class TestPipelinePanelIndexPropagation:
    """MatchResult.panel_index -> PanelRecord.pipeline_panel_index."""

    def _make_match(self, panel_index):
        from rlpe.types import MatchResult

        return MatchResult(
            paper_id="p", figure_id="f", panel_id="1",
            species="Genus species", panel_path="/tmp/1.png",
            bbox=[10, 20, 100, 200], confidence=0.7,
            label_text="1", caption_snippet="Plate 1",
            ocr_text=None, metadata={}, paper_metadata=None,
            panel_index=panel_index,
        )

    def test_one_based_index_propagates(self):
        from rlpe.converters import panel_record_from_match

        rec = panel_record_from_match(self._make_match(1))
        assert rec.pipeline_panel_index == 1

    def test_high_index_propagates(self):
        from rlpe.converters import panel_record_from_match

        rec = panel_record_from_match(self._make_match(35))
        assert rec.pipeline_panel_index == 35

    def test_none_index_stays_none(self):
        from rlpe.converters import panel_record_from_match

        rec = panel_record_from_match(self._make_match(None))
        assert rec.pipeline_panel_index is None

    def test_default_index_when_attr_missing(self):
        # MatchResults built without the panel_index kwarg default to None.
        from rlpe.types import MatchResult

        m = MatchResult(
            paper_id="p", figure_id="f", panel_id="1",
            species="Genus species", panel_path="/tmp/1.png",
            bbox=[10, 20, 100, 200], confidence=0.7,
            label_text="1", caption_snippet="Plate 1",
            ocr_text=None, metadata={}, paper_metadata=None,
        )
        from rlpe.converters import panel_record_from_match

        rec = panel_record_from_match(m)
        assert rec.pipeline_panel_index is None


class TestPipelinePanelIndexSourceGuard:
    """Source guard: pipeline.py must populate panel_index in the
    LLM-first MatchResult sites so the field lands on the published
    PanelRecord (audit 2026-08-05 Fix 5).
    """

    def _src(self):
        return Path(_SRC, "rlpe", "pipeline.py").read_text(encoding="utf-8")

    def test_primary_llm_first_uses_enumerate(self):
        text = self._src()
        # The primary LLM-first loop (after the parsed JSON) must
        # use ``enumerate(panels_data, start=1)`` so panel_index is
        # populated. A naive ``for p in panels_data`` would silently
        # leave panel_index at None on every row.
        assert "enumerate(panels_data, start=1)" in text, (
            "Fix 5 expects the primary LLM-first loop to enumerate "
            "panels_data with start=1 so pipeline_panel_index is "
            "populated. If a future refactor removes this, the "
            "panel_index field falls back to None."
        )

    def test_hybrid_caption_enrichment_uses_post_append_index(self):
        text = self._src()
        # The hybrid caption-enrichment loop must compute a 1-based
        # panel_index from the pre_append_count + 1 formula so the
        # value reflects the row's final position in llm_results.
        assert "_hybrid_panel_idx = pre_append_count + 1" in text, (
            "Fix 5 expects the hybrid path to compute the new row's "
            "1-based position via _hybrid_panel_idx = pre_append_count + 1."
        )

    def test_no_hard_coded_panel_index_none_in_llm_first(self):
        # The audit 2026-08-05 fix removed ``panel_index=None`` from
        # both LLM-first sites. If a future regression re-introduces
        # ``panel_index=None`` at one of them, this guard fires.
        text = self._src()
        # We can't simply assert "panel_index=None" doesn't appear —
        # other code paths legitimately use None (e.g. classical
        # fallback stubs at line 776). Instead assert the specific
        # comment marker is gone.
        assert (
            "Phase 55 audit CRITICAL-2 fix: explicitly pass panel_index=None"
            not in text
        ), (
            "Fix 5 removed the Phase 55 hard-coded panel_index=None "
            "from the primary LLM-first site. If this comment is back, "
            "panel_index regressed to None on every LLM-first row."
        )