"""Regression tests for audit 2026-08-05 (Fill Gaps) — Fix 1 + Fix 2.

Fix 1: Wilson 95% CI on PanelRecord.confidence_interval_low / _high.
  - Audit 2026-08-02 declared ``Schema v1.1.0`` shipped these two
    fields but never populated them. The audit 2026-08-05 follow-up
    confirmed: on Beccaro 2006 (4 panels), both fields were ``None``.
  - ``src/rlpe/evaluation/metrics.py:wilson_score_interval`` is the
    new producer. ``src/rlpe/converters.py:panel_record_from_match``
    consumes it.

Fix 2: ``PanelRecord.review_priority`` heuristic.
  - Same audit showed ``review_priority`` was always 0.
  - ``_review_priority_from_reasons`` in ``src/rlpe/converters.py``
    buckets critical / non-critical / no reasons into 2 / 1 / 0.

These tests assert the helpers behave correctly under the four
production-relevant scenarios:

  - default (no metadata hints) → computed from confidence + n=5
  - explicit metadata override → meta wins over heuristic
  - degenerate (n=0) → returns widest interval
  - clamp (p_hat outside [0, 1]) → no crash; bounds clamped

Plus a converter round-trip asserting every CI/priority bucket
lands on the published PanelRecord.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class TestWilsonScoreInterval:
    """Pure-function tests for ``wilson_score_interval``."""

    def test_basic_95pct_n5_p0_5(self):
        # The textbook Wilson 95% CI for p=0.5 with n=5 is
        # approximately [0.150, 0.850] (z=1.96). Allow a generous
        # tolerance for the n=5 "matcher-evidence-count"
        # approximation the audit uses as a default.
        from rlpe.evaluation.metrics import wilson_score_interval

        low, high = wilson_score_interval(0.5, n=5, z=1.96)
        assert 0.10 <= low <= 0.22, f"low={low} outside [0.10, 0.22]"
        assert 0.78 <= high <= 0.90, f"high={high} outside [0.78, 0.90]"
        # Symmetric around 0.5 (within rounding).
        assert abs((low + high) / 2 - 0.5) < 1e-9
        # Width shrinks as n grows.
        assert (high - low) > 0.55  # n=5 is fairly wide

    def test_n0_returns_widest(self):
        from rlpe.evaluation.metrics import wilson_score_interval

        low, high = wilson_score_interval(0.9, n=0)
        assert (low, high) == (0.0, 1.0)

    def test_p_hat_clamped(self):
        from rlpe.evaluation.metrics import wilson_score_interval

        # p_hat > 1.0 must not crash; bounds must stay in [0, 1].
        low, high = wilson_score_interval(1.5, n=5)
        assert 0.0 <= low <= 1.0
        assert 0.0 <= high <= 1.0
        # For p=1 the upper bound collapses to 1.
        low, high = wilson_score_interval(1.0, n=10)
        assert high == 1.0
        # For p=0 the lower bound collapses to 0.
        low, high = wilson_score_interval(0.0, n=10)
        assert low == 0.0

    def test_n_increases_tightens_interval(self):
        from rlpe.evaluation.metrics import wilson_score_interval

        width_small = wilson_score_interval(0.5, n=5)[1] - wilson_score_interval(0.5, n=5)[0]
        width_large = wilson_score_interval(0.5, n=50)[1] - wilson_score_interval(0.5, n=50)[0]
        assert width_large < width_small, (
            f"n=50 width ({width_large}) should be < n=5 width ({width_small})"
        )


class TestReviewPriorityBucketer:
    """Pure-function tests for ``_review_priority_from_reasons``."""

    def test_no_reasons_is_zero(self):
        from rlpe.converters import _review_priority_from_reasons

        assert _review_priority_from_reasons([]) == 0

    def test_critical_reasons_is_two(self):
        from rlpe.converters import _review_priority_from_reasons

        for reason in (
            "missing_species",
            "missing_bbox",
            "missing_printed_panel_id",
            "missing_panel_image",
        ):
            assert _review_priority_from_reasons([reason]) == 2, reason
            assert _review_priority_from_reasons(["other", reason]) == 2, reason

    def test_non_critical_reasons_is_one(self):
        from rlpe.converters import _review_priority_from_reasons

        assert _review_priority_from_reasons(["llm_first_without_visual_evidence"]) == 1
        assert _review_priority_from_reasons(["needs_translation_check"]) == 1


class TestPanelRecordCiAndPriority:
    """End-to-end: MatchResult → PanelRecord carries Wilson CI + priority."""

    def _make_match(self, **overrides):
        from rlpe.types import MatchResult

        meta = {
            "extraction_source": "opendataloader",
            "matcher_used": False,
            "matcher_type": "heuristic",
            "matcher_conf": 0.0,
            "caption_pairs_used": True,
            "panel_id_source": "caption",
            "printed_panel_id": None,
        }
        meta.update(overrides.pop("meta", {}))
        defaults = dict(
            paper_id="p", figure_id="f", panel_id="1",
            species="Genus species", panel_path="/tmp/panel.png",
            bbox=[10, 20, 100, 200], confidence=0.7,
            label_text="1", caption_snippet="Plate 1, figs 1-2",
            ocr_text=None, metadata=meta, paper_metadata=None,
        )
        defaults.update(overrides)
        return MatchResult(**defaults)

    def test_default_populates_ci_and_priority(self):
        from rlpe.converters import panel_record_from_match

        m = self._make_match()
        rec = panel_record_from_match(m)
        # CI filled with Wilson bounds around p_hat=0.7, n=5.
        assert rec.confidence_interval_low is not None
        assert rec.confidence_interval_high is not None
        assert 0.0 <= rec.confidence_interval_low < 0.7
        assert 0.7 < rec.confidence_interval_high <= 1.0
        # No review reasons → priority 0 (species + bbox + printed_id all OK).
        assert rec.review_priority == 0
        assert rec.image_verified is False

    def test_explicit_meta_wins_over_heuristic(self):
        from rlpe.converters import panel_record_from_match

        m = self._make_match(meta={
            "confidence_interval_low": 0.42,
            "confidence_interval_high": 0.99,
            "review_priority": 1,  # explicit even though heuristic would say 0
        })
        rec = panel_record_from_match(m)
        assert rec.confidence_interval_low == 0.42
        assert rec.confidence_interval_high == 0.99
        assert rec.review_priority == 1

    def test_missing_species_bumps_priority_to_2(self):
        from rlpe.converters import panel_record_from_match

        m = self._make_match(species=None)
        rec = panel_record_from_match(m)
        assert "missing_species" in rec.review_reasons
        assert rec.review_priority == 2

    def test_missing_printed_panel_id_bumps_priority_to_2(self):
        from rlpe.converters import panel_record_from_match

        # panel_id_source='caption' is the explicit "we have a label
        # but NOT from image OCR" tag; the heuristic should still
        # honour the panel_id_source whitelist and NOT add
        # missing_printed_panel_id. To force the flag, use legacy:
        m = self._make_match(meta={"panel_id_source": "legacy"})
        rec = panel_record_from_match(m)
        assert "missing_printed_panel_id" in rec.review_reasons
        assert rec.review_priority == 2

    def test_llm_first_with_no_image_evidence_is_priority_2(self):
        from rlpe.converters import panel_record_from_match

        m = self._make_match(
            panel_path=None,
            meta={"extraction_method": "llm_first", "panel_id_source": "llm_first"},
        )
        rec = panel_record_from_match(m)
        assert rec.review_priority == 2  # missing_panel_image is critical
        assert "missing_panel_image" in rec.review_reasons

    def test_evidence_count_meta_widens_or_narrows_ci(self):
        from rlpe.converters import panel_record_from_match

        m_small = self._make_match(meta={"matcher_evidence_count": 1})
        m_large = self._make_match(meta={"matcher_evidence_count": 50})
        rec_small = panel_record_from_match(m_small)
        rec_large = panel_record_from_match(m_large)
        width_small = rec_small.confidence_interval_high - rec_small.confidence_interval_low
        width_large = rec_large.confidence_interval_high - rec_large.confidence_interval_low
        assert width_large < width_small, (
            f"n=50 CI width ({width_large}) should be < n=1 CI width ({width_small})"
        )

    def test_image_verified_meta_set_true(self):
        from rlpe.converters import panel_record_from_match

        m = self._make_match(meta={"image_verified": True})
        rec = panel_record_from_match(m)
        assert rec.image_verified is True