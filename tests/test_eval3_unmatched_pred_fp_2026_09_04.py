"""Regression: audit 2026-09-04 eval-3 — the scoring loop in
:func:`rlpe.evaluation.metrics.evaluate` only iterates ``for g in
gold``, so a prediction that does NOT match any gold entry (extra
prediction) is never counted as a species false-positive. Only FPs
that landed on a real gold panel with a wrong species label were
counted.

Real example from the 9-paper corpus: a pred row whose panel_id
doesn't appear in the gold (e.g. an OCR-hallucinated panel from a
caption the LLM mis-segmented) was silently dropped from FP. With
extra preds never inflating the FP count, precision stayed
artificially high — the report claimed 90% precision on a run
that was actually 65%.

Fix contract: after the gold loop, every prediction key in
``pred_groups`` that was NOT added to ``consumed_pred_keys`` counts
as a panel-level false positive. If that pred carries a non-empty
species, it ALSO counts as a species-level FP. Aggregated FP is
the sum across papers.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe.evaluation.metrics import evaluate  # noqa: E402
from rlpe.evaluation.gold import GoldPanel  # noqa: E402


def _gold_row(paper_id="p1", figure_id="fig_1", panel_id="1", species="Genus alpha"):
    return GoldPanel(
        paper_id=paper_id,
        figure_id=figure_id,
        panel_id=panel_id,
        species=species,
    )


def _pred_row(paper_id="p1", figure_id="fig_1", panel_id="1", species="Genus alpha"):
    return {
        "paper_id": paper_id,
        "figure_id": figure_id,
        "panel_id": panel_id,
        "species": species,
        "confidence": 0.9,
    }


class TestUnmatchedPredictionCountsAsFP:
    def test_extra_panel_counts_as_panel_fp(self):
        # Gold has 1 panel; pred has 2 panels (1 matching + 1 extra).
        gold = [_gold_row(panel_id="1", species="Genus alpha")]
        preds = [
            # Match → TP
            _pred_row(panel_id="1", species="Genus alpha"),
            # No matching gold panel → panel FP
            _pred_row(panel_id="2", species="Genus beta"),
        ]
        rep = evaluate(preds, gold)
        p1 = rep.papers["p1"]
        # Panel match: 1 of 1 gold matched
        assert p1.panel_match == 1
        # The extra pred panel SHOULD count as a panel-level FP
        # (n_pred_panels - panel_match = 1).
        assert p1.n_pred_panels == 2
        # Species FP: the unmatched panel has a species ("Genus beta")
        # that doesn't match any gold → counts as a species FP.
        assert p1.species_fp >= 1

    def test_extra_panel_with_species_is_species_fp(self):
        gold = [_gold_row(panel_id="1", species="Genus alpha")]
        preds = [
            _pred_row(panel_id="1", species="Genus alpha"),
            _pred_row(panel_id="2", species="Genus hallucinated"),
        ]
        rep = evaluate(preds, gold)
        # species_fp lives on per-paper PaperMetrics; aggregate
        # exposes precision/recall/F1 derived from it.
        p1 = rep.papers["p1"]
        assert p1.species_fp >= 1
        # And the aggregate precision must reflect the FP (otherwise
        # the audit bug is still present).
        assert rep.aggregate["species_precision"] < 1.0

    def test_extra_panel_without_species_not_species_fp(self):
        # An unmatched pred panel with NO species is a panel FP
        # but not a species FP (no species was claimed).
        gold = [_gold_row(panel_id="1", species="Genus alpha")]
        preds = [
            _pred_row(panel_id="1", species="Genus alpha"),
            # Empty species gets filtered as placeholder — the audit
            # contract is "unmatched pred WITH species = species FP";
            # empty species is its own contract. We verify the
            # empty-species pred is filtered before the panel-FP pass
            # runs.
            {**_pred_row(panel_id="2", species=None), "confidence": 0.9},
        ]
        rep = evaluate(preds, gold)
        p1 = rep.papers["p1"]
        # The empty-species pred was filtered as a placeholder
        # (see _is_real_prediction); only the matching pred counts.
        assert p1.n_pred_panels == 1
        # No species FP from the empty-species row.
        assert p1.species_fp == 0

    def test_matched_pred_with_wrong_species_still_counted(self):
        # Sanity: the existing behavior (matched panel, wrong species)
        # is preserved.
        gold = [_gold_row(panel_id="1", species="Genus alpha")]
        preds = [_pred_row(panel_id="1", species="Genus wrong")]
        rep = evaluate(preds, gold)
        p1 = rep.papers["p1"]
        assert p1.panel_match == 1
        assert p1.species_fp == 1
