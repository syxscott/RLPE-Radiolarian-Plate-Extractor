"""Regression tests for the Bragin 2025 paper_id alias fix (audit 2026-08-02).

The Bragin figure_id schema variant was normalised by commit ``f97f33a``,
but the paper_id side was missed: the gold standard records Bragin 2025
with the human-readable paper slug ``bragin2025`` while the upstream
OpenDataLoader extractor emits the 16-char content hash
``2e85364a3c605326`` in its prediction rows. Eval-side matching is keyed
on (paper_id, figure_id, panel_id), so the paper_id mismatch alone made
Bragin report 0% panel_match even after the figure_id fix.

This file pins the alias map and the ``normalize_paper_id_for_eval``
helper so future refactors cannot silently drop the asymmetric pair.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class TestBraginPaperIdAlias:
    """Pin the Bragin paper_id alias map and its public helper."""

    def test_alias_resolves_to_bragin2025(self):
        """The alias map must contain the Bragin hash -> slug entry.

        Without this entry the Bragin pred rows (keyed by the OD
        content hash) never reach the Bragin gold rows (keyed by the
        paper slug), so Bragin silently reports 0% panel_match.
        """
        from rlpe.evaluation.metrics import _PAPER_ID_ALIASES

        assert _PAPER_ID_ALIASES, (
            "Expected the paper_id alias map to be non-empty; "
            "the Bragin hash->slug alias must be present."
        )
        assert _PAPER_ID_ALIASES.get("2e85364a3c605326") == "bragin2025", (
            "Expected the Bragin OD hash '2e85364a3c605326' to alias "
            "to the gold slug 'bragin2025'."
        )

    def test_normalize_paper_id_function(self):
        """``normalize_paper_id_for_eval`` is True for (hash, slug) and
        True for the symmetric (slug, hash), False for unrelated ids.

        The helper is the public API used by callers that need a
        boolean match decision without re-implementing the alias
        logic (e.g. cross-paper audits, reporting layers).
        """
        from rlpe.evaluation.metrics import normalize_paper_id_for_eval

        # The canonical mismatch that motivated this fix.
        assert normalize_paper_id_for_eval(
            "2e85364a3c605326", "bragin2025"
        ) is True
        # Symmetry: a caller may pass either side first.
        assert normalize_paper_id_for_eval(
            "bragin2025", "2e85364a3c605326"
        ) is True
        # Identical ids always match (no alias needed).
        assert normalize_paper_id_for_eval("bragin2025", "bragin2025") is True
        # Unrelated papers never match through the alias map.
        assert normalize_paper_id_for_eval(
            "5d5264c7bf0b0a43", "bragin2025"
        ) is False
        # Empty / None inputs are non-matches (defensive — the eval
        # loop already filters these, but the helper should be safe
        # to call from arbitrary contexts).
        assert normalize_paper_id_for_eval("", "bragin2025") is False
        assert normalize_paper_id_for_eval(None, "bragin2025") is False

    def test_aggregate_eval_includes_bragin_after_fix(self):
        """End-to-end: the 9-paper eval now contains Bragin with
        non-zero panel_match.

        Before the fix, Bragin reported ``panel_match == 0`` because
        pred paper_id (``2e85364a3c605326``) and gold paper_id
        (``bragin2025``) compared unequal under strict string match.
        After the fix, the per-paper Bragin row in the eval JSON
        must have at least 1 panel_match (the corpus has 11 Bragin
        panels).
        """
        from rlpe.evaluation import GoldPanel, evaluate
        from rlpe.evaluation.metrics import _figure_id_logical_key

        PAPER = "bragin2025"
        GOLD_FIGURE = "od_plate_bragin2025_p001_pl01"
        PRED_FIGURE = "od_plate_2e85364a3c605326_p006_pl01"

        # Sanity: the figure_id alias must already be in place
        # (commit f97f33a) so this test isolates the paper_id fix.
        assert _figure_id_logical_key(GOLD_FIGURE) == "bragin2025_pl01"
        assert _figure_id_logical_key(PRED_FIGURE) == "bragin2025_pl01"

        # 11 gold panels on Bragin pl01; pred side uses the hash
        # paper_id (mirroring what the real OpenDataLoader extractor
        # emits). The species ``Pantanellium moscowiense`` matches
        # what the gold side expects on a couple of those panels so
        # we can also see ``species_tp > 0``.
        gold = [
            GoldPanel(PAPER, GOLD_FIGURE, str(i), "Pantanellium moscowiense")
            for i in range(1, 12)
        ]
        preds = [
            {
                "paper_id": "2e85364a3c605326",  # pred-side hash
                "figure_id": PRED_FIGURE,
                "panel_id": str(i),
                "species": "Pantanellium moscowiense",
                "metadata": {},
            }
            for i in range(1, 12)
        ]
        report = evaluate(preds, gold)
        assert PAPER in report.papers, (
            "Expected Bragin to appear in the per-paper report "
            "after the paper_id alias fix."
        )
        metrics = report.papers[PAPER]
        assert metrics.n_gold == 11
        assert metrics.panel_match == 11, (
            "Expected all 11 Bragin panels to match once the "
            "paper_id alias is in place; got panel_match="
            f"{metrics.panel_match}/{metrics.n_gold}."
        )
        assert metrics.panel_match_rate == 1.0
        assert metrics.species_tp == 11
        assert metrics.species_f1 == 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))