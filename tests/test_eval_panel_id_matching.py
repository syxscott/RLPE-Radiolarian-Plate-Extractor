"""Regression tests for the figure_id schema-variant fallback in
:func:`rlpe.evaluation.metrics.evaluate`.

Background
~~~~~~~~~~
The RLPE corpus emits figure_ids under two schemas depending on which
extraction path produced them:

* ``od_plate_<pid>_p<page>_pl<N>`` — the plate-caption matcher (the
  modern path).
* ``od_fig_<pid>_p<page>_<idx>`` — the per-figure matcher (legacy).

When :mod:`scripts.build_gold_bandini_pl070809` re-keyed Bandini 2011
plates 7/8/9 from the legacy ``od_fig_*`` schema to the verified
``od_plate_*`` schema, the gold file's pl08 (22 panels) and pl09 (18
panels) could no longer be matched against predictions that still
carried the legacy schema — every legacy pred was filtered out by the
strict ``figure_id != gold.figure_id`` guard in
:func:`rlpe.evaluation.metrics.evaluate`, inflating the unmatched
count and dragging the paper-level F1 down by ~30pp.

The fix relaxes the guard: when ``figure_id`` strings differ, fall
back to a canonical ``<pid>_p<page>`` key. If both sides share the
same (paper, page), the prediction is allowed to satisfy the gold
panel; the per-panel species compare still governs TP/FP/FN.

These tests pin down that fallback behaviour so a future refactor of
the guard cannot silently regress Bandini 2011 (or any other paper
whose gold was partially re-keyed).

audit 2026-08-02.
"""

from __future__ import annotations

from rlpe.evaluation import GoldPanel, evaluate
from rlpe.evaluation.metrics import _figure_id_logical_key


def _pred(paper_id: str, figure_id: str, panel_id: str, species: str) -> dict:
    return {
        "paper_id": paper_id,
        "figure_id": figure_id,
        "panel_id": panel_id,
        "species": species,
        "metadata": {},
    }


class TestFigureIdLogicalKey:
    """Direct unit tests on the helper used by the figure_id guard."""

    def test_od_plate_normalises_to_pid_page(self):
        # New-schema figure_id keeps everything up to and including _pNNN.
        fid = "od_plate_4f1bf415485765b8_p026_pl09"
        assert _figure_id_logical_key(fid) == "4f1bf415485765b8_p026"

    def test_od_fig_normalises_to_pid_page(self):
        # Legacy-schema figure_id strips the trailing _NN index.
        fid = "od_fig_4f1bf415485765b8_p026_04"
        assert _figure_id_logical_key(fid) == "4f1bf415485765b8_p026"

    def test_both_schemas_collapse_to_same_key(self):
        # The whole point of the fix: the two schemas refer to the
        # same logical figure when (paper, page) match.
        new = "od_plate_4f1bf415485765b8_p025_pl08"
        old = "od_fig_4f1bf415485765b8_p025_03"
        assert _figure_id_logical_key(new) == _figure_id_logical_key(old)

    def test_different_pages_do_not_collapse(self):
        a = "od_plate_paperA_p010_pl01"
        b = "od_fig_paperA_p011_01"
        assert _figure_id_logical_key(a) != _figure_id_logical_key(b)

    def test_non_od_figure_id_is_returned_unchanged(self):
        # The legacy gold files use simple ids like "plate_1"; the
        # helper must leave those alone so the strict-equality path
        # still gates their matches.
        assert _figure_id_logical_key("plate_1") == "plate_1"
        assert _figure_id_logical_key("fig_3") == "fig_3"

    def test_empty_figure_id_returns_empty(self):
        assert _figure_id_logical_key("") == ""


class TestFigureIdSchemaVariantMatching:
    """End-to-end tests on :func:`evaluate` exercising the new fallback."""

    PAPER_ID = "4f1bf415485765b8"

    def test_od_plate_gold_matches_od_fig_pred(self):
        """Regression: Bandini 2011 pl08 — gold uses ``od_plate_*``, pred
        uses ``od_fig_*`` (legacy schema). Without the fix, every pred
        was rejected and the paper-level F1 dropped by 30pp.
        """
        gold = [
            GoldPanel(
                paper_id=self.PAPER_ID,
                figure_id="od_plate_4f1bf415485765b8_p025_pl08",
                panel_id="1",
                species="Archaeodictyomitra cf. immenhauseri",
            ),
        ]
        pred = [
            _pred(
                self.PAPER_ID,
                "od_fig_4f1bf415485765b8_p025_03",  # legacy schema
                "1",
                "Archaeodictyomitra cf. immenhauseri",
            ),
        ]
        report = evaluate(pred, gold)
        m = report.papers[self.PAPER_ID]
        assert m.panel_match == 1
        assert m.species_tp == 1
        assert m.species_fn == 0

    def test_schema_variant_match_with_wrong_species(self):
        """Panel-match is granted (same logical figure), species compare
        still governs TP/FP/FN — so a wrong species on a schema-variant
        match counts as a mismatch (FP + FN), not a silent TP.
        """
        gold = [
            GoldPanel(
                paper_id=self.PAPER_ID,
                figure_id="od_plate_4f1bf415485765b8_p026_pl09",
                panel_id="5",
                species="Squinabollum aff. fossile",
            ),
        ]
        pred = [
            _pred(
                self.PAPER_ID,
                "od_fig_4f1bf415485765b8_p026_04",  # legacy schema
                "5",
                "Squinabollum aff. veneta",  # wrong species
            ),
        ]
        report = evaluate(pred, gold)
        m = report.papers[self.PAPER_ID]
        assert m.panel_match == 1  # panel gates passed
        assert m.species_tp == 0
        assert m.species_fp == 1
        assert m.species_fn == 1
        assert len(m.mismatches) == 1

    def test_same_schema_still_matches_strictly(self):
        """The fix is purely a fallback — the existing exact-match path
        must keep working for predictions and gold that already use the
        same schema."""
        gold = [
            GoldPanel(
                paper_id="p1",
                figure_id="od_plate_p1_p012_pl01",
                panel_id="1",
                species="Genus species",
            ),
        ]
        pred = [
            _pred("p1", "od_plate_p1_p012_pl01", "1", "Genus species"),
        ]
        report = evaluate(pred, gold)
        m = report.papers["p1"]
        assert m.species_tp == 1

    def test_cross_page_schema_variant_does_not_collide(self):
        """The fallback must NOT collapse different pages just because
        the schemas differ. Pred on page 026 must NOT satisfy gold on
        page 025 even when their prefix tokens happen to share the
        ``od_fig_``/``od_plate_`` schema.
        """
        gold = [
            GoldPanel(
                paper_id="p1",
                figure_id="od_plate_p1_p025_pl08",
                panel_id="1",
                species="Genus A",
            ),
        ]
        pred = [
            _pred("p1", "od_fig_p1_p026_04", "1", "Genus A"),
        ]
        report = evaluate(pred, gold)
        m = report.papers["p1"]
        assert m.panel_match == 0
        assert m.species_fn == 1

    def test_bandini2011_real_gold_subset_recovers_matches(self):
        """Integration-style regression on the actual Bandini 2011
        gold+pred files. The fix should turn the previously-unmatched
        pl08 (22 panels) and pl09 (18 panels) into panel_matches.

        We pin the expected lift to a small constant so future edits to
        the pred or gold files are caught explicitly (e.g. if someone
        drops a panel from the pred, the assertion fails loudly
        instead of silently masking the regression).
        """
        import json
        from pathlib import Path

        repo = Path(__file__).resolve().parents[1]
        gold_path = repo / "data" / "gold" / "bandini2011.jsonl"
        pred_path = repo / "work" / "llm_first_bandini2011.jsonl"
        if not (gold_path.exists() and pred_path.exists()):
            import pytest

            pytest.skip("bandini2011 gold/pred files not present in this checkout")

        gold_rows = [
            GoldPanel(
                paper_id=str(g["paper_id"]),
                figure_id=str(g["figure_id"]),
                panel_id=g.get("panel_id"),
                species=g.get("species"),
            )
            for g in (json.loads(l) for l in gold_path.read_text().splitlines() if l.strip())
        ]
        preds = []
        for line in pred_path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            preds.append(
                {
                    "paper_id": d.get("paper_id"),
                    "figure_id": d.get("figure_id"),
                    "panel_id": d.get("panel_id"),
                    "species": d.get("species"),
                    "metadata": d.get("metadata") or {},
                }
            )

        report = evaluate(preds, gold_rows)
        m = report.papers[self.PAPER_ID]

        # Pre-fix baseline: 172 panel_matches (63.0% panel_match_rate).
        # Post-fix: >= 201 panel_matches (73.6% panel_match_rate) per
        # the Phase 68 audit measurement on 2026-08-02. We pin at 195
        # to leave headroom for minor upstream changes (the lift over
        # the strict-match baseline is what matters).
        assert m.panel_match >= 195, (
            f"Bandini 2011 panel_match regressed: got {m.panel_match}, "
            f"expected >= 195 (pre-fix baseline was 172)"
        )
        # The fix should lift panel_match_rate from 63% to >= 70%.
        assert m.panel_match_rate >= 0.70, (
            f"panel_match_rate regressed: got {m.panel_match_rate:.3f}, expected >= 0.70"
        )
