"""Tests for the evaluation harness (metrics.py + report.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlpe.evaluation import (
    EvaluationReport,
    GoldPanel,
    PaperMetrics,
    compare_before_after,
    evaluate,
    evaluate_run,
    load_predictions_jsonl,
    write_json_report,
    write_markdown_report,
)
from rlpe.evaluation.metrics import _norm_species  # noqa: E402

# _levenshtein and _species_close_enough were removed (commit follow-up):
# the species-fallback never fires on the real 7-paper eval set, so the
# dead code (and its 8 unit tests) are deleted. The 7 tests that
# exercise the actual evaluate() path remain.


GOLD_DIR = Path(__file__).resolve().parents[1] / "data" / "gold"


def _pred(paper_id, panel_id, species, matcher_type="heuristic"):
    return {
        "paper_id": paper_id,
        "panel_id": panel_id,
        "species": species,
        "metadata": {"matcher_type": matcher_type},
    }


class TestEvaluate:
    def test_perfect_match(self):
        gold = [GoldPanel("p1", "f1", "1", "Genus species")]
        preds = [_pred("p1", "1", "Genus species")]
        report = evaluate(preds, gold)
        assert report.papers["p1"].species_tp == 1
        assert report.papers["p1"].species_fp == 0
        assert report.papers["p1"].species_fn == 0
        assert report.aggregate["species_precision"] == 1.0
        assert report.aggregate["species_recall"] == 1.0
        assert report.aggregate["species_f1"] == 1.0

    def test_wrong_species(self):
        gold = [GoldPanel("p1", "f1", "1", "Genus species")]
        preds = [_pred("p1", "1", "Other species")]
        report = evaluate(preds, gold)
        # TP=0, FP=1 (predicted), FN=1 (gold unmatched)
        assert report.papers["p1"].species_tp == 0
        assert report.papers["p1"].species_fp == 1
        assert report.papers["p1"].species_fn == 1
        assert report.papers["p1"].exact_match == 0

    def test_missing_panel(self):
        gold = [GoldPanel("p1", "f1", "1", "Genus species")]
        preds = [_pred("p1", "2", "Genus species")]  # wrong panel
        report = evaluate(preds, gold)
        assert report.papers["p1"].species_fn == 1
        assert report.papers["p1"].panel_match == 0

    def test_extra_predicted_panel(self):
        gold = [GoldPanel("p1", "f1", "1", "Genus species")]
        preds = [
            _pred("p1", "1", "Genus species"),
            _pred("p1", "99", "Other"),  # extra
        ]
        report = evaluate(preds, gold)
        # Extra panel doesn't add to TP/FP/FN unless it matches a gold panel
        assert report.papers["p1"].n_pred_panels == 2
        assert report.papers["p1"].species_tp == 1

    def test_species_normalization(self):
        gold = [GoldPanel("p1", "f1", "1", "Genus species")]
        preds = [_pred("p1", "1", "  Genus   species  ")]
        report = evaluate(preds, gold)
        assert report.papers["p1"].species_tp == 1

    def test_question_prefix_normalization(self):
        """The leading "?" is an uncertainty marker on the genus and
        is non-significant. Gold may have "?Sethocapsa sp." while
        predictions have "Sethocapsa sp" (or vice versa); both must
        count as a species TP."""
        gold = [GoldPanel("p1", "f1", "1", "?Sethocapsa sp.")]
        preds = [_pred("p1", "1", "Sethocapsa sp")]
        report = evaluate(preds, gold)
        assert report.papers["p1"].species_tp == 1
        # Reverse: gold without "?", pred with "?" also matches.
        gold2 = [GoldPanel("p1", "f1", "1", "Sethocapsa sp")]
        preds2 = [_pred("p1", "1", "?Sethocapsa sp")]
        report2 = evaluate(preds2, gold2)
        assert report2.papers["p1"].species_tp == 1

    def test_case_insensitive_match(self):
        gold = [GoldPanel("p1", "f1", "1", "Genus Species")]
        preds = [_pred("p1", "1", "genus species")]
        report = evaluate(preds, gold)
        assert report.papers["p1"].species_tp == 1

    def test_empty_gold_species(self):
        # Gold has no species (e.g. scale bar) but predicted a species → FP
        gold = [GoldPanel("p1", "f1", "1", None)]
        preds = [_pred("p1", "1", "Genus species")]
        report = evaluate(preds, gold)
        assert report.papers["p1"].species_fp == 1

    def test_multiple_papers(self):
        gold = [
            GoldPanel("p1", "f1", "1", "A"),
            GoldPanel("p2", "f1", "1", "B"),
        ]
        preds = [
            _pred("p1", "1", "A"),
            _pred("p2", "1", "wrong"),
        ]
        report = evaluate(preds, gold)
        assert "p1" in report.papers
        assert "p2" in report.papers
        assert report.papers["p1"].species_tp == 1
        assert report.papers["p2"].species_tp == 0

    def test_panel_match_property(self):
        gold = [GoldPanel("p1", "f1", "1", "A")]
        preds = [_pred("p1", "1", "B")]  # panel matches, species doesn't
        report = evaluate(preds, gold)
        m = report.papers["p1"]
        assert m.panel_match == 1
        assert m.exact_match == 0
        assert m.panel_match_rate == 1.0
        assert m.exact_match_rate == 0.0

    def test_same_panel_label_in_different_figures_does_not_collide(self):
        """Regression test for eval bug #3: pred "1" in fig_1 must not
        match gold "1" in fig_2. Before the fix, pred_groups was keyed
        on (paper_id, panel_id) only, so a single pred "1" was
        overcounting against every figure that contained a "1" panel.
        In bandini2011, where "1" appears in 6 figures, this single
        bug inflated species recall by ~5x.
        """
        gold = [
            GoldPanel("p1", "fig_1", "1", "Species in fig 1"),
            GoldPanel("p1", "fig_2", "1", "Species in fig 2"),
            GoldPanel("p1", "fig_3", "1", "Species in fig 3"),
        ]
        # Single prediction: panel "1" in fig_1 only.
        preds = [{**_pred("p1", "1", "Species in fig 1"), "figure_id": "fig_1"}]
        report = evaluate(preds, gold)
        m = report.papers["p1"]
        # Only fig_1 should match. fig_2 and fig_3 are NOT matched
        # (no prediction in those figures), so species_fn = 2.
        assert m.species_tp == 1
        assert m.species_fn == 2
        assert m.species_fp == 0
        assert m.n_pred_panels == 1

    def test_figure_id_must_match_for_match_to_count(self):
        """The figure_id gate: a pred in fig_X must NOT count for a
        gold entry in fig_Y, even with the same panel_id."""
        gold = [GoldPanel("p1", "fig_1", "1", "Species A")]
        preds = [{**_pred("p1", "1", "Species A"), "figure_id": "fig_2"}]  # wrong figure
        report = evaluate(preds, gold)
        m = report.papers["p1"]
        # panel_match counts gold panels that have at least one pred in
        # the same figure with the right panel_id; the pred is in the
        # wrong figure so it doesn't count.
        assert m.panel_match == 0
        assert m.species_fn == 1
        assert m.species_tp == 0

    def test_alphabetic_suffix_panel_label_does_not_count_as_match(self):
        """Regression test for eval bug #1: prefix-match collapse.
        Gold "5" must NOT match pred "10" (different panel).
        Gold "5" + pred "5a" IS allowed (alphabetic suffix is a sub-label).
        """
        gold = [GoldPanel("p1", "f1", "5", "Species five")]
        # Numeric suffix: "10" should NOT match gold "5"
        preds = [{**_pred("p1", "10", "Species ten"), "figure_id": "f1"}]
        report = evaluate(preds, gold)
        m = report.papers["p1"]
        assert m.panel_match == 0
        assert m.species_tp == 0
        assert m.species_fn == 1

        # Alphabetic suffix: "5a" should match gold "5"
        preds2 = [{**_pred("p1", "5a", "Species five-a"), "figure_id": "f1"}]
        report2 = evaluate(preds2, gold)
        m2 = report2.papers["p1"]
        assert m2.panel_match == 1
        assert m2.species_fn == 1  # species doesn't match but panel does


class TestPaperMetrics:
    def test_zero_division_safe(self):
        m = PaperMetrics(paper_id="p1")
        assert m.species_precision == 0.0
        assert m.species_recall == 0.0
        assert m.species_f1 == 0.0
        assert m.panel_match_rate == 0.0

    def test_f1_calculation(self):
        m = PaperMetrics(paper_id="p1", species_tp=3, species_fp=1, species_fn=1)
        # P = 3/4, R = 3/4, F1 = 0.75
        assert abs(m.species_f1 - 0.75) < 1e-9


class TestLoadPredictions:
    def test_load_jsonl(self, tmp_path):
        path = tmp_path / "p.jsonl"
        path.write_text(
            '{"paper_id":"p1","panel_id":"1","species":"A"}\n'
            '{"paper_id":"p1","panel_id":"2","species":"B"}\n'
        )
        preds = load_predictions_jsonl(path)
        assert len(preds) == 2
        assert preds[0]["paper_id"] == "p1"
        # metadata is now passed through so _is_real_prediction can filter
        assert "metadata" in preds[0]

    def test_skip_empty_lines(self, tmp_path):
        path = tmp_path / "p.jsonl"
        path.write_text('\n{"paper_id":"p1","panel_id":"1","species":"A"}\n\n')
        preds = load_predictions_jsonl(path)
        assert len(preds) == 1


class TestPlaceholderFilter:
    """Real pipeline output often contains rows where the upstream caption
    parser failed (matcher_type='skipped-placeholder-caption'). These rows
    carry no species and no real signal. The eval harness filters them out
    so they don't inflate the denominator."""

    def test_skipped_placeholder_does_not_count_as_pred(self, capsys):
        gold = [GoldPanel("p1", "f1", "1", "Genus species")]
        preds = [
            _pred("p1", "1", None, matcher_type="skipped-placeholder-caption"),
            # the real prediction comes from a different (real) source
            _pred("p1", "1", "Genus species", matcher_type="heuristic"),
        ]
        report = evaluate(preds, gold)
        # species_tp should be 1 — the placeholder row was filtered out
        m = report.papers["p1"]
        assert m.species_tp == 1
        assert m.species_fp == 0
        assert m.species_fn == 0
        # the filter message is surfaced
        out = capsys.readouterr().out
        assert "filtered 1 placeholder" in out

    def test_only_placeholder_records_means_no_match(self, capsys):
        gold = [GoldPanel("p1", "f1", "1", "Genus")]
        preds = [
            _pred("p1", "1", None, matcher_type="skipped-placeholder-caption"),
        ]
        report = evaluate(preds, gold)
        m = report.papers["p1"]
        # No real prediction → species_fn counts the missed gold species
        assert m.species_fn == 1
        assert m.species_tp == 0

    def test_heuristic_with_species_survives_filter(self, capsys):
        gold = [GoldPanel("p1", "f1", "1", "Genus")]
        preds = [_pred("p1", "1", "Genus", matcher_type="heuristic")]
        report = evaluate(preds, gold)
        # No filter log printed when nothing is filtered
        out = capsys.readouterr().out
        assert "filtered" not in out
        m = report.papers["p1"]
        assert m.species_tp == 1


class TestEvaluateRun:
    def test_against_real_gold(self):
        """Smoke test: evaluate batch4_v2 predictions against the gold set."""
        preds_path = Path(__file__).resolve().parents[1] / "work" / "batch4_v2" / "results.jsonl"
        if not preds_path.exists():
            pytest.skip(f"{preds_path} not available")
        report = evaluate_run(preds_path, GOLD_DIR)
        # Each paper should appear
        assert len(report.papers) > 0
        # Aggregate F1 should be in [0, 1]
        assert 0.0 <= report.aggregate["species_f1"] <= 1.0


class TestReports:
    def test_markdown_report(self, tmp_path):
        m = PaperMetrics(
            paper_id="p1",
            n_gold=10,
            n_pred_panels=12,
            panel_match=8,
            species_tp=5,
            species_fp=3,
            species_fn=5,
            exact_match=4,
        )
        report = EvaluationReport(
            papers={"p1": m},
            aggregate={
                "n_papers": 1,
                "n_gold": 10,
                "species_precision": 0.625,
                "species_recall": 0.5,
                "species_f1": 0.555,
                "panel_match_rate": 0.8,
                "exact_match_rate": 0.4,
            },
        )
        out = write_markdown_report(report, tmp_path / "r.md", title="Test")
        text = out.read_text()
        assert "# Test" in text
        assert "p1" in text
        assert "Species F1" in text

    def test_json_report(self, tmp_path):
        m = PaperMetrics(paper_id="p1", n_gold=5, species_tp=3)
        report = EvaluationReport(papers={"p1": m}, aggregate={"n_papers": 1})
        out = write_json_report(report, tmp_path / "r.json")
        data = json.loads(out.read_text())
        assert "p1" in data["papers"]
        assert "aggregate" in data


class TestMissLists:
    """Per-panel miss lists (mismatches, unmatched) on the eval report.

    The miss lists let callers (CI scripts, hand-audits) drill into
    *which* gold panels were matched-but-wrong or not matched at all
    without re-diffing predictions and gold by hand. Without this,
    the 12 bandini2011 misses and the 9 bandini2006 unmatched
    Plate-1 panels are uncategorizable from the eval output.
    """

    def test_mismatch_recorded_when_species_differs(self):
        gold = [GoldPanel("p1", "f1", "1", "Genus species")]
        preds = [_pred("p1", "1", "Other species")]
        report = evaluate(preds, gold)
        m = report.papers["p1"]
        assert len(m.mismatches) == 1
        assert m.mismatches[0]["panel_id"] == "1"
        assert m.mismatches[0]["figure_id"] == "f1"
        assert m.mismatches[0]["expected"] == "Genus species"
        assert m.mismatches[0]["predicted"] == "Other species"
        assert m.unmatched == []

    def test_unmatched_recorded_when_no_pred(self):
        gold = [GoldPanel("p1", "f1", "5", "Genus species")]
        preds = [_pred("p1", "1", "Other species")]
        report = evaluate(preds, gold)
        m = report.papers["p1"]
        assert m.mismatches == []
        assert len(m.unmatched) == 1
        assert m.unmatched[0]["panel_id"] == "5"
        assert m.unmatched[0]["expected"] == "Genus species"
        assert "predicted" not in m.unmatched[0]

    def test_perfect_match_has_no_miss_lists(self):
        gold = [GoldPanel("p1", "f1", "1", "Genus species")]
        preds = [_pred("p1", "1", "Genus species")]
        report = evaluate(preds, gold)
        m = report.papers["p1"]
        assert m.mismatches == []
        assert m.unmatched == []

    def test_miss_lists_appear_in_to_dict(self):
        gold = [GoldPanel("p1", "f1", "1", "Genus species")]
        preds = [_pred("p1", "1", "Other species")]
        report = evaluate(preds, gold)
        d = report.papers["p1"].to_dict()
        assert "mismatches" in d
        assert "unmatched" in d
        assert d["mismatches"][0]["expected"] == "Genus species"

    def test_unmatched_gold_with_no_species_not_counted(self):
        """A gold panel with an empty species string is not counted as
        a missed species in either direction, so it should not appear
        in the unmatched list. (It is, however, still a panel-miss if
        no prediction matches by panel_id.)"""
        gold = [GoldPanel("p1", "f1", "1", "")]
        preds = []
        report = evaluate(preds, gold)
        m = report.papers["p1"]
        # No species to mismatch on, so neither list records it.
        assert m.mismatches == []
        assert m.unmatched == []


class TestSpeciesNormAsymmetric:
    """Asymmetric gold/pred qualifier stripping. The caption parser
    captures optional qualifiers that the gold annotator drops (or
    vice versa). For 11 of the 18 v18 mismatches, the asymmetry is
    a parser-vs-annotator convention, not a real species difference:

        gold  = "Theocampe"                   (bare genus)
        pred  = "Theocampe sp"                (parser added "sp")
        gold  = "Eucyrtidiellum unumaense"    (no subspecies)
        pred  = "Eucyrtidiellum unumaense pustulatum"   (subspecies)
        gold  = "Spumellarian gen. et sp. indet"   (long form)
        pred  = "Spumellarian gen"            (parser truncation)
        gold  = "Archaeodictyomitra sp. aff. minoensis"  (spelling)
        pred  = "Archeodictyomitra sp. aff. minoensis"

    These four pairs all refer to the same species. The 7 remaining
    mismatches (e.g. "Pseudoeucyrtis sp" vs "Pseudoeucyrtis sp. B")
    are kept as-is because "B" is a meaningful list identifier — not
    a parser convention — and collapsing them would over-match.
    """

    @pytest.mark.parametrize(
        "gold,pred",
        [
            # bandini2011: gold has bare genus, pred has genus + "sp"
            ("Theocampe", "Theocampe sp"),
            ("Obeliscoites", "Obeliscoites sp"),
            ("Hiscocapsa", "Hiscocapsa sp"),
            ("Parahsuum", "Parahsuum sp"),
            ("Canoptum", "Canoptum sp"),
            # bandini2011: trinomial — gold has binomial, pred has subspecies
            ("Eucyrtidiellum unumaense", "Eucyrtidiellum unumaense pustulatum"),
            ("Deviatus diamphidius", "Deviatus diamphidius hipposidericus"),
            # boughdiri: spelling variant "Archaeo" / "Archeo"
            ("Archaeodictyomitra sp. aff. minoensis", "Archeodictyomitra sp. aff. minoensis"),
            # hollis: "X gen" parser truncation ↔ "X indet" gold long form
            ("Spumellarian gen. et sp. indet", "Spumellarian gen"),
        ],
    )
    def test_asymmetric_qualifier_normalization(self, gold, pred):
        # audit 2026-07-31: equivalence is now judged by
        # ``_species_compatible`` — subspecies preds match their
        # species-level gold (subspecies is a refinement), while two
        # DIFFERENT subspecies remain a mismatch. Pure string equality
        # of the normalised forms no longer holds for trinomials.
        from rlpe.evaluation.metrics import _species_compatible

        g = _norm_species(gold)
        p = _norm_species(pred)
        assert _species_compatible(g, p), (
            f"gold={gold!r} → {g!r}, pred={pred!r} → {p!r} must be compatible"
        )

    @pytest.mark.parametrize(
        "gold,pred",
        [
            # beccaro: "sp. B" is a paper-list identifier (B-th undetermined
            # species). Collapsing it to bare "sp" would over-match two
            # genuinely different species. The eval should report this as
            # a real mismatch.
            ("Pseudoeucyrtis sp", "Pseudoeucyrtis sp. B"),
            # danelian: same shape — "sp. A" is a list identifier.
            ("Archaeodictyomitra sp", "Archaeodictyomitra sp. A"),
            # bandini2011: "sp. aff. robustum" is more specific than bare
            # "sp" (it means "species affinis P. robustum"). Real mismatch.
            ("Praewilliriedellum sp", "Praewilliriedellum sp. aff. robustum"),
        ],
    )
    def test_list_identifier_not_collapsed(self, gold, pred):
        """A "sp. X" identifier (letter/digit) is a real species
        differentiator; the normalization must not collapse it.
        """
        assert _norm_species(gold).lower() != _norm_species(pred).lower()

    def test_end_to_end_match_via_evaluate(self):
        """Smoke test that the asymmetric normalization actually
        makes the eval TP a real prediction that was previously a
        mismatch. Mirrors the bandini2011 'Theocampe' case."""
        gold = [GoldPanel("p1", "f1", "1", "Theocampe")]
        preds = [_pred("p1", "1", "Theocampe sp")]
        report = evaluate(preds, gold)
        assert report.papers["p1"].species_tp == 1
        assert report.papers["p1"].mismatches == []

    def test_does_not_over_match_unrelated_species(self):
        """The normalization must not cause unrelated species to
        compare equal. The 'sp' stripping is suffix-only — it cannot
        collapse two distinct genera."""
        assert _norm_species("Theocampe sp") != _norm_species("Obeliscoites sp")

    def test_handles_none_and_empty(self):
        assert _norm_species(None) == ""
        assert _norm_species("") == ""


# ---------------------------------------------------------------------------
# Round 9: compare_before_after merge-key fix
# ---------------------------------------------------------------------------


class TestCompareBeforeAfterRound9:
    """Round 9 (Bug-M2): ``compare_before_after`` previously merged on
    ``(paper_id, figure_id, panel_path)``. The LLM-first path leaves
    ``panel_path=None`` while the classical rules path writes a real
    path, so the merge silently dropped every row and returned
    ``n_samples=0, match_improvement=0.0`` regardless of actual
    performance. Post-fix the merge key is ``(paper_id, figure_id,
    panel_id)`` and ``n_samples`` reflects the real comparison size.
    """

    def test_llm_first_with_none_panel_path_drops_no_rows(self):
        """The regression case: before (classical) has panel_path set,
        after (LLM-first) has panel_path=None. Pre-fix this dropped
        the row and returned n_samples=0; post-fix the merge succeeds
        because the key is panel_id, not panel_path."""
        before = [
            {
                "paper_id": "p1",
                "figure_id": "fig1",
                "panel_id": "1",
                "species": "A",
                "panel_path": "/work/p1/fig1_panel_01.png",
            },
        ]
        after = [
            {
                "paper_id": "p1",
                "figure_id": "fig1",
                "panel_id": "1",
                "species": "A",
                "panel_path": None,
            },
        ]
        gold = [{"paper_id": "p1", "figure_id": "fig1", "panel_id": "1", "species": "A"}]
        result = compare_before_after(before, after, gold)
        assert result["n_samples"] == 1, "LLM-first vs rules comparison must not silently drop rows"
        assert result["match_acc_before"] == 1.0
        assert result["match_acc_after"] == 1.0
        assert result["match_improvement"] == 0.0

    def test_match_improvement_detects_llm_fix(self):
        """The motivating use case: rules miss panel 1's species, LLM
        catches it. Pre-fix n_samples=0 → improvement=0. Post-fix
        n_samples=1 → improvement reflects the actual delta."""
        before = [
            {
                "paper_id": "p1",
                "figure_id": "fig1",
                "panel_id": "1",
                "species": "wrong_species",
                "panel_path": "x.png",
            },
        ]
        after = [
            {
                "paper_id": "p1",
                "figure_id": "fig1",
                "panel_id": "1",
                "species": "right_species",
                "panel_path": None,
            },
        ]
        gold = [
            {"paper_id": "p1", "figure_id": "fig1", "panel_id": "1", "species": "right_species"}
        ]
        result = compare_before_after(before, after, gold)
        assert result["n_samples"] == 1
        assert result["match_acc_before"] == 0.0
        assert result["match_acc_after"] == 1.0
        assert result["match_improvement"] == 1.0

    def test_placeholder_rows_excluded_from_merge(self):
        """Rows with panel_id=None (the placeholder-caption skip path)
        must NOT participate in the merge — they're junk and would
        inflate the denominator with non-panel rows."""
        before = [
            {
                "paper_id": "p1",
                "figure_id": "fig1",
                "panel_id": None,
                "species": None,
                "panel_path": None,
            },  # placeholder
            {
                "paper_id": "p1",
                "figure_id": "fig1",
                "panel_id": "1",
                "species": "A",
                "panel_path": "x.png",
            },
        ]
        after = [
            {
                "paper_id": "p1",
                "figure_id": "fig1",
                "panel_id": "1",
                "species": "A",
                "panel_path": None,
            },
        ]
        gold = [{"paper_id": "p1", "figure_id": "fig1", "panel_id": "1", "species": "A"}]
        result = compare_before_after(before, after, gold)
        # Only the panel_id="1" row participates (1 from each side).
        assert result["n_samples"] == 1

    def test_gemma_confidence_mean_reported(self):
        """The gemma_confidence_mean field must surface when the
        after-side rows carry metadata.gemma_confidence. Pre-fix the
        merge dropped too many rows to surface anything meaningful."""
        before = [
            {
                "paper_id": "p1",
                "figure_id": "fig1",
                "panel_id": "1",
                "species": "A",
                "panel_path": "x.png",
            },
        ]
        after = [
            {
                "paper_id": "p1",
                "figure_id": "fig1",
                "panel_id": "1",
                "species": "A",
                "panel_path": None,
                "metadata": {"gemma_confidence": 0.85},
            },
        ]
        gold = [{"paper_id": "p1", "figure_id": "fig1", "panel_id": "1", "species": "A"}]
        result = compare_before_after(before, after, gold)
        assert result["gemma_confidence_mean"] == 0.85

    def test_empty_inputs_return_zero_dict(self):
        """Both sides empty → no rows merged → no crash, sensible defaults."""
        result = compare_before_after([], [], [])
        assert result["n_samples"] == 0
        assert result["match_improvement"] == 0.0
