"""Tests for the evaluation harness (metrics.py + report.py)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlpe.evaluation import (
    EvaluationReport,
    GoldPanel,
    PaperMetrics,
    evaluate,
    evaluate_run,
    load_gold,
    load_predictions_jsonl,
    write_json_report,
    write_markdown_report,
)


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
            paper_id="p1", n_gold=10, n_pred_panels=12, panel_match=8,
            species_tp=5, species_fp=3, species_fn=5, exact_match=4,
        )
        report = EvaluationReport(papers={"p1": m}, aggregate={
            "n_papers": 1, "n_gold": 10,
            "species_precision": 0.625, "species_recall": 0.5,
            "species_f1": 0.555, "panel_match_rate": 0.8,
            "exact_match_rate": 0.4,
        })
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
