"""Regression tests for audit 2026-07-31 batch 6 (evaluation & export).

Covers:
  - _best_pred prefers species over confidence
  - one prediction cannot satisfy two gold panels (prefix-match double count)
  - DwC occurrence rows carry PBDB higher-rank classification
  - scientificNameAuthorship populated from the species string
  - low-confidence rows flagged needs_review
  - garbage journal values dropped
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class TestBestPred:
    def test_species_presence_beats_confidence(self):
        from rlpe.evaluation.gold import GoldPanel
        from rlpe.evaluation.metrics import evaluate

        gold = [GoldPanel("p", "f", "1", "Unuma echinatus")]
        preds = [
            # empty species, high confidence — must NOT win
            {
                "paper_id": "p",
                "figure_id": "f",
                "panel_id": "1",
                "species": None,
                "confidence": 0.99,
            },
            # correct species, lower confidence — must win
            {
                "paper_id": "p",
                "figure_id": "f",
                "panel_id": "1",
                "species": "Unuma echinatus",
                "confidence": 0.5,
            },
        ]
        report = evaluate(preds, gold)
        m = report.papers["p"]
        assert m.species_tp == 1, f"expected TP, got tp={m.species_tp} fn={m.species_fn}"


class TestPrefixDoubleCount:
    def test_one_pred_cannot_satisfy_two_gold(self):
        from rlpe.evaluation.gold import GoldPanel
        from rlpe.evaluation.metrics import evaluate

        gold = [
            GoldPanel("p", "f", "5", "A species"),
            GoldPanel("p", "f", "5a", "A species"),
        ]
        preds = [
            {
                "paper_id": "p",
                "figure_id": "f",
                "panel_id": "5",
                "species": "A species",
                "confidence": 0.9,
            },
        ]
        report = evaluate(preds, gold)
        m = report.papers["p"]
        assert m.panel_match <= 1, f"one prediction matched {m.panel_match} gold panels"
        assert m.species_tp <= 1


class TestDwcaHigherClassification:
    def _make_panel(self):
        from rlpe.schema_models import PanelRecord

        return PanelRecord(
            paper_id="p",
            figure_id="f",
            panel_id="1",
            species="Unuma echinatus",
            panel_path="/x.png",
            confidence=0.9,
            metadata={
                "paleodb_taxonomy": {
                    "kingdom": "Chromista",
                    "phylum": "Retaria",
                    "class": "Polycystinea",
                    "order": "Spumellaria",
                    "family": "Xiphostylidae",
                }
            },
        )

    def test_occurrence_row_higher_ranks(self):
        from rlpe.exporters.archive import _occurrence_row

        row = _occurrence_row(self._make_panel())
        assert row["kingdom"] == "Chromista"
        assert row["phylum"] == "Retaria"
        assert row["class"] == "Polycystinea"
        assert row["order"] == "Spumellaria"
        assert row["family"] == "Xiphostylidae"

    def test_authorship_from_species_string(self):
        from rlpe.exporters.archive import _occurrence_row

        p = self._make_panel()
        p.species = "Podocyrtis (Podocyrtites) amphora Haeckel, 1887"
        row = _occurrence_row(p)
        assert "Haeckel" in row["scientificNameAuthorship"]
        assert row["genus"] == "Podocyrtis"
        assert row["specificEpithet"] == "amphora"

    def test_no_taxonomy_no_crash(self):
        from rlpe.exporters.archive import _occurrence_row

        p = self._make_panel()
        p.metadata.paleodb_taxonomy = None
        row = _occurrence_row(p)
        assert row["family"] == ""
        assert row["kingdom"] == ""


class TestLowConfidenceReviewFlag:
    def test_low_confidence_flagged(self, tmp_path):
        from unittest.mock import patch

        from rlpe.config import PipelineConfig

        with (
            patch("rlpe.pipeline.GrobidClient"),
            patch("rlpe.pipeline.OCRBackend"),
            patch("rlpe.pipeline.TaxonRecognizer"),
            patch("rlpe.pipeline.PanelSegmenter"),
        ):
            cfg = PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "work")
            from rlpe.pipeline import RadiolarianPipeline

            pipe = RadiolarianPipeline(cfg)
            rows = [
                {
                    "paper_id": "p",
                    "figure_id": "f",
                    "panel_id": "1",
                    "species": "X",
                    "confidence": 0.3,
                    "panel_path": "/x.png",
                },
                {
                    "paper_id": "p",
                    "figure_id": "f",
                    "panel_id": "2",
                    "species": "Y",
                    "confidence": 0.8,
                    "panel_path": "/y.png",
                },
            ]
            out = pipe._finalize_rows(rows)
            low = [r for r in out if r["panel_id"] == "1"][0]
            assert low["metadata"]["needs_review"] is True
            assert "low_confidence" in low["metadata"]["review_reasons"]
            hi = [r for r in out if r["panel_id"] == "2"][0]
            assert not (hi.get("metadata") or {}).get("needs_review", False)


class TestJournalGarbage:
    def test_caption_fragment_journal_dropped(self):
        from rlpe.paper_metadata_cleanup import cleanup_paper_metadata

        cleaned, reasons = cleanup_paper_metadata(
            {
                "paper_id": "p",
                "title": "A real title",
                "journal": "Explanation of Plate",
                "doi": None,
            }
        )
        assert cleaned["journal"] is None
        assert "journal_extraction_failed" in reasons

    def test_real_journal_kept(self):
        from rlpe.paper_metadata_cleanup import cleanup_paper_metadata

        cleaned, _ = cleanup_paper_metadata(
            {
                "paper_id": "p",
                "title": "A real title",
                "journal": "Journal of Micropalaeontology",
                "doi": None,
            }
        )
        assert cleaned["journal"] == "Journal of Micropalaeontology"
