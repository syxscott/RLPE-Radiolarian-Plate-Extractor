"""Regression: audit 2026-09-04 eval-1 — evaluation reports carried no
gold-provenance marker.

The gold set is derived from caption-parser output
(``scripts/build_gold_from_captions.py``), so every F1 number the
evaluator prints measures **parser self-consistency**, not accuracy
against manually annotated panels — the 2026-09-02 re-measurement
confirmed it (claimed F1 0.84 → image-verified 0.075). ``gold.py``'s
docstring says this, but the Markdown/JSON evaluation reports — the
artifacts people actually read and cite — shipped bare F1 tables with
no caveat, which is how README ended up advertising "82.96% F1" as
unqualified progress.

Contract: every evaluation report (Markdown AND JSON) now carries the
provenance statement by default; a caller may override it (e.g. once
a genuinely image-verified gold exists), but never silently omit it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe.evaluation.metrics import EvaluationReport, PaperMetrics
from rlpe.evaluation.report import write_json_report, write_markdown_report


def _report() -> EvaluationReport:
    pm = PaperMetrics(
        paper_id="dummy",
        n_gold=10,
        n_pred_panels=8,
        panel_match=6,
        species_tp=5,
        species_fp=3,
        species_fn=5,
        exact_match=4,
    )
    return EvaluationReport(
        papers={"dummy": pm},
        aggregate={"n_papers": 1, "species_f1": 0.5},
    )


class TestJsonReportProvenance:
    def test_json_report_carries_gold_provenance(self, tmp_path):
        target = write_json_report(_report(), tmp_path / "r.json")
        data = json.loads(target.read_text(encoding="utf-8"))
        prov = data.get("gold_provenance")
        assert isinstance(prov, str) and prov
        low = prov.lower()
        assert "parser" in low and ("self-consistency" in low or "self consistency" in low)
        assert "not image-verified" in low or "not image verified" in low

    def test_provenance_override_preserved(self, tmp_path):
        rep = _report()
        rep.gold_provenance = "image-verified by two human annotators (2026-09)"
        target = write_json_report(rep, tmp_path / "r.json")
        data = json.loads(target.read_text(encoding="utf-8"))
        assert data["gold_provenance"] == ("image-verified by two human annotators (2026-09)")


class TestMarkdownReportProvenance:
    def test_markdown_report_has_caveat_before_f1_table(self, tmp_path):
        target = write_markdown_report(_report(), tmp_path / "r.md")
        text = target.read_text(encoding="utf-8")
        assert "gold" in text.lower() and "parser" in text.lower()
        assert "self-consistency" in text.lower()
        # The caveat must appear BEFORE the Species F1 row, not buried
        # after the table a reader has already consumed.
        caveat_pos = text.lower().index("self-consistency")
        f1_pos = text.lower().index("species f1")
        assert caveat_pos < f1_pos

    def test_markdown_default_notes_not_required(self, tmp_path):
        # Caller passing no notes still gets the provenance caveat.
        target = write_markdown_report(_report(), tmp_path / "r.md", notes="")
        text = target.read_text(encoding="utf-8")
        assert "parser" in text.lower()


class TestReportDataclassDefault:
    def test_default_provenance_present_on_fresh_report(self):
        rep = EvaluationReport()
        assert rep.gold_provenance
        assert "parser" in rep.gold_provenance.lower()

    def test_to_dict_includes_provenance(self):
        data = _report().to_dict()
        assert "gold_provenance" in data
