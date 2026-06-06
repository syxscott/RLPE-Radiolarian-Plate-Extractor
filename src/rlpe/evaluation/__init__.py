"""Evaluation harness — see gold.py and metrics.py for the public API."""
from .gold import GOLD_SCHEMA_VERSION, GoldPanel, load_gold, match_panel, write_gold
from .metrics import (
    EvaluationReport,
    PaperMetrics,
    evaluate,
    evaluate_run,
    load_predictions_jsonl,
)
from .report import write_json_report, write_markdown_report

__all__ = [
    "GOLD_SCHEMA_VERSION",
    "GoldPanel",
    "load_gold",
    "match_panel",
    "write_gold",
    "EvaluationReport",
    "PaperMetrics",
    "evaluate",
    "evaluate_run",
    "load_predictions_jsonl",
    "write_json_report",
    "write_markdown_report",
]
