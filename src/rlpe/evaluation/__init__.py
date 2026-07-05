"""Evaluation harness — see gold.py and metrics.py for the public API."""

from .gold import GOLD_SCHEMA_VERSION, GoldPanel, load_gold, match_panel, write_gold
from .image_label_check import ImageLabelStats, run_image_label_check
from .metrics import (
    EvaluationReport,
    PaperMetrics,
    compare_before_after,
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
    "compare_before_after",
    "load_predictions_jsonl",
    "write_json_report",
    "write_markdown_report",
    "ImageLabelStats",
    "run_image_label_check",
]
