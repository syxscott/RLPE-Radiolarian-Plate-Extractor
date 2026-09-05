"""Markdown + JSON report writer for evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path

from .metrics import EvaluationReport, PaperMetrics


def write_json_report(report: EvaluationReport, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, sort_keys=True, ensure_ascii=False)
    return target


def _paper_row(m: PaperMetrics) -> str:
    return (
        f"| {m.paper_id} "
        f"| {m.n_gold} "
        f"| {m.n_pred_panels} "
        f"| {m.panel_match_rate:.1%} "
        f"| {m.species_precision:.1%} "
        f"| {m.species_recall:.1%} "
        f"| {m.species_f1:.1%} "
        f"| {m.exact_match_rate:.1%} |"
    )


def write_markdown_report(
    report: EvaluationReport,
    target: Path,
    title: str = "RLPE Evaluation Report",
    notes: str = "",
) -> Path:
    """Write a human-readable Markdown summary."""
    target.parent.mkdir(parents=True, exist_ok=True)
    agg = report.aggregate
    lines: list[str] = []
    lines.append(f"# {title}\n")
    # Audit 2026-09-04 eval-1: the gold-provenance caveat goes at the
    # TOP of the report — before any F1 table — so nobody reads a
    # bare "Species F1: 82.96%" without knowing the gold is
    # parser-derived (self-consistency), not image-verified.
    lines.append(f"> **Gold provenance:** {report.gold_provenance}\n")
    if notes:
        lines.append(notes + "\n")
    lines.append("## Aggregate\n")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Papers | {agg.get('n_papers', 0)} |")
    lines.append(f"| Gold panels | {agg.get('n_gold', 0)} |")
    lines.append(f"| Species precision | {agg.get('species_precision', 0):.1%} |")
    lines.append(f"| Species recall | {agg.get('species_recall', 0):.1%} |")
    # Audit 2026-09-04 eval-5: the aggregate no longer carries a
    # single ``species_f1`` key — the micro and macro variants are
    # surfaced separately so reviewers cannot be confused by two
    # different "Species F1" numbers in the same report. ``agg.get``
    # fallback to ``species_f1`` keeps old callers rendering a value
    # rather than 0.0%.
    lines.append(
        f"| Species F1 (micro) | {agg.get('species_f1_micro', agg.get('species_f1', 0)):.1%} |"
    )
    lines.append(
        f"| Species F1 (macro) | {agg.get('species_f1_macro', agg.get('species_f1', 0)):.1%} |"
    )
    lines.append(f"| Panel match rate | {agg.get('panel_match_rate', 0):.1%} |")
    lines.append(f"| Exact match rate | {agg.get('exact_match_rate', 0):.1%} |")
    lines.append("")
    lines.append("## Per-paper\n")
    lines.append(
        "| Paper | Gold | Pred | Panel-match | Species P | Species R | Species F1 | Exact |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for m in sorted(report.papers.values(), key=lambda x: x.paper_id):
        lines.append(_paper_row(m))
    lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")
    return target
