"""Evaluation metrics for RLPE.

Compares a predicted JSONL of panels against a gold JSONL of (panel_id,
species) and reports:

    - panel_match_rate:   fraction of gold panels where predicted panel exists
                          (regardless of species)
    - species_prf:        precision/recall/F1 on species assignment
    - exact_match_rate:   fraction of gold panels where both panel and
                          species match
    - paper_breakdown:    per-paper species_prf and panel counts

The metrics are designed for the batch4_v2 test set but generalise to
any paper with a gold JSONL.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .gold import GoldPanel, load_gold, match_panel


@dataclass(slots=True)
class PaperMetrics:
    paper_id: str
    n_gold: int = 0
    n_pred_panels: int = 0
    panel_match: int = 0
    species_tp: int = 0
    species_fp: int = 0
    species_fn: int = 0
    exact_match: int = 0

    @property
    def species_precision(self) -> float:
        return self.species_tp / max(1, self.species_tp + self.species_fp)

    @property
    def species_recall(self) -> float:
        return self.species_tp / max(1, self.species_tp + self.species_fn)

    @property
    def species_f1(self) -> float:
        p, r = self.species_precision, self.species_recall
        return 2 * p * r / max(1e-9, p + r)

    @property
    def panel_match_rate(self) -> float:
        return self.panel_match / max(1, self.n_gold)

    @property
    def exact_match_rate(self) -> float:
        return self.exact_match / max(1, self.n_gold)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "n_gold": self.n_gold,
            "n_pred_panels": self.n_pred_panels,
            "panel_match": self.panel_match,
            "species_precision": self.species_precision,
            "species_recall": self.species_recall,
            "species_f1": self.species_f1,
            "panel_match_rate": self.panel_match_rate,
            "exact_match_rate": self.exact_match_rate,
        }


@dataclass(slots=True)
class EvaluationReport:
    papers: dict[str, PaperMetrics] = field(default_factory=dict)
    aggregate: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "papers": {k: v.to_dict() for k, v in self.papers.items()},
            "aggregate": dict(self.aggregate),
        }


def _norm_species(s: str | None) -> str:
    if not s:
        return ""
    # Normalize whitespace and trim trailing punctuation
    return " ".join(s.split()).rstrip(".,;")


def evaluate(predictions: list[dict[str, Any]], gold: list[GoldPanel]) -> EvaluationReport:
    """Score predictions against a gold set.

    Predictions are dicts with keys: paper_id, panel_id, species.
    A prediction is considered to "match" a gold panel if
    :func:`match_panel` returns True. Species comparison is
    case-insensitive whitespace-normalized equality.

    When multiple predictions have the same (paper_id, panel_id), the
    one with a non-empty species (or with the highest ``confidence``)
    is preferred. This matters when both a taxon-recognizer hit and a
    caption-parser hit exist for the same panel.
    """
    by_paper: dict[str, PaperMetrics] = defaultdict(lambda: PaperMetrics(paper_id=""))
    for g in gold:
        m = by_paper[g.paper_id]
        m.paper_id = g.paper_id
        m.n_gold += 1

    # Build a list of predictions per (paper_id, panel_id), then pick
    # the best one (highest confidence; if tied, prefer non-empty species).
    pred_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for p in predictions:
        pid = p.get("paper_id")
        plabel = p.get("panel_id")
        if not pid or not plabel:
            continue
        pred_groups.setdefault((pid, plabel), []).append(p)

    def _best_pred(preds: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not preds:
            return None
        return max(
            preds,
            key=lambda p: (
                float(p.get("confidence") or 0.0),
                bool((p.get("species") or "").strip()),
            ),
        )

    for g in gold:
        m = by_paper[g.paper_id]
        gold_species = _norm_species(g.species)
        # Find a matching prediction
        matched_pred: dict[str, Any] | None = None
        for (pid, plabel), preds in pred_groups.items():
            if match_panel(g, pid, plabel):
                cand = _best_pred(preds)
                if cand is None:
                    continue
                if matched_pred is None:
                    matched_pred = cand
                else:
                    # Prefer the candidate that matches the gold species
                    cand_sp = _norm_species(cand.get("species"))
                    cur_sp = _norm_species(matched_pred.get("species"))
                    if cand_sp.lower() == gold_species.lower() and cur_sp.lower() != gold_species.lower():
                        matched_pred = cand
        matched_pred_species = (
            _norm_species(matched_pred.get("species")) if matched_pred else None
        )
        if matched_pred is not None:
            m.panel_match += 1
            if gold_species and matched_pred_species and gold_species.lower() == matched_pred_species.lower():
                m.species_tp += 1
                m.exact_match += 1
            elif matched_pred_species and not gold_species:
                m.species_fp += 1
            elif gold_species and not matched_pred_species:
                m.species_fn += 1
            else:
                m.species_fp += 1
                m.species_fn += 1
        else:
            if gold_species:
                m.species_fn += 1

    # n_pred_panels per paper (count unique panel labels, not raw rows)
    pred_per_paper: dict[str, int] = defaultdict(int)
    for (pid, _plabel) in pred_groups.keys():
        pred_per_paper[pid] += 1
    for pid, n in pred_per_paper.items():
        if pid not in by_paper:
            by_paper[pid] = PaperMetrics(paper_id=pid)
        by_paper[pid].n_pred_panels = n

    # Aggregate
    total_gold = sum(m.n_gold for m in by_paper.values())
    total_tp = sum(m.species_tp for m in by_paper.values())
    total_fp = sum(m.species_fp for m in by_paper.values())
    total_fn = sum(m.species_fn for m in by_paper.values())
    total_panel_match = sum(m.panel_match for m in by_paper.values())
    total_exact = sum(m.exact_match for m in by_paper.values())

    agg = {
        "n_papers": len(by_paper),
        "n_gold": total_gold,
        "species_precision": total_tp / max(1, total_tp + total_fp),
        "species_recall": total_tp / max(1, total_tp + total_fn),
        "species_f1": (
            2 * total_tp / max(1, 2 * total_tp + total_fp + total_fn)
        ),
        "panel_match_rate": total_panel_match / max(1, total_gold),
        "exact_match_rate": total_exact / max(1, total_gold),
    }

    return EvaluationReport(papers=dict(by_paper), aggregate=agg)


def load_predictions_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append({
                "paper_id": d.get("paper_id"),
                "panel_id": d.get("panel_id"),
                "species": d.get("species"),
            })
    return out


def evaluate_run(predictions_path: Path, gold_dir: Path) -> EvaluationReport:
    """Convenience: load a predictions JSONL + all gold files in a dir."""
    preds = load_predictions_jsonl(predictions_path)
    all_gold: list[GoldPanel] = []
    for gold_path in sorted(gold_dir.glob("*.jsonl")):
        all_gold.extend(load_gold(gold_path))
    return evaluate(preds, all_gold)
