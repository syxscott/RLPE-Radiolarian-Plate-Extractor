from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .metrics import MetricScore, iou, prf, matching_accuracy
from .utils import write_json


@dataclass(slots=True)
class EvaluationSummary:
    figure_prf: MetricScore | None = None
    panel_prf: MetricScore | None = None
    match_accuracy: float = 0.0
    mean_iou: float = 0.0
    gemma_confidence_mean: float = 0.0
    match_improvement: float = 0.0
    n_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "figure_prf": asdict(self.figure_prf) if self.figure_prf else None,
            "panel_prf": asdict(self.panel_prf) if self.panel_prf else None,
            "match_accuracy": self.match_accuracy,
            "mean_iou": self.mean_iou,
            "gemma_confidence_mean": self.gemma_confidence_mean,
            "match_improvement": self.match_improvement,
            "n_samples": self.n_samples,
        }


def evaluate_predictions(
    predicted: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> EvaluationSummary:
    if not predicted or not gold:
        return EvaluationSummary()

    pred_boxes = [tuple(item["bbox"]) for item in predicted if item.get("bbox")]
    gold_boxes = [tuple(item["bbox"]) for item in gold if item.get("bbox")]

    matched = 0
    iou_scores: list[float] = []
    used = set()
    for g in gold_boxes:
        best_iou = 0.0
        best_idx = None
        for i, p in enumerate(pred_boxes):
            if i in used:
                continue
            score = iou(p, g)
            if score > best_iou:
                best_iou = score
                best_idx = i
        if best_idx is not None and best_iou >= iou_threshold:
            matched += 1
            used.add(best_idx)
            iou_scores.append(best_iou)

    figure_prf = prf(matched, max(0, len(pred_boxes) - matched), max(0, len(gold_boxes) - matched))
    panel_prf = figure_prf
    pred_labels = [item.get("panel_id") for item in predicted]
    gold_labels = [item.get("panel_id") for item in gold]
    summary = EvaluationSummary(
        figure_prf=figure_prf,
        panel_prf=panel_prf,
        match_accuracy=matching_accuracy(pred_labels, gold_labels),
        mean_iou=sum(iou_scores) / len(iou_scores) if iou_scores else 0.0,
        gemma_confidence_mean=_gemma_confidence_mean(predicted),
        n_samples=len(gold),
    )
    return summary


def compare_before_after(
    before_rows: list[dict[str, Any]],
    after_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare baseline and Gemma-enhanced rows against gold labels."""
    import pandas as pd

    df_b = pd.DataFrame(before_rows).copy()
    df_a = pd.DataFrame(after_rows).copy()
    df_g = pd.DataFrame(gold_rows).copy()

    key_cols = ["paper_id", "figure_id", "panel_path"]
    for col in key_cols:
        if col not in df_b:
            df_b[col] = None
        if col not in df_a:
            df_a[col] = None
        if col not in df_g:
            df_g[col] = None

    if "panel_id" not in df_g:
        df_g["panel_id"] = None
    if "species" not in df_g:
        df_g["species"] = None

    merged = df_b.merge(df_a, on=key_cols, suffixes=("_before", "_after"))
    merged = merged.merge(df_g[key_cols + ["panel_id", "species"]], on=key_cols, how="left")
    merged = merged.rename(columns={"panel_id": "gold_panel_id", "species": "gold_species"})

    merged["correct_before"] = (merged["panel_id_before"] == merged["gold_panel_id"]) & (
        merged["species_before"] == merged["gold_species"]
    )
    merged["correct_after"] = (merged["panel_id_after"] == merged["gold_panel_id"]) & (
        merged["species_after"] == merged["gold_species"]
    )

    before_acc = float(merged["correct_before"].mean()) if len(merged) else 0.0
    after_acc = float(merged["correct_after"].mean()) if len(merged) else 0.0

    gemma_col = "gemma_confidence_after" if "gemma_confidence_after" in merged.columns else None
    gemma_mean = float(merged[gemma_col].fillna(0).mean()) if gemma_col else 0.0

    return {
        "n_samples": int(len(merged)),
        "match_acc_before": round(before_acc, 4),
        "match_acc_after": round(after_acc, 4),
        "match_improvement": round(after_acc - before_acc, 4),
        "gemma_confidence_mean": round(gemma_mean, 4),
    }


def _gemma_confidence_mean(rows: list[dict[str, Any]]) -> float:
    values = []
    for row in rows:
        meta = row.get("metadata") if isinstance(row, dict) else None
        if isinstance(meta, dict) and "gemma_confidence" in meta:
            try:
                values.append(float(meta["gemma_confidence"]))
            except Exception:
                pass
        elif "gemma_confidence" in row:
            try:
                values.append(float(row["gemma_confidence"]))
            except Exception:
                pass
    if not values:
        return 0.0
    return sum(values) / len(values)


def save_evaluation(summary: EvaluationSummary, path: Path) -> None:
    write_json(path, summary.to_dict())
