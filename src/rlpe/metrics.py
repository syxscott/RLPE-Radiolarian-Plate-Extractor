from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class MetricScore:
    precision: float
    recall: float
    f1: float


def safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def prf(tp: int, fp: int, fn: int) -> MetricScore:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return MetricScore(precision=precision, recall=recall, f1=f1)


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    a1x, a1y = ax, ay
    a2x, a2y = ax + aw, ay + ah
    b1x, b1y = bx, by
    b2x, b2y = bx + bw, by + bh

    ix1, iy1 = max(a1x, b1x), max(a1y, b1y)
    ix2, iy2 = min(a2x, b2x), min(a2y, b2y)
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter = inter_w * inter_h
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    return safe_div(inter, union)


def matching_accuracy(predicted: Iterable[str | None], target: Iterable[str | None]) -> float:
    pred_list = list(predicted)
    tgt_list = list(target)
    n = min(len(pred_list), len(tgt_list))
    if n == 0:
        return 0.0
    correct = sum(1 for p, t in zip(pred_list[:n], tgt_list[:n]) if p == t)
    return correct / n
