"""Phase 61 Plan 4 (Bug 4.6): scale bar 10x+ disagreement must be dropped.

Previously ``merge_scale_info`` picked the higher-confidence of
caption / OCR-derived scale bars without checking the ratio between
them. 100 µm vs 1 µm (a 100x discrepancy) was silently accepted — the
resulting ``um_per_px`` is garbage and would cascade into PBDB
coord-radius validation downstream.

The fix:
  * When the ratio is > 10x: drop BOTH values, return
    ``ScaleInfo(value=None, warning="scale_bar_10x_disagreement")``.
  * When the ratio is in [2x, 10x]: keep the higher-confidence value
    but stamp ``warning="scale_bar_disagreement"`` for downstream
    visibility.
  * When the ratio is < 2x: keep the higher-confidence value unchanged.
"""
from __future__ import annotations

import pytest

from rlpe.scale_bar import ScaleInfo, merge_scale_info


def _to_um(value, unit):
    """Shortcut for µm conversion used in tests (mirrors scale_bar.to_um)."""
    if value is None or unit is None:
        return None
    u = unit.lower()
    if u == "um":
        return value
    if u == "mm":
        return value * 1000.0
    if u == "nm":
        return value / 1000.0
    return value


def test_merge_scale_100x_disagreement_dropped():
    """100 µm caption vs 1 µm OCR (100x) → both dropped, warning set."""
    cap = ScaleInfo(value=100.0, unit="um", source="caption", confidence=0.8)
    ocr = ScaleInfo(value=1.0, unit="um", source="ocr", confidence=0.7)
    out = merge_scale_info(cap, ocr)
    assert out.value is None
    assert out.warning == "scale_bar_10x_disagreement"


def test_merge_scale_5x_disagreement_warns():
    """5x ratio → keep higher-confidence but stamp disagreement warning."""
    cap = ScaleInfo(value=50.0, unit="um", source="caption", confidence=0.9)
    ocr = ScaleInfo(value=10.0, unit="um", source="ocr", confidence=0.7)
    out = merge_scale_info(cap, ocr)
    assert out.value == 50.0  # caption had higher confidence
    assert out.warning == "scale_bar_disagreement"


def test_merge_scale_agreeing_unchanged():
    """Values within 2x of each other → no warning."""
    cap = ScaleInfo(value=100.0, unit="um", source="caption", confidence=0.9)
    ocr = ScaleInfo(value=80.0, unit="um", source="ocr", confidence=0.7)
    out = merge_scale_info(cap, ocr)
    assert out.value == 100.0
    assert out.warning is None or out.warning == ""


def test_merge_scale_one_missing_passes_through():
    """When only one source has a value, no ratio check, no warning."""
    cap = ScaleInfo(value=100.0, unit="um", source="caption", confidence=0.8)
    ocr = ScaleInfo()  # empty
    out = merge_scale_info(cap, ocr)
    assert out.value == 100.0
    assert not getattr(out, "warning", None)