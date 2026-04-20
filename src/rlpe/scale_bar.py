from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np


SCALE_PATTERN = re.compile(
    r"(?:scale\s*bar\s*(?:=|:)?\s*|bar\s*=\s*)(\d+(?:\.\d+)?)\s*(μm|um|mm|nm)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ScaleInfo:
    value: float | None = None
    unit: str | None = None
    source: str = "none"
    pixel_length: float | None = None
    um_per_px: float | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_scale_from_caption(caption_text: str) -> ScaleInfo:
    if not caption_text:
        return ScaleInfo()
    m = SCALE_PATTERN.search(caption_text)
    if not m:
        return ScaleInfo()
    val = float(m.group(1))
    unit = normalize_unit(m.group(2))
    return ScaleInfo(value=val, unit=unit, source="caption", confidence=0.8)


def extract_scale_from_ocr_text(ocr_text: str) -> ScaleInfo:
    if not ocr_text:
        return ScaleInfo()
    m = SCALE_PATTERN.search(ocr_text)
    if not m:
        return ScaleInfo()
    val = float(m.group(1))
    unit = normalize_unit(m.group(2))
    return ScaleInfo(value=val, unit=unit, source="ocr", confidence=0.7)


def detect_scale_bar_length_px(image: np.ndarray) -> float | None:
    """Estimate scale bar pixel length by detecting horizontal long segments."""
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, 80, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=5)
    if lines is None:
        return None
    best = 0.0
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = line
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dy > 6:  # 倾斜较大的线段过滤
            continue
        length = float(np.hypot(dx, dy))
        if length > best:
            best = length
    return best if best > 0 else None


def estimate_um_per_px(scale_value: float | None, scale_unit: str | None, pixel_length: float | None) -> float | None:
    if scale_value is None or scale_unit is None or pixel_length is None or pixel_length <= 0:
        return None
    um_value = to_um(scale_value, scale_unit)
    if um_value is None:
        return None
    return um_value / pixel_length


def merge_scale_info(caption_info: ScaleInfo, ocr_info: ScaleInfo, pixel_length: float | None = None) -> ScaleInfo:
    base = caption_info if caption_info.confidence >= ocr_info.confidence else ocr_info
    if base.value is None and ocr_info.value is not None:
        base = ocr_info
    if base.value is None and caption_info.value is not None:
        base = caption_info

    out = ScaleInfo(
        value=base.value,
        unit=base.unit,
        source=base.source,
        pixel_length=pixel_length,
        confidence=base.confidence,
    )
    out.um_per_px = estimate_um_per_px(out.value, out.unit, out.pixel_length)
    return out


def normalize_unit(unit: str) -> str:
    u = unit.lower().strip()
    if u in {"μm", "um"}:
        return "um"
    return u


def to_um(value: float, unit: str) -> float | None:
    u = normalize_unit(unit)
    if u == "um":
        return value
    if u == "mm":
        return value * 1000.0
    if u == "nm":
        return value / 1000.0
    return None
