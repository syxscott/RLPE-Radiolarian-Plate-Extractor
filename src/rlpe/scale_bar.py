from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SCALE_PATTERN = re.compile(
    r"(?:scale\s*bar(?:\s*(?:length|=|:))?\s*|bar\s*=\s*|bars?\s+are\s+|"
    r"scale\s+bar\s+of\s+|"
    r"(?<![A-Za-z0-9_]))"  # bare-number form, not preceded by word char
    r"(\d+(?:\.\d+)?)\s*(?:[–—\-]\s*(\d+(?:\.\d+)?)\s*)?"
    r"(μm|µm|um|microns?|micron|mm|cm|nm)",
    re.IGNORECASE,
)

# Words whose presence near a number-unit pair indicate a NON-scale-bar
# context: specimen sizes, sieve apertures, sediment depths, etc. When
# the SCALE_PATTERN matches a "bare number + unit" (no "scale bar"
# prefix) and one of these tokens appears in the immediate left context,
# we reject the match — otherwise "specimen 250 µm long" or
# "100 µm sieve" gets stored as the figure's scale bar.
_NON_SCALE_CONTEXT_WORDS = (
    "specimen", "specimens", "sieve", "sample", "depth", "length",
    "long", "wide", "tall", "diameter", "radius", "thick",
    "aperture", "mesh", "grain", "test",
)


def _is_real_scale_match(text: str, match: re.Match[str]) -> bool:
    """Return True only if the SCALE_PATTERN match looks like a genuine
    figure-scale-bar mention (not a specimen size measurement).

    A match is "real" if either:
      - it has the "scale bar" / "bar =" / "bars are" prefix (the
        regex captures that as part of the group-0 span), OR
      - the 30 chars immediately before the match contain none of the
        non-scale-context words above.
    """
    span = match.group(0).lower()
    # Has explicit "scale" / "bar" prefix in the matched text → real.
    if "scale" in span or "bar" in span:
        return True
    # Otherwise check the left-context for specimen-size words.
    left = text[max(0, match.start() - 30):match.start()].lower()
    return not any(w in left for w in _NON_SCALE_CONTEXT_WORDS)


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
    # Iterate matches and pick the first one that survives the
    # specimen-size context filter. This prevents "specimen 250 µm
    # long" from being recorded as the figure's scale bar when no
    # real scale-bar mention exists in the caption.
    m = None
    for cand in SCALE_PATTERN.finditer(caption_text):
        if _is_real_scale_match(caption_text, cand):
            m = cand
            break
    if not m:
        return ScaleInfo()
    val = float(m.group(1))
    unit = normalize_unit(m.group(3))
    info = ScaleInfo(value=val, unit=unit, source="caption", confidence=0.8)
    # Range form: 5–10 µm → use midpoint
    if m.group(2):
        try:
            hi = float(m.group(2))
            info.value = (val + hi) / 2.0
            info.confidence = 0.7
        except (TypeError, ValueError) as exc:
            # The regex matched a range shape but the second group
            # wasn't a valid float (e.g. unicode minus, OCR noise
            # injected between the digits). The single-value
            # confidence is the right fallback — log at debug so the
            # operator can see this happened without spamming the
            # warning level.
            logger.debug(
                "scale caption: range form matched but group(2)=%r is not a "
                "float: %s — keeping single value", m.group(2), exc,
            )
    return info


def extract_scale_from_ocr_text(ocr_text: str) -> ScaleInfo:
    if not ocr_text:
        return ScaleInfo()
    m = None
    for cand in SCALE_PATTERN.finditer(ocr_text):
        if _is_real_scale_match(ocr_text, cand):
            m = cand
            break
    if not m:
        return ScaleInfo()
    val = float(m.group(1))
    unit = normalize_unit(m.group(3))
    info = ScaleInfo(value=val, unit=unit, source="ocr", confidence=0.7)
    if m.group(2):
        try:
            hi = float(m.group(2))
            info.value = (val + hi) / 2.0
            info.confidence = 0.6
        except (TypeError, ValueError) as exc:
            # See caption variant above. OCR text is much noisier so
            # this is more common — log at debug level.
            logger.debug(
                "scale ocr: range form matched but group(2)=%r is not a "
                "float: %s — keeping single value", m.group(2), exc,
            )
    return info


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
    u = (unit or "").lower().strip()
    if u in {"μm", "µm", "um", "micron", "microns"}:
        return "um"
    return u


def to_um(value: float, unit: str) -> float | None:
    u = normalize_unit(unit)
    if u == "um":
        return value
    if u == "mm":
        return value * 1000.0
    if u == "cm":
        return value * 10000.0
    if u == "nm":
        return value / 1000.0
    return None
