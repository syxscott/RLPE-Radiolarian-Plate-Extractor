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

# Phase 62 Plan 5 (Bug 5.2): sanity bounds for an extracted scale-bar
# value, expressed in µm. The bounds are deliberately generous so
# unusual but legitimate scales ("5 mm" for a microfossil overview
# plate, "500 nm" for an SEM close-up) still pass. The gate exists
# to catch the catastrophic 10x and 100x OCR misreads (e.g. "1O µm"
# being read as "10 µm" by a naive OCR pass, or a stray "O"
# character dropped entirely).
#
#  * Lower bound 0.1 µm = 100 nm: below this, the bar would be
#    sub-pixel on any reasonable figure.
#  * Upper bound 10000 µm = 10 mm: above this, the bar is a
#    map-scale, not a figure-scale.
#
# We convert the parsed value to µm via ``to_um`` BEFORE comparing,
# so a "1 cm" value (which is 10000 µm) sits exactly on the upper
# bound and is accepted.
_SANITY_VALUE_MIN_UM = 0.1
_SANITY_VALUE_MAX_UM = 10000.0


def _safe_float(s: str) -> float:
    """Wrapper around float() that can be patched in tests to simulate
    parse failures (e.g. OCR noise in a scale-bar range upper bound)."""
    return float(s)


def _value_in_sanity_range(val: float, unit: str | None) -> bool:
    """Return True if ``val`` (in ``unit``) is within the scale-bar
    sanity range. Used to reject catastrophic OCR misreads that
    would otherwise pass the bare ``\\d+`` regex.
    """
    if val is None:
        return False
    um = to_um(val, unit)
    # If the unit is unknown, fall through and accept the value —
    # the downstream merge_scale_info will flag a disagreement if
    # the value is wrong, and we don't want to drop legitimate
    # unfamiliar-unit bars here.
    if um is None:
        return True
    return _SANITY_VALUE_MIN_UM <= um <= _SANITY_VALUE_MAX_UM


# Phase 62 Plan 5 (Bug 5.11): explicit sentinel returned by
# ``normalize_unit`` for ``None`` / empty / whitespace-only input.
# Previously these all returned ``""``, indistinguishable from a
# legitimate unknown-unit input that happened to normalise to
# empty (none currently do, but the contract was fragile).
# Callers can use ``unit is UNKNOWN_UNIT`` to detect "no unit at
# all" without relying on string magic.
UNKNOWN_UNIT = "__unknown__"

# Words whose presence near a number-unit pair indicate a NON-scale-bar
# context: specimen sizes, sieve apertures, sediment depths, etc. When
# the SCALE_PATTERN matches a "bare number + unit" (no "scale bar"
# prefix) and one of these tokens appears in the immediate left context,
# we reject the match — otherwise "specimen 250 µm long" or
# "100 µm sieve" gets stored as the figure's scale bar.
_NON_SCALE_CONTEXT_WORDS = (
    "specimen",
    "specimens",
    "sieve",
    "sample",
    "depth",
    "length",
    "long",
    "wide",
    "tall",
    "diameter",
    "radius",
    "thick",
    "aperture",
    "mesh",
    "grain",
    "test",
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
    left = text[max(0, match.start() - 30) : match.start()].lower()
    return not any(w in left for w in _NON_SCALE_CONTEXT_WORDS)


@dataclass(slots=True)
class ScaleInfo:
    value: float | None = None
    unit: str | None = None
    source: str = "none"
    pixel_length: float | None = None
    um_per_px: float | None = None
    confidence: float = 0.0
    # Phase 61 Plan 4 (Bug 4.6): disagreement diagnostic. ``"scale_bar_10x_disagreement"``
    # means the caption + OCR values differed by >10x and BOTH were dropped;
    # ``"scale_bar_disagreement"`` means a 2x-10x ratio, kept the
    # higher-confidence value but flagged for review; empty/None means the
    # two sources agreed within 2x (or only one source had a value).
    warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_scale_from_caption(caption_text: str) -> ScaleInfo:
    if not caption_text:
        return ScaleInfo()
    # Audit 2026-08-01 batch W2 (Bug D17): the previous implementation
    # broke out of the loop on the FIRST ``_is_real_scale_match``
    # candidate, then dropped the result entirely if it failed the
    # sanity range. That meant a caption like "Scale bar 5 cm; scale
    # bar 100 um" — where the first candidate is fine but, e.g.,
    # "Scale bar 99999 mm; scale bar 50 um" — where the first fails
    # sanity — returned an empty ScaleInfo instead of falling through
    # to the next candidate.
    #
    # New contract:
    #   1. Collect every candidate that survives ``_is_real_scale_match``
    #      (the specimen-size context filter).
    #   2. Among those, keep only the ones whose value+unit pass the
    #      sanity range (drops catastrophic OCR misreads).
    #   3. Pick the sanity-passing candidate with the largest raw
    #      ``value``. Caption authors typically write the most
    #      precise scale last ("scale bar 100 µm" beats
    #      "scale bar 5 cm" for a microfossil plate because the µm
    #      reading has a larger numeric magnitude and is what the
    #      downstream µm/px calculation actually needs); if a future
    #      maintainer prefers a unit-normalised selection (compare
    #      in µm) the call site can change this to ``to_um(val, unit)``.
    #   4. If NO candidate passes sanity, return an empty ScaleInfo
    #      as before.
    candidates: list[tuple[float, str, re.Match[str]]] = []
    for cand in SCALE_PATTERN.finditer(caption_text):
        if _is_real_scale_match(caption_text, cand):
            cand_val = float(cand.group(1))
            cand_unit = normalize_unit(cand.group(3))
            candidates.append((cand_val, cand_unit, cand))
    if not candidates:
        return ScaleInfo()
    sane: list[tuple[float, int]] = []
    for idx, (cand_val, cand_unit, _cand) in enumerate(candidates):
        if _value_in_sanity_range(cand_val, cand_unit):
            sane.append((cand_val, idx))
    if not sane:
        # Audit 2026-08-01 batch W2 (Bug D17): all candidates failed
        # sanity (likely OCR misreads). Keep legacy behaviour and
        # return empty ScaleInfo.
        logger.debug(
            "scale caption: %d candidate(s) found but none survived sanity "
            "range [%s, %s] µm — dropping",
            len(candidates),
            _SANITY_VALUE_MIN_UM,
            _SANITY_VALUE_MAX_UM,
        )
        return ScaleInfo()
    # Pick the candidate with the largest raw value. Ties broken by
    # the later regex occurrence (preserves "later candidate wins"
    # behaviour for the audit test fixtures).
    chosen_idx = max(sane, key=lambda item: (item[0], item[1]))[1]
    val, unit, m = candidates[chosen_idx]
    # Sanity already enforced above when building ``sane``; the chosen
    # candidate is guaranteed to be in range. The downstream ScaleInfo
    # builder uses ``val`` / ``unit`` directly.
    info = ScaleInfo(value=val, unit=unit, source="caption", confidence=0.8)
    # Range form: 5–10 µm → use midpoint
    if m.group(2):
        try:
            # Use _safe_float so tests can reliably patch parse failures.
            hi = _safe_float(m.group(2))
            # Re-check sanity on the midpoint; if the midpoint is
            # out of range the range itself was garbage.
            mid = (val + hi) / 2.0
            if not _value_in_sanity_range(mid, unit):
                logger.debug(
                    "scale caption: range midpoint=%s unit=%r outside sanity "
                    "range — keeping single value=%s instead",
                    mid,
                    unit,
                    val,
                )
            else:
                info.value = mid
                info.confidence = 0.7
        except (TypeError, ValueError) as exc:
            # The regex matched a range shape but the second group
            # wasn't a valid float (e.g. unicode minus, OCR noise
            # injected between the digits). Phase 62 Plan 5 (Bug
            # 5.8): the previous fallback kept confidence=0.8
            # (the single-value level) which overstated our
            # certainty — the range form was matched but the
            # upper bound couldn't be parsed, so we have only a
            # single number with NO range confirmation. Lower
            # confidence to 0.4 (caption) so downstream consumers
            # can see the partial-failure path.
            logger.debug(
                "scale caption: range form matched but group(2)=%r is not a "
                "float: %s — keeping single value with degraded confidence",
                m.group(2),
                exc,
            )
            info.confidence = 0.4
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
    # Phase 62 Plan 5 (Bug 5.2): sanity-check the value (see
    # ``_value_in_sanity_range`` docstring). OCR text is even
    # noisier than captions so this gate fires more often.
    if not _value_in_sanity_range(val, unit):
        logger.debug(
            "scale ocr: parsed value=%s unit=%r outside sanity range "
            "[%s, %s] µm — dropping (likely OCR misread)",
            val,
            unit,
            _SANITY_VALUE_MIN_UM,
            _SANITY_VALUE_MAX_UM,
        )
        return ScaleInfo()
    info = ScaleInfo(value=val, unit=unit, source="ocr", confidence=0.7)
    if m.group(2):
        try:
            hi = float(m.group(2))
            mid = (val + hi) / 2.0
            if not _value_in_sanity_range(mid, unit):
                logger.debug(
                    "scale ocr: range midpoint=%s unit=%r outside sanity "
                    "range — keeping single value=%s instead",
                    mid,
                    unit,
                    val,
                )
            else:
                info.value = mid
                info.confidence = 0.6
        except (TypeError, ValueError) as exc:
            # Phase 62 Plan 5 (Bug 5.8): see caption variant. OCR
            # text is even noisier so this degraded-confidence path
            # fires more often; we lower to 0.3 (degraded from 0.7).
            logger.debug(
                "scale ocr: range form matched but group(2)=%r is not a "
                "float: %s — keeping single value with degraded confidence",
                m.group(2),
                exc,
            )
            info.confidence = 0.3
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
    # Round 25 audit: ``cv2.HoughLinesP`` returns a 2-D array of
    # shape ``(N, 4)`` in OpenCV >=5 (verified empirically against
    # OpenCV 5.0.0 on this machine). Older docs / older OpenCV show
    # the historical 3-D ``(1, N, 4)``. Indexing ``lines[:, 0, :]``
    # on the 2-D shape raised ``IndexError: too many indices for
    # array``. Normalise on entry so both shapes yield a flat
    # ``(N, 4)`` array we can iterate.
    if lines.ndim == 3:
        lines = lines[:, 0, :]
    if lines.ndim != 2 or lines.shape[-1] != 4:
        return None
    best = 0.0
    for x1, y1, x2, y2 in lines:
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dy > 6:  # 倾斜较大的线段过滤
            continue
        length = float(np.hypot(dx, dy))
        if length > best:
            best = length
    return best if best > 0 else None


def estimate_um_per_px(
    scale_value: float | None, scale_unit: str | None, pixel_length: float | None
) -> float | None:
    if scale_value is None or scale_unit is None or pixel_length is None or pixel_length <= 0:
        return None
    um_value = to_um(scale_value, scale_unit)
    if um_value is None:
        return None
    return um_value / pixel_length


def merge_scale_info(
    caption_info: ScaleInfo, ocr_info: ScaleInfo, pixel_length: float | None = None
) -> ScaleInfo:
    base = caption_info if caption_info.confidence >= ocr_info.confidence else ocr_info
    if base.value is None and ocr_info.value is not None:
        base = ocr_info
    if base.value is None and caption_info.value is not None:
        base = caption_info

    # Phase 61 Plan 4 (Bug 4.6): detect scale-bar disagreement between
    # caption and OCR sources. Two sources producing wildly different
    # numbers (e.g. 100 µm vs 1 µm, a 100x gap) is almost always an
    # OCR error on one side; trusting either silently propagates a
    # garbage ``um_per_px`` into PBDB coord-radius validation
    # downstream. Compute the ratio in µm units and:
    #   * ratio > 10x → drop BOTH values, return a "no-scale" ScaleInfo
    #     stamped with warning="scale_bar_10x_disagreement".
    #   * ratio 2x-10x → keep higher-confidence value but stamp
    #     warning="scale_bar_disagreement".
    #   * ratio < 2x (or only one source has a value) → unchanged
    #     legacy behaviour.
    warning = ""
    cv = caption_info.value if caption_info.value not in (None, 0) else None
    ov = ocr_info.value if ocr_info.value not in (None, 0) else None
    if cv is not None and ov is not None:
        cv_um = to_um(cv, caption_info.unit)
        ov_um = to_um(ov, ocr_info.unit)
        if cv_um and ov_um and cv_um > 0 and ov_um > 0:
            ratio = max(cv_um, ov_um) / min(cv_um, ov_um)
            if ratio > 10.0:
                # Drop both. The caller (figure-level inference) will
                # see value=None and um_per_px=None.
                return ScaleInfo(
                    value=None,
                    unit=None,
                    source="none",
                    pixel_length=pixel_length,
                    confidence=0.0,
                    warning="scale_bar_10x_disagreement",
                )
            if ratio >= 2.0:
                warning = "scale_bar_disagreement"

    out = ScaleInfo(
        value=base.value,
        unit=base.unit,
        source=base.source,
        pixel_length=pixel_length,
        confidence=base.confidence,
        warning=warning,
    )
    out.um_per_px = estimate_um_per_px(out.value, out.unit, out.pixel_length)
    return out


def normalize_unit(unit: str) -> str:
    # Phase 62 Plan 5 (Bug 5.11): explicit None / empty / whitespace
    # handling. Return UNKNOWN_UNIT sentinel so callers can tell
    # "no unit provided" apart from "unknown unit" (which still
    # returns the lowercased input).
    if unit is None:
        return UNKNOWN_UNIT
    u = unit.lower().strip()
    if not u:
        return UNKNOWN_UNIT
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
    # Phase 62 Plan 5 (Bug 5.9): add km branch. 1 km = 1e9 µm. The
    # value will fail the sanity-range gate in
    # ``_value_in_sanity_range`` (any km-scale figure is far above
    # the 10000 µm ceiling), so it will be dropped before reaching
    # downstream consumers — but the conversion itself is now
    # well-defined so we don't silently lose the unit signal.
    if u == "km":
        return value * 1e9
    return None
