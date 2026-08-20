"""Phase 62 Plan 5 (Bug 5.8): scale-bar range fallback confidence.

When ``SCALE_PATTERN`` matches a range form (e.g. ``5-10 µm``) and
the second group's float() fails (e.g. the OCR misread the ``-``
as a ``—`` em-dash and the second number is unreadable), the
extractor keeps the single-value ``val`` with ``confidence=0.8``
(caption) or ``0.7`` (OCR).

This is too high: the parse partially failed (the range form was
matched but the upper bound couldn't be parsed), so the value is
really just a single number with no range confirmation. 0.8 (or
0.7 for OCR) overstates our certainty in the upper bound.

The fix: lower the confidence on this fallback path to 0.4 (caption)
and 0.3 (OCR). The original 0.8/0.7 was for a fully-validated range;
the degraded single-value path deserves a degraded confidence.
"""

from __future__ import annotations

from unittest.mock import patch

from rlpe.scale_bar import SCALE_PATTERN, extract_scale_from_caption


def test_caption_range_parse_failure_low_confidence():
    """When the range form matched but _safe_float(m.group(2)) raises,
    the fallback single-value confidence must be 0.4 (not 0.8)."""
    # We need a text that matches SCALE_PATTERN with a range form
    # (both group 1 and group 2 numeric), then make _safe_float raise
    # on the second group so the try/except (TypeError, ValueError)
    # branch fires and degrades confidence to 0.4.
    #
    # Approach: patch _safe_float in the rlpe.scale_bar module. This
    # is the cleanest approach that works consistently across all
    # Python versions (the previous globals()/builtins.float lookup
    # was unreliable on Python 3.11 + pytest-cov due to __builtins__
    # caching at function-definition time).
    with patch("rlpe.scale_bar._safe_float", side_effect=ValueError("simulated bad range value")):
        text = "scale bar 5–10 µm"
        info = extract_scale_from_caption(text)

    # The range form matched; _safe_float raised; fallback
    # confidence is 0.4.
    assert info.confidence <= 0.5, (
        f"range-fallback confidence too high: {info.confidence} "
        f"(expected <=0.5; pre-fix value was 0.8)"
    )
    assert info.confidence == 0.4, (
        f"range-fallback confidence should be 0.4 (degraded from 0.8), got {info.confidence}"
    )


def test_caption_successful_range_keeps_higher_confidence():
    """Regression: a fully-parsed range form (5-10 µm) keeps
    confidence 0.7."""
    text = "scale bar 5-10 µm"
    info = extract_scale_from_caption(text)
    # The range form parsed cleanly; midpoint = 7.5.
    assert info.value == 7.5
    # Confidence for a clean range is 0.7 (not the degraded 0.4).
    assert info.confidence >= 0.6, (
        f"clean-range confidence regressed: {info.confidence} "
        f"(expected >=0.6; pre-fix value was 0.7)"
    )
