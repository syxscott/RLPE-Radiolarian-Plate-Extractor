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


def _range_match_with_bad_second_group(text: str):
    """Find a SCALE_PATTERN range match in ``text`` and patch the
    second group to a value that fails float()."""
    m = SCALE_PATTERN.search(text)
    assert m is not None
    assert m.group(2) is not None, f"expected range form in {text!r}"
    return m


def test_caption_range_parse_failure_low_confidence():
    """When the range form matched but group(2) fails float(), the
    fallback single-value confidence must be 0.4 (not 0.8)."""
    # Construct text that matches SCALE_PATTERN with a range form,
    # then patch group(2) to a non-float value before calling
    # extract_scale_from_caption. The simplest way is to use a
    # text whose range second group IS a non-float (the regex
    # captures \d+ so we need to manipulate the input).
    #
    # Instead: patch float() within extract_scale_from_caption to
    # raise on the second call, leaving the first (val) intact.
    real_float = __builtins__["float"] if isinstance(__builtins__, dict) else __builtins__.float
    call_count = {"n": 0}

    def patched_float(value):
        call_count["n"] += 1
        # The first call inside extract is `float(m.group(1))` for val.
        # The second call is `float(m.group(2))` for the range upper.
        if call_count["n"] >= 2:
            raise ValueError("simulated bad range value")
        return real_float(value)

    # We need a text with a real range form. "5–10 µm" uses the
    # em-dash which the regex accepts.
    text = "scale bar 5–10 µm"
    with patch("rlpe.scale_bar.float", side_effect=patched_float):
        info = extract_scale_from_caption(text)
    # The range form matched; group(2) couldn't be parsed; fallback
    # confidence is 0.4.
    assert info.confidence <= 0.5, (
        f"range-fallback confidence too high: {info.confidence} "
        f"(expected <=0.5; pre-fix value was 0.8)"
    )
    assert info.confidence == 0.4, (
        f"range-fallback confidence should be 0.4 (degraded from 0.8), "
        f"got {info.confidence}"
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