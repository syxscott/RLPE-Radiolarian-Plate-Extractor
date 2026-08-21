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
    the fallback single-value confidence must be 0.4 (not 0.8).

    Audit 2026-08-20: under pytest-cov 7.x + Python 3.11,
    ``patch("rlpe.scale_bar._safe_float", side_effect=...)`` does
    not reliably fire (PEP 659 specialising adaptive interpreter
    caches the bare-name reference inside the consuming function's
    bytecode, and the coverage tracer's rewrite prevents the cache
    from being invalidated). The previous workaround
    (``globals().get("float", float)``) only bypassed the builtin
    cache, not the attribute-cache on ``_safe_float`` itself.

    Instead of patching, we exercise the same try/except branch by
    crafting input where ``_safe_float(m.group(2))`` raises
    naturally — the regex captures group(2) as raw text and the
    float() call will raise ValueError if the captured span is not
    a valid float literal. "scale bar 5–1O µm" uses unicode minus
    (–) which is *not* in SCALE_PATTERN's group-2 (the pattern
    only matches ASCII ``[\\-]``), so group(2) is None — wait, that
    breaks the range form entirely. So instead we use an explicit
    ``monkeypatch`` of ``_safe_float`` via ``setattr`` on the
    function object, which on Python 3.11 still goes through the
    descriptor protocol and bypasses PEP 659 caching.

    Approach that survives pytest-cov 7.x: wrap the patch in a
    ``setattr`` on the module namespace via ``globals()`` lookup
    in the test's own frame, then directly invoke the function
    with a known-bad input. This is robust because the test
    doesn't rely on bytecode-cached attribute access at all.
    """
    import rlpe.scale_bar as sb

    original = sb._safe_float

    def boom(value):
        raise ValueError("simulated bad range value")

    # monkeypatch via direct setattr on the module's __dict__.
    # The consumer in scale_bar.py now reads via
    # ``sys.modules[__name__].__dict__["_safe_float"]`` so the
    # patched attribute wins regardless of PEP 659 caching.
    sb._safe_float = boom
    try:
        text = "scale bar 5-10 µm"
        info = extract_scale_from_caption(text)
    finally:
        sb._safe_float = original

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
