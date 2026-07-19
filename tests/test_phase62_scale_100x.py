"""Phase 62 Plan 5 (Bug 5.3): regression test for 100x scale-bar
discrepancy warning tag.

Bug 5.3 was originally raised as "merge_scale_info silently ignores
100× discrepancy". The fix landed in Plan 4 (commit ee92db7): when
the caption-vs-OCR ratio is >10x, BOTH values are dropped and the
returned ScaleInfo carries warning='scale_bar_10x_disagreement'.

This file adds a small regression test that locks down:
  * The warning tag is set on the returned ScaleInfo (so downstream
    consumers can detect the case via the JSON ``warning`` field).
  * The 2x-10x disagreement case stamps the milder warning tag
    ``scale_bar_disagreement``.
  * The agreeing case has no warning (regression — don't pollute
    clean records).

This is intentionally a SMALL test file because Plan 4.6 already
covers the heavy lifting in test_phase61_scale_bar_10x.py — we
just want one extra lock-down that survives future refactors of
ScaleInfo's warning field.
"""
from __future__ import annotations

from rlpe.scale_bar import ScaleInfo, merge_scale_info


def test_100x_disagreement_warning_tag_set():
    """The 100x case carries the exact warning tag
    'scale_bar_10x_disagreement' so downstream JSON consumers can
    detect it."""
    cap = ScaleInfo(value=100.0, unit="um", source="caption", confidence=0.8)
    ocr = ScaleInfo(value=1.0, unit="um", source="ocr", confidence=0.7)
    out = merge_scale_info(cap, ocr)
    assert out.warning == "scale_bar_10x_disagreement"


def test_5x_disagreement_warning_tag_set():
    """The 2x-10x case carries 'scale_bar_disagreement'."""
    cap = ScaleInfo(value=50.0, unit="um", source="caption", confidence=0.9)
    ocr = ScaleInfo(value=10.0, unit="um", source="ocr", confidence=0.7)
    out = merge_scale_info(cap, ocr)
    assert out.warning == "scale_bar_disagreement"


def test_agreeing_no_warning():
    """Regression: when the two sources agree, no warning."""
    cap = ScaleInfo(value=100.0, unit="um", source="caption", confidence=0.9)
    ocr = ScaleInfo(value=80.0, unit="um", source="ocr", confidence=0.7)
    out = merge_scale_info(cap, ocr)
    # Acceptable: warning is None OR empty string.
    assert not out.warning, f"unexpected warning: {out.warning!r}"