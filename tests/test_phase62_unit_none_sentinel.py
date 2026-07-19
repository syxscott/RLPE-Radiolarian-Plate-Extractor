"""Phase 62 Plan 5 (Bug 5.11): normalize_unit must distinguish None
from empty-string.

Previously ``normalize_unit(unit)`` did::

    u = (unit or "").lower().strip()

So:
  - ``normalize_unit(None)`` returned ``""`` (empty string).
  - ``normalize_unit("")`` returned ``""`` (empty string).

The two were indistinguishable to callers. ``to_um`` then
fell through all branches and returned ``None`` — which is also
indistinguishable from "value was None" or "unit was unknown".

The fix: ``normalize_unit`` returns the explicit sentinel
``"__unknown__"`` when given ``None`` or empty string, so callers
can tell "no unit at all" from "unknown unit" (which still returns
the lowercased input as a string).

A new ``UNKNOWN_UNIT`` constant is exported so callers can check
without hard-coding the string.
"""
from __future__ import annotations

from rlpe.scale_bar import UNKNOWN_UNIT, normalize_unit


def test_normalize_unit_none_returns_sentinel():
    """normalize_unit(None) returns the explicit sentinel."""
    out = normalize_unit(None)
    assert out == UNKNOWN_UNIT, f"expected UNKNOWN_UNIT sentinel, got {out!r}"
    assert out != "", "sentinel must differ from empty string"


def test_normalize_unit_empty_returns_sentinel():
    """normalize_unit('') returns the explicit sentinel."""
    out = normalize_unit("")
    assert out == UNKNOWN_UNIT


def test_normalize_unit_whitespace_returns_sentinel():
    """normalize_unit('   ') returns the explicit sentinel."""
    out = normalize_unit("   ")
    assert out == UNKNOWN_UNIT


def test_normalize_unit_recognised_unit():
    """Regression: 'µm' still normalises to 'um'."""
    assert normalize_unit("µm") == "um"
    assert normalize_unit("µM") == "um"
    assert normalize_unit("microns") == "um"
    assert normalize_unit("MM") == "mm"


def test_normalize_unit_unknown_unit_passes_through():
    """Unknown (non-empty) units pass through lowercased — distinct
    from the None/empty sentinel."""
    out = normalize_unit("lightyear")
    assert out == "lightyear"
    assert out != UNKNOWN_UNIT


def test_unknown_unit_constant_exported():
    """The UNKNOWN_UNIT constant is importable from rlpe.scale_bar."""
    assert UNKNOWN_UNIT == "__unknown__"