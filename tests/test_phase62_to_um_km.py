"""Phase 62 Plan 5 (Bug 5.9): to_um must handle 'km'.

Previously ``to_um(value, unit)`` had explicit branches for
``um`` / ``mm`` / ``cm`` / ``nm`` and silently returned ``None``
for any other unit — including ``km``. A ``"1 km"`` scale bar
(a map-scale locality overview) would parse cleanly via
``SCALE_PATTERN`` (km is a valid length unit) but the resulting
``um_per_px`` was ``None`` with no diagnostic for the operator.

The fix: add a ``km`` branch. 1 km = 1e9 µm. This is far above
the sanity range (0.1..10000 µm) so the sanity gate from Bug 5.2
will still drop the resulting ScaleInfo, but the conversion itself
is now well-defined and the operator can see the unit was
recognised.
"""

from __future__ import annotations

from rlpe.scale_bar import to_um


def test_to_um_km_conversion():
    """1 km = 1e9 µm."""
    out = to_um(1.0, "km")
    assert out == 1e9, f"expected 1e9 µm for 1 km, got {out}"


def test_to_um_km_fractional():
    """0.5 km = 5e8 µm."""
    out = to_um(0.5, "km")
    assert out == 5e8


def test_to_um_normalizes_uppercase_km():
    """Unit normalization happens via normalize_unit; 'KM' must
    also produce 1e9 µm."""
    out = to_um(1.0, "KM")
    # normalize_unit lowercases before matching, so this works.
    assert out == 1e9


def test_to_um_regression_known_units():
    """Regression: existing units still convert correctly."""
    assert to_um(100.0, "um") == 100.0
    assert to_um(1.0, "mm") == 1000.0
    assert to_um(1.0, "cm") == 10000.0
    assert to_um(1000.0, "nm") == 1.0


def test_to_um_unknown_unit_returns_none():
    """Unknown units still return None (no silent coercion)."""
    assert to_um(1.0, "lightyear") is None
    assert to_um(1.0, "") is None
