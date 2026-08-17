"""Phase 62 Plan 5 (Bug 5.6): biozone regex must accept numbered zones.

Previously ``_BIOZONE_RE`` accepted only:
  - ``N. optima Zone`` (first-letter abbrev. + lowercase + Zone)
  - ``Zone 5`` / ``Subzone 5a`` (numbered zones)
  - ``Morozovella aragonensis Zone`` (full taxon-named zone)

It MISSED the very common numbered-zone variants used in
radiolarian biostratigraphy:
  - ``UAZ 1`` / ``UAZ 7`` (Unitary Association Zones — a
    standard radiolarian biozone scheme)
  - ``UAZ 1-7`` / ``UAZ 5-6`` (UAZ range form)
  - ``Pessagno Zone A`` / ``Pessagno Zone B`` (Pessagno's
    Jurassic-Cretaceous zonation)
  - ``Pessagno Zone 1`` (numbered variant)
  - ``Pessagno 1977 Zone A`` (citation form)

The fix: extend ``_BIOZONE_RE`` with these patterns.
"""

from __future__ import annotations

from rlpe.geology_extraction import _BIOZONE_RE


def test_biozone_uaz_numbered():
    """UAZ 1 / UAZ 7 / UAZ 12 must match."""
    for txt in ("UAZ 1", "UAZ 7", "UAZ 12", "UAZ 100"):
        m = _BIOZONE_RE.search(txt)
        assert m is not None, f"UAZ numbered zone not matched: {txt!r}"


def test_biozone_uaz_range():
    """UAZ 1-7 (range form) must match."""
    m = _BIOZONE_RE.search("UAZ 1-7")
    assert m is not None, "UAZ 1-7 range form not matched"


def test_biozone_pessagno_zone_letter():
    """Pessagno Zone A must match."""
    m = _BIOZONE_RE.search("Pessagno Zone A")
    assert m is not None, "Pessagno Zone A not matched"


def test_biozone_pessagno_zone_number():
    """Pessagno Zone 1 must match."""
    m = _BIOZONE_RE.search("Pessagno Zone 1")
    assert m is not None, "Pessagno Zone 1 not matched"


def test_biozone_pessagno_range():
    """Pessagno Zones A-B (range form) must match."""
    m = _BIOZONE_RE.search("Pessagno Zones A-B")
    assert m is not None, "Pessagno Zones A-B not matched"


def test_biozone_regression_existing_still_works():
    """Regression: existing patterns still match."""
    for txt in (
        "N. optima Zone",
        "Zone 5a",
        "Subzone 5a",
        "Morozovella aragonensis Zone",
    ):
        m = _BIOZONE_RE.search(txt)
        assert m is not None, f"regression: {txt!r} no longer matches"
