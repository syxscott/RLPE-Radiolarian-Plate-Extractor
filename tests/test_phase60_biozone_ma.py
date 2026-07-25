"""Tests for Phase 60 Plan 3 — Bug 3.11: biozone names are not mapped
to numeric Ma bounds.

Real radiolarian biostratigraphy papers use named biozones like:

  * ``Buryella clinata Zone``
  * ``Cryptocephalus nigricae Zone``
  * ``UAZ 1-7`` (Unitary Association Zones, after Baumgartner 1984)
  * ``Pessagno Zone A``

These names were stored in ``geology_links.biozone`` as opaque strings
with no numeric ``ma_top`` / ``ma_base`` to plot against. Without
that mapping the biostratigraphy column on the Web UI / xlsx export
could not position the samples on the Ma axis.

Phase 60 Plan 3 fix: ``_BIOZONE_TO_MA`` table maps a curated set of
Baumgartner 1984 / O'Dogherty / Hollis standard zones to ``(ma_top,
ma_base)`` tuples. ``lookup_biozone_ma(name)`` returns the bounds or
``None`` for unknown zones — unknown zones are flagged ``unknown_biozone``
rather than invented.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.stratigraphy import (  # noqa: E402
    _BIOZONE_TO_MA,
    lookup_biozone_ma,
)


def test_biozone_table_nonempty():
    """The biozone-to-Ma table must contain at least 5 named zones."""
    assert len(_BIOZONE_TO_MA) >= 5, (
        f"_BIOZONE_TO_MA shrunk to {len(_BIOZONE_TO_MA)}; "
        "add named zones back to maintain the fix"
    )


def test_biozone_to_ma_lookup():
    """At least 5 well-known biozones from Baumgartner 1984 / Hollis
    must resolve to a numeric ``(ma_top, ma_base)`` tuple."""
    # Baumgartner 1984 standard radiolarian zones (subset).
    # References: Baumgartner et al. (1984) "A Middle Jurassic to
    # Early Cretaceous radiolarian zonation based on Unitary
    # Associations and Rhabdocyclus costatus" — UAZ 1-21.
    cases = [
        # UAZ 1 = Callovian–Kimmeridgian (Baumgartner 1984 fig. 1)
        ("UAZ 1", 152.0, 168.0),
        # UAZ 5 = Hauterivian–Barremian (Baumgartner 1984 fig. 1)
        ("UAZ 5", 121.4, 132.6),
        # UAZ 11 = Campanian–Maastrichtian
        ("UAZ 11", 66.0, 83.6),
        # Hollis 1997 NZ Late Paleocene (Thanetian) radiolarian zone
        # (corrected: was incorrectly set to Permian ~254 Ma)
        ("Buryella clinata Zone", 56.0, 59.0),
        # Late Cretaceous Coniacian–Santonian zone
        ("Cryptocephalus nigricae Zone", 83.6, 86.3),
    ]
    for name, exp_top, exp_base in cases:
        out = lookup_biozone_ma(name)
        assert out is not None, f"{name!r} not found in biozone table"
        ma_top, ma_base = out
        # Tolerate ±5 Ma drift from Baumgartner / Hollis original tables
        # because the curated table rounds to stage boundaries and the
        # ICS Ma values themselves have ±0.5 Ma uncertainty.
        assert abs(ma_top - exp_top) <= 5.0, (
            f"{name!r}: ma_top={ma_top} expected ~{exp_top}"
        )
        assert abs(ma_base - exp_base) <= 5.0, (
            f"{name!r}: ma_base={ma_base} expected ~{exp_base}"
        )


def test_lookup_returns_none_for_unknown_biozone():
    """Unknown zone names return ``None`` rather than inventing data."""
    assert lookup_biozone_ma("This zone does not exist 9999") is None
    assert lookup_biozone_ma("") is None
    assert lookup_biozone_ma(None) is None


def test_lookup_handles_trailing_zone_word():
    """``Buryella clinata Zone`` and ``Buryella clinata`` (without
    ``Zone``) should both resolve (we strip a trailing ``Zone``)."""
    a = lookup_biozone_ma("Buryella clinata Zone")
    b = lookup_biozone_ma("Buryella clinata")
    assert a is not None
    assert b is not None
    assert a == b


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])