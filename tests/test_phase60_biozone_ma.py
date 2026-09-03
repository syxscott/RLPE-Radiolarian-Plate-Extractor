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
Baumgartner 1984 / O'Dogherty / Hollis standard zones to
``BiozoneMa`` NamedTuples (top_ma, base_ma, confidence).
``lookup_biozone_ma(name)`` returns the NamedTuple or ``None`` for
unknown zones — unknown zones are flagged ``unknown_biozone`` rather
than invented.

Audit 2026-09-03 (BLOCKER-#7) added a ``confidence`` field so
downstream consumers (PBDB exporter, find_ages_in_text) can
distinguish well-anchored zones (0.95 for UAZ 1-12) from interpolated
zones (0.5 for UAZ 13-21). A ``lookup_biozone_ma_legacy`` shim
returns the historical 2-tuple for callers that haven't migrated.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.stratigraphy import (  # noqa: E402
    BiozoneMa,
    _BIOZONE_TO_MA,
    lookup_biozone_ma,
    lookup_biozone_ma_legacy,
)


def test_biozone_table_nonempty():
    """The biozone-to-Ma table must contain at least 5 named zones."""
    assert len(_BIOZONE_TO_MA) >= 5, (
        f"_BIOZONE_TO_MA shrunk to {len(_BIOZONE_TO_MA)}; add named zones back to maintain the fix"
    )


def test_biozone_to_ma_lookup():
    """At least 5 well-known biozones from Baumgartner 1984 / Hollis
    must resolve to a numeric ``(ma_top, ma_base)`` NamedTuple."""
    # Baumgartner 1984 standard radiolarian zones (subset).
    # References: Baumgartner et al. (1984) "A Middle Jurassic to
    # Early Cretaceous radiolarian zonation based on Unitary
    # Associations and Rhabdocyclus costatus" — UAZ 1-21.
    # audit 2026-07-31: UAZ values recalibrated to the published
    # Baumgartner et al. 1995 scheme (UAZones95, Aalenian→Hauterivian/
    # Barremian). The old values placed UAZ 5/11 in the Late
    # Cretaceous — 40-80 Myr too young.
    cases = [
        # UAZ 1 = early–middle Aalenian (~172-174.7 Ma)
        ("UAZ 1", 172.0, 174.7),
        # UAZ 5 = latest Bajocian–early Bathonian (~166-168 Ma)
        ("UAZ 5", 166.2, 167.7),
        # UAZ 11 = late Kimmeridgian–early Tithonian (~147-152 Ma)
        ("UAZ 11", 147.5, 152.0),
        # Hollis 1997 NZ Late Paleocene (Thanetian) radiolarian zone
        # (corrected: was incorrectly set to Permian ~254 Ma)
        ("Buryella clinata Zone", 56.0, 59.0),
        # Late Cretaceous Coniacian–Santonian zone
        ("Cryptocephalus nigricae Zone", 83.6, 86.3),
    ]
    for name, exp_top, exp_base in cases:
        out = lookup_biozone_ma(name)
        assert out is not None, f"{name!r} not found in biozone table"
        # Audit 2026-09-03 (BLOCKER-#7): return type is now BiozoneMa
        # (top_ma, base_ma, confidence). Tolerate ±5 Ma drift from
        # Baumgartner / Hollis original tables because the curated
        # table rounds to stage boundaries and the ICS Ma values
        # themselves have ±0.5 Ma uncertainty.
        assert isinstance(out, BiozoneMa), (
            f"{name!r} returned {type(out).__name__}, expected BiozoneMa"
        )
        ma_top, ma_base = out.top_ma, out.base_ma
        assert abs(ma_top - exp_top) <= 5.0, f"{name!r}: ma_top={ma_top} expected ~{exp_top}"
        assert abs(ma_base - exp_base) <= 5.0, f"{name!r}: ma_base={ma_base} expected ~{exp_base}"


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


# ---------------------------------------------------------------------------
# Audit 2026-09-03 (BLOCKER-#7): confidence propagation
# ---------------------------------------------------------------------------


def test_uaz_1_to_12_high_confidence():
    """UAZ 1-12 are calibrated against ICS 2023 stage boundaries —
    confidence must be 0.95 (the highest tier in the table)."""
    for i in range(1, 13):
        bm = lookup_biozone_ma(f"UAZ {i}")
        assert bm is not None, f"UAZ {i} not found"
        assert bm.confidence == 0.95, (
            f"UAZ {i} confidence={bm.confidence}, expected 0.95 "
            "(calibrated against ICS 2023 stages)"
        )


def test_uaz_13_to_21_low_confidence():
    """UAZ 13-21 are spaced evenly over the 145-123 Ma interval
    with the source comments explicitly marking them "(approx.)" —
    confidence must be 0.5 so downstream consumers (PBDB exporter)
    can fall back to section-measured ages."""
    for i in range(13, 22):
        bm = lookup_biozone_ma(f"UAZ {i}")
        assert bm is not None, f"UAZ {i} not found"
        assert bm.confidence == 0.5, (
            f"UAZ {i} confidence={bm.confidence}, expected 0.5 "
            "(spaced evenly — marked (approx.) in source comments)"
        )


def test_legacy_shim_returns_plain_tuple():
    """``lookup_biozone_ma_legacy`` returns the historical 2-tuple
    for backward-compat with code that does
    ``ma_top, ma_base = lookup_biozone_ma(name)``."""
    out = lookup_biozone_ma_legacy("UAZ 1")
    assert out == (172.0, 174.7)
    # Plain tuple, not NamedTuple.
    assert type(out) is tuple
    # Unknown zones propagate as None.
    assert lookup_biozone_ma_legacy("BOGUS 9999") is None


def test_age_classification_carries_confidence():
    """When find_ages_in_text sees a UAZ 17 in free text, the
    resulting AgeClassification must carry confidence 0.5 so
    downstream consumers can detect the (approx.) zones."""
    from rlpe.stratigraphy import find_ages_in_text
    out = find_ages_in_text("Sample S1 from UAZ 17 zone in the Valanginian.")
    # Find the biozone classification.
    biozone = [c for c in out if c.rank == "biozone"]
    assert biozone, f"No biozone classification in {out}"
    biozone = biozone[0]
    assert biozone.confidence < 0.7, (
        f"UAZ 17 (approx.) AgeClassification confidence={biozone.confidence}, "
        "expected < 0.7 so PBDB exporter triggers section-based fallback"
    )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
