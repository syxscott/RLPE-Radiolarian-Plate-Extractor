"""Tests for Phase 63 Plan 6 — Bug 6.14: ``_locality_id`` must include
modern coords when legacy coords are absent.

Before: ``_locality_id`` hashed ``(paper_id, locality, latitude,
longitude)`` where ``latitude/longitude`` are the LEGACY fields. Round
25+ converters only populate ``modern_latitude/modern_longitude``,
leaving the legacy fields as ``None``. Two distinct modern localities
that shared a name (rare in nature, common in deep-time rock units
with same name in two basins) collapsed onto the SAME hash and the
export silently dropped one.

After: ``_locality_id`` (and the dedup key in
``locality_records_from_geology``) prefer ``modern_latitude /
modern_longitude`` when set, falling back to legacy when not.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.converters import _locality_id  # noqa: E402


def test_locality_id_modern_coords_used():
    """Two localities with the same name but different modern coords
    produce different ``_locality_id`` hashes."""
    geo_a = {
        "locality": "Lone Mountain",
        "modern_latitude": 39.5,
        "modern_longitude": -117.0,
        # legacy fields left None
        "latitude": None,
        "longitude": None,
    }
    geo_b = {
        "locality": "Lone Mountain",
        "modern_latitude": 39.6,
        "modern_longitude": -117.1,
        # legacy fields left None
        "latitude": None,
        "longitude": None,
    }
    lid_a = _locality_id(geo_a, "p1")
    lid_b = _locality_id(geo_b, "p1")
    assert lid_a != lid_b, (
        f"_locality_id collapsed on identical-name localities with "
        f"distinct modern coords: {lid_a!r} == {lid_b!r}. "
        "Phase 63 Plan 6.14 fix regressed."
    )


def test_locality_id_legacy_fallback_still_works():
    """When only the legacy fields are set, _locality_id still
    produces a stable hash for backward-compat with Round 24 runs."""
    geo = {
        "locality": "Italy",
        "latitude": 46.5,
        "longitude": 11.5,
        "modern_latitude": None,
        "modern_longitude": None,
    }
    lid = _locality_id(geo, "p1")
    assert lid.startswith("loc_"), lid
    # Sanity: same inputs -> same hash
    assert lid == _locality_id(geo, "p1")


def test_locality_id_modern_takes_precedence():
    """When BOTH modern and legacy are set, modern wins (Round 25+)."""
    geo_modern_first = {
        "locality": "Italy",
        "latitude": None,         # legacy missing
        "longitude": None,
        "modern_latitude": 36.5,  # modern present
        "modern_longitude": 4.8,
    }
    geo_legacy_only = {
        "locality": "Italy",
        "latitude": 46.5,
        "longitude": 11.5,
        "modern_latitude": None,
        "modern_longitude": None,
    }
    lid_modern = _locality_id(geo_modern_first, "p1")
    lid_legacy = _locality_id(geo_legacy_only, "p1")
    # They must differ — one is at Italian coordinates, the other at
    # North African coordinates for the same-named locale.
    assert lid_modern != lid_legacy


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
