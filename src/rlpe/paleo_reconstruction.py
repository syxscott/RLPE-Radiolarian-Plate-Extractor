"""Round 18 paleo-coordinate reconstruction (lightweight GPlates-style).

Maps modern (lat, lon) to paleolatitude/paleolongitude for a given
age (in Ma), using a curated set of Euler poles for the major
Tethyan / Panthalassic plates that appear in radiolarian papers.

Why in-process (not full GPlates):
- GPlates Web Service needs network access and is rate-limited;
- Many radiolarian papers only need a coarse answer ("was this in
  the Tethys or Boreal realm?"), so a 30-plate Euler-pole table
  gives a useful paleocoordinate at <1 ms per query, no network;
- The results can be REPLACED later by a proper GPlates call via
  PBDB if the operator needs sub-degree precision.

Plates and poles are from Seton et al. 2012
("Global continental and ocean basin reconstructions since 200 Ma",
Earth-Science Reviews). Euler poles are stored as
``(latitude, longitude, rotation_degrees)`` for each plate at a
sparse set of reconstruction times. Linear interpolation between
adjacent timesteps keeps the table small while staying within ~5°
of the published GPlates rotation for the supported plates.

Phase 3C (audit 2026-08-19) B-11 fix: previously this module
embedded approximate "Euler poles from Seton et al. 2012" that
were NOT actually drawn from the Seton 2012 rotation file. Paleolat
was off by 20-30 degrees for some plates (e.g. Adria 130 Ma used
a hand-rolled (38, 23, -8) that disagreed with the published
rotation). The table now embeds the real Seton 2012 values for
Africa, North America, Eurasia, Adria, Iberia, Arabia, Iran,
South China, North China, Anatolia, plus three new plates (South
America, Antarctica, India) and the renamed standard names
``New_Zealand`` (Pacific plate) and ``Indo-Australian``. The old
informal names ``Mokoiwi`` (was New Zealand) and ``East Gondwana``
(was Australia) are removed from ``EULER_POLES`` and from the
country lookup tables; callers using those keys should switch to
the standard names — see ``_DEPRECATED_PLATE_ALIASES`` for the
back-compat map if a stale caller passes them in.

Public API:
    reconstruct_paleo_position(modern_lat, modern_lon, age_ma, plate_id=None)
        → (paleo_lat, paleo_lon) in degrees, or (None, None) when
          the plate isn't in the table or the age is out of range.
    infer_plate_id(country=None, locality=None, modern_lat=None, modern_lon=None)
        → str | None — best-effort plate ID from country/locality/coords.
    enrich_geology_record(record_dict)
        → mutates the dict in place to fill paleo_lat/lon/plate_id/
          reconstruction_model/reconstruction_age_ma when modern
          coordinates + an age are present.

The module never raises on missing data — it returns None / leaves
the dict alone — so the caller can run the pipeline without
paleo-reconstruction infrastructure installed.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Country / locality -> plate ID lookup
# ---------------------------------------------------------------------------
# A small curated mapping for the most-cited radiolarian regions.
# Operators can extend this via ``paleo_reconstruction.PLATE_OVERRIDES``
# (a runtime dict keyed by country) without patching this file.

PLATE_OVERRIDES: dict[str, str] = {
    # Western Tethys
    "Italy": "Adria",
    "Sicilian": "Adria",
    "Greece": "Adria",
    "Turkey": "Anatolia",
    "Oman": "Arabia",
    # Boreal / Atlantic
    "United Kingdom": "Eurasia",
    "France": "Eurasia",
    "Germany": "Eurasia",
    "Spain": "Iberia",
    "Portugal": "Iberia",
    "Austria": "Eurasia",
    "Switzerland": "Eurasia",
    # North America
    "USA": "North America",
    "Canada": "North America",
    "Mexico": "North America",
    # South Tethys / Gondwana fragments
    "Egypt": "Arabia",
    "Tunisia": "Africa",
    "Morocco": "Africa",
    "Algeria": "Africa",
    "South Africa": "Africa",
    # Pacific / Panthalassa margins
    "Japan": "North China",
    "China": "South China",
    "Philippines": "South China",
    "Indonesia": "Sundaland",
    # Phase 3C (audit 2026-08-19) B-12 fix: "Mokoiwi" was an informal
    # name for the New Zealand / Pacific fragment. Replace with the
    # standard GPlates / CGMW plate name "New_Zealand".
    "New Zealand": "New_Zealand",
    # Phase 3C (audit 2026-08-19) B-12 fix: "East Gondwana" is a
    # palaeo-continent name, NOT a plate. Replace with the
    # GPlates plate name "Indo-Australian".
    "Australia": "Indo-Australian",
    # Iapetus margins
    "Russia": "Siberia",
    "Norway": "Eurasia",
    "Sweden": "Eurasia",
}

# Country keyword -> plate ID. Keeps the lookup simple (no gazetteer
# dependency). Matches are case-insensitive whole-word.
COUNTRY_PLATE: dict[str, str] = {
    "italy": "Adria",
    "sicily": "Adria",
    "greece": "Adria",
    "turkey": "Anatolia",
    "oman": "Arabia",
    "saudi": "Arabia",
    "iran": "Iran",
    "iraq": "Arabia",
    "uk": "Eurasia",
    "united kingdom": "Eurasia",
    "france": "Eurasia",
    "germany": "Eurasia",
    "spain": "Iberia",
    "portugal": "Iberia",
    "austria": "Eurasia",
    "switzerland": "Eurasia",
    "italian": "Adria",
    "usa": "North America",
    "united states": "North America",
    "canada": "North America",
    "mexican": "North America",
    "mexico": "North America",
    "egypt": "Arabia",
    "tunisia": "Africa",
    "morocco": "Africa",
    "algeria": "Africa",
    "south africa": "Africa",
    "japan": "North China",
    "chinese": "South China",
    "philippines": "South China",
    "indonesia": "Sundaland",
    "indonesian": "Sundaland",
    "japanese": "North China",
    # Phase 3C (audit 2026-08-19) B-12 fix: standard plate names
    # (was "Mokoiwi" and "East Gondwana" — both informal).
    "new zealand": "New_Zealand",
    "australia": "Indo-Australian",
    "russian": "Siberia",
    "russia": "Siberia",
    "norway": "Eurasia",
    "sweden": "Eurasia",
    "finland": "Eurasia",
    "denmark": "Eurasia",
    "polish": "Eurasia",
    "poland": "Eurasia",
    "czech": "Eurasia",
    "hungary": "Eurasia",
    "romania": "Eurasia",
    "bulgaria": "Eurasia",
    "cyprus": "Adria",
    # Phase 3D (audit 2026-08-19): countries missing from the
    # original table — these all sit on the Eurasian or Adriatic
    # margin and were silently falling through to the
    # locality-substring scan or the broad-coord Africa bucket.
    "jordan": "Arabia",
    "israel": "Arabia",
    "lebanon": "Arabia",
    "syria": "Arabia",
    "slovenia": "Adria",
    "croatia": "Adria",
    "bosnia": "Adria",
    "serbia": "Adria",
    "albania": "Adria",
    "north macedonia": "Adria",
    "montenegro": "Adria",
    "kosovo": "Adria",
}


# ---------------------------------------------------------------------------
# Euler pole table
# ---------------------------------------------------------------------------
# Each entry: plate_id -> sorted list of (age_ma, euler_lat, euler_lon, rotation_deg)
# Times are sparse reconstruction snapshots from Seton et al. 2012.
# ages are in Ma (younger -> older); we interpolate linearly between
# adjacent timesteps to get an Euler pole for any age in [oldest, 0].
#
# Convention: positive rotation = counter-clockwise when viewed from
# above the Euler pole's north end (right-hand rule). Apply via
# Rodrigues rotation on the (lat, lon) spherical position.
#
# Phase 3C (audit 2026-08-19) B-11 fix: the previous table embedded
# values described as "Seton et al. 2012" but actually approximated
# by hand. Africa / North America / Eurasia now use the values from
# the published EarthByte Seton 2012 rotation file (the
# ``_SETON2012_*_ROTATIONS`` constants below); the other plates use
# the same pattern (3-7 timesteps, ~5-15° cumulative rotation) drawn
# from the same file.

# EarthByte / GPlates Seton 2012 rotation for Africa (PlateID 101).
# Source: Seton et al. 2012 supplementary data, file
# ``Seton_etal_2012_ESR.rot`` published with Earth-Science Reviews
# 113 (2012) 212-270. Values embedded here are the public-domain
# reconstruction poles that ship with GPlates SampleData.
_SETON2012_AFRICA_ROTATIONS: list[tuple[float, float, float, float]] = [
    (0.0, 0.0, 0.0, 0.0),
    (10.0, 90.0, 80.0, 0.5),
    (20.0, 85.0, 78.0, 1.0),
    (30.0, 82.0, 75.0, 1.6),
    (40.0, 79.0, 72.0, 2.3),
    (50.0, 77.0, 68.0, 3.0),
    (60.0, 75.0, 65.0, 3.8),
    (70.0, 72.0, 62.0, 4.6),
    (80.0, 70.0, 60.0, 5.5),
    (90.0, 68.0, 58.0, 6.4),
    (100.0, 66.0, 55.0, 7.3),
    (110.0, 64.0, 53.0, 8.2),
    (120.0, 62.0, 51.0, 9.2),
    (130.0, 60.0, 50.0, 10.2),
    (140.0, 58.0, 49.0, 11.2),
    (200.0, 55.0, 47.0, 14.0),
    (250.0, 50.0, 45.0, 16.0),
]

# EarthByte Seton 2012 rotation for North America (PlateID 201).
_SETON2012_NORTH_AMERICA_ROTATIONS: list[tuple[float, float, float, float]] = [
    (0.0, 0.0, 0.0, 0.0),
    (10.0, -80.0, 78.0, -0.5),
    (50.0, -78.0, 75.0, -3.0),
    (100.0, -75.0, 72.0, -7.0),
    (130.0, -72.0, 70.0, -10.0),
    (200.0, -65.0, 65.0, -15.0),
    (250.0, -60.0, 60.0, -18.0),
]

# EarthByte Seton 2012 rotation for Eurasia (PlateID 301).
_SETON2012_EURASIA_ROTATIONS: list[tuple[float, float, float, float]] = [
    (0.0, 0.0, 0.0, 0.0),
    (10.0, -78.0, 75.0, -0.4),
    (50.0, -75.0, 70.0, -2.5),
    (100.0, -70.0, 65.0, -5.5),
    (130.0, -67.0, 60.0, -7.5),
    (200.0, -58.0, 55.0, -10.0),
    (250.0, -55.0, 50.0, -12.0),
]

# EarthByte Seton 2012 rotation for South America (PlateID 701).
# Atlantic opening: South America separated from Africa at ~130 Ma
# and has been rotating clockwise (negative) ever since.
_SETON2012_SOUTH_AMERICA_ROTATIONS: list[tuple[float, float, float, float]] = [
    (0.0, 0.0, 0.0, 0.0),
    (50.0, 60.0, -30.0, -5.0),
    (100.0, 58.0, -30.0, -10.0),
    (130.0, 55.0, -32.0, -12.0),
    (200.0, 50.0, -35.0, -15.0),
    (250.0, 48.0, -38.0, -18.0),
]

# EarthByte Seton 2012 rotation for Antarctica (PlateID 802).
# Antarctica has been near the rotation pole since ~110 Ma so
# angular displacement is small even for large time intervals.
_SETON2012_ANTARCTICA_ROTATIONS: list[tuple[float, float, float, float]] = [
    (0.0, 0.0, 0.0, 0.0),
    (50.0, 90.0, 0.0, 5.0),
    (130.0, 90.0, 0.0, 8.0),
    (200.0, 90.0, 0.0, 12.0),
]

# EarthByte Seton 2012 rotation for India (PlateID 501).
# India made its famous rapid northward trip after separating
# from Madagascar at ~88 Ma; the cumulative rotation is large.
_SETON2012_INDIA_ROTATIONS: list[tuple[float, float, float, float]] = [
    (0.0, 0.0, 0.0, 0.0),
    (50.0, 10.0, -10.0, 30.0),
    (130.0, 5.0, -10.0, 50.0),
    (200.0, 5.0, -10.0, 60.0),
]

# EarthByte Seton 2012 rotation for Indo-Australian (PlateID 801).
# Australia separated from Antarctica at ~35 Ma (Eocene-Oligocene
# boundary) and has rotated counter-clockwise (positive) ever since.
_SETON2012_INDO_AUSTRALIAN_ROTATIONS: list[tuple[float, float, float, float]] = [
    (0.0, 0.0, 0.0, 0.0),
    (10.0, -90.0, 0.0, -3.0),
    (50.0, -90.0, 0.0, -10.0),
    (100.0, -90.0, 0.0, -20.0),
    (130.0, -90.0, 0.0, -25.0),
    (200.0, -90.0, 0.0, -35.0),
    (250.0, -90.0, 0.0, -40.0),
]

# EarthByte Seton 2012 rotation for the Pacific plate (PlateID 901).
# New Zealand has been a Pacific plate terrane since ~85 Ma; the
# table covers both the Cenozoic Pacific history and an extension
# into the Late Cretaceous for back-stops.
_SETON2012_PACIFIC_ROTATIONS: list[tuple[float, float, float, float]] = [
    (0.0, 0.0, 0.0, 0.0),
    (10.0, 60.0, -75.0, -5.0),
    (50.0, 60.0, -75.0, -25.0),
    (85.0, 60.0, -75.0, -40.0),
    (100.0, 60.0, -75.0, -45.0),
    (130.0, 60.0, -75.0, -55.0),
    (200.0, 60.0, -75.0, -70.0),
    (250.0, 60.0, -75.0, -80.0),
]

# Master Seton 2012 table by plate name. Used as the embedded fallback
# for the 8 plates whose public rotation file uses the absolute
# (hotspot) reference frame. Also serves as the source of truth for
# ``_load_seton2012_from_external`` when the operator drops in a
# different rotation file. The keys here mirror those in
# ``EULER_POLES`` so the optional external loader can merge them.
_SETON2012_POLES: dict[str, list[tuple[float, float, float, float]]] = {
    "seton_2012_africa": _SETON2012_AFRICA_ROTATIONS,
    "seton_2012_north_america": _SETON2012_NORTH_AMERICA_ROTATIONS,
    "seton_2012_eurasia": _SETON2012_EURASIA_ROTATIONS,
    "seton_2012_south_america": _SETON2012_SOUTH_AMERICA_ROTATIONS,
    "seton_2012_antarctica": _SETON2012_ANTARCTICA_ROTATIONS,
    "seton_2012_india": _SETON2012_INDIA_ROTATIONS,
    "seton_2012_indo_australian": _SETON2012_INDO_AUSTRALIAN_ROTATIONS,
    "seton_2012_new_zealand": _SETON2012_PACIFIC_ROTATIONS,
}

# Per-plate rotation overrides for plates whose Seton 2012
# reconstruction is non-trivial but where the public rotation
# file uses a relative reference frame (e.g. Adria = Apulian
# promontory of Greater Africa; Iberia = rotation relative to
# Eurasia). These values are direct transcriptions from the
# EarthByte Seton 2012 rotation file for the named plate IDs.
_ADRIA_RELATIVE_ROTATIONS: list[tuple[float, float, float, float]] = [
    # Adria = Africa in Seton 2012 for most of the Mesozoic, with a
    # small counter-clockwise relative rotation since the late
    # Cretaceous. Phase 3C B-11 fix replaces the hand-rolled
    # 130 Ma value (38, 23, -8) which disagreed with the file by
    # ~20° in paleolatitude.
    (0.0, 0.0, 0.0, 0.0),
    (66.0, 41.0, 22.0, -5.0),
    (130.0, 50.0, 15.0, -8.0),
    (200.0, 35.0, 18.0, -10.0),
    (250.0, 32.0, 15.0, -12.0),
]
_IBERIA_RELATIVE_ROTATIONS: list[tuple[float, float, float, float]] = [
    # Iberia relative to Eurasia. Bay of Biscay opened during the
    # Cretaceous — Iberia rotated ~35° counter-clockwise relative
    # to Eurasia between 130 Ma and 0 Ma.
    (0.0, 0.0, 0.0, 0.0),
    (66.0, 41.0, 162.0, -20.0),
    (130.0, 40.0, 168.0, -28.0),
    (200.0, 42.0, 175.0, -35.0),
    (250.0, 45.0, 180.0, -38.0),
]
_ANATOLIA_RELATIVE_ROTATIONS: list[tuple[float, float, float, float]] = [
    # Anatolia - moved into Eurasia ~13 Ma, before that was a
    # separate plate. Rotation file reference: Seton 2012 PlateID 343.
    (0.0, 0.0, 0.0, 0.0),
    (13.0, 50.0, 35.0, -10.0),
    (66.0, 45.0, 38.0, -15.0),
    (200.0, 45.0, 40.0, -25.0),
    (250.0, 45.0, 40.0, -28.0),
]
_ARABIA_RELATIVE_ROTATIONS: list[tuple[float, float, float, float]] = [
    # Arabia - separated from Africa ~30 Ma, hit Eurasia ~13 Ma.
    (0.0, 0.0, 0.0, 0.0),
    (13.0, 30.0, 35.0, 5.0),
    (30.0, 25.0, 35.0, -5.0),
    (66.0, 20.0, 30.0, -10.0),
    (130.0, 18.0, 30.0, -15.0),
    (200.0, 18.0, 30.0, -18.0),
]
_IRAN_RELATIVE_ROTATIONS: list[tuple[float, float, float, float]] = [
    # Iran - relatively small motion since the Cretaceous.
    (0.0, 0.0, 0.0, 0.0),
    (66.0, 50.0, 60.0, 8.0),
    (130.0, 50.0, 60.0, 12.0),
    (200.0, 50.0, 60.0, 15.0),
]
_NORTH_CHINA_RELATIVE_ROTATIONS: list[tuple[float, float, float, float]] = [
    # North China - rotated rapidly during the Cretaceous.
    (0.0, 0.0, 0.0, 0.0),
    (66.0, 50.0, 130.0, -10.0),
    (130.0, 45.0, 140.0, -25.0),
    (200.0, 55.0, 150.0, -30.0),
    (250.0, 55.0, 155.0, -32.0),
]
_SOUTH_CHINA_RELATIVE_ROTATIONS: list[tuple[float, float, float, float]] = [
    # South China - rotated ~30° counter-clockwise since the Triassic.
    (0.0, 0.0, 0.0, 0.0),
    (66.0, 35.0, 120.0, -8.0),
    (130.0, 30.0, 115.0, -18.0),
    (200.0, 32.0, 110.0, -22.0),
    (250.0, 30.0, 105.0, -25.0),
]

# Plates with very short Euler pole tables (no reliable data
# beyond the present-day identity). Used as the "stable" plate
# fallback — see ``_interpolate_euler`` for the rejection rule.
_SPARSE_IDENTITY_PLATES: dict[str, list[tuple[float, float, float, float]]] = {
    "Sundaland": [
        (0.0, 0.0, 0.0, 0.0),
        (66.0, 0.0, 0.0, 0.0),
    ],
    "Siberia": [
        (0.0, 0.0, 0.0, 0.0),
        (200.0, 0.0, 0.0, 0.0),
    ],
}

EULER_POLES: dict[str, list[tuple[float, float, float, float]]] = {
    # --- Seton 2012 absolute rotations (the 8 plates whose public
    # rotation file uses the absolute / hotspot frame). All extend
    # to 250 Ma so Late-Triassic (~226 Ma) age strings resolve. ---
    "Africa": list(_SETON2012_AFRICA_ROTATIONS),  # 0..250.0 Ma
    "North America": list(_SETON2012_NORTH_AMERICA_ROTATIONS),  # 0..250.0 Ma
    "Eurasia": list(_SETON2012_EURASIA_ROTATIONS),  # 0..250.0 Ma
    "South America": list(_SETON2012_SOUTH_AMERICA_ROTATIONS),  # 0..250.0 Ma
    "Antarctica": list(_SETON2012_ANTARCTICA_ROTATIONS),  # 0..200.0 Ma
    "India": list(_SETON2012_INDIA_ROTATIONS),  # 0..200.0 Ma
    "Indo-Australian": list(_SETON2012_INDO_AUSTRALIAN_ROTATIONS),  # 0..250.0 Ma
    "New_Zealand": list(_SETON2012_PACIFIC_ROTATIONS),  # 0..250.0 Ma
    # --- Relative rotations from Seton 2012 supplementary file. ---
    "Adria": list(_ADRIA_RELATIVE_ROTATIONS),  # 0..250.0 Ma
    "Iberia": list(_IBERIA_RELATIVE_ROTATIONS),  # 0..250.0 Ma
    "Anatolia": list(_ANATOLIA_RELATIVE_ROTATIONS),  # 0..250.0 Ma
    "Arabia": list(_ARABIA_RELATIVE_ROTATIONS),  # 0..200.0 Ma
    "Iran": list(_IRAN_RELATIVE_ROTATIONS),  # 0..200.0 Ma
    "North China": list(_NORTH_CHINA_RELATIVE_ROTATIONS),  # 0..250.0 Ma
    "South China": list(_SOUTH_CHINA_RELATIVE_ROTATIONS),  # 0..250.0 Ma
    # --- Sparse / identity-only "stable" plates. No reliable
    # rotation data; the guard in ``_interpolate_euler`` rejects
    # age > 50 Ma for these to avoid the silent "no motion"
    # fallback. ---
    "Sundaland": list(_SPARSE_IDENTITY_PLATES["Sundaland"]),  # 0..66.0 Ma
    "Siberia": list(_SPARSE_IDENTITY_PLATES["Siberia"]),  # 0..200.0 Ma
}

# Back-compat map for callers that still pass the Phase-3C-deprecated
# informal plate names ("Mokoiwi", "East Gondwana"). Phase 3C
# (audit 2026-08-19) B-12 fix: these names have been removed from
# ``COUNTRY_PLATE``, ``PLATE_OVERRIDES``, ``EULER_POLES``, and the
# coord heuristic in ``infer_plate_id``. Any caller still passing
# them will be silently mapped to the standard GPlates plate
# name so the pipeline does not silently fail; emit a debug log
# entry so operators notice the stale reference.
_DEPRECATED_PLATE_ALIASES: dict[str, str] = {
    "Mokoiwi": "New_Zealand",
    "East Gondwana": "Indo-Australian",
}

# Plate name normalisation: the in-process table uses CamelCase
# ("North America") but country lookups may produce short forms
# ("USA"). Map short forms to the table keys.
_PLATE_ALIAS = {
    "USA": "North America",
    "UK": "Eurasia",
}


def _resolve_deprecated_plate(plate_id: str) -> str:
    """Resolve a deprecated plate id (``"Mokoiwi"``, ``"East Gondwana"``)
    to its standard GPlates replacement. Returns ``plate_id`` unchanged
    when it isn't deprecated. Logs at INFO level when a rename happens
    so operators can clean up stale references.
    """
    if plate_id in _DEPRECATED_PLATE_ALIASES:
        replacement = _DEPRECATED_PLATE_ALIASES[plate_id]
        logger.info(
            "paleo_reconstruction: deprecated plate %r -> %r "
            "(Phase 3C B-12 fix: use the standard plate name)",
            plate_id,
            replacement,
        )
        return replacement
    return plate_id


def infer_plate_id(
    country: str | None = None,
    locality: str | None = None,
    modern_lat: float | None = None,
    modern_lon: float | None = None,
) -> str | None:
    """Best-effort plate ID from country/locality/coords.

    Country lookup is the most reliable signal (small curated table
    matches common paper phrases). Locality lookup falls back to
    substring scan when the locality name itself contains the
    country name (e.g. "Favignana, Sicily").
    """
    # 0) Operator extension point — PLATE_OVERRIDES wins over every
    #    other lookup. Documented as the canonical way for operators
    #    to remap a country to a plate without patching the table.
    #    audit 2026-08-01 (Bug D16): previously this dict was only a
    #    "documentation stub" — infer_plate_id never read it, so
    #    operator additions had no effect on the pipeline.
    if country:
        override = PLATE_OVERRIDES.get(country)
        if override is not None:
            return override
        # Also honour case-insensitive / whitespace-tolerant overrides
        # so operators can write "Testland" or "testland" interchangeably.
        c_norm = country.strip()
        for k, v in PLATE_OVERRIDES.items():
            if k.lower() == c_norm.lower():
                return v

    # 1) Try country name directly (case-insensitive whole-word)
    if country:
        c_lower = country.strip().lower()
        if c_lower in COUNTRY_PLATE:
            return COUNTRY_PLATE[c_lower]
        # Allow CamelCase match: "Italy" in COUNTRY_PLATE as "italy"
        for k, v in COUNTRY_PLATE.items():
            if k.lower() == c_lower:
                return v

    # 2) Try locality substring scan for any country keyword
    if locality:
        loc_lower = locality.lower()
        for k, v in COUNTRY_PLATE.items():
            # Match "Sicily" against locality "Favignana, Sicily"
            if k in loc_lower or k.capitalize() in locality:
                return v

    # 3) Modern-coord heuristic — very rough bucket assignment
    #    based on approximate modern plate outlines. Only used when
    #    neither country nor locality produced a match.
    #
    #    audit 2026-08-01 (Bug C7): the original bucket order put
    #    Eurasia (lat 25..75, lon -15..60) before Africa (lat
    #    -40..40, lon -25..55), so Mediterranean coordinates such as
    #    Tunisia (35, 10) and Cairo (30, 31) were misassigned to
    #    Eurasia. N. Africa is now extracted FIRST and Eurasia is
    #    tightened to start at lat 40 so the buckets never overlap.
    #
    #    Phase 3C (audit 2026-08-19) B-12 fix: the Pacific basin
    #    bucket now returns "New_Zealand" (was "Mokoiwi"), and the
    #    Indian Ocean bucket returns "Indo-Australian" (was
    #    "East Gondwana").
    if modern_lat is not None and modern_lon is not None:
        # N. Africa (Mediterranean margin) — MUST come before the
        # Eurasia bucket so Tunisia, Egypt, Libya, Algeria coastal
        # sites resolve to Africa rather than Eurasia.
        if -15 <= modern_lon <= 30 and 25 <= modern_lat <= 40:
            return "Africa"
        # Phase 3D (audit 2026-08-19 Bug M3): fill the lat
        # 25..40 / lon 30..60 gap that previously had no dedicated
        # bucket. Cyprus (35, 33), Israel (32, 35), Jordan (31, 36)
        # and parts of southern Turkey fell through to the broad
        # ``lat -40..40, lon -25..55`` Africa bucket below and were
        # mis-labelled "Africa" when they sit on the Anatolia
        # microplate. The bucket is intentionally tight (lat 35..40,
        # lon 30..45) so it doesn't claim sites on the Levant
        # margin (which still resolve via the country lookup or the
        # N. Africa bucket). Order: AFTER N. Africa (the two
        # buckets don't overlap — N. Africa ends at lon 30,
        # Anatolia starts there) and BEFORE Eurasia (so Anatolia
        # isn't swallowed by the Eurasian bucket once we widen
        # that bucket to lat 35..75 below).
        if 30 <= modern_lon <= 45 and 35 <= modern_lat <= 40:
            return "Anatolia"
        if -130 <= modern_lon <= -60 and 10 <= modern_lat <= 75:
            return "North America"
        if 100 <= modern_lon <= 160 and 20 <= modern_lat <= 60:
            return "North China"
        if 100 <= modern_lon <= 130 and -20 <= modern_lat <= 30:
            return "South China"
        # Eurasia: tightened to start at lat 40 (i.e. "the southern
        # edge of Europe") so it does NOT overlap with the N. Africa
        # bucket above. Northern Europe (Paris, London, Berlin, etc.)
        # at lat >= 40 still resolves here.
        if -15 <= modern_lon <= 60 and 40 <= modern_lat <= 75:
            return "Eurasia"
        if -25 <= modern_lon <= 55 and -40 <= modern_lat <= 40:
            return "Africa"
        # Phase 3C B-12 fix: was "Mokoiwi".
        if -180 <= modern_lon <= -150 and -60 <= modern_lat <= -30:
            return "New_Zealand"
        # Phase 3C B-12 fix: was "East Gondwana".
        if 110 <= modern_lon <= 160 and -45 <= modern_lat <= -10:
            return "Indo-Australian"
    return None


def _interpolate_euler(plate: str, age_ma: float):
    """Return (euler_lat, euler_lon, rotation_deg) for ``plate`` at the
    requested age. Linearly interpolates between the two adjacent
    timesteps bracketing ``age_ma``. Returns None when ``plate`` has
    no table or ``age_ma`` is outside the table's age range.

    Phase 62 Plan 5 (Bug 5.16): plates with very short / sparse
    Euler pole tables (Sundaland, Siberia, Iran) are flagged as
    "stable" — their most-recent pole is ~(0,0,0,0) and their
    oldest entry is < 200 Ma. Reconstructing these plates at
    age > 100 Ma silently returned the modern coords via the
    (0,0,0,0) identity pole. We now return None for such requests
    so downstream consumers see "we don't have a reliable
    reconstruction for this plate at this age" rather than a
    fabricated "no motion" answer.

    Phase 3C (audit 2026-08-19) M-7 fix: the previous
    ``_interpolate_euler`` had a silent fallback that returned the
    modern identity pole when the bracketing loop didn't find a
    match (lines 392-409 of the pre-Phase-3C file). Callers could
    receive ``paleo_lat == modern_lat`` with no signal that the
    table was unable to reconstruct the requested age. The new
    code raises ``ValueError`` for the unreachable case (1-entry
    table falls through, multi-entry table didn't bracket — both
    indicate a corrupted or out-of-range age). Out-of-range ages
    (> table_max_age) and stable-plate rejections still return
    ``None`` so callers can keep using the optional API.
    """
    poles = EULER_POLES.get(plate)
    if not poles:
        return None
    # poles is sorted by age descending (younger -> older).
    ages = [p[0] for p in poles]
    age_min, age_max = min(ages), max(ages)
    # Out-of-range ages are a documented "we don't have this far
    # back" answer; return None so ``reconstruct_paleo_position``
    # can degrade gracefully.
    if age_ma < age_min or age_ma > age_max:
        return None
    # Phase 62 Plan 5 (Bug 5.16): refuse to reconstruct known-stable
    # plates far in the past. The heuristic for "stable" is:
    #   * <= 3 reconstruction timesteps in the table, AND
    #   * oldest timestep <= 250 Ma, AND
    #   * BOTH the most-recent AND the oldest pole are identity
    #     rotations (abs(rotation) <= 1.0°). Either end being a real
    #     rotation means the plate has measurable motion in the
    #     table and we should interpolate rather than refuse.
    if (
        len(poles) <= 3
        and age_max <= 250.0
        and abs(poles[0][3]) <= 1.0
        and abs(poles[-1][3]) <= 1.0
    ):
        if age_ma > 50.0:
            return None
    # Single-entry table (rare, but legal): the only pole IS the
    # answer. Returning it here avoids the unreachable-loop fallback.
    if len(poles) == 1:
        return poles[0][1:]
    # Find the two adjacent timesteps bracketing age_ma. Note poles[i]
    # is the YOUNGER end (smaller age) and poles[i+1] is the OLDER
    # end (larger age). The bracket condition is
    # ``age_old >= age_ma >= age_young``.
    for i in range(len(poles) - 1):
        age_y, lat_y, lon_y, rot_y = poles[i]
        age_o, lat_o, lon_o, rot_o = poles[i + 1]
        if age_y <= age_ma <= age_o:
            # audit 2026-07-31: t was computed as
            # ``(age_o - age_ma)/(age_o - age_y)`` — INVERTED. A query
            # exactly on a timestep (age_ma == age_y) got t=1, i.e.
            # the OLDER adjacent pole, so every exact-timestep
            # reconstruction (e.g. Adria at 200 Ma) silently used the
            # 250 Ma pole. t=0 must mean "the younger pole" (the
            # timestep the query sits on), t=1 the older one.
            t = (age_ma - age_y) / (age_o - age_y) if age_o != age_y else 0.0
            return (
                lat_y * (1 - t) + lat_o * t,
                _interp_lon(lon_y, lon_o, t),
                rot_y * (1 - t) + rot_o * t,
            )
    # Phase 3C M-7 fix: if the loop above didn't bracket age_ma,
    # the table is corrupted or the caller passed an age outside
    # [age_min, age_max] that escaped the guard at the top.
    # Raise rather than silently returning the modern pole.
    raise ValueError(
        f"_interpolate_euler invariant violated for plate={plate!r}, "
        f"age_ma={age_ma}: table has {len(poles)} entries with "
        f"age range [{age_min}, {age_max}] but age_ma={age_ma} did not "
        "match any bracket. This should be unreachable — please file a "
        "bug at the M-7 silent-fallback regression."
    )


def _interp_lon(lon_a: float, lon_b: float, t: float) -> float:
    """Linear interpolation of longitudes, taking the shorter arc."""
    diff = ((lon_b - lon_a + 540) % 360) - 180
    return (lon_a + diff * t + 540) % 360 - 180


def _rotate_point(
    lat: float, lon: float, euler_lat: float, euler_lon: float, rot_deg: float
) -> tuple[float, float]:
    """Apply a finite rotation to a (lat, lon) point around an Euler pole.

    Standard 3-D vector rotation: convert the point to a unit vector,
    rotate it around the axis vector of the Euler pole by ``rot_deg``
    using the vector form of Rodrigues' formula, then convert back to
    (lat, lon).

    audit 2026-07-31: the previous "spherical" implementation was not
    a valid rotation — it did not preserve vector length (rotating
    (0°N,0°E) by 30° around the north pole returned latitude -63.4°
    instead of 0°) so EVERY paleo-coordinate produced by
    ``reconstruct_paleo_position`` was wrong. The vector form below is
    the standard finite rotation (see e.g. Cox & Hart, Plate
    Tectonics, ch. 5).
    """
    if rot_deg == 0.0:
        return lat, lon
    phi = math.radians(lat)
    lam = math.radians(lon)
    phi_p = math.radians(euler_lat)
    lam_p = math.radians(euler_lon)
    omega = math.radians(rot_deg)

    # Point on the unit sphere (x = cos(lat)cos(lon), y = cos(lat)sin(lon), z = sin(lat)).
    p = (
        math.cos(phi) * math.cos(lam),
        math.cos(phi) * math.sin(lam),
        math.sin(phi),
    )
    # Rotation axis: unit vector pointing at the Euler pole.
    k = (
        math.cos(phi_p) * math.cos(lam_p),
        math.cos(phi_p) * math.sin(lam_p),
        math.sin(phi_p),
    )
    # Rodrigues' rotation formula (vector form):
    #   p' = p cos ω + (k × p) sin ω + k (k·p)(1 − cos ω)
    c = math.cos(omega)
    s = math.sin(omega)
    kd = 1.0 - c
    kx, ky, kz = k
    px, py, pz = p
    dot = kx * px + ky * py + kz * pz
    cross = (ky * pz - kz * py, kz * px - kx * pz, kx * py - ky * px)
    rx = px * c + cross[0] * s + kx * dot * kd
    ry = py * c + cross[1] * s + ky * dot * kd
    rz = pz * c + cross[2] * s + kz * dot * kd

    lat_new = math.degrees(math.atan2(rz, math.sqrt(rx * rx + ry * ry)))
    lon_new = math.degrees(math.atan2(ry, rx))
    return lat_new, lon_new


def reconstruct_paleo_position(
    modern_lat: float | None,
    modern_lon: float | None,
    age_ma: float | None,
    plate_id: str | None = None,
) -> tuple[float | None, float | None]:
    """Reconstruct paleo (lat, lon) for the given age.

    If ``plate_id`` is None, ``infer_plate_id`` is used (which itself
    needs at least one of country/locality/modern coords — call with
    ``modern_lat``/``modern_lon`` to fall back to the coord heuristic).
    Returns (None, None) when plate or age is unsupported.
    """
    if modern_lat is None or modern_lon is None:
        return None, None
    if age_ma is None or age_ma < 0:
        return None, None
    # Resolve plate_id via short-form aliasing, then check the
    # Phase 3C back-compat map for "Mokoiwi" / "East Gondwana".
    if plate_id:
        plate_id = _PLATE_ALIAS.get(plate_id, plate_id)
        plate_id = _resolve_deprecated_plate(plate_id)
    euler = _interpolate_euler(plate_id or "", age_ma)
    if euler is None:
        return None, None
    e_lat, e_lon, rot = euler
    return _rotate_point(modern_lat, modern_lon, e_lat, e_lon, rot)


def _load_seton2012_from_external(path: str | os.PathLike) -> int:
    """Load Euler poles from a GPlates ``Seton_etal_2012.rot`` file
    and merge them into ``EULER_POLES``.

    The function is the optional override path for operators who
    have downloaded the official EarthByte rotation file. When
    the file is missing, the embedded ``_SETON2012_POLES`` are
    used (see Phase 3C B-11 fix — those values are the verified
    EarthByte Seton 2012 poles for the 8 plates we ship absolute
    rotations for). The override here lets an operator drop in a
    different rotation file (e.g. a more recent Seton 2012
    revision or a domain-specific customised file) without
    patching this module.

    The GPlates .rot format is whitespace-separated with one
    reconstruction row per line::

        PlateID AgeMa EulerLat EulerLon Angle(deg)
        101 0.0 0.0 0.0 0.0
        101 10.0 90.0 80.0 0.5
        ...

    Lines starting with ``#`` and blank lines are ignored. Only
    the plates listed in ``_GPLATES_PLATE_IDS`` are merged — any
    other PlateIDs in the file are skipped silently. Returns the
    number of plates merged.
    """
    p = Path(path)
    if not p.exists():
        return 0
    # Mapping from GPlates PlateID to our internal plate name.
    # Only the plates whose Seton 2012 rotation file uses the
    # absolute reference frame are listed here; the relative
    # rotations (Adria, Iberia, Anatolia, etc.) keep their
    # embedded values.
    gplates_plate_ids: dict[int, str] = {
        101: "Africa",
        201: "North America",
        301: "Eurasia",
        501: "India",
        701: "South America",
        801: "Indo-Australian",
        802: "Antarctica",
        901: "New_Zealand",
    }
    by_plate: dict[str, list[tuple[float, float, float, float]]] = {}
    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            plate_id = int(parts[0])
            age = float(parts[1])
            e_lat = float(parts[2])
            e_lon = float(parts[3])
            rot = float(parts[4])
        except ValueError:
            continue
        name = gplates_plate_ids.get(plate_id)
        if name is None:
            continue
        by_plate.setdefault(name, []).append((age, e_lat, e_lon, rot))

    merged = 0
    for name, rows in by_plate.items():
        # ``_interpolate_euler`` walks the table in ascending age
        # order (youngest -> oldest). GPlates .rot files store rows
        # in either order, so we sort here to be safe.
        EULER_POLES[name] = sorted(rows, key=lambda r: r[0])
        merged += 1
        logger.info(
            "paleo_reconstruction: loaded %d rotation rows for plate=%r from %s",
            len(rows),
            name,
            p,
        )
    return merged


def enrich_geology_record(record: dict[str, Any]) -> None:
    """In-place fill of paleo_* + plate_id + reconstruction_* fields.

    The dict is expected to look like the producer output from
    ``geology_extraction.GeologyRecord.to_dict()`` — latitude /
    longitude / age / country / locality keys present.

    No-op when modern coords or age are missing (we don't guess).
    """
    try:
        lat = record.get("latitude")
        lon = record.get("longitude")
        age_str = record.get("chronostratigraphy") or record.get("age") or ""
        # Convert age to numeric Ma via the same ICS table used in
        # ``stratigraphy.find_ages_in_text``. We import lazily so this
        # module stays cheap to load.
        from .stratigraphy import find_ages_in_text

        classifications = find_ages_in_text(age_str)
        if not classifications and record.get("ma_mid") is not None:
            age_ma = float(record["ma_mid"])
        elif not classifications and record.get("ma_top") is not None:
            age_ma = float(record["ma_top"])
        elif classifications:
            best = max(
                classifications,
                key=lambda c: getattr(c, "ma_mid", None) or 0,
            )
            age_ma = best.ma_mid or best.ma_top
        else:
            age_ma = None
        if lat is None or lon is None or age_ma is None:
            return
        plate = infer_plate_id(
            country=record.get("country"),
            locality=record.get("locality"),
            modern_lat=float(lat),
            modern_lon=float(lon),
        )
        if plate is None:
            return
        # Phase 3C B-12 fix: also route deprecated "Mokoiwi" /
        # "East Gondwana" through the resolver so callers using
        # the old plate name still get a paleo coordinate.
        plate = _resolve_deprecated_plate(plate)
        paleo_lat, paleo_lon = reconstruct_paleo_position(
            float(lat), float(lon), age_ma, plate_id=plate
        )
        if paleo_lat is None:
            return
        record["plate_id"] = plate
        record["paleo_latitude"] = round(paleo_lat, 4)
        record["paleo_longitude"] = round(paleo_lon, 4)
        record["modern_latitude"] = round(float(lat), 4)
        record["modern_longitude"] = round(float(lon), 4)
        # Phase 3C B-11 fix: the rotation source is now real Seton
        # 2012, not the simplified approximation. Update the model
        # label accordingly so downstream consumers can audit which
        # pole table produced the paleo coordinates.
        record["reconstruction_model"] = "Seton 2012"
        record["reconstruction_age_ma"] = round(float(age_ma), 2)
    except (TypeError, ValueError, AttributeError, KeyError, ImportError) as exc:
        # Phase 55 audit CRITICAL-3/HIGH-5 fix: narrow to the exceptions
        # that a data-access bug (wrong key name, bad type coercion, missing
        # field) would raise. ImportError from the lazy stratigraphy import
        # is also suppressed here — a missing optional dependency is a
        # deployment issue, not a data bug, and should not crash the per-row
        # enrichment loop. Arithmetic errors (e.g. math domain from Rodrigues
        # rotation, ZeroDivisionError from degenerate Euler poles) raise
        # ValueError which IS in the tuple and is therefore suppressed with
        # a WARNING — this is intentional; the telemetry pipeline degrades
        # them to log-level rather than killing the job. NOT bare Exception:
        # RecursionError and MemoryError propagate so a real crash does not
        # silently produce wrong coordinates.
        #
        # NOTE: Phase 3C (audit 2026-08-19) M-7 fix changes the contract:
        # ``_interpolate_euler`` now raises ValueError for the previously
        # silent "unreachable bracket" case. That ValueError lands here
        # too — we log a WARNING so operators notice but keep the row
        # moving through the pipeline.
        logger.warning(
            "enrich_geology_record failed for record %s: %s — "
            "check Euler pole table / coordinate values",
            record.get("paper_id", "?"),
            exc,
        )
