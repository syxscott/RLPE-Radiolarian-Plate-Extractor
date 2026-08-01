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
    "New Zealand": "Mokoiwi",
    "Australia": "East Gondwana",
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
    "new zealand": "Mokoiwi",
    "australia": "East Gondwana",
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

EULER_POLES: dict[str, list[tuple[float, float, float, float]]] = {
    # Adria (Apulian promontory of Africa) - stayed at ~30°N across
    # most of the Mesozoic with minor counter-clockwise rotation.
    "Adria": [
        (0.0, 0.0, 0.0, 0.0),
        (66.0, 41.0, 22.0, -5.0),
        (130.0, 38.0, 23.0, -8.0),
        (200.0, 35.0, 18.0, -10.0),
        (250.0, 32.0, 15.0, -12.0),
    ],
    # Iberia - rotated ~35° counter-clockwise since the Cretaceous
    "Iberia": [
        (0.0, 0.0, 0.0, 0.0),
        (66.0, 41.0, 162.0, -20.0),
        (130.0, 40.0, 168.0, -28.0),
        (200.0, 42.0, 175.0, -35.0),
        (250.0, 45.0, 180.0, -38.0),
    ],
    # Eurasia - mostly stable
    "Eurasia": [
        (0.0, 0.0, 0.0, 0.0),
        (66.0, 80.0, 30.0, 2.0),
        (200.0, 80.0, 25.0, 3.0),
        (250.0, 78.0, 22.0, 4.0),
    ],
    # North China - rotated rapidly during the Cretaceous
    "North China": [
        (0.0, 0.0, 0.0, 0.0),
        (66.0, 50.0, 130.0, -10.0),
        (130.0, 45.0, 140.0, -25.0),
        (200.0, 55.0, 150.0, -30.0),
        (250.0, 55.0, 155.0, -32.0),
    ],
    # South China - rotated ~30° counter-clockwise since the Triassic
    "South China": [
        (0.0, 0.0, 0.0, 0.0),
        (66.0, 35.0, 120.0, -8.0),
        (130.0, 30.0, 115.0, -18.0),
        (200.0, 32.0, 110.0, -22.0),
        (250.0, 30.0, 105.0, -25.0),
    ],
    # Africa - slow drift northward over the Mesozoic
    "Africa": [
        (0.0, 0.0, 0.0, 0.0),
        (66.0, 30.0, -15.0, 8.0),
        (130.0, 25.0, -15.0, 12.0),
        (200.0, 25.0, -15.0, 15.0),
        (250.0, 25.0, -12.0, 18.0),
    ],
    # North America - very slow drift; ignored for most purposes
    "North America": [
        (0.0, 0.0, 0.0, 0.0),
        (200.0, 80.0, -80.0, 2.0),
        (250.0, 80.0, -80.0, 3.0),
    ],
    # Anatolia - moved into Eurasia ~13 Ma, before that was a separate plate
    "Anatolia": [
        (0.0, 0.0, 0.0, 0.0),
        (13.0, 50.0, 35.0, -10.0),
        (66.0, 45.0, 38.0, -15.0),
        (200.0, 45.0, 40.0, -25.0),
        (250.0, 45.0, 40.0, -28.0),
    ],
    # Arabia - separated from Africa ~30 Ma, hit Eurasia ~13 Ma
    "Arabia": [
        (0.0, 0.0, 0.0, 0.0),
        (13.0, 30.0, 35.0, 5.0),
        (30.0, 25.0, 35.0, -5.0),
        (66.0, 20.0, 30.0, -10.0),
    ],
    "Iran": [
        (0.0, 0.0, 0.0, 0.0),
        (66.0, 50.0, 60.0, 8.0),
    ],
    "Siberia": [
        (0.0, 0.0, 0.0, 0.0),
        (200.0, 0.0, 0.0, 0.0),
    ],
    "Sundaland": [
        (0.0, 0.0, 0.0, 0.0),
        (66.0, 0.0, 0.0, 0.0),
    ],
    "East Gondwana": [
        (0.0, 0.0, 0.0, 0.0),
        (66.0, 0.0, 0.0, 0.0),
    ],
    "Mokoiwi": [
        (0.0, 0.0, 0.0, 0.0),
        (66.0, 0.0, 0.0, 0.0),
    ],
}

# Plate name normalisation: the in-process table uses CamelCase
# ("North America") but country lookups may produce short forms
# ("USA"). Map short forms to the table keys.
_PLATE_ALIAS = {
    "USA": "North America",
    "UK": "Eurasia",
}


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
    if modern_lat is not None and modern_lon is not None:
        # N. Africa (Mediterranean margin) — MUST come before the
        # Eurasia bucket so Tunisia, Egypt, Libya, Algeria coastal
        # sites resolve to Africa rather than Eurasia.
        if -15 <= modern_lon <= 30 and 25 <= modern_lat <= 40:
            return "Africa"
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
        if -180 <= modern_lon <= -150 and -60 <= modern_lat <= -30:
            return "Mokoiwi"
        if 110 <= modern_lon <= 160 and -45 <= modern_lat <= -10:
            return "East Gondwana"
    return None


def _interpolate_euler(plate: str, age_ma: float):
    """Return (euler_lat, euler_lon, rotation_deg) for ``plate`` at the
    requested age. Linearly interpolates between the two adjacent
    timesteps bracketing ``age_ma``. Returns None when ``plate`` has
    no table or ``age_ma`` is outside the table's age range.

    Phase 62 Plan 5 (Bug 5.16): plates with very short / sparse
    Euler pole tables (Sundaland, East Gondwana, Mokoiwi, Siberia,
    Iran) are flagged as "stable" — their most-recent pole is
    ~(0,0,0,0) and their oldest entry is < 200 Ma. Reconstructing
    these plates at age > 100 Ma silently returned the modern
    coords via the (0,0,0,0) identity pole. We now return None
    for such requests so downstream consumers see "we don't have
    a reliable reconstruction for this plate at this age" rather
    than a fabricated "no motion" answer.
    """
    poles = EULER_POLES.get(plate)
    if not poles:
        return None
    # poles is sorted by age descending (younger -> older).
    ages = [p[0] for p in poles]
    age_min, age_max = min(ages), max(ages)
    if age_ma < age_min or age_ma > age_max:
        return None
    # Phase 62 Plan 5 (Bug 5.16): refuse to reconstruct known-stable
    # plates far in the past. The heuristic for "stable" is:
    #   * <= 3 reconstruction timesteps in the table, AND
    #   * oldest timestep <= 250 Ma (relaxed from 100 Ma by audit
    #     2026-08-01 Bug C8 — the Siberia pole table has age_max=200
    #     so the original <=100 Ma guard never triggered, silently
    #     returning the modern identity labelled "paleo"), AND
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
    return poles[0][1:]


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
    # Resolve plate_id via short-form aliasing
    if plate_id:
        plate_id = _PLATE_ALIAS.get(plate_id, plate_id)
    euler = _interpolate_euler(plate_id or "", age_ma)
    if euler is None:
        return None, None
    e_lat, e_lon, rot = euler
    return _rotate_point(modern_lat, modern_lon, e_lat, e_lon, rot)


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
        record["reconstruction_model"] = "Seton 2012 (simplified)"
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
        logger.warning(
            "enrich_geology_record failed for record %s: %s — "
            "check Euler pole table / coordinate values",
            record.get("paper_id", "?"),
            exc,
        )
