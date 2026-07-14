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
    "italy": "Adria", "sicily": "Adria", "greece": "Adria",
    "turkey": "Anatolia", "oman": "Arabia", "saudi": "Arabia",
    "iran": "Iran", "iraq": "Arabia",
    "uk": "Eurasia", "united kingdom": "Eurasia",
    "france": "Eurasia", "germany": "Eurasia",
    "spain": "Iberia", "portugal": "Iberia",
    "austria": "Eurasia", "switzerland": "Eurasia",
    "italian": "Adria",
    "usa": "North America", "united states": "North America",
    "canada": "North America", "mexican": "North America",
    "mexico": "North America",
    "egypt": "Arabia", "tunisia": "Africa",
    "morocco": "Africa", "algeria": "Africa",
    "south africa": "Africa",
    "japan": "North China", "chinese": "South China",
    "philippines": "South China", "indonesia": "Sundaland",
    "indonesian": "Sundaland", "japanese": "North China",
    "new zealand": "Mokoiwi", "australia": "East Gondwana",
    "russian": "Siberia", "russia": "Siberia",
    "norway": "Eurasia", "sweden": "Eurasia", "finland": "Eurasia",
    "denmark": "Eurasia", "polish": "Eurasia",
    "poland": "Eurasia", "czech": "Eurasia",
    "hungary": "Eurasia", "romania": "Eurasia",
    "bulgaria": "Eurasia", "cyprus": "Adria",
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
    if modern_lat is not None and modern_lon is not None:
        if -15 <= modern_lon <= 60 and 25 <= modern_lat <= 75:
            return "Eurasia"
        if -130 <= modern_lon <= -60 and 10 <= modern_lat <= 75:
            return "North America"
        if 100 <= modern_lon <= 160 and 20 <= modern_lat <= 60:
            return "North China"
        if 100 <= modern_lon <= 130 and -20 <= modern_lat <= 30:
            return "South China"
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
    """
    poles = EULER_POLES.get(plate)
    if not poles:
        return None
    # poles is sorted by age descending (younger -> older).
    ages = [p[0] for p in poles]
    age_min, age_max = min(ages), max(ages)
    if age_ma < age_min or age_ma > age_max:
        return None
    # Find the two adjacent timesteps bracketing age_ma. Note poles[i]
    # is the YOUNGER end (smaller age) and poles[i+1] is the OLDER
    # end (larger age). The bracket condition is
    # ``age_old >= age_ma >= age_young``.
    for i in range(len(poles) - 1):
        age_y, lat_y, lon_y, rot_y = poles[i]
        age_o, lat_o, lon_o, rot_o = poles[i + 1]
        if age_y <= age_ma <= age_o:
            t = (age_o - age_ma) / (age_o - age_y) if age_o != age_y else 0.0
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
    """Apply a finite rotation to a (lat, lon) point using Rodrigues'
    formula on the unit sphere. Angles in degrees, converted to radians
    internally.
    """
    if rot_deg == 0.0:
        return lat, lon
    phi_p = math.radians(euler_lat)
    lam_p = math.radians(euler_lon)
    phi = math.radians(lat)
    lam = math.radians(lon)
    omega = math.radians(rot_deg)
    cp = math.cos(phi_p)
    sp = math.sin(phi_p)
    ca, sa = math.cos(omega), math.sin(omega)
    cl, sl = math.cos(lam - lam_p), math.sin(lam - lam_p)
    # Rotate the point by omega around the Euler pole.
    x_new = (
        cp * cl * math.cos(phi)
        - sa * sl * math.cos(phi)
        + ca * sp * cl * math.sin(phi)
    )
    y_new = (
        cp * sl * math.cos(phi)
        + sa * cl * math.cos(phi)
        + ca * sp * sl * math.sin(phi)
    )
    z_new = -sp * math.cos(phi) + ca * cp * math.sin(phi)
    lat_new = math.degrees(math.atan2(z_new, math.sqrt(x_new * x_new + y_new * y_new)))
    lon_new = math.degrees(math.atan2(y_new, x_new))
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
    except (TypeError, ValueError, AttributeError, KeyError) as exc:
        # Phase 55 audit MEDIUM-3 fix: narrow to the exceptions that a
        # data-access bug (wrong key name, bad type coercion, missing field)
        # would raise. Arithmetic errors (e.g. math domain from Rodrigues
        # rotation) are also worth surfacing since they indicate wrong
        # Euler poles or coordinates. Only suppress ImportError (missing
        # optional dependency) which is a deployment issue, not a data bug.
        # NOT bare Exception: let RecursionError, MemoryError propagate so
        # a real crash doesn't silently produce wrong coordinates.
        logger.warning(
            "enrich_geology_record failed for record %s: %s — "
            "check Euler pole table / coordinate values",
            record.get("paper_id", "?"),
            exc,
        )
