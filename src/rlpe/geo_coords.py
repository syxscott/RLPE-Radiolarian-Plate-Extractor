"""Coordinate parsing for locality strings in paleontology papers.

Supports the common coordinate formats used in geological / paleontological
literature:

1. Decimal degrees with hemisphere:
     "35.7°N, 110.3°E"   "35.7N 110.3E"   "35.7, -110.3"
2. Degrees-minutes-seconds (DMS):
     "35°42'12\"N 110°18'00\"E"
3. Bracketed tuple form (used in some Asian papers):
     "(35.7 N, 110.3 E)"

Returned ``Coordinate`` has both signed decimal degrees and a normalised
representation, so downstream consumers can pick whichever they need.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class Coordinate:
    latitude: float
    longitude: float
    source: str = ""        # the snippet where it was found
    raw: str = ""           # the original match
    confidence: float = 0.9

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Decimal degrees with optional hemisphere letter
_DECIMAL_RE = re.compile(
    r"""
    (?P<lat>\d{1,3}(?:\.\d+)?)\s*°?\s*(?P<lat_h>[NSEWnsew])?
    \s*[,;\s]\s*
    (?P<lon>-?\d{1,3}(?:\.\d+)?)\s*°?\s*(?P<lon_h>[NSEWnsew])?
    """,
    re.VERBOSE,
)

# DMS form: 35°42'12"N   110°18'00"E
_DMS_RE = re.compile(
    r"""
    (?P<lat_d>\d{1,3})[°]\s*(?P<lat_m>\d{1,2})['′]\s*(?P<lat_s>\d{1,2}(?:\.\d+)?)["″]?\s*(?P<lat_h>[NSEWnsew])?
    \s*[,;\s]\s*
    (?P<lon_d>\d{1,3})[°]\s*(?P<lon_m>\d{1,2})['′]\s*(?P<lon_s>\d{1,2}(?:\.\d+)?)["″]?\s*(?P<lon_h>[NSEWnsew])?
    """,
    re.VERBOSE,
)


def parse_coordinate(text: str) -> Coordinate | None:
    """Return the first parseable coordinate in ``text`` or ``None``.

    Tries DMS first (more specific) then decimal-degrees.
    """
    if not text:
        return None
    # 1. DMS
    m = _DMS_RE.search(text)
    if m:
        try:
            lat_d = int(m.group("lat_d"))
            lat_m = int(m.group("lat_m"))
            lat_s = float(m.group("lat_s"))
            lat = lat_d + lat_m / 60.0 + lat_s / 3600.0
            if m.group("lat_h") and m.group("lat_h").upper() in ("S", "W"):
                lat = -lat

            lon_d = int(m.group("lon_d"))
            lon_m = int(m.group("lon_m"))
            lon_s = float(m.group("lon_s"))
            lon = lon_d + lon_m / 60.0 + lon_s / 3600.0
            if m.group("lon_h") and m.group("lon_h").upper() in ("W", "S"):
                lon = -lon
            if _valid(lat, lon):
                return Coordinate(latitude=lat, longitude=lon, source=text[:200], raw=m.group(0))
        except Exception:
            pass
    # 2. Decimal
    m = _DECIMAL_RE.search(text)
    if m:
        try:
            lat = float(m.group("lat"))
            lon = float(m.group("lon"))
            if m.group("lat_h") and m.group("lat_h").upper() in ("S", "W"):
                lat = -lat
            if m.group("lon_h") and m.group("lon_h").upper() in ("W", "S"):
                lon = -lon
            if _valid(lat, lon):
                return Coordinate(latitude=lat, longitude=lon, source=text[:200], raw=m.group(0))
        except Exception:
            pass
    return None


def parse_all_coordinates(text: str) -> list[Coordinate]:
    """Return every parseable coordinate in ``text``."""
    if not text:
        return []
    out: list[Coordinate] = []
    for m in _DMS_RE.finditer(text):
        try:
            lat = int(m.group("lat_d")) + int(m.group("lat_m")) / 60.0 + float(m.group("lat_s")) / 3600.0
            lon = int(m.group("lon_d")) + int(m.group("lon_m")) / 60.0 + float(m.group("lon_s")) / 3600.0
            if m.group("lat_h") and m.group("lat_h").upper() in ("S", "W"):
                lat = -lat
            if m.group("lon_h") and m.group("lon_h").upper() in ("W", "S"):
                lon = -lon
            if _valid(lat, lon):
                out.append(Coordinate(latitude=lat, longitude=lon, source=text[:200], raw=m.group(0)))
        except Exception:
            pass
    for m in _DECIMAL_RE.finditer(text):
        try:
            lat = float(m.group("lat"))
            lon = float(m.group("lon"))
            if m.group("lat_h") and m.group("lat_h").upper() in ("S", "W"):
                lat = -lat
            if m.group("lon_h") and m.group("lon_h").upper() in ("W", "S"):
                lon = -lon
            if _valid(lat, lon):
                out.append(Coordinate(latitude=lat, longitude=lon, source=text[:200], raw=m.group(0)))
        except Exception:
            pass
    return out


def _valid(lat: float, lon: float) -> bool:
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0
