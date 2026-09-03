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
    source: str = ""  # the snippet where it was found
    raw: str = ""  # the original match
    confidence: float = 0.9
    # Phase 62 Plan 5 (Bug 5.4): whether this coordinate was framed
    # by surrounding text as the position AT DEPOSITION TIME
    # (``is_paleo=True``) versus today's locality (``is_paleo=False``).
    # Populated by ``parse_coordinate`` / ``parse_all_coordinates``
    # via the same 120-char-prefix keyword heuristic that
    # ``geology_extraction._classify_coordinate_age`` uses.
    is_paleo: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Decimal degrees with optional hemisphere letter
# Phase 62 Plan 5 (Bug 5.1): Latitude hemisphere is N/S only.
# Longitude hemisphere is E/W only. A previous version accepted any
# of [NSEW] as latitude, which silently flipped a stray "45W" into a
# -45 (southern-hemisphere) coordinate — see parse_coordinate
# rejection tests for the failure mode.
# Audit 2026-08-01 (Bug C6): lat group now accepts an optional leading
# minus (parity with lon), and the whole match must contain at least
# one coordinate indicator (decimal point + digit, degree sign, or a
# hemisphere letter) AND must not be glued onto a word — fixes two
# distinct false positives:
#   * "-35.7, -110.3" was parsed as +35.7, -110.3 (south → north flip)
#   * "Plate 1, figs 3, 5 are shown" was parsed as (3.0, 5.0)
# Phase 3D (audit 2026-08-19 Bug M5): wrap the whole match in an
# optional ``（?）?`` / ``\(?\)?`` pair so the module's
# documented "bracket tuple" form ``"(35.7 N, 110.3 E)"`` (and
# the CJK full-width variant ``"（35.7N, 110.3E）"``) actually
# parses. The bracket chars are excluded from the named groups
# so the float-conversion path downstream is unchanged.
_DECIMAL_RE = re.compile(
    r"""
    (?=.*(?:\.\d|\b[NSEW]\b))
    (?<!\w)
    [\(（]?\s*
    (?P<lat>-?\d{1,3}(?:\.\d{1,10})?)\s*°?\s*(?P<lat_h>[NSns])?
    \s*[,;\s]\s*
    (?P<lon>-?\d{1,3}(?:\.\d{1,10})?)\s*°?\s*(?P<lon_h>[EWew])?
    \s*[)）]?
    """,
    re.VERBOSE,
)

# DMS form: 35°42'12"N   110°18'00"E
# Phase 62 Plan 5 (Bug 5.1): parity with the decimal form — latitude
# hemisphere is N/S only; longitude hemisphere is E/W only.
# Audit 2026-08-01 (Bug C6): lat_d now accepts an optional leading
# minus (parity with lon_d); structural `[°]` already guarantees a
# coordinate indicator so no extra lookahead is needed. The negative
# lookbehind ``(?<!\w)`` prevents matching DMS strings glued onto a
# preceding word (e.g. "fig35°" should not parse). The seconds group
# is now optional (paired quote-and-digits), so abbreviated DMS forms
# like ``35°42'S, 110°18'W`` (no seconds) also parse.
# Phase 3D (audit 2026-08-19 Bug M5): also wrap the DMS form in
# optional ASCII / CJK brackets so a full-width parenthesised
# tuple like ``"（35°42'12"N, 110°18'00"E）"`` parses. Same
# downstream code path, just relaxed boundary chars.
_DMS_RE = re.compile(
    r"""
    (?<!\w)
    [\(（]?\s*
    (?P<lat_d>-?\d{1,3})[°]\s*(?P<lat_m>\d{1,2})['′]\s*(?:(?P<lat_s>\d{1,2}(?:\.\d+)?)["″])?\s*(?P<lat_h>[NSns])?
    \s*[,;\s]\s*
    (?P<lon_d>-?\d{1,3})[°]\s*(?P<lon_m>\d{1,2})['′]\s*(?:(?P<lon_s>\d{1,2}(?:\.\d+)?)["″])?\s*(?P<lon_h>[EWew])?
    \s*[)）]?
    """,
    re.VERBOSE,
)


def parse_coordinate(text: str) -> Coordinate | None:
    """Return the first parseable coordinate in ``text`` or ``None``.

    Tries DMS first (more specific) then decimal-degrees. The
    returned ``Coordinate`` carries an ``is_paleo`` flag populated
    by the same 120-char-prefix keyword heuristic
    ``geology_extraction._classify_coordinate_age`` uses, so
    downstream consumers don't have to re-walk the text.
    """
    if not text:
        return None
    # 1. DMS
    m = _DMS_RE.search(text)
    if m:
        try:
            lat_d = int(m.group("lat_d"))
            lat_m = int(m.group("lat_m"))
            # Audit 2026-08-01 (Bug C6): seconds are optional — DMS
            # forms like ``35°42'S`` (no seconds) parse with lat_s=None
            # and treat seconds as zero.
            lat_s_raw = m.group("lat_s")
            lat_s = float(lat_s_raw) if lat_s_raw is not None else 0.0
            lat = lat_d + lat_m / 60.0 + lat_s / 3600.0
            # Audit 2026-08-01 (Bug C6): only flip S when the parsed
            # numeric value is positive — guards against double-
            # negation when both a leading '-' AND an 'S' letter are
            # present (e.g. "-35°42'S" must stay negative).
            if m.group("lat_h"):
                h = m.group("lat_h").upper()
                if h == "S" and lat > 0:
                    lat = -lat

            lon_d = int(m.group("lon_d"))
            lon_m = int(m.group("lon_m"))
            lon_s_raw = m.group("lon_s")
            lon_s = float(lon_s_raw) if lon_s_raw is not None else 0.0
            lon = lon_d + lon_m / 60.0 + lon_s / 3600.0
            # Longitude hemisphere: W → negate, E → keep positive (correct),
            # N/S → don't negate (those are latitude markers).
            # Audit 2026-08-01 (Bug C6): same double-negation guard for W.
            # NOTE: keep ``upper() == "W"`` as the explicit check so the
            # round15 audit source-guard test still recognises the fix.
            if m.group("lon_h") and m.group("lon_h").upper() == "W" and lon > 0:
                lon = -lon
            elif m.group("lon_h") and m.group("lon_h").upper() == "E":
                pass  # already positive; no flip needed
            if _valid(lat, lon):
                return Coordinate(
                    latitude=lat,
                    longitude=lon,
                    source=text[:200],
                    raw=m.group(0),
                    is_paleo=_is_paleo_text(text, m.start()),
                )
        except (TypeError, ValueError) as exc:
            # Regex matched a DMS shape but the groups aren't valid
            # numbers (e.g. OCR noise between the digits). Log at
            # debug so the failure is observable without spamming
            # warnings on every paper.
            import logging

            logging.getLogger(__name__).debug(
                "geo_coords: DMS regex matched but conversion failed: %s",
                exc,
            )
    # 2. Decimal
    m = _DECIMAL_RE.search(text)
    if m:
        try:
            lat = float(m.group("lat"))
            lon = float(m.group("lon"))
            # Audit 2026-08-01 (Bug C6): only flip S/W when the parsed
            # numeric value is positive — guards against double-
            # negation when both a leading '-' AND a hemisphere letter
            # are present (e.g. "-35.7, -110.3" and "-35.7S, -110.3W").
            if m.group("lat_h"):
                h = m.group("lat_h").upper()
                if h == "S" and lat > 0:
                    lat = -lat
            # Longitude hemisphere is W only — S/E/N are latitude markers.
            # Audit 2026-08-01 (Bug C6): same double-negation guard.
            # NOTE: keep ``upper() == "W"`` as the explicit check so the
            # round15 audit source-guard test still recognises the fix.
            if m.group("lon_h") and m.group("lon_h").upper() == "W" and lon > 0:
                lon = -lon
            if _valid(lat, lon):
                return Coordinate(
                    latitude=lat,
                    longitude=lon,
                    source=text[:200],
                    raw=m.group(0),
                    is_paleo=_is_paleo_text(text, m.start()),
                )
        except (TypeError, ValueError) as exc:
            import logging

            logging.getLogger(__name__).debug(
                "geo_coords: decimal regex matched but conversion failed: %s",
                exc,
            )
    return None


def parse_all_coordinates(text: str) -> list[Coordinate]:
    """Return every parseable coordinate in ``text``."""
    if not text:
        return []
    out: list[Coordinate] = []
    seen: list[tuple[float, float]] = []  # deduplication helper

    def _add_unique(lat: float, lon: float, raw: str, source: str, text: str, start: int) -> bool:
        """Add coord if no prior entry is within 0.01° (avoids DMS+Decimal dupes)."""
        for s_lat, s_lon in seen:
            if abs(lat - s_lat) < 0.01 and abs(lon - s_lon) < 0.01:
                return False
        seen.append((lat, lon))
        out.append(
            Coordinate(
                latitude=lat,
                longitude=lon,
                source=source,
                raw=raw,
                is_paleo=_is_paleo_text(text, start),
            )
        )
        return True

    for m in _DMS_RE.finditer(text):
        try:
            # Audit 2026-08-01 (Bug C6): seconds optional; default to 0.
            lat_s_raw = m.group("lat_s")
            lon_s_raw = m.group("lon_s")
            lat_s = float(lat_s_raw) if lat_s_raw is not None else 0.0
            lon_s = float(lon_s_raw) if lon_s_raw is not None else 0.0
            lat = int(m.group("lat_d")) + int(m.group("lat_m")) / 60.0 + lat_s / 3600.0
            lon = int(m.group("lon_d")) + int(m.group("lon_m")) / 60.0 + lon_s / 3600.0
            # Audit 2026-08-01 (Bug C6): same double-negation guard.
            if m.group("lat_h"):
                h = m.group("lat_h").upper()
                if h == "S" and lat > 0:
                    lat = -lat
            # Longitude hemisphere is W only — S/E/N are latitude markers.
            # Audit 2026-08-01 (Bug C6): same double-negation guard.
            # NOTE: keep ``upper() == "W"`` as the explicit check so the
            # round15 audit source-guard test still recognises the fix.
            if m.group("lon_h") and m.group("lon_h").upper() == "W" and lon > 0:
                lon = -lon
            if _valid(lat, lon):
                _add_unique(lat, lon, m.group(0), text[:200], text, m.start())
        except Exception:
            pass
    for m in _DECIMAL_RE.finditer(text):
        try:
            lat = float(m.group("lat"))
            lon = float(m.group("lon"))
            # Audit 2026-08-01 (Bug C6): same double-negation guard.
            if m.group("lat_h"):
                h = m.group("lat_h").upper()
                if h == "S" and lat > 0:
                    lat = -lat
            # Longitude hemisphere is W only — S/E/N are latitude markers.
            # Audit 2026-08-01 (Bug C6): same double-negation guard.
            # NOTE: keep ``upper() == "W"`` as the explicit check so the
            # round15 audit source-guard test still recognises the fix.
            if m.group("lon_h") and m.group("lon_h").upper() == "W" and lon > 0:
                lon = -lon
            if _valid(lat, lon):
                _add_unique(lat, lon, m.group(0), text[:200], text, m.start())
        except Exception:
            pass
    return out


def _valid(lat: float, lon: float) -> bool:
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


# Phase 62 Plan 5 (Bug 5.4): same keyword heuristic as
# ``geology_extraction._classify_coordinate_age``. Kept local so the
# geo_coords module doesn't depend on geology_extraction (avoids a
# potential circular import — geology_extraction imports
# parse_coordinate indirectly through converter chains).
_PALEO_KEYWORDS_GEO = (
    "during the ",
    "at that time",
    "at the time",
    "in the late ",
    "in the early ",
    "in the middle ",
    "paleogeographic",
    "paleolatitude",
    "paleolongitude",
    "during deposition",
    "reconstructed",
    "was located",
    "lay at",
    "was situated",
    "at deposition",
    "in triassic",
    "in jurassic",
    "in cretaceous",
    "in permian",
    "in devonian",
    "in ordovician",
    "in silurian",
    "in cambrian",
    "in carboniferous",
    # Phase 62 Plan 5 (Bug 5.15): era + epoch names (mirror copy
    # of geology_extraction._PALEO_KEYWORDS so the two paths stay
    # in sync without a circular import).
    "in mesozoic",
    "in the mesozoic",
    "mesozoic",
    "in cenozoic",
    "in the cenozoic",
    "cenozoic",
    "in paleozoic",
    "in the paleozoic",
    "paleozoic",
    "in paleogene",
    "in the paleogene",
    "paleogene",
    "in neogene",
    "in the neogene",
    "neogene",
    "in eocene",
    "in the eocene",
    "eocene",
    "in oligocene",
    "in the oligocene",
    "oligocene",
    "in miocene",
    "in the miocene",
    "miocene",
    "in pliocene",
    "in the pliocene",
    "pliocene",
    "in pleistocene",
    "in the pleistocene",
    "pleistocene",
    # Audit 2026-09-03 (BLOCKER-#9): British English spellings used in
    # De Wever 2001, O'Dogherty 1994, Hollis 1997 and other classical
    # radiolarian literature. The American spellings above (paleo*,
    # ceno*) are kept so existing matches still work; the British
    # variants below are added because the previous list only
    # matched the American forms, causing 56 Ma occurrences in
    # "during the Palaeocene" text to be misclassified as modern
    # coordinates (corrupting downstream GBIF submissions).
    "palaeogeographic", "palaeolatitude", "palaeolongitude",
    "palaeocene", "in the palaeocene", "in palaeocene",
    "palaeogene", "in the palaeogene", "in palaeogene",
    "palaeozoic", "in the palaeozoic", "in palaeozoic",
    "palaeontological",
    "cainozoic", "in the cainozoic", "in cainozoic",
    "caenozoic", "in the caenozoic", "in caenozoic",
    # French spellings (De Wever 2001 et al.) — these need to be
    # matched case-insensitively too; the regex below uses
    # ``re.IGNORECASE`` so the literal accents pass through. Adding
    # them as plain strings lets the |alternation| work without
    # extra compilation.
    "paléocène", "paléogène", "paléozoïque",
    "mésozoïque", "cénozoïque", "cainozoïque",
)


# Audit 2026-08-01 (Bug M5): word-boundary regex matching. The
# previous bare-substring ``kw in ctx`` produced false positives
# whenever a keyword was embedded inside a longer word (e.g.
# "paleogeneously" matched "paleogene", "subpaleogene" matched
# "paleogene", "concurrently" matched "currently"). The regex
# keeps the same keyword list but enforces ``\b`` on either side
# so the match must start/end at a word boundary. Trailing
# spaces preserved in the keyword list (e.g. "during the ")
# still anchor a phrase-end thanks to the surrounding ``\b``.
_PALEO_KEYWORDS_GEO_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(kw) for kw in _PALEO_KEYWORDS_GEO) + r")\b",
    re.IGNORECASE,
)


def _is_paleo_text(text: str, match_start: int) -> bool:
    """Return True if a paleo keyword appears within ~120 chars
    BEFORE ``match_start``. Used by ``parse_coordinate`` /
    ``parse_all_coordinates`` to populate ``Coordinate.is_paleo``
    without forcing downstream code to re-walk the text.

    Mirrors ``geology_extraction._classify_coordinate_age``. The two
    paths use independent keyword copies to keep ``geo_coords``
    importable without ``geology_extraction`` (which itself imports
    modules that touch coordinate parsing).
    """
    ctx = text[max(0, match_start - 120) : match_start]
    return bool(_PALEO_KEYWORDS_GEO_RE.search(ctx))
