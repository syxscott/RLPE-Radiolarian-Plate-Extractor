"""Tests for Plan B: geology_extraction delegates to geo_coords.parse_coordinate.

Audit 2026-08-16 (fill-gaps): previously geology_extraction._extract_first_coord
held its own ``COORDINATE_PATTERN`` regex that duplicated the DMS +
decimal-degrees parsing already implemented in ``rlpe.geo_coords``. The
two paths had already drifted once (audit 2026-08-01 Bug C6 had to be
fixed in both). Now the inline regex is removed and the function calls
``parse_coordinate`` directly, with the same hemisphere/degree-symbol
guard preserved.
"""
from __future__ import annotations

from rlpe.geo_coords import parse_coordinate
from rlpe.geology_extraction import _extract_first_coord


def test_dms_with_seconds():
    """Standard DMS: 35°42'30"N, 139°46'15"E."""
    text = "Locality at 35°42'30\"N, 139°46'15\"E, central Honshu"
    lat, lon, start, end = _extract_first_coord(text)
    assert lat is not None and abs(lat - 35.7083) < 0.01
    assert lon is not None and abs(lon - 139.7708) < 0.01
    assert start is not None and end is not None
    # Match starts at the "35°42'30"N" token
    assert text[start:end].startswith("35°42'")


def test_dms_without_seconds():
    """DMS without seconds: 35°42'N, 139°46'E (audit 2026-08-01 Bug C6 case)."""
    text = "Locality at 35°42'N, 139°46'E, central Honshu"
    lat, lon, start, end = _extract_first_coord(text)
    assert lat is not None and abs(lat - 35.7) < 0.01
    assert lon is not None and abs(lon - 139.7667) < 0.01


def test_decimal_with_degree_symbol():
    """Decimal degrees with the ° symbol: 35.7°N, 139.77°E."""
    text = "Locality at 35.7°N, 139.77°E"
    lat, lon, _, _ = _extract_first_coord(text)
    assert lat is not None and abs(lat - 35.7) < 0.01
    assert lon is not None and abs(lon - 139.77) < 0.01


def test_southern_hemisphere_negated():
    """Southern latitude must come back negative."""
    text = "Type locality 33.9°S, 151.2°E, near Sydney"
    lat, lon, _, _ = _extract_first_coord(text)
    assert lat is not None and lat < 0
    assert abs(lat + 33.9) < 0.01
    assert lon is not None and abs(lon - 151.2) < 0.01


def test_western_longitude_negated():
    """Western longitude must come back negative."""
    text = "Locality at 37.4°N, 122.1°W"
    lat, lon, _, _ = _extract_first_coord(text)
    assert lat is not None and abs(lat - 37.4) < 0.01
    assert lon is not None and lon < 0
    assert abs(lon + 122.1) < 0.01


def test_bare_decimal_pair_rejected():
    """45, 90 (page numbers, specimen dimensions) must NOT be parsed as coords."""
    text = "See Fig. 5, page 45, 90 specimens found"
    result = _extract_first_coord(text)
    assert result == (None, None, None, None), (
        "bare decimal pairs without hemisphere/degree should not parse"
    )


def test_bare_decimal_pair_with_degree_accepted():
    """Decimal with degree symbol AND hemisphere letter must parse."""
    text = "Specimen at 45°N, 90°W, isolated"
    lat, lon, _, _ = _extract_first_coord(text)
    assert lat is not None and lon is not None
    assert abs(lat - 45.0) < 0.01
    assert abs(lon + 90.0) < 0.01


def test_empty_text_returns_none():
    """Defensive: empty / whitespace input → None tuple."""
    assert _extract_first_coord("") == (None, None, None, None)
    assert _extract_first_coord("   ") == (None, None, None, None)


def test_no_coordinate_returns_none():
    """No numeric pair at all → None tuple."""
    assert _extract_first_coord("Random text without coordinates.") == (
        None, None, None, None
    )


def test_offsets_match_source_text():
    """The (start, end) offsets must point at the matched substring."""
    text = "Type locality: 35°42'N, 139°46'E; central Honshu"
    _, _, start, end = _extract_first_coord(text)
    matched = text[start:end]
    # Should parse to the same coordinate we get back from parse_coordinate
    coord = parse_coordinate(text)
    assert coord is not None
    assert matched == coord.raw


def test_no_inline_coordinate_pattern_in_module():
    """Source-guard: geology_extraction must NOT redeclare the
    COORDINATE_PATTERN module-level constant. We allow the name to
    appear in a docstring (the audit comment mentions it), but the
    re.compile() declaration must be gone.
    """
    import inspect

    from rlpe import geology_extraction

    src = inspect.getsource(geology_extraction)
    assert "COORDINATE_PATTERN = re.compile" not in src, (
        "geology_extraction should delegate to geo_coords.parse_coordinate, "
        "not redeclare its own COORDINATE_PATTERN = re.compile(...)"
    )
