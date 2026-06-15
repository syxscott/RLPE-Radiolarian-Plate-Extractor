"""Tests for locality coordinate parsing."""
from __future__ import annotations

from rlpe.geo_coords import (
    parse_all_coordinates,
    parse_coordinate,
)


class TestDecimalCoordinates:
    def test_simple_decimal(self):
        c = parse_coordinate("35.7, 110.3")
        assert c is not None
        assert abs(c.latitude - 35.7) < 0.01
        assert abs(c.longitude - 110.3) < 0.01

    def test_decimal_with_NSEW(self):
        c = parse_coordinate("35.7°N, 110.3°E")
        assert c is not None
        assert c.latitude == 35.7
        assert c.longitude == 110.3

    def test_decimal_negative_longitude(self):
        c = parse_coordinate("35.7, -110.3")
        assert c is not None
        assert c.latitude == 35.7
        assert c.longitude == -110.3

    def test_decimal_with_S_hemisphere(self):
        c = parse_coordinate("35.7 S, 110.3 E")
        assert c is not None
        assert c.latitude == -35.7
        assert c.longitude == 110.3

    def test_decimal_with_W_hemisphere(self):
        c = parse_coordinate("35.7 N, 110.3 W")
        assert c is not None
        assert c.longitude == -110.3

    def test_invalid_latitude_rejected(self):
        c = parse_coordinate("999.0, 100.0")
        # If regex matches, validation should reject
        assert c is None or abs(c.latitude) <= 90

    def test_invalid_longitude_rejected(self):
        c = parse_coordinate("35.7, 999.0")
        assert c is None or abs(c.longitude) <= 180

    def test_empty_text(self):
        assert parse_coordinate("") is None

    def test_no_coordinates(self):
        assert parse_coordinate("No coordinates here") is None


class TestDMSCoordinates:
    def test_basic_dms(self):
        c = parse_coordinate("35°42'12\"N 110°18'00\"E")
        assert c is not None
        assert abs(c.latitude - 35.70333) < 0.001
        assert abs(c.longitude - 110.30) < 0.01

    def test_dms_with_negative_lon(self):
        c = parse_coordinate("35°42'12\"N 110°18'00\"W")
        assert c is not None
        assert c.longitude < 0


class TestParseAllCoordinates:
    def test_returns_multiple(self):
        text = "Site 1 at 35.7, 110.3 and Site 2 at 36.0, 111.0"
        coords = parse_all_coordinates(text)
        assert len(coords) >= 2

    def test_empty(self):
        assert parse_all_coordinates("") == []

    def test_single(self):
        coords = parse_all_coordinates("Coordinates: 35.7, 110.3")
        assert len(coords) >= 1
        assert abs(coords[0].latitude - 35.7) < 0.01
