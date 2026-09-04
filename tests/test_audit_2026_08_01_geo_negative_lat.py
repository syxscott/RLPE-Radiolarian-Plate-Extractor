"""Regression tests for audit 2026-08-01 batch W1 — C6 geo_coords.py lat regex missing -?."""

from __future__ import annotations

from rlpe.geo_coords import (
    _DECIMAL_RE,
    _DMS_RE,
    parse_all_coordinates,
    parse_coordinate,
)


class TestGeoCoordsNegativeLat:
    def test_negative_lat_string(self):
        # Audit 2026-09-04 geo-4: hemisphere letter required (or
        # negative sign on both — that's an unambiguous hemisphere
        # indicator). The "-35.7, -110.3" form still parses because
        # both signs are explicit.
        c = parse_coordinate("-35.7, -110.3")
        assert c is not None
        assert c.latitude == -35.7
        assert c.longitude == -110.3

    def test_negative_lat_with_s_letter(self):
        """``35.7°S, 110.3°W`` must parse with negative coordinates."""
        c = parse_coordinate("35.7°S, 110.3°W")
        assert c is not None
        assert c.latitude == -35.7
        assert c.longitude == -110.3

    def test_no_double_negation(self):
        """``-35.7S, -110.3W`` must NOT be flipped back to positive
        (a pre-existing bug: literal '-' AND hemisphere letter
        double-negated, silently sending a south-hemisphere point to
        the northern hemisphere)."""
        c = parse_coordinate("-35.7S, -110.3W")
        assert c is not None
        assert c.latitude == -35.7, f"expected lat=-35.7 (no double-negation), got {c.latitude}"
        assert c.longitude == -110.3, f"expected lon=-110.3 (no double-negation), got {c.longitude}"

    def test_positive_no_letter(self):
        # Audit 2026-09-04 geo-4: bare decimals without hemisphere or
        # explicit sign are no longer accepted — the parser cannot
        # disambiguate N/S/E/W without a marker. Update to use an
        # explicit sign on longitude (unambiguous hemisphere indicator).
        c = parse_coordinate("35.7, 110.3E")
        assert c is not None
        assert c.latitude == 35.7
        assert c.longitude == 110.3

    def test_plate_refs_not_matched(self):
        """``Plate 1, figs 3, 5 are shown`` must NOT be mis-parsed as
        ``Coordinate(3.0, 5.0)`` — there is no coordinate indicator
        (decimal point, degree sign, or standalone hemisphere letter)
        in this sentence, so the regex must reject the integer pair."""
        c = parse_coordinate("Plate 1, figs 3, 5 are shown")
        assert c is None, (
            f"expected None (plate/figure ref must not parse as a coordinate), got {c!r}"
        )

    def test_dms_negative(self):
        """DMS form ``35°42'S, 110°18'W`` (no seconds) must parse with
        negative coordinates. The DMS regex previously required seconds
        so this was silently rejected; the C6 audit also accepts
        abbreviated DMS."""
        c = parse_coordinate("35°42'S, 110°18'W")
        assert c is not None
        # 35°42' = 35.7 exactly; 110°18' = 110.3
        assert abs(c.latitude - (-35.7)) < 1e-6
        assert abs(c.longitude - (-110.3)) < 1e-6

    def test_decimal_with_point_only(self):
        # Audit 2026-09-04 geo-4: bare decimals without a hemisphere
        # marker are rejected. The qualified form ("12.5N, 45.2E")
        # still parses.
        c = parse_coordinate("12.5N, 45.2E")
        assert c is not None
        assert c.latitude == 12.5
        assert c.longitude == 45.2

    def test_decimal_regex_has_negative_lat_group(self):
        """Source-guard: the decimal regex ``lat`` group must accept an
        optional leading minus. A future refactor that drops the ``-?``
        would silently break the C6 fix.

        Audit 2026-09-04 geo-4: the regex now also requires a
        hemisphere letter (or both signs) so we use "-35.7, -110.3"
        (both negative) as the probe — the hemisphere requirement is
        satisfied by the explicit signs."""
        m = _DECIMAL_RE.search("-35.7, -110.3")
        assert m is not None, "decimal regex must accept leading '-' on lat"
        assert m.group("lat") == "-35.7"
        assert float(m.group("lat")) == -35.7

    def test_dms_regex_has_negative_lat_d_group(self):
        """Source-guard: the DMS regex ``lat_d`` group must accept an
        optional leading minus (parity with lon_d)."""
        m = _DMS_RE.search("-35°42'S, 110°18'W")
        assert m is not None, "DMS regex must accept leading '-' on lat_d"
        assert m.group("lat_d") == "-35"
        assert int(m.group("lat_d")) == -35

    def test_parse_all_coordinates_negative(self):
        """``parse_all_coordinates`` must also avoid double-negation."""
        out = parse_all_coordinates("First at -35.7S, -110.3W and second at 36.0, 111.0")
        # At least the negative pair must round-trip with both coords negative.
        negatives = [c for c in out if c.latitude < 0 and c.longitude < 0]
        assert negatives, f"expected at least one negative coord in {out!r}"
        first_neg = negatives[0]
        assert first_neg.latitude == -35.7
        assert first_neg.longitude == -110.3
