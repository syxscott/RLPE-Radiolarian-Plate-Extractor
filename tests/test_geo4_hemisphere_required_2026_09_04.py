"""Regression: audit 2026-09-04 geo-4 — the decimal-coord regex in
:mod:`rlpe.geo_coords` accepted a bare pair of numbers as lat/lon if
EITHER side carried a decimal point, even when neither side carried a
hemisphere letter. Examples:

    "35, 110.5"   → (lat=35, lon=110.5)  accepted; should be rejected
    "35.5, 110"   → (lat=35.5, lon=110)  accepted; should be rejected

Without a hemisphere letter the parser cannot tell whether 35 is N or
S (or 110 is E or W). The current behaviour silently assumed positive
(N/E), which makes a real Southern Hemisphere or Western Hemisphere
coordinate read as the wrong hemisphere — e.g. an Argentinean site at
"35.5, 110W" written without the "W" in OCR was routed through the
modern-rotation path with the wrong sign.

Fix contract: a coordinate pair must carry at least one hemisphere
letter (``N``/``S``/``E``/``W``) to be accepted as parseable. Bare
numeric pairs are still returned as ``None`` (callers fall back to
country-centroid heuristic or skip the record).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe.geo_coords import _DECIMAL_RE, parse_coordinate  # noqa: E402

import pytest  # noqa: E402


class TestHemisphereRequired:
    def test_decimal_only_one_side_rejected(self):
        # Decimal present on one side is not enough — must have a
        # hemisphere letter.
        assert parse_coordinate("35, 110.5") is None
        assert parse_coordinate("35.5, 110") is None

    def test_both_decimals_no_hemisphere_rejected(self):
        assert parse_coordinate("35.5, 110.5") is None

    def test_hemisphere_present_still_accepted(self):
        # Sanity: the qualified form still parses.
        c = parse_coordinate("35.5N, 110.5E")
        assert c is not None
        assert c.latitude == pytest.approx(35.5)
        assert c.longitude == pytest.approx(110.5)

    def test_single_hemisphere_letter_accepted(self):
        # One hemisphere letter suffices (it's enough to disambiguate
        # the missing side — common in OCR-truncated captions).
        c = parse_coordinate("35.5, 110.5E")
        assert c is not None
        assert c.latitude == pytest.approx(35.5)
        assert c.longitude == pytest.approx(110.5)

    def test_dms_form_unaffected(self):
        # DMS form has its own regex (``_DMS_RE``) that already
        # requires the degree sign — must still parse.
        c = parse_coordinate("35°42'N, 110°18'E")
        assert c is not None
        assert c.latitude == pytest.approx(35.7, abs=0.01)
        assert c.longitude == pytest.approx(110.3, abs=0.01)


class TestRegexLookaheadEnforcesHemisphere:
    def test_decimal_only_one_side_no_match(self):
        assert _DECIMAL_RE.search("35, 110.5") is None
        assert _DECIMAL_RE.search("35.5, 110") is None

    def test_both_decimals_no_hemisphere_no_match(self):
        assert _DECIMAL_RE.search("35.5, 110.5") is None

    def test_hemisphere_present_matches(self):
        assert _DECIMAL_RE.search("35.5N, 110.5E") is not None
        assert _DECIMAL_RE.search("35.5, 110.5E") is not None
        assert _DECIMAL_RE.search("35.5N, 110.5") is not None
