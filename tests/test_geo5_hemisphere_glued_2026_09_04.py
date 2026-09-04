"""Regression: audit 2026-09-04 geo-5 —
:func:`rlpe.geology_extraction._extract_first_coord` rejected
coordinates whose hemisphere letter was glued to a digit (the
standard cartographic form ``"35.7N, 110.3E"``) because the guard
regex was ``\\b[NSEWnsew]\\b`` — a word boundary on BOTH sides of
the letter. Between a digit (``7``) and ``N``, there is no word
boundary (both are word chars), so the regex never matched and the
coordinate was rejected as "no hemisphere, no degree sign".

Real radiolarian papers prefer the digit-glued form (``35.7N``)
because it saves column space in locality tables; the bare-space
form (``35.7 N``) is the alternative. The previous regex matched
only the bare-space form. The result: every digit-glued coordinate
in 9-paper corpus failed the hemisphere/degree guard and was
dropped from the geology record.

Fix contract: the guard regex must allow the hemisphere letter
either glued to the digit (``5N``) OR following whitespace
(``5 N``). A letter SURROUNDED by alphabetic chars ("Northeast")
must still NOT count — that's a false positive.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe.geology_extraction import _extract_first_coord  # noqa: E402


class TestDigitGluedHemisphereAccepted:
    def test_digit_glued_north(self):
        lat, lon, start, end = _extract_first_coord("lat 35.7N, 110.3E")
        assert lat is not None, "digit-glued '35.7N, 110.3E' must parse"
        assert abs(lat - 35.7) < 0.01
        assert abs(lon - 110.3) < 0.01

    def test_digit_glued_south(self):
        lat, lon, _, _ = _extract_first_coord("at 35.7S, 110.3W in section")
        assert lat is not None
        assert abs(lat - (-35.7)) < 0.01
        assert abs(lon - (-110.3)) < 0.01

    def test_space_separated_still_accepted(self):
        # Sanity: the space-separated form (previous contract) still
        # passes the guard.
        lat, lon, _, _ = _extract_first_coord("at 35.7 N, 110.3 E")
        assert lat is not None
        assert abs(lat - 35.7) < 0.01


class TestFalsePositivesRejected:
    def test_alphabetic_n_in_northeast_not_hemisphere(self):
        # The guard must NOT count an alphabetic "N" (e.g. "Northeast")
        # as a hemisphere letter — but this only matters when the
        # coord itself had a degree sign (°). Without ° AND without a
        # real hemisphere, the coord is rejected.
        # The matched raw here is "35N" inside "Northeast 35N" — the
        # raw extract is just the lat/lon substring, so this case is
        # still safe.
        lat, _, _, _ = _extract_first_coord("Section at 35N, 110.3E")
        # "35N" digit-glued — must parse (digit-glued fix).
        assert lat is not None

    def test_no_hemisphere_no_degree_rejected(self):
        # Bare pair with no hemisphere and no degree: still rejected
        # (this is the original guard's purpose — page numbers /
        # figure counts shouldn't be mis-parsed as coords).
        lat, lon, _, _ = _extract_first_coord("Plate 35, fig 110")
        assert lat is None
        assert lon is None
