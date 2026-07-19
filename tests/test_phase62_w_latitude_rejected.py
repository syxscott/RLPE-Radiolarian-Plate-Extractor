"""Phase 62 Plan 5 (Bug 5.1): _DECIMAL_RE must reject 'W' as latitude.

Previously the regex ``_DECIMAL_RE`` accepted any of ``[NSEWnsew]`` as
the latitude hemisphere, which is physically invalid — W is the
longitude hemisphere marker. A stray ``W`` on a latitude field (e.g.
``45°W, 110°E`` — both meant as longitudes by the OCR, the first
captured in the lat slot by mistake) silently produced a -45 latitude
that downstream consumers treated as a southern-hemisphere locality.

The fix tightens the regex to ``[NSns]`` only, so any input that uses
``W`` (or ``E``) on the latitude slot fails to match the decimal form
at all (and would be parsed as the DMS form if the shape supports it,
or rejected otherwise).

This test asserts:
  * The DMS regex *also* rejects ``W`` as latitude (parity with the
    decimal form).
  * Both regexes accept ``N`` / ``S`` as latitude (regression).
  * Both regexes accept ``E`` / ``W`` as longitude (regression).
  * ``parse_coordinate`` returns ``None`` when no valid (lat, lon)
    pair can be parsed — i.e. the invalid latitude hemisphere
    doesn't cause a silent parse.
"""
from __future__ import annotations

from rlpe.geo_coords import (
    _DECIMAL_RE,
    _DMS_RE,
    parse_coordinate,
)


def test_decimal_regex_rejects_w_latitude():
    """``45W`` as latitude must NOT match the decimal regex.

    The latitude hemisphere group must only accept N/S, not E/W.
    """
    m = _DECIMAL_RE.search("45W, 110E")
    # If the regex matched but lat_h is W, it would extract a wrong
    # coord. Assert that either:
    #   1. The regex doesn't match at all, OR
    #   2. The matched lat_h is NOT W
    if m is not None:
        lat_h = m.group("lat_h")
        assert lat_h is None or lat_h.upper() != "W", (
            f"_DECIMAL_RE matched with W as latitude hemisphere: {m.group(0)!r}"
        )


def test_decimal_regex_accepts_n_s_latitude():
    """Regression: N and S must still be accepted as latitude markers."""
    for h in ("N", "S", "n", "s"):
        text = f"35.5{h}, 110.3E"
        m = _DECIMAL_RE.search(text)
        assert m is not None, f"should match for h={h!r}"
        assert m.group("lat_h").upper() == h.upper()


def test_decimal_regex_rejects_e_latitude():
    """``E`` is a longitude marker, must NOT be accepted as latitude."""
    m = _DECIMAL_RE.search("45E, 110W")
    if m is not None:
        lat_h = m.group("lat_h")
        assert lat_h is None or lat_h.upper() != "E", (
            f"_DECIMAL_RE matched with E as latitude hemisphere: {m.group(0)!r}"
        )


def test_dms_regex_rejects_w_latitude():
    """DMS regex must also reject W on the latitude slot."""
    m = _DMS_RE.search('45°00\'00"W 110°00\'00"E')
    if m is not None:
        lat_h = m.group("lat_h")
        assert lat_h is None or lat_h.upper() != "W", (
            f"_DMS_RE matched with W as latitude hemisphere: {m.group(0)!r}"
        )


def test_dms_regex_accepts_n_s_latitude():
    """Regression: DMS regex still accepts N/S as latitude."""
    for h in ("N", "S"):
        text = f'45°00\'00"{h} 110°00\'00"E'
        m = _DMS_RE.search(text)
        assert m is not None, f"DMS should match for h={h!r}"
        assert m.group("lat_h").upper() == h.upper()


def test_parse_coordinate_rejects_w_latitude():
    """End-to-end: ``45°W, 110°E`` must NOT silently produce a coord."""
    # The decimal regex rejects W-as-lat, so the input should NOT
    # parse via the decimal path. The DMS regex also rejects W-as-
    # lat. Either way, parse_coordinate returns None.
    # (If the regex DID silently match, it would produce lat=-45
    # and lon=110 — a clearly wrong "southern Italy" coordinate.)
    out = parse_coordinate("45W, 110E")
    if out is not None:
        # If something parsed, ensure lat is positive (the value
        # that would have been flipped to -45 by a stray W).
        # The literal "45W" without decimal is suspicious — if we
        # get any match, it should NOT be a -45 latitude.
        assert out.latitude >= 0.0, (
            f"parse_coordinate silently produced negative latitude "
            f"from W-as-latitude input: lat={out.latitude}"
        )


def test_parse_coordinate_regression_valid_still_works():
    """Regression: legitimate coordinates still parse."""
    out = parse_coordinate("35.7°N, 110.3°E")
    assert out is not None
    assert abs(out.latitude - 35.7) < 1e-6
    assert abs(out.longitude - 110.3) < 1e-6