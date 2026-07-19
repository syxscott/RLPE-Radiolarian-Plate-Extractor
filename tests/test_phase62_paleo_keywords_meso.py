"""Phase 62 Plan 5 (Bug 5.15): paleo keyword heuristic must include
Mesozoic / Cenozoic / epoch names.

``_PALEO_KEYWORDS`` in ``geology_extraction.py`` covers period
names (Triassic, Jurassic, etc.) but not:

  - Era names: Mesozoic, Cenozoic
  - Period names: Paleogene, Neogene (sometimes used directly
    without "in the X" framing)
  - Epoch names: Eocene, Oligocene, Miocene, Pliocene, Pleistocene

A sentence like "the basin was located at 23°S, 47°W in the
Mesozoic" or "the locality was at 38°N, 14°E in the Eocene" was
failing the paleo-keyword heuristic and being mis-tagged as a
modern coordinate — the downstream paleo_reconstruction
would silently produce modern coords for an Eocene-age record.

The fix: extend ``_PALEO_KEYWORDS`` (and the mirror copy in
geo_coords._PALEO_KEYWORDS_GEO) with the era + epoch names.
"""
from __future__ import annotations

from rlpe.geology_extraction import (
    _PALEO_KEYWORDS,
    _classify_coordinate_age,
)


def test_paleo_keyword_mesozoic():
    """'in the Mesozoic' must flip is_paleo=True."""
    assert "in the mesozoic" in _PALEO_KEYWORDS


def test_paleo_keyword_cenozoic():
    """'in the Cenozoic' must flip is_paleo=True."""
    assert "in the cenozoic" in _PALEO_KEYWORDS


def test_paleo_keyword_paleogene():
    """'in the Paleogene' must flip is_paleo=True."""
    assert "in the paleogene" in _PALEO_KEYWORDS


def test_paleo_keyword_neogene():
    """'in the Neogene' must flip is_paleo=True."""
    assert "in the neogene" in _PALEO_KEYWORDS


def test_paleo_keyword_eocene():
    """'in the Eocene' must flip is_paleo=True."""
    assert "in the eocene" in _PALEO_KEYWORDS


def test_paleo_keyword_oligocene():
    """'in the Oligocene' must flip is_paleo=True."""
    assert "in the oligocene" in _PALEO_KEYWORDS


def test_paleo_keyword_miocene():
    """'in the Miocene' must flip is_paleo=True."""
    assert "in the miocene" in _PALEO_KEYWORDS


def test_paleo_keyword_pliocene():
    """'in the Pliocene' must flip is_paleo=True."""
    assert "in the pliocene" in _PALEO_KEYWORDS


def test_paleo_keyword_pleistocene():
    """'in the Pleistocene' must flip is_paleo=True."""
    assert "in the pleistocene" in _PALEO_KEYWORDS


def test_classify_eocene_age_text():
    """End-to-end: 'in the Eocene, the basin was located at 23N, 47W'
    must classify as paleo (keyword BEFORE the match)."""
    text = "in the Eocene, the basin was located at 23N, 47W"
    # _classify_coordinate_age looks 120 chars BEFORE the match —
    # place the coordinate at the end so "in the Eocene" is in the
    # prefix.
    age = _classify_coordinate_age(text, match_start=len(text) - 6, match_end=len(text))
    assert age == "paleo", f"expected paleo for Eocene framing, got {age!r}"


def test_classify_mesozoic_age_text():
    """End-to-end: 'in the Mesozoic, the locality was at 38N, 14E'."""
    text = "in the Mesozoic, the locality was at 38N, 14E"
    age = _classify_coordinate_age(text, match_start=len(text) - 6, match_end=len(text))
    assert age == "paleo", f"expected paleo for Mesozoic framing, got {age!r}"


def test_classify_cenozoic_age_text():
    """End-to-end: 'in the Cenozoic, the locality was at 38N, 14E'."""
    text = "in the Cenozoic, the locality was at 38N, 14E"
    age = _classify_coordinate_age(text, match_start=len(text) - 6, match_end=len(text))
    assert age == "paleo"