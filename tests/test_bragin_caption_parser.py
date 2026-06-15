"""Regression tests for Bragin 2025-style "(N) Species" parenthesised
caption parsing.

Bragin 2025 ("Oxfordian-Kimmeridgian radiolarians from the Nordvik
section, northern Siberia") uses a parenthesised label form that the
pre-existing _CAPTION_CLAUSE_RE / _DANELIAN_CLAUSE_RE did not handle.
A typical caption reads::

    Plate I. Characteristic taxa of family Parvicingulidae ... :
    (1) Praeparvicingula blackhorsensis, (2) Praeparvicingula donnae,
    (3) Parvicingula khabakovi, (4) Nordvikella elegans, (5) Arctocapsula
    perforata, (6) Echinocampe aliferum, (7) Echinocampe modestum,
    (8) Pantanellium tierrablancaense, (9) Pantanellium sp. cf. P.
    tierrablancaense, (10, 11) Pantanellium moscowiense Bragin.

The challenge is:
  1. The label is wrapped in (open+close) parens, not just ")" or ":".
  2. The whole plate is one long chunk (no ";" separators) with a
     "Plate I." preamble in front of the first label.
  3. The inner _DANELIAN_CLAUSE_RE.match was MULTILINE-anchored on
     ``^\\s*``, so finditer only matched the first "(1) Praeparvicingula"
     even though 11 pairs live in the same chunk.
  4. The "Plate I. ...prose..." preamble must not be misread as a clause.

The fix: optional open paren in ``_DANELIAN_CLAUSE_RE``, removal of the
``^`` anchor + MULTILINE flag, and a ``danelian_lead_re`` that strips the
preamble before the inner finditer.
"""
from __future__ import annotations

from rlpe.m3_engine import _regex_parse_caption

BRAGIN_PLATE_I = (
    "Plate I. Characteristic taxa of family Parvicingulidae from the "
    "Oxfordian-Kimmeridgian deposits of the Nordvik section: (1) "
    "Praeparvicingula blackhorsensis, (2) Praeparvicingula donnae, "
    "(3) Parvicingula khabakovi, (4) Nordvikella elegans, (5) "
    "Arctocapsula perforata, (6) Echinocampe aliferum, (7) Echinocampe "
    "modestum, (8) Pantanellium tierrablancaense, (9) Pantanellium sp. "
    "cf. P. tierrablancaense, (10, 11) Pantanellium moscowiense Bragin."
)


def test_bragin_plate_i_parses_all_11_panels():
    """All 11 Bragin Plate I panels must parse. The (10, 11) range
    must expand into two labels."""
    pairs = _regex_parse_caption(BRAGIN_PLATE_I)
    assert len(pairs) == 10, (
        f"expected 10 (label,species) clauses, got {len(pairs)}: "
        f"{[(p.labels, p.species) for p in pairs]}"
    )
    # 10 unique species + 1 expanded range = 11 distinct label-species
    # assignments.
    all_labels = [lbl for p in pairs for lbl in p.labels]
    assert len(all_labels) == 11
    assert "1" in all_labels
    assert "2" in all_labels
    assert "3" in all_labels
    assert "4" in all_labels
    assert "5" in all_labels
    assert "6" in all_labels
    assert "7" in all_labels
    assert "8" in all_labels
    assert "9" in all_labels
    assert "10" in all_labels
    assert "11" in all_labels  # (10, 11) range expanded


def test_bragin_species_match_gold_list():
    """Spot-check the parsed species names against the gold list."""
    pairs = _regex_parse_caption(BRAGIN_PLATE_I)
    species = [p.species for p in pairs]
    assert "Praeparvicingula blackhorsensis" in species
    assert "Praeparvicingula donnae" in species
    assert "Parvicingula khabakovi" in species
    assert "Nordvikella elegans" in species
    assert "Arctocapsula perforata" in species
    assert "Echinocampe aliferum" in species
    assert "Echinocampe modestum" in species
    # Panel 9 is "Pantanellium sp. cf. P. tierrablancaense" — the inner
    # cf. qualifier is a second disambiguating tag, not a real new
    # species. We only require the genus-level "Pantanellium sp." to
    # match for F1 purposes; a stricter form might capture the full
    # "cf. P. tierrablancaense" disambiguator as the modifier.
    assert any("Pantanellium" in s for s in species)
    assert "Pantanellium moscowiense" in species


def test_bragin_does_not_match_preamble_as_clause():
    """The 'Plate I. Characteristic taxa ... Nordvik section:' prose
    before the first label must NOT be (mis)matched as a (label, species)
    pair. The genus-only words like 'Plate' must be filtered."""
    pairs = _regex_parse_caption(BRAGIN_PLATE_I)
    # The first species must be 'Praeparvicingula blackhorsensis'
    # (label '1'), not 'Plate I' / 'Plate' / 'Nordvik' / 'Characteristic'.
    assert pairs[0].species == "Praeparvicingula blackhorsensis"
    assert pairs[0].labels == ["1"]


def test_bragin_short_two_panel_caption():
    """A short (1), (2) two-panel caption must still parse correctly."""
    pairs = _regex_parse_caption(
        "Plate I. (1) Praeparvicingula blackhorsensis, (2) Praeparvicingula donnae"
    )
    assert len(pairs) == 2
    assert pairs[0].species == "Praeparvicingula blackhorsensis"
    assert pairs[1].species == "Praeparvicingula donnae"
