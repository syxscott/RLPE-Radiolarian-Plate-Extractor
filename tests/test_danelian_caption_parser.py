"""Regression tests for Danelian-style "1) Species; 2-3) Species" caption parsing.

Danelian 2006 has 2 plates (Plate 1, Plate 2) with 23 and 19 species
respectively, formatted as a single line per plate:
  "1) Acastea sp.cf. A. remusa HULL, Mg-100; 2-3) Archaeodictyomitra
   apiarium (RÜST), Mg-2; 4) Archaeodictyomitra etrusca CHIARI et al.
   Mg-29; 5-6) Archaeodictyomitra patricki KOCHER, Mg-29; 7) A. patricki,
   Mg-2; ..."

The pre-existing _CAPTION_CLAUSE_RE only matches the "fig N. Species"
form, and a line-anchored single-pass scan only finds the first clause.
This test covers the splitter-based parser: split on ";" + newlines,
match each clause independently, and accept abbreviated "X. epithet"
genera (e.g. "A. patricki" for Archaeodictyomitra patricki).
"""
from __future__ import annotations

from rlpe.m3_engine import _regex_parse_caption

DANELIAN_PLATE_1 = (
    "Plate 1\n\n"
    "Scanning Electron Micrographs of Radiolaria extracted from samples "
    "of the Méouge section. Bar scale (upper right) is equal to 100 µm "
    "for all figures.\n\n"
    "1) Acastea sp.cf. A. remusa HULL, Mg-100; 2-3) Archaeodictyomitra "
    "apiarium (RÜST), Mg-2; 4) Archaeodictyomitra etrusca CHIARI et al. "
    "Mg-29; 5-6) Archaeodictyomitra patricki KOCHER, Mg-29; 7) A. patricki, "
    "Mg-2; 8) Archaeodictyomitra sp.aff. A. patricki KOCHER, Mg-2 ; 9) "
    "Archaeodictyomitra shengi YANG, Mg-29; 10) A. shengi, Mg-77; 11) "
    "Archaeodictyomitra spelae CHIARI et al., Mg-77; 12) Archaeodictyomitra "
    "sp. A, Mg-2; 13) Cinguloturris fusiforma HORI, Mg-100; 14) "
    "Cinguloturris sp.cf. C. fusiforma HORI, Mg-77; 15) Emiluvia "
    "pentaporata STEIGER & STEIGER, Mg-100; 16) Eucyrtidiellum ptyctum "
    "(SANFILIPPO & RIEDEL), Morphotype A, Mg-77 ; 17) E. ptyctum, "
    "Morphotype B, Mg-2; 18) Gongylothorax favosus DUMITRICA, Mg-37; 19) "
    "Loopus doliolum DUMITRICA, Mg-37; 20) L. doliomum, Mg-29; 21) "
    "Loopus venustus (CHIARI et al.), Mg-2; 22) L. venustus, Mg-2; 23) "
    "L. venustus, Mg-133."
)


def test_danelian_plate1_full_parse():
    """All 23 species clauses on Danelian Plate 1 should parse."""
    pairs = _regex_parse_caption(DANELIAN_PLATE_1)
    species = [p.species for p in pairs]
    # Spot-check 8 specific ones — they cover the three sub-cases
    # (full binomial, abbreviated, "sp." only).
    # Note: the captured "sp" form gets a period added back by
    # _normalize_species (historical v9 behaviour, preserved by N3).
    # The eval harness strips trailing period before comparing to
    # gold, so the v17 baseline matches v16 here.
    assert "Acastea sp." in species
    assert "Archaeodictyomitra apiarium" in species
    assert "Archaeodictyomitra patricki" in species
    assert "A. patricki" in species  # abbreviated genus
    assert "Cinguloturris fusiforma" in species
    assert "Cinguloturris sp." in species  # "sp.cf." truncated to "sp."
    assert "Loopus venustus" in species
    assert "L. venustus" in species  # abbreviated
    # 23 distinct species clauses: 21 unique species, 2-3 / 5-6 ranges
    # should expand.
    all_labels = []
    for p in pairs:
        all_labels.extend(p.labels)
    assert "1" in all_labels
    assert "2" in all_labels and "3" in all_labels  # range expanded
    assert "5" in all_labels and "6" in all_labels  # range expanded
    assert "23" in all_labels


def test_danelian_abbreviated_genus_supported():
    """A clause like "7) A. patricki, Mg-2" must match — the regex used
    to require a full Genus name and missed abbreviated references."""
    pairs = _regex_parse_caption(
        "7) A. patricki, Mg-2; 10) A. shengi, Mg-77"
    )
    assert len(pairs) == 2
    assert pairs[0].labels == ["7"]
    assert pairs[0].species == "A. patricki"
    assert pairs[1].labels == ["10"]
    assert pairs[1].species == "A. shengi"


def test_danelian_range_label_supported():
    """A range like "2-3) Species" must be split into two labels."""
    pairs = _regex_parse_caption(
        "2-3) Archaeodictyomitra apiarium (RÜST), Mg-2"
    )
    assert len(pairs) == 1
    assert pairs[0].labels == ["2", "3"]
    assert pairs[0].species == "Archaeodictyomitra apiarium"


def test_danelian_ignores_preamble():
    """The "100 µm for all figures" preamble must not be matched as a clause
    (its "1" is part of a number, not a label)."""
    pairs = _regex_parse_caption(
        "Plate 1\n\nBar scale is 100 µm for all figures.\n\n"
        "1) Acastea sp, Mg-100"
    )
    # Only 1 pair from the real clause. _normalize_species adds
    # the period back on "sp" (historical v9 behaviour, preserved
    # by N3). Eval harness strips it before gold match.
    assert len(pairs) == 1
    assert pairs[0].species == "Acastea sp."


def test_danelian_does_not_break_pouille_or_fig_pattern():
    """Adding the danelian splitter must not regress the other caption
    forms. Sample pouille and inverse "Fig. N. Species" patterns here."""
    pouille_text = (
        "Syntagentactinia biocculosa (Pl. 1, figs 5)\n"
        "Haplentactinia juncta (Pl. 1, fig. 1)"
    )
    pairs = _regex_parse_caption(pouille_text)
    assert len(pairs) == 2
    fig_text = "Figs 1-3. Entactinia itsukichiensis: test spherical."
    pairs = _regex_parse_caption(fig_text)
    assert len(pairs) == 1
    assert pairs[0].species == "Entactinia itsukichiensis"


def test_danelian_sp_aff_epithet_tail():
    """Regression: boughdiri2007 uses "Genus sp. aff. <epithet>" form
    (e.g. "Orbiculiforma sp. aff. mclaughlini", "Archaeodictyomitra sp.
    aff. minoensis"). The DANELIAN clause regex used to truncate these
    at " sp." and drop the " aff. <epithet>" tail, yielding
    "Orbiculiforma sp." instead of the full gold form. The DANELIAN
    and CAPTION clause regexes now share the BAUMGARTNER Shape 1
    "sp. (cf.|aff.) <W>?. <epithet>" tail branch.
    """
    pairs = _regex_parse_caption(
        "1) Ristola altissima; 3) Orbiculiforma sp. aff. mclaughlini "
        "PESSAGNO, OTA5; 27) Archaeodictyomitra sp. aff. minoensis"
    )
    by_label = {p.labels[0]: p for p in pairs}
    assert by_label["3"].species == "Orbiculiforma sp. aff. mclaughlini"
    assert by_label["27"].species == "Archaeodictyomitra sp. aff. minoensis"
    # Negative check: the bare " sp." (no aff./cf. tail) must still
    # parse correctly when the species has no aff. form.
    assert by_label["1"].species == "Ristola altissima"


def test_caption_clause_sp_aff_tail():
    """The same sp. cf./aff. tail must work for the "fig N. Species"
    form (CAPTION_CLAUSE_RE path). Mirrors test_danelian_sp_aff_epithet_tail
    for the inverse shape used by the Baumgartner/fig-style papers."""
    pairs = _regex_parse_caption(
        "fig 3 Orbiculiforma sp. aff. mclaughlini; fig 27 "
        "Archaeodictyomitra sp. aff. minoensis"
    )
    assert len(pairs) == 2
    by_label = {p.labels[0]: p for p in pairs}
    assert by_label["3"].species == "Orbiculiforma sp. aff. mclaughlini"
    assert by_label["27"].species == "Archaeodictyomitra sp. aff. minoensis"
