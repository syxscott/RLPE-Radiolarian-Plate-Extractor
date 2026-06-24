"""Regression tests for Pouille-style "Species (Pl. N, figs M)" caption parsing.

The OpenDataLoader pass can't find a real "Plate N" header in Pouille
2014 (the paper has none — the species live in the systematic
paleontology descriptions), so the synthesis pass builds a caption
shaped like

  "Plate 1. (Reconstructed from systematic descriptions)
   Syntagentactinia biocculosa (Pl. 1, figs 5)
   Syntagentactinia? angulata n. sp. (Pl. 1, figs 12–14b)
   Polyentactinia spinulenta n. sp. (Pl. 1, figs 8–11) ..."

with the species name FIRST and the (Pl. N, figs M) reference SECOND.
The pre-existing _CAPTION_CLAUSE_RE only matches the inverse "Fig. N.
Species" form, so without this pass the parser returned zero pairs
and the order-based fallback tagged every panel with taxa[0] (one
species per plate, instead of one per fig range).
"""

from __future__ import annotations

from rlpe.m3_engine import _regex_parse_caption


def test_pouille_species_before_label_pattern():
    text = (
        "Plate 1. (Reconstructed from systematic descriptions)\n"
        "Syntagentactinia biocculosa (Pl. 1, figs 5)\n"
        "Syntagentactinia? angulata n. sp. (Pl. 1, figs 12–14b)\n"
        "Polyentactinia spinulenta n. sp. (Pl. 1, figs 8–11)\n"
        "Haplentactinia juncta (Pl. 1, fig. 1)"
    )
    pairs = _regex_parse_caption(text)
    # 4 distinct species on plate 1.
    assert len(pairs) == 4
    # First pair: single fig, plain binomial, no modifier.
    p0 = pairs[0]
    assert p0.species == "Syntagentactinia biocculosa"
    assert p0.modifier == ""
    assert p0.labels == ["5"]
    # Second pair: uncertainty marker "?" + modifier "n. sp.",
    # range "12-14b" expanded into ["12", "13", "14b"] (suffix on last).
    p1 = pairs[1]
    assert p1.species == "Syntagentactinia? angulata"
    assert p1.modifier == "n. sp."
    assert p1.labels == ["12", "13", "14b"]
    # Third pair: range "8-11" expanded into ["8", "9", "10", "11"].
    p2 = pairs[2]
    assert p2.species == "Polyentactinia spinulenta"
    assert p2.labels == ["8", "9", "10", "11"]
    # Fourth pair: "fig." (singular) with single label.
    p3 = pairs[3]
    assert p3.species == "Haplentactinia juncta"
    assert p3.labels == ["1"]


def test_pouille_synthetic_header_is_ignored():
    text = (
        "Plate 1. (Reconstructed from systematic descriptions)\n"
        "Haplentactinia juncta (Pl. 1, fig. 1)"
    )
    pairs = _regex_parse_caption(text)
    # The synthetic header line should not be picked up as a species pair.
    assert len(pairs) == 1
    assert pairs[0].species == "Haplentactinia juncta"


def test_inverse_pattern_still_works():
    """The original "Fig. N. Species" pattern must still parse."""
    text = (
        "Fig. 1. Entactinia itsukichiensis: test spherical with cortical shell.\n"
        "Figs 2-3. Trilonche crassispinosa Sashida & Tonishi: spicule robust.\n"
    )
    pairs = _regex_parse_caption(text)
    species = [p.species for p in pairs]
    assert "Entactinia itsukichiensis" in species
    assert "Trilonche crassispinosa" in species


def test_pouille_label_dedup_against_existing_pairs():
    """If a label range overlaps an earlier pair, skip it (Pouille 2014
    has only one fig per species, so the second occurrence of "fig 1"
    in the same caption should not overwrite the first assignment)."""
    text = (
        "Syntagentactinia biocculosa (Pl. 1, fig. 1)\n"
        "Haplentactinia juncta (Pl. 1, fig. 1)"  # duplicate label
    )
    pairs = _regex_parse_caption(text)
    # First occurrence wins.
    assert len(pairs) == 1
    assert pairs[0].species == "Syntagentactinia biocculosa"


def test_base_label_alias_rescues_ocr_misread():
    """A pair with label "14b" should also index under bare "14" so
    that an OCR that misread "14b" as "14" still maps to the right
    species. This is the _add_label_base_aliases helper in
    association.py, exercised here through match_panels."""
    from rlpe.association import PanelCandidate, match_panels
    from rlpe.types import CaptionRecord

    cap = CaptionRecord(
        paper_id="t",
        figure_id="f1",
        caption=("Plate 1.\nSyntagentactinia? angulata n. sp. (Pl. 1, figs 12–14b)"),
        entities=[],
        figure_number="1",
        page_index=1,
        panel_labels=[],
        source_xml=None,
    )
    panels = [
        PanelCandidate(panel_id=str(i), bbox=(0, 0, 100, 100), score=0.5) for i in range(12, 15)
    ]
    matches = match_panels(
        paper_id="t",
        figure_id="f1",
        caption=cap,
        panels=panels,
        ocr_tokens=[],
        taxon_entities=[],
        caption_pairs=None,
    )
    by_id = {m.panel_id: m.species for m in matches}
    # All three panels — 12, 13, 14 — should map to the species,
    # even though the caption key is "14b" and the OCR read "14".
    assert by_id["12"] == "Syntagentactinia? angulata"
    assert by_id["13"] == "Syntagentactinia? angulata"
    assert by_id["14"] == "Syntagentactinia? angulata"
