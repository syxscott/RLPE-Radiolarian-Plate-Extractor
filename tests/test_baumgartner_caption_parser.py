"""Tests for the Baumgartner-style caption parser added in commit (TBD).

Handles the "1, 2- Species; 3- Species" convention used in Baumgartner
2008 (IRIS) and other Mesozoic radiolarian papers. Without this fallback
the parser returns 0 pairs for an entire paper and the order-based
fallback tags every panel with taxa[0] from the truncated caption
preamble.
"""
from __future__ import annotations

from rlpe.m3_engine import _regex_parse_caption


def test_baumgartner_plate1_full_caption():
    caption = (
        "Plate 1 - Middle and Upper Jurassic Radiolaria from the Siuna "
        "serpentinite mélange (NE Nicaragua) scale bar = 100µm for all "
        "figures. 1-7) Sample 0501-21-03, Middle Jurassic (UAZ 4-6) red "
        "radiolarite associated with greenstones. 8-13) Sample 05-01-16-02, "
        "Upper Jurassic (UAZ 9-11) black chert. 1, 2- Williriedellum "
        "marcucciae Cortese, UAZ 4-8; 3- Williriedellum sp. S (= "
        "Tricolocapsa sp. S, sensu Baumgartner et al., 1995a), UAZ 4-5 "
        "(4-6 after Prela et al., 2000); 4- Williriedellum sp. cf. W. sp. S "
        "(= Tricolocapsa sp. S, sensu Baumgartner et al., 1995a); 5- "
        "Linaresia sp. cf. L. chrafatensis (El Kadiri); 6, 7- Zhamoidellum "
        "sp.; 8, 9- Xitus spp.; 10- Pseudodictyomitra primitiva Matsuoka "
        "and Yao, UAZ 7-12; 11- Archaeodictyomitra (Mizutani), UAZ 9-12; "
        "12- Mirifusus dianae s. l. (Karrer), UAZ 9-20; 13- Sethocapsa "
        "sp. cf. S. dorysphaeroides Neviani, sensu Schaaf."
    )
    pairs = _regex_parse_caption(caption)
    by_label = {}
    for p in pairs:
        for lbl in p.labels:
            by_label[lbl] = p.species
    assert by_label.get("1") == "Williriedellum marcucciae"
    assert by_label.get("2") == "Williriedellum marcucciae"
    assert by_label.get("3") == "Williriedellum sp. S" or "Williriedellum sp" in by_label.get("3", "")
    assert by_label.get("6") == "Zhamoidellum sp."
    assert by_label.get("7") == "Zhamoidellum sp."
    assert by_label.get("10") == "Pseudodictyomitra primitiva"
    assert by_label.get("12") == "Mirifusus dianae"
    assert by_label.get("13") == "Sethocapsa sp. cf. S. dorysphaeroides"
    # Plate preamble ("Radiolaria from", "Siuna serpentinite") must NOT
    # be matched as species — those words appear in the title, not the
    # species clauses.
    assert "Radiolaria" not in by_label.values()
    assert "serpentinite" not in by_label.values()


def test_baumgartner_single_clause():
    """Single label + single species, semicolon-separated."""
    caption = "1- Triactoma jonesi; 2- Triactoma blakei; 3- Triactoma sp."
    pairs = _regex_parse_caption(caption)
    by_label = {lbl: p.species for p in pairs for lbl in p.labels}
    assert by_label.get("1") == "Triactoma jonesi"
    assert by_label.get("2") == "Triactoma blakei"
    assert "sp" in by_label.get("3", "")


def test_baumgartner_does_not_match_prose_numbers():
    """The boundary constraint prevents the regex from matching the
    "1" inside "100 µm for all figures" or similar prose."""
    caption = "Scale bar = 100 µm for all illustrations. 1- Triactoma sp."
    pairs = _regex_parse_caption(caption)
    by_label = {lbl: p.species for p in pairs for lbl in p.labels}
    # The first match would be the "1" inside "100 µm" if the boundary
    # were missing; with (?<![\dA-Za-z]) it must be "1" + dash + species.
    assert by_label.get("1") == "Triactoma sp." or "Triactoma" in by_label.get("1", "")


def test_baumgartner_returns_no_pairs_for_empty_caption():
    assert _regex_parse_caption("") == []
    assert _regex_parse_caption(None) == []  # type: ignore[arg-type]


def test_baumgartner_genus_only_with_author_citation():
    """Some panels are listed as "Genus (Author)" with no epithet —
    e.g. Baumgartner 2008 plate 1 panel 11: "Archaeodictyomitra
    (Mizutani)". The genus-only shape must be accepted when
    immediately followed by an author citation, and rejected
    otherwise (e.g. preamble "Plate 1 - Middle" must not be
    captured as a "Middle" species)."""
    # Genus + author citation — should match.
    pairs = _regex_parse_caption("1- Archaeodictyomitra (Mizutani)")
    assert len(pairs) == 1
    assert pairs[0].labels == ["1"]
    assert pairs[0].species == "Archaeodictyomitra"

    # Genus without author citation — must NOT match.
    pairs = _regex_parse_caption("1- Middle Jurassic red chert")
    by_label = {lbl: p.species for p in pairs for lbl in p.labels}
    assert "Middle" not in by_label.get("1", ""), (
        "genus-only without author citation must not match prose"
    )

    # Preamble "Plate 1 - Middle and Upper..." — must not match.
    pairs = _regex_parse_caption(
        "Plate 1 - Middle and Upper Jurassic Radiolaria. 1- Triactoma jonesi"
    )
    by_label = {lbl: p.species for p in pairs for lbl in p.labels}
    assert "Middle" not in by_label.values()
    assert by_label.get("1") == "Triactoma jonesi"


def test_baumgartner_range_labels():
    """Numeric range "8-10- Species" must expand to 8, 9, 10.
    The dash BEFORE the species is a separator; the dash INSIDE the
    label-list is part of the range. This is the baum pl02 pattern
    ("8-10- Zhamoidellum spp.") which was previously captured as
    just ["10"] and dropped panels 8 and 9."""
    pairs = _regex_parse_caption("1, 2- Triactoma; 8-10- Zhamoidellum spp.; 16-17- Sethocapsa spp.")
    by_label = {}
    for p in pairs:
        for lbl in p.labels:
            by_label[lbl] = p.species
    assert by_label.get("1") == "Triactoma"
    assert by_label.get("2") == "Triactoma"
    assert by_label.get("8") == "Zhamoidellum spp."
    assert by_label.get("9") == "Zhamoidellum spp."
    assert by_label.get("10") == "Zhamoidellum spp."
    assert by_label.get("16") == "Sethocapsa spp."
    assert by_label.get("17") == "Sethocapsa spp."


def test_baumgartner_no_space_between_label_and_species():
    """The caption "7Williriedellum sp." (no space between label and
    species) must still match — the dash-separator is made optional
    so a tight-set caption with zero gap between "7" and the genus
    is captured. This is the baum pl02 panel 7 pattern."""
    pairs = _regex_parse_caption("7Williriedellum sp.; 8- Triactoma jonesi")
    by_label = {lbl: p.species for p in pairs for lbl in p.labels}
    assert by_label.get("7") == "Williriedellum sp."
    assert by_label.get("8") == "Triactoma jonesi"


def test_baumgartner_uncertainty_marker_after_genus():
    """Species with a "(?)" uncertainty marker between genus and
    epithet — "Stichomitra (?) sp.", "Acaeniotyle (?) sp.",
    "Hiscocapsa (?) sp." — must be captured. The "(?)" is a
    genus-level marker; the post-parse _normalize_species pass
    strips it so the captured species matches the gold convention
    (which omits the uncertainty marker)."""
    pairs = _regex_parse_caption(
        "1- Stichomitra (?) sp. cf. S. (?) acuta; "
        "2- Acaeniotyle (?) sp.; "
        "3- Hiscocapsa (?) sp."
    )
    by_label = {lbl: p.species for p in pairs for lbl in p.labels}
    assert by_label.get("1") == "Stichomitra sp. cf. S. acuta"
    assert by_label.get("2") == "Acaeniotyle sp."
    assert by_label.get("3") == "Hiscocapsa sp."


def test_baumgartner_trailing_single_letter_species_identifier():
    """One-letter species identifiers like "S" in "Williriedellum sp. S"
    or "W. sp. S" must be preserved in the captured species — without
    this, the eval fails to match gold that records the "sp. S"
    identifier (baum2008 panels 3, 4). The trailing identifier is
    " <uppercase_letter>" optionally preceded by "." for the
    cf./aff. shape (". S" after the cf.-epithet)."""
    pairs = _regex_parse_caption(
        "3- Williriedellum sp. S (= Tricolocapsa sp. S, sensu); "
        "4- Williriedellum sp. cf. W. sp. S (= Tricolocapsa sp. S)"
    )
    by_label = {lbl: p.species for p in pairs for lbl in p.labels}
    # Panel 3: trailing "S" after "sp."
    assert by_label.get("3") == "Williriedellum sp. S"
    # Panel 4: trailing "S" after "W. sp." (dot, space, S)
    assert by_label.get("4") == "Williriedellum sp. cf. W. sp. S"
