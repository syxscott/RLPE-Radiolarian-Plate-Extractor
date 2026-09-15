"""Fixes for the external-review round (2026-09-15).

Two reviewers read six Bragin/Bragina 2020 papers against the extracted
database and confirmed systemic metadata faults. Each test pins one root
cause fixed in this round:

  * caption-first geology link priority (Cyprus/Perapedhi setting text
    contaminating West Serbia rows)
  * chrono/Ma derived from the SAME age mention as the record's age field
    (age="upper Albian" + chronostratigraphy="Maastrichtian" contradictions)
  * no more country-centroid coordinates
  * paper-level geology anchor (Bragin 2020 Omolon all-NaN rows)
  * junk-species gate expansion ("Struganik limestone", "Legend same")
  * truncation fixes (hyphenated epithets, "(?)" markers, "nov. sp.",
    "?" uncertainty preserved)
  * synonymy-citation guard in the systematic-section harvest
    (Dumitricaia maxwellensis minted onto Plate 4 fig. 10)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ============================================================
# ages_consistent + anchor filter
# ============================================================
def test_ages_consistent_basic():
    from rlpe.geology_extraction import ages_consistent

    assert ages_consistent("upper Albian", "late Albian")
    assert ages_consistent("Upper Santonian", "Santonian")
    # Boundary-adjacent stages overlap at 83.6.
    assert ages_consistent("Santonian", "Campanian")
    assert not ages_consistent("Upper Albian", "Maastrichtian")
    assert not ages_consistent("Upper Triassic", "Upper Santonian")
    assert not ages_consistent("Pliensbachian", "Cenomanian")
    # Unclassifiable / missing → consistent (gate never guesses).
    assert ages_consistent(None, "Albian")
    assert ages_consistent("some unknown words", "Albian")


def test_filter_links_consistent_with_anchor():
    from rlpe.geology_extraction import filter_links_consistent_with_anchor

    links = [
        {"age": "Upper Triassic", "locality": "Cyprus"},
        {"age": "upper Santonian", "locality": "Struganik"},
        {"formation": "Perapedhi Formation"},  # no age → kept
    ]
    kept = filter_links_consistent_with_anchor(links, "Upper Santonian")
    assert len(kept) == 2
    assert all(gl.get("age") != "Upper Triassic" for gl in kept)
    assert filter_links_consistent_with_anchor(links, None) == links


# ============================================================
# Chrono/Ma from the SAME age mention (field consistency)
# ============================================================
def test_chrono_follows_record_age_not_most_specific_mention():
    from rlpe.geology_extraction import extract_geology_from_sections

    # A setting section whose headline is "upper Albian" but which also
    # contains a stage-rank mention ("Campanian–Maastrichtian") further in —
    # the old most-specific heuristic let Maastrichtian win the chrono and
    # Ma fields while age stayed Albian.
    sec = {
        "title": "Geological setting",
        "section_type": "geological_setting",
        "text": (
            "Upper Albian radiolarians were recovered from black calcareous "
            "clays of the sections near Mar'ino. The sequence is dated as "
            "upper Albian. Siliciclastic flysch of Campanian-Maastrichtian "
            "age overlies the section unconformably."
        ),
    }
    recs = extract_geology_from_sections([sec])
    assert recs, "expected at least one geology record"
    rec = recs[0].to_dict()
    age, chrono = rec.get("age"), rec.get("chronostratigraphy")
    ma_top, ma_base = rec.get("ma_top"), rec.get("ma_base")
    assert age and "lbian" in str(age).lower(), f"age={age!r}"
    if chrono:
        assert "lbian" in str(chrono).lower(), (
            f"chrono {chrono!r} must describe the record's own age {age!r}"
        )
        assert ma_top == 100.5 and ma_base == 113.0, (ma_top, ma_base)


# ============================================================
# No country-centroid coordinates
# ============================================================
def test_no_country_centroid_coordinates():
    from rlpe.geology_extraction import extract_geology_from_sections

    sec = {
        "title": "Material",
        "section_type": "geological_setting",
        "text": (
            "Radiolarians were extracted from samples collected in Russia near the Omolon Massif."
        ),
    }
    recs = extract_geology_from_sections([sec])
    assert recs
    for rec in recs:
        d = rec.to_dict()
        assert d.get("coord_source") != "country_centroid"
        # Centroid values must never be filled from the lookup table.
        if d.get("latitude") is not None:
            assert (d.get("latitude"), d.get("longitude")) != (60.0, 100.0)


# ============================================================
# Junk-species gate expansion
# ============================================================
def test_junk_species_gate_expansion():
    from rlpe.semantic_engine import _species_candidate_rejected as rej

    assert rej("Struganik limestone") is not None
    assert rej("Legend same") is not None
    assert rej("Correlation of") is not None
    assert rej("When only") is not None
    assert rej("Field photographs") is not None
    assert rej("Cyclastrum sp. aff") is not None  # truncated open nomenclature
    # Real taxa stay untouched.
    assert rej("Afens perapediensis") is None
    assert rej("Drotus annosus") is None
    assert rej("Crolanium triangulare") is None
    assert rej("Savaryella? nikishini nov. sp.") is None
    assert rej("Cyclastrum sp. aff. C. trigonum") is None


# ============================================================
# Caption parse: "?" kept, nov. sp. kept, "(?)" accepted
# ============================================================
def test_composite_parser_keeps_uncertainty_and_nov_sp():
    from rlpe.semantic_engine import _parse_composite_caption

    cap = (
        "Plate 6. 1, 2. Cavaspongia euganea (Squinabol); 3. Crucella messinae "
        "Pessagno. 4, 5. Savaryella? nikishini nov. sp.; 6. Hexapyramis sp.; "
        "17–Noritus? sp."
    )
    pairs = _parse_composite_caption(cap)
    by_species = {p.species: p for p in pairs}
    assert any("Savaryella?" in s and "nov. sp." in s for s in by_species), by_species.keys()
    assert any("Noritus? sp." in s for s in by_species), by_species.keys()
    # No doubled modifier from the two tail matchers.
    assert not any("sp. sp" in s for s in by_species)


def test_clause_regex_accepts_parenthesised_question_genus():
    from rlpe.semantic_engine import _regex_parse_caption

    pairs = _regex_parse_caption("Plate 6. 6. Hexapyramis(?) sp. cf. H. perforatum")
    assert pairs, "clause regex must match Hexapyramis(?)"
    assert any("Hexapyramis" in p.species for p in pairs)


def test_parse_caption_joins_hyphenated_line_break():
    from rlpe.semantic_engine import _regex_parse_caption

    pairs = _regex_parse_caption("Plate III. (1) Afens perapedien-\nsis Bragina; (2) Tubilustrella")
    assert any("perapediensis" in p.species for p in pairs), [p.species for p in pairs]


# ============================================================
# Synonymy-citation guard in the inline-ref harvest
# ============================================================
def test_harvest_skips_year_prefixed_synonymy_refs():
    from rlpe.opendataloader_extractor import _harvest_inline_plate_refs

    kids = [
        {
            "type": "paragraph",
            "page number": 8,
            "content": (
                "Cavaspongia euganea (Squinabol). Synonymy: 1976 Dumitricaia "
                "maxwellensis nov. sp. - Pessagno, pl. 4, figs. 10, 11. "
                "Remarks: close to Becus regius."
            ),
        }
    ]
    out = _harvest_inline_plate_refs(kids)
    plate4 = out.get(4, [])
    assert not any("Dumitricaia" in sp for sp, _ref, _pg in plate4), plate4


def test_harvest_kept_for_own_paper_descriptions():
    from rlpe.opendataloader_extractor import _harvest_inline_plate_refs

    kids = [
        {
            "type": "paragraph",
            "page number": 9,
            "content": (
                "Becus naidini nov. sp. (Plate 4, figs. 6-10). Holotype in "
                "sample 4870/285. Becus regius Lipman (Plate 4, fig. 3)."
            ),
        }
    ]
    out = _harvest_inline_plate_refs(kids)
    plate4 = out.get(4, [])
    assert any("naidini" in sp and "nov" in sp for sp, _ref, _pg in plate4), plate4


# ============================================================
# Caption-first link priority in _enrich_llm_first_results
# ============================================================
def _bare_pipeline():
    from rlpe.pipeline import RadiolarianPipeline

    return object.__new__(RadiolarianPipeline)


def _caption_rec(text: str):
    from rlpe.types import CaptionRecord

    return CaptionRecord(
        paper_id="p",
        figure_id="f1",
        caption=text,
        entities=[],
        figure_number="1",
        page_index=5,
        panel_labels=[],
        source_xml=None,
    )


def test_caption_link_outranks_setting_links():
    pipe = _bare_pipeline()
    rows = [
        {"panel_id": "1", "species": "Alievium gallowayi", "metadata": {}},
        {"panel_id": "2", "species": "Pseudoaulophacus floresensis", "metadata": {}},
    ]
    section_links = {
        "Alievium gallowayi": [
            {
                "age": "Upper Triassic",
                "formation": "Perapedhi Formation",
                "locality": "Cyprus",
            }
        ],
        "Pseudoaulophacus floresensis": [],
    }
    out = pipe._enrich_llm_first_results(
        rows,
        caption=_caption_rec(
            "Plate I. Upper Santonian radiolarians from the Petrovića Brdo "
            "section (Serbia). (1) Alievium gallowayi; (2) Pseudoaulophacus "
            "floresensis."
        ),
        region_img=None,
        section_links=section_links,
        grobid_sections=[],
    )
    md0 = out[0]["metadata"]
    ages = [gl.get("age") for gl in md0["geology_links"]]
    assert "Upper Triassic" not in ages, md0["geology_links"]
    assert any("Santonian" in str(a) for a in ages if a), md0["geology_links"]
    # Second row falls back to the same caption (species has no setting links).
    ages1 = [gl.get("age") for gl in out[1]["metadata"]["geology_links"]]
    assert any("Santonian" in str(a) for a in ages1 if a)


# ============================================================
# Paper-level geology anchor
# ============================================================
def test_paper_anchor_fills_unlinked_rows():
    pipe = _bare_pipeline()
    rows = [
        {"paper_id": "p", "panel_id": "1", "species": "Praenanina? hirsuta", "metadata": {}},
        {
            "paper_id": "p",
            "panel_id": "2",
            "species": "Nodotrisphaera ossispina",
            "metadata": {"geology_links": [{"age": "Carnian", "locality": "x"}]},
        },
    ]
    sections = [
        {
            "title": "Abstract",
            "section_type": "abstract",
            "text": (
                "Late Ladinian to Early Carnian radiolarians were recovered "
                "from black siliceous shales of the Pravyi Vodopadnyi Creek "
                "section, Omolon Massif, Northeastern Russia."
            ),
        },
        {
            "title": "Introduction",
            "section_type": "geological_setting",
            "text": (
                "The Omolon Massif section exposes Upper Triassic to Jurassic "
                "deposits. Radiolarians of the late Ladinian-early Carnian "
                "were found in bed 4."
            ),
        },
    ]
    out = pipe._attach_paper_geology_anchor(rows, "p", sections)
    md0 = out[0]["metadata"]
    assert md0["geology_links"], "empty row must inherit the paper anchor"
    link0 = md0["geology_links"][0]
    assert link0["section_type"] == "paper_anchor"
    assert link0.get("age"), link0
    # Linked rows untouched.
    assert out[1]["metadata"]["geology_links"][0]["age"] == "Carnian"


def test_paper_anchor_noop_when_all_rows_linked():
    pipe = _bare_pipeline()
    rows = [
        {
            "paper_id": "p",
            "panel_id": "1",
            "species": "x",
            "metadata": {"geology_links": [{"age": "Albian"}]},
        }
    ]
    out = pipe._attach_paper_geology_anchor(rows, "p", [])
    assert out[0]["metadata"]["geology_links"] == [{"age": "Albian"}]
