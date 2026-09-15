"""Fixes for the audit round (2026-09-15): abbreviation-expansion
verification, chart-figure routing, systematic backfill of header-only
plate captions, and the 4th unit->age prose shape.

Pins the three audit findings:
  #8  Hernandez-Almeida FAIL — "A. setosa" expanded to the real-but-wrong
      genus "Acanthodesmia setosa" on an abundance-chart figure.
  #9  Suzuki & Gawlick P3 — OD detached the 50-item list from the caption
      header; the systematic section carries the mapping as inline refs.
  #10 Ozkan UAZ 9 — "dated this as middle Oxfordian–Tithonian (UAZ 9 to
      11–12; Late Jurassic)" matched no unit->age shape, so rows fell
      back to the wrong paper-level age.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ============================================================
# Fix 1 — genus-support verification
# ============================================================
def test_expanded_genus_rejected_when_absent_from_context():
    from rlpe.semantic_engine import species_supported_by_text

    cap = (
        "Fig. 10. Abundance of A. setosa in cores in the high-latitude "
        "North Pacific and marginal seas, between MIS 6 and 4. (A) IODP "
        "Site U1417 (Matsuzaki and Suzuki, 2017)."
    )
    assert not species_supported_by_text("Acanthodesmia setosa", cap)


def test_full_genus_in_context_passes():
    from rlpe.semantic_engine import species_supported_by_text

    cap = "Plate 1. 1. Glomeropyle algidum Neyman; 2. Glomeropyle cuneum Neyman."
    assert species_supported_by_text("Glomeropyle algidum", cap)
    # abbreviated caption + full genus elsewhere in the SAME context
    cap2 = (
        "Fig. 5. 14. A. umbilicata (Rüst, 1898).\n\n"
        "Acaeniotyle umbilicata (Rüst, 1898). Figs 5.3–4."
    )
    assert species_supported_by_text("Acaeniotyle umbilicata", cap2)


def test_open_nomenclature_and_bare_genus():
    from rlpe.semantic_engine import species_supported_by_text

    assert species_supported_by_text("gen. et sp. indet. A", "anything")
    assert species_supported_by_text("Parvicingula sp.", "Plate 2. 1. Parvicingula sp.")
    # bare genus absent from context → rejected
    assert not species_supported_by_text("Nagrum", "Plate 1. 1. A. setosa")


def test_pdf_ligature_and_linebreak_normalization():
    from rlpe.semantic_engine import species_supported_by_text

    cap = "1. Williriedellum dierschei Suzuki and Gawlick, 2004"
    assert species_supported_by_text("Williriedellum dierschei", cap)
    cap2 = "1. Williriedel-\nlum crystallinum Dumitrica, 1970"
    assert species_supported_by_text("Williriedellum crystallinum", cap2)


def test_stringified_null_species_rejected():
    """38fd148 follow-up: literal 'None'/'null' strings reaching the
    species field are never taxa."""
    from rlpe.semantic_engine import _species_candidate_rejected

    assert _species_candidate_rejected("None") == "stringified_null"
    assert _species_candidate_rejected("null") == "stringified_null"
    assert _species_candidate_rejected("Glomeropyle algidum") is None


# ============================================================
# Fix 2 — chart captions classify as diagram
# ============================================================
def test_abundance_chart_classifies_as_diagram():
    from rlpe.range_chart_extractor import classify_figure_type

    cap = (
        "Fig. 10. Abundance of A. setosa in cores in the high-latitude "
        "North Pacific and marginal seas, between MIS 6 and 4. (A) IODP "
        "Site U1417 (Matsuzaki and Suzuki, 2017). (B) ODP Site 883."
    )
    assert classify_figure_type(cap) == "diagram"


def test_real_plate_still_classifies_as_plate():
    from rlpe.range_chart_extractor import classify_figure_type

    assert (
        classify_figure_type(
            "Plate 1. Scanning electron micrographs of radiolarians. "
            "1. Williriedellum dierschei; 2. Archaeodictyomitra apiarium."
        )
        == "plate"
    )


def test_species_clause_override_still_works_without_chart_words():
    from rlpe.range_chart_extractor import classify_figure_type

    cap = (
        "Taxonomic list. Dictyomitra montisserei Plate 1, Figures 7-8, "
        "11-12. Archaeodictyomitra spp. Plate 1, Figures 2, 6."
    )
    assert classify_figure_type(cap) == "plate"


# ============================================================
# Fix 3 — systematic backfill of header-only captions
# ============================================================
def _kids_with(plate_caption: str, systematic_paragraphs: list[str]) -> list[dict]:
    kids: list[dict] = [
        {"type": "caption", "content": plate_caption, "page number": 34, "kids": []}
    ]
    for t in systematic_paragraphs:
        kids.append({"type": "paragraph", "content": t, "page number": 20, "kids": []})
    return kids


def test_header_only_plate_caption_backfilled_from_systematic_section():
    from rlpe.opendataloader_extractor import _find_plate_captions

    header = (
        "Plate 3. Scanning electron micrographs of radiolarians from the "
        "samples D1052 (1–6), EW146 (7–17) and D1025 (18–45), basal "
        "horizons of the Fludergraben section, Austria. A 50 µm scale bar "
        "applies to all photos."
    )
    kids = _kids_with(
        header,
        [
            "Tetracapsa sp. A sensu Suzuki and Gawlick, 2003b (Plate 3, figs. 1, 32)",
            "Saitoum pagei Pessagno, 1977 (Plate 3, figs. 18, 19)",
            "Parvicingula spinata Vinassa, 1899 (Plate 3, fig. 13)",
            "Droltus galerus Suzuki, 1995b (Plate 3, fig. 49)",
            "Archaeodictyomitra sixi Yang, 1993 (Plate 3, fig. 50)",
        ],
    )
    found = _find_plate_captions(kids, caption_window=2)
    p3 = [f for f in found if f["plate_number"] == 3]
    assert p3, "plate 3 caption must exist"
    content = p3[0]["content"]
    assert "systematic backfill" in content
    assert "1. Tetracapsa sp. A" in content
    assert "13. Parvicingula spinata" in content
    assert "50. Archaeodictyomitra sixi" in content


def test_complete_caption_not_backfilled():
    from rlpe.opendataloader_extractor import _find_plate_captions

    header = (
        "Plate 2. Scanning electron micrographs.\n"
        "1. Archaeospongoprunum cortinaense Wu, 1993\n"
        "2. Parahsuum sp. S sensu Matsuoka, 1986\n"
        "3. Cinguloturris carpatica Dumitrica, 1982"
    )
    kids = _kids_with(
        header,
        [
            "Archaeospongoprunum cortinaense Wu (Plate 2, fig. 1)",
            "Parahsuum sp. S (Plate 2, fig. 2)",
            "Cinguloturris carpatica (Plate 2, fig. 3)",
        ],
    )
    found = _find_plate_captions(kids, caption_window=2)
    p2 = [f for f in found if f["plate_number"] == 2]
    assert p2 and "systematic backfill" not in p2[0]["content"]


def test_fig_numbers_from_ref():
    from rlpe.opendataloader_extractor import _fig_numbers_from_ref

    assert _fig_numbers_from_ref("(Plate 3, figs. 1, 32)") == [1, 32]
    assert _fig_numbers_from_ref("Pl. 1, figs 5–7") == [5, 6, 7]
    assert _fig_numbers_from_ref("(Plate 2, figure 7)") == [7]
    assert _fig_numbers_from_ref("no figs") == []


# ============================================================
# Fix 4 — 4th unit→age prose shape ("dated ... as ... (UAZ 9 to 11–12; X)")
# ============================================================
def test_dated_shape_ozkan_sentence():
    from rlpe.geology_extraction import extract_unit_age_map_regex

    sections = [
        {
            "title": "Geological evolution",
            "text": (
                "The presence of Parvicingula sp. and Svinitzium sp. in "
                "sample MET-267 dated this as middle Oxfordian–Tithonian "
                "(UAZ 9 to 11–12; Late Jurassic) (see Baumgartner et al., "
                "1995)."
            ),
        }
    ]
    out = extract_unit_age_map_regex(sections, {"UAZ 9"})
    assert "UAZ 9" in out
    assert out["UAZ 9"]["age_text"] == "middle Oxfordian–Tithonian"


def test_dated_shape_no_range_and_unwanted_filtered():
    from rlpe.geology_extraction import extract_unit_age_map_regex

    sections = [
        {
            "title": "x",
            "text": "The assemblage is dated as late Bathonian (UAZ 7; late Middle Jurassic).",
        }
    ]
    out = extract_unit_age_map_regex(sections, {"UAZ 7"})
    assert out["UAZ 7"]["age_text"] == "late Bathonian"
    # unwanted unit → dropped
    assert extract_unit_age_map_regex(sections, {"UAZ 1"}) == {}


def test_stage_phrase_allows_qualifierless_join():
    """Pre-existing bug: 'Oxfordian–Tithonian' (no early/late qualifier on
    the second stage) never matched the dash-joined continuation."""
    from rlpe.geology_extraction import _STAGE_PHRASE_RE

    assert _STAGE_PHRASE_RE.fullmatch("Oxfordian–Tithonian")
    assert _STAGE_PHRASE_RE.fullmatch("Bathonian-early Callovian")
    assert _STAGE_PHRASE_RE.fullmatch("middle Oxfordian–Tithonian")


def test_prompt_documents_dated_shape():
    from rlpe.geology_extraction import build_unit_resolution_prompt

    sys_p, _u = build_unit_resolution_prompt(None, {"UAZ 9"}, None)
    assert "dated" in sys_p and "UAZ 9" in sys_p
