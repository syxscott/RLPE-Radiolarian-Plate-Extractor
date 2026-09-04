import sys

sys.path.insert(0, "scripts")
from post_process import (
    dedup_panels,
    filter_low_confidence,
    normalize_panel_id,
    parse_open_nomenclature,
)


def test_parse_open_nomenclature_cf():
    # Audit 2026-09-04 taxon-8: parse_open_nomenclature preserves the
    # species string verbatim and surfaces the qualifier label only.
    # The previous assertion ``sp == "Genus species"`` pinned the
    # pseudo-trinomial corruption that fused "cf. species" into the
    # canonical binomial.
    sp, qual = parse_open_nomenclature("Genus cf. species")
    assert sp == "Genus cf. species"
    assert qual == "cf."


def test_parse_open_nomenclature_aff():
    sp, qual = parse_open_nomenclature("Genus aff. species")
    assert qual == "aff."


def test_parse_open_nomenclature_no_qualifier():
    sp, qual = parse_open_nomenclature("Genus species")
    assert qual is None


def test_dedup_panels_removes_duplicates():
    panels = [
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "1",
            "species": "A sp",
            "confidence": 0.9,
        },
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "1",
            "species": "A sp",
            "confidence": 0.85,
        },
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "2",
            "species": "B sp",
            "confidence": 0.9,
        },
    ]
    deduped = dedup_panels(panels)
    assert len(deduped) == 2
    # Higher confidence kept
    assert deduped[0]["confidence"] == 0.9


def test_filter_low_confidence():
    panels = [
        {"confidence": 0.5, "species": "A"},
        {"confidence": 0.85, "species": "B"},
        {"confidence": 0.71, "species": "C"},
    ]
    filtered = filter_low_confidence(panels, threshold=0.7)
    assert len(filtered) == 2
    assert all(p["confidence"] >= 0.7 for p in filtered)


def test_normalize_panel_id_strips_fig():
    assert normalize_panel_id("Fig. 1") == "1"
    assert normalize_panel_id("Figs. 24 and 25") == "24 and 25"
    assert normalize_panel_id("1") == "1"


def test_normalize_panel_id_strips_plate():
    """'Plate N' / 'Plates N' / 'Pl. N' all normalize to just 'N'."""
    assert normalize_panel_id("Plate 3") == "3"
    assert normalize_panel_id("Plates 3") == "3"
    assert normalize_panel_id("Plate 3.5") == "3.5"
    # Reorder: 'Pl' should still match (no false positive for 'Plate')
    assert normalize_panel_id("Pl. 3") == "3"
    assert normalize_panel_id("Pl 3") == "3"


def test_parse_open_nomenclature_whitespace():
    """Pure-whitespace input returns (None, None), not the original whitespace."""
    sp, qual = parse_open_nomenclature("   ")
    assert sp is None
    assert qual is None


def test_dedup_panels_tie_keeps_first():
    """Same key + same confidence → first encountered row is kept (stable)."""
    panels = [
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "1",
            "species": "A sp",
            "confidence": 0.5,
            "marker": "first",
        },
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "1",
            "species": "A sp",
            "confidence": 0.5,
            "marker": "second",
        },
    ]
    deduped = dedup_panels(panels)
    assert len(deduped) == 1
    assert deduped[0]["marker"] == "first"


def test_filter_low_confidence_string_confidence():
    """String confidence values (e.g. '0.8') are coerced to float."""
    panels = [
        {"confidence": "0.8", "species": "A"},
        {"confidence": "0.5", "species": "B"},
    ]
    filtered = filter_low_confidence(panels, threshold=0.7)
    assert len(filtered) == 1
    assert filtered[0]["species"] == "A"
