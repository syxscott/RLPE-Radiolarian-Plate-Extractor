"""Tests for panel-label normalization and the strict caption-pair lookup
in association.match_panels.

These cover the two regressions that surfaced on Feng 2007 Plate 1:
  1. The caption parser missed "ﬁgs" (U+FB01 ligature) and fell back to
     positional assignment, tagging panel 2 as the wrong species.
  2. Carry-forward wrongly assigned species to 14 SEM-metadata fragments
     that don't have a label in any caption clause.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.association import (  # noqa: E402
    _normalize_panel_label,
    _label_in_pair_lookup,
    _label_sort_key,
    match_panels,
    assign_panels_to_labels,
    extract_panel_labels,
    extract_taxa_from_caption,
    is_placeholder_caption,
)
from rlpe.types import CaptionRecord, PanelCandidate  # noqa: E402


# ---------------------------------------------------------------------------
# _normalize_panel_label
# ---------------------------------------------------------------------------


def test_normalize_panel_label_strips_leading_zeros():
    """PaddleOCR commonly reads "0" as "00" and "4" as "04" when the glyph
    is small. Strip leading zeros so caption lookup still works."""
    assert _normalize_panel_label("0") == "0"
    assert _normalize_panel_label("00") == "0"
    assert _normalize_panel_label("04") == "4"
    assert _normalize_panel_label("07") == "7"
    assert _normalize_panel_label("10") == "10"
    assert _normalize_panel_label("13") == "13"


def test_normalize_panel_label_keeps_alphabetic():
    """"A", "B" labels must not become "0"."""
    assert _normalize_panel_label("A") == "A"
    assert _normalize_panel_label("AB") == "AB"


def test_normalize_panel_label_handles_none_and_empty():
    assert _normalize_panel_label(None) is None
    assert _normalize_panel_label("") is None
    assert _normalize_panel_label("   ") is None


def test_normalize_panel_label_strips_whitespace():
    assert _normalize_panel_label(" 3 ") == "3"
    assert _normalize_panel_label("\t04\n") == "4"


# ---------------------------------------------------------------------------
# _label_in_pair_lookup
# ---------------------------------------------------------------------------


def test_label_in_pair_lookup_direct_hit():
    pl = {"1": "Foo", "2": "Bar"}
    assert _label_in_pair_lookup("1", pl) == "1"
    assert _label_in_pair_lookup("2", pl) == "2"


def test_label_in_pair_lookup_falls_back_to_normalized():
    """OCR misread "04" should still match pair_lookup key "4"."""
    pl = {"4": "Foo", "5": "Bar"}
    assert _label_in_pair_lookup("04", pl) == "4"
    assert _label_in_pair_lookup("00", pl) is None  # "0" not in lookup
    assert _label_in_pair_lookup("07", pl) is None  # "7" not in lookup


def test_label_in_pair_lookup_no_match():
    pl = {"1": "Foo"}
    assert _label_in_pair_lookup("99", pl) is None
    assert _label_in_pair_lookup(None, pl) is None
    assert _label_in_pair_lookup("", pl) is None


# ---------------------------------------------------------------------------
# _label_sort_key
# ---------------------------------------------------------------------------


def test_label_sort_key_numeric_vs_alpha():
    """Pure digits sort first (rank 0), alpha sort after (rank 1)."""
    # Rank 0 = numeric, rank 1 = alpha. Within rank, sort is stable.
    assert _label_sort_key("1")[0] == 0
    assert _label_sort_key("2")[0] == 0
    assert _label_sort_key("10")[0] == 0
    assert _label_sort_key("A")[0] == 1
    assert _label_sort_key("Z")[0] == 1
    # All numeric labels share rank 0; sort tie-breaks by caller's stable
    # sort on a secondary key, but the rank itself is what we test here.
    assert _label_sort_key("9") == _label_sort_key("10")
    # Alpha > numeric.
    assert _label_sort_key("9") < _label_sort_key("A")
    assert _label_sort_key("100") < _label_sort_key("A")


# ---------------------------------------------------------------------------
# match_panels: strict caption-pair lookup (no carry-forward to no-label)
# ---------------------------------------------------------------------------


def _panel(pid: str | None, bbox=(0, 0, 100, 100)) -> PanelCandidate:
    return PanelCandidate(
        panel_id=pid,
        bbox=bbox,
        score=0.5,
        metadata={"method": "test"},
    )


def _caption(text: str) -> CaptionRecord:
    return CaptionRecord(
        paper_id="p1",
        figure_id="fig1",
        caption=text,
        page_index=6,
        figure_number="1",
    )


def test_match_panels_ligature_caption_assigns_correct_species():
    """End-to-end: real Feng 2007 Plate 1 caption shape (with U+FB01 'ﬁgs').
    Before the ligature fix, caption_pairs_used was False and panel 2 got
    the wrong species. After the fix, panels 1-2 → itsukichiensis and
    panels 3-4 → reticulata."""
    caption_text = (
        "Explanation of Plate 1. ﬁgs 1–2. Entactinia itsukichiensis "
        "Sashida & Tonishi: 1, DP2/B024; 2, DP4/P016. "
        "ﬁgs 3–4. Entactinia reticulata: 3, DP1/B005; 4, DP3/P009."
    )
    caption = _caption(caption_text)
    panels = [_panel("1"), _panel("2"), _panel("3"), _panel("4")]
    matches = match_panels(
        paper_id="p1",
        figure_id="fig1",
        caption=caption,
        panels=panels,
        ocr_tokens=[],
        taxon_entities=[],
    )
    assert len(matches) == 4
    by_label = {m.panel_id: m.species for m in matches}
    assert by_label["1"] == "Entactinia itsukichiensis"
    assert by_label["2"] == "Entactinia itsukichiensis"
    assert by_label["3"] == "Entactinia reticulata"
    assert by_label["4"] == "Entactinia reticulata"


def test_match_panels_strict_no_carry_forward_to_unlabeled():
    """The 14 SEM-metadata fragments on Plate 1 have no label and should
    be assigned species=None, not the last seen species (regression
    from the previous carry-forward behaviour)."""
    caption_text = (
        "Explanation of Plate 1. ﬁgs 1–2. Entactinia itsukichiensis: 1, ...; 2, ... "
        "ﬁgs 3–4. Entactinia reticulata: 3, ...; 4, ..."
    )
    caption = _caption(caption_text)
    panels = [
        _panel("1"),
        _panel("2"),
        _panel("3"),
        _panel("4"),
        _panel(None),  # metadata fragment
        _panel(None),  # metadata fragment
        _panel(None),  # metadata fragment
    ]
    matches = match_panels(
        paper_id="p1",
        figure_id="fig1",
        caption=caption,
        panels=panels,
        ocr_tokens=[],
        taxon_entities=[],
    )
    species = [m.species for m in matches]
    # First 4 panels correctly assigned.
    assert species[0:4] == [
        "Entactinia itsukichiensis",
        "Entactinia itsukichiensis",
        "Entactinia reticulata",
        "Entactinia reticulata",
    ]
    # The 3 unlabeled fragments must NOT be filled in via carry-forward.
    assert species[4] is None
    assert species[5] is None
    assert species[6] is None


def test_match_panels_strict_no_carry_forward_to_ocr_misread():
    """OCR misread "63" must not be filled in with the last seen species
    if "63" is not in any caption clause. The user wants false assignments
    over false misses here, but carry-forward to non-existent labels is
    strictly wrong."""
    caption_text = (
        "Explanation of Plate 4. ﬁgs 1–5. Triaenosphaera variabilis: 1, ...; 2, ... "
        "ﬁgs 6–9. Triaenosphaera minutus: 6, ...; 7, ... "
        "ﬁgs 10–13. Triaenosphaera sp.: 10, ... "
        "ﬁgs 14–16. Triaenosphaera megacantha: 14, ..."
    )
    caption = _caption(caption_text)
    panels = [
        _panel("1"),
        _panel("63"),  # OCR misread of "6" or some non-existent label
    ]
    matches = match_panels(
        paper_id="p1",
        figure_id="fig1",
        caption=caption,
        panels=panels,
        ocr_tokens=[],
        taxon_entities=[],
    )
    # Panel 1 → variabilis (in caption). Panel "63" → None (not in any clause).
    assert matches[0].species == "Triaenosphaera variabilis"
    assert matches[1].species is None


def test_match_panels_leading_zero_ocr_misread_resolves():
    """OCR misread "04" must resolve to caption key "4"."""
    caption_text = (
        "Explanation of Plate 4. ﬁgs 1–5. Triaenosphaera variabilis: 1, ...; 4, ... "
    )
    caption = _caption(caption_text)
    panels = [_panel("04"), _panel("1")]
    matches = match_panels(
        paper_id="p1",
        figure_id="fig1",
        caption=caption,
        panels=panels,
        ocr_tokens=[],
        taxon_entities=[],
    )
    # Both should resolve to "Triaenosphaera variabilis" because
    # "04" normalises to "4" which is in the pair_lookup.
    assert matches[0].species == "Triaenosphaera variabilis"
    assert matches[0].panel_id == "4"  # normalised
    assert matches[1].species == "Triaenosphaera variabilis"


# ---------------------------------------------------------------------------
# assign_panels_to_labels
# ---------------------------------------------------------------------------


def test_assign_panels_to_labels_normalises_panel_id():
    panels = [_panel("00"), _panel("04"), _panel("7"), _panel(None)]
    out = assign_panels_to_labels(panels, labels=[], ocr_tokens=[])
    assert out == ["0", "4", "7", None]


def test_assign_panels_to_labels_uses_caption_labels_fallback():
    panels = [_panel(None), _panel(None), _panel(None)]
    out = assign_panels_to_labels(panels, labels=["1", "2", "3"], ocr_tokens=[])
    assert out == ["1", "2", "3"]


# ---------------------------------------------------------------------------
# is_placeholder_caption + extractors skip placeholder text
# ---------------------------------------------------------------------------


def test_is_placeholder_caption_recognises_auto_generated():
    """The pipeline emits "Auto-generated figure for page N" when the
    upstream extractor (GROBID / OpenDataLoader) returns no caption.
    The binomial regex used to match "Auto-generated figure" as a
    species — gate it at the boundary."""
    assert is_placeholder_caption("Auto-generated figure for page 17") is True
    assert is_placeholder_caption("auto-generated figure for page 1") is True
    assert is_placeholder_caption("AUTO-GENERATED FIGURE FOR PAGE 99") is True
    # Variants we also want to reject.
    assert is_placeholder_caption("placeholder caption") is True
    assert is_placeholder_caption("N/A") is True
    assert is_placeholder_caption("missing caption") is True
    # Real captions are kept.
    assert is_placeholder_caption("Explanation of Plate 1. fig. 1. Foo bar") is False
    assert is_placeholder_caption("figs 1-3. Entactinia itsukichiensis") is False


def test_is_placeholder_caption_handles_empty():
    assert is_placeholder_caption(None) is True
    assert is_placeholder_caption("") is True
    assert is_placeholder_caption("   ") is True


def test_extract_taxa_from_caption_rejects_placeholder():
    """The species extractor must not return 'Auto-generated figure' as a
    taxon when the caption is a placeholder."""
    out = extract_taxa_from_caption("Auto-generated figure for page 17")
    assert out == []
    out = extract_taxa_from_caption("Explanation of Plate 1. fig. 1. Entactinia itsukichiensis")
    assert "Entactinia itsukichiensis" in out


def test_extract_panel_labels_rejects_placeholder():
    out = extract_panel_labels("Auto-generated figure for page 17")
    assert out == []


def test_match_panels_skips_placeholder_caption():
    """When the caption is a placeholder, match_panels must NOT tag any
    panel with 'Auto-generated figure' as a species (regression that
    produced 63 bogus rows for Hollis 2006 in the 4-paper batch test)."""
    caption = _caption("Auto-generated figure for page 17")
    panels = [_panel("1"), _panel("2"), _panel(None)]
    matches = match_panels(
        paper_id="p1",
        figure_id="fig1",
        caption=caption,
        panels=panels,
        ocr_tokens=[],
        taxon_entities=[],
    )
    assert len(matches) == 3
    # Every species is None — no positional fallback to "Auto-generated".
    for m in matches:
        assert m.species is None
        assert (m.metadata or {}).get("matcher_type") == "skipped-placeholder-caption"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
