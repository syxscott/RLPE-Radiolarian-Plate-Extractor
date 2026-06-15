"""Tests for cross-figure reference parsing."""
from __future__ import annotations

from rlpe.cross_refs import parse_cross_refs


class TestBasicCrossRefs:
    def test_simple_fig_reference(self):
        refs = parse_cross_refs("See Fig. 3 for the type species.")
        assert len(refs) == 1
        assert refs[0].target_figure == "Fig. 3"
        assert refs[0].target_figure_num == "3"

    def test_figure_with_capital(self):
        refs = parse_cross_refs("Compared with Figure 5.")
        assert len(refs) == 1
        assert refs[0].target_figure_num == "5"

    def test_plate_reference(self):
        refs = parse_cross_refs("Pl. 2 shows additional specimens.")
        assert len(refs) == 1
        assert refs[0].target_figure_num == "2"
        assert "Pl" in refs[0].target_figure

    def test_figure_with_letter_suffix(self):
        refs = parse_cross_refs("As shown in Fig. 2C-E")
        assert len(refs) >= 1
        # The number group captures the digits, and the trailing letter range
        # is recorded in the context.
        assert refs[0].target_figure_num == "2"
        assert "C-E" in refs[0].context or "C" in refs[0].context

    def test_multiple_refs(self):
        refs = parse_cross_refs("Compare Fig. 1 and Fig. 2 and Fig. 3.")
        assert len(refs) == 3
        nums = {r.target_figure_num for r in refs}
        assert {"1", "2", "3"} <= nums


class TestSelfReferenceFiltering:
    def test_self_ref_skipped(self):
        refs = parse_cross_refs("See Fig. 2 in this plate.", current_fig_id="fig_2")
        assert refs == []

    def test_other_ref_kept(self):
        refs = parse_cross_refs("See Fig. 3 in this plate.", current_fig_id="fig_2")
        assert len(refs) == 1
        assert refs[0].target_figure_num == "3"

    def test_no_fig_id_keeps_all(self):
        refs = parse_cross_refs("See Fig. 2.", current_fig_id="")
        assert len(refs) == 1


class TestSpeciesHint:
    def test_species_hint_right_side(self):
        refs = parse_cross_refs("Fig. 2C shows Cromyomma sp.")
        assert len(refs) == 1
        assert refs[0].species_hint is not None
        assert "Cromyomma" in refs[0].species_hint

    def test_no_species_in_caption(self):
        refs = parse_cross_refs("Fig. 2C")
        assert len(refs) == 1
        # No species hint since no species name near the reference
        assert refs[0].species_hint is None


class TestEdgeCases:
    def test_empty_text(self):
        assert parse_cross_refs("") == []

    def test_no_figure_references(self):
        assert parse_cross_refs("Just a normal sentence without figure refs.") == []

    def test_context_field_populated(self):
        refs = parse_cross_refs("See Fig. 3 for details.")
        assert len(refs) == 1
        assert len(refs[0].context) > 0
        assert "Fig. 3" in refs[0].context

    def test_to_dict_serialization(self):
        refs = parse_cross_refs("See Fig. 3.")
        d = refs[0].to_dict()
        assert "target_figure" in d
        assert "span" in d
        assert isinstance(d["span"], list)
