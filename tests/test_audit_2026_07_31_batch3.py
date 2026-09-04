"""Regression tests for audit 2026-07-31 batch 3 (taxon extraction &
figure routing).

Covers:
  - figure-type routing: "plate" must not match inside plateau/platform
  - letter-suffixed caption labels ("figs 12-14b", "figs 1a-b")
  - cf./aff. author initials not swallowed ("cf. S. excelsa")
  - English phrase fragments never parsed as binomials
  - paper-whitelist corrections no longer corrupt correct output
  - subgenus / uncertainty forms keep the epithet
  - SEM/TEM acronyms and "gen" truncation rejected
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class TestFigureTypeRouting:
    def test_plateau_not_plate(self):
        from rlpe.range_chart_extractor import classify_figure_type

        cap = (
            "Fig. 1. An attempt of paleogeographic reconstruction of the "
            "Sicilian area at the Late Triassic time (Catalano et al. 1996). "
            "Favignana, Balata di Baida and Inici Mt. sections belong to the "
            "pelagic plateau of the Trapanese Domain"
        )
        assert classify_figure_type(cap) == "paleogeographic_map"

    def test_platform_not_plate(self):
        from rlpe.range_chart_extractor import classify_figure_type

        cap = "Fig. 2. Paleogeographic map of the carbonate platform area."
        assert classify_figure_type(cap) == "paleogeographic_map"

    def test_real_plate_still_plate(self):
        from rlpe.range_chart_extractor import classify_figure_type

        assert classify_figure_type("Plate 1. Scanning electron micrographs.") == "plate"
        assert classify_figure_type("Plates 1-3. SEM images of radiolarians.") == "plate"


class TestLetterSuffixLabels:
    def test_figs_12_14b(self):
        from rlpe.m3_engine import _regex_parse_caption

        pairs = _regex_parse_caption("figs 12-14b. Hiscocapsa lugeoni n. sp.")
        assert [(p.labels, p.species) for p in pairs] == [
            (["12", "13", "14b"], "Hiscocapsa lugeoni")
        ]

    def test_figs_1a_b(self):
        from rlpe.m3_engine import _regex_parse_caption

        pairs = _regex_parse_caption("Pl. 1, figs 1a–b: Cenodiscinus amphitectus Haeckel")
        assert [(p.labels, p.species) for p in pairs] == [
            (["1a", "1b"], "Cenodiscinus amphitectus")
        ]

    def test_expand_label_list_letter_suffix(self):
        from rlpe.m3_engine import _regex_expand_label_list

        assert _regex_expand_label_list("1a-b") == ["1a", "1b"]
        assert _regex_expand_label_list("12-14b") == ["12", "13", "14b"]
        assert _regex_expand_label_list("5-3b") == ["3b", "4", "5"]


class TestAuthorInitialNotSwallowed:
    def test_main_regex_path(self):
        from rlpe.m3_engine import _regex_parse_caption

        pairs = _regex_parse_caption("figs 1-2. Stichocapsa excelsa cf. S. excelsa")
        assert [(p.labels, p.species) for p in pairs] == [(["1", "2"], "Stichocapsa excelsa")]

    def test_danelian_path(self):
        from rlpe.m3_engine import _regex_parse_caption

        pairs = _regex_parse_caption("1) Stichocapsa excelsa cf. S. excelsa")
        assert pairs[0].species == "Stichocapsa excelsa"

    def test_hollis_specimen_codes_kept(self):
        from rlpe.m3_engine import _regex_parse_caption

        pairs = _regex_parse_caption("1. Corythomelissa sp. A. B-F36/0")
        assert pairs[0].species == "Corythomelissa sp. A. B-F36/0"
        pairs = _regex_parse_caption("2. Haliomma gr. A-K47/4")
        assert pairs[0].species == "Haliomma gr. A-K47/4"


class TestPhraseFalsePositives:
    def test_an_attempt_of(self):
        from rlpe.association import extract_taxa_from_caption

        cap = (
            "Fig. 1. An attempt of paleogeographic reconstruction of the "
            "Sicilian area at the Late Triassic time (Catalano et al. 1996)."
        )
        assert extract_taxa_from_caption(cap) == []

    def test_explanation_of(self):
        from rlpe.association import extract_taxa_from_caption

        taxa = extract_taxa_from_caption(
            "Explanation of Plate 1. figs 1-2. Entactinia itsukichiensis ..."
        )
        assert "Explanation of" not in taxa
        assert "Entactinia itsukichiensis" in taxa

    def test_pipeline_entity_extractor(self):
        from rlpe.pipeline import _extract_taxon_entities_from_text

        ents = _extract_taxon_entities_from_text(
            "Fig. 1. An attempt of paleogeographic reconstruction of the "
            "Sicilian area at the Late Triassic time."
        )
        assert ents == []
        ents = _extract_taxon_entities_from_text("Explanation of Plate 1. Unuma echinatus Kocher")
        assert any(e.text == "Unuma echinatus" for e in ents)


class TestPaperWhitelistNoCorruption:
    def test_complete_specimen_not_duplicated(self):
        from rlpe.ocr_corrections import apply_corrections

        s = apply_corrections("Corythomelissa sp. A. B-F36/0", "hollis2006")
        assert s == "Corythomelissa sp. A. B-F36/0"

    def test_truncated_form_not_forced_to_inject_sample_code(self):
        from rlpe.ocr_corrections import apply_corrections

        # Audit 2026-09-04 taxon-5: the previous rule injected the
        # sample-code "B-F36/0" into the species field; that turned
        # the species string into a label-plus-sample-code, which is
        # not a taxon name. The rule is removed — the truncated
        # "sp. A" must survive unchanged.
        s = apply_corrections("Corythomelissa sp. A", "hollis2006")
        assert s == "Corythomelissa sp. A"
        assert "B-F36/0" not in s

    def test_indet_long_form_not_broken(self):
        from rlpe.ocr_corrections import apply_corrections

        s = apply_corrections("Spumellarian gen. et sp. indet", "hollis2006")
        assert s == "Spumellarian gen. et sp. indet"

    def test_truncated_gen_fixed(self):
        from rlpe.ocr_corrections import apply_corrections

        s = apply_corrections("Spumellarian gen", "hollis2006")
        assert s == "Spumellarian indet"

    def test_beccaro_open_nomenclature_preserved(self):
        from rlpe.ocr_corrections import apply_corrections

        # Audit 2026-09-04 taxon-5: "Pseudoeucyrtis sp." is a real
        # open-nomenclature label in ``data/gold/beccaro2006.jsonl``
        # (undetermined species, NOT a named informal morphogroup);
        # the previous forcing rule rewrote it to "sp. B" which
        # destroyed that distinction. Both forms now survive intact.
        assert apply_corrections("Pseudoeucyrtis sp. B", "beccaro2006") == "Pseudoeucyrtis sp. B"
        assert apply_corrections("Pseudoeucyrtis sp.", "beccaro2006") == "Pseudoeucyrtis sp."


class TestSubgenusForms:
    def test_subgenus_keeps_epithet(self):
        # audit 2026-08-01 W3 M1: subgenus now goes to ``generic_name``
        # (DwC subgenus column) instead of ``qualifier`` — Phase 63 introduced
        # the ``generic_name`` field for proper ICZN subgenus handling.
        from rlpe.converters import _taxon_parts
        from rlpe.taxon import _is_valid_species

        parts = _taxon_parts("Podocyrtis (Podocyrtites) amphora")
        assert parts["specific_epithet"] == "amphora"
        assert parts["generic_name"] == "Podocyrtites"
        assert parts.get("qualifier") is None
        assert _is_valid_species("Podocyrtis (Podocyrtites) amphora") is True

    def test_uncertainty_marker_keeps_epithet(self):
        # The ``(?)`` marker is an uncertainty qualifier (NOT a subgenus),
        # so it correctly stays in the ``qualifier`` field per ICZN.
        from rlpe.converters import _taxon_parts
        from rlpe.taxon import _is_valid_species

        parts = _taxon_parts("Sethoconus (?) amphora")
        assert parts["specific_epithet"] == "amphora"
        assert parts["qualifier"] == "(?)"
        assert _is_valid_species("Sethoconus (?) amphora") is True


class TestMicroscopyAndGen:
    def test_sem_photographs_rejected(self):
        from rlpe.taxon import TaxonRecognizer

        rec = TaxonRecognizer()
        ents = rec._fallback_predict("SEM photographs: 1, 2. specimens")
        assert not any(e.text == "SEM photographs" for e in ents)

    def test_tem_micrographs_rejected(self):
        from rlpe.taxon import TaxonRecognizer

        rec = TaxonRecognizer()
        ents = rec._fallback_predict("TEM micrographs of radiolarians")
        assert not any(e.text == "TEM micrographs" for e in ents)

    def test_gen_truncation_invalid(self):
        from rlpe.converters import _taxon_parts
        from rlpe.taxon import _is_valid_species

        parts = _taxon_parts("Spumellarian gen")
        assert parts["specific_epithet"] is None
        # Audit 2026-09-04 taxon-1 behavioural change: "gen" is now an
        # accepted open-nomenclature token (bandini 2011's real gold
        # convention "Spumellaria gen" / "Nassellaria gen" was being
        # silently dropped alongside the truncation false-positives
        # this test used to pin). Truncated-prose false positives are
        # the responsibility of the extractor layer, not this shape
        # check, which has no species-name vocabulary to consult.
        assert _is_valid_species("Spumellarian gen") is True
