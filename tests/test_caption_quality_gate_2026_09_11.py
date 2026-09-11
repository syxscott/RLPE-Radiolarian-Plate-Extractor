"""2026-09-11 caption-quality gate regression tests.

Root cause: ``_CAPTION_CLAUSE_RE``'s genus branch matched the Capitalised
geologic stage name "Asselian" and its epithet branch matched " and",
yielding species="Asselian and" from the real caption
"Fig. 2. Asselian and Sakmarian radiolarians of the Lower Permian South
Urals in the Kondurovka (1-3)" (Afanasieva 2020c). That garbage then
propagated into matches.jsonl and the panel image file names.
"""

from __future__ import annotations

import pytest


CAPTION = (
    "Fig. 2. Asselian and Sakmarian radiolarians of the Lower Permian "
    "South Urals in the Kondurovka (1\u20133)"
)


class TestRegexParseCaptionGate:
    def test_asselian_and_no_longer_produced(self):
        from rlpe.semantic_engine import _regex_parse_caption

        pairs = _regex_parse_caption(CAPTION)
        species_values = [p.species for p in pairs]
        assert "Asselian and" not in species_values

    def test_rejection_reason_reported(self):
        from rlpe.semantic_engine import _species_candidate_rejected

        reason = _species_candidate_rejected("Asselian and")
        assert reason in {"stopword_token", "geologic_time_term"}

    def test_geologic_genus_rejected_even_without_stopword(self):
        from rlpe.semantic_engine import _species_candidate_rejected

        assert _species_candidate_rejected("Asselian sakmariana") == "geologic_time_term"


class TestIsValidSpeciesGate:
    def test_asselian_and_invalid(self):
        from rlpe.taxon import _is_valid_species

        assert _is_valid_species("Asselian and") is False

    def test_geologic_stage_genus_invalid(self):
        from rlpe.taxon import _is_valid_species

        assert _is_valid_species("Asselian sakmariana") is False

    def test_real_binomial_still_valid(self):
        from rlpe.taxon import _is_valid_species

        assert _is_valid_species("Spinodeflandrella tetraspinosa") is True

    def test_open_nomenclature_still_valid(self):
        from rlpe.taxon import _is_valid_species

        assert _is_valid_species("Entactinia sp. 1") is True


class TestPouillePositiveSurvives:
    """The Plate-3 style caption must keep producing the correct pair —
    the gate guards against prose, not against real taxonomy."""

    def test_pouille_pair_kept(self):
        from rlpe.semantic_engine import _regex_parse_caption

        caption = (
            "Plate 3. (Reconstructed from systematic descriptions)\n"
            "Spinodeflandrella tetraspinosa (pl. 3, figs. 1)"
        )
        pairs = _regex_parse_caption(caption)
        assert any(
            "tetraspinosa" in (p.species or "") and "Spinodeflandrella" in (p.species or "")
            for p in pairs
        )


class TestEntityExtractionGeologicGate:
    """2026-09-12: the position-fallback entity scan read
    "Sakmarian radiolarians" (stage name + common noun) out of the
    Kondurovka caption as a species candidate. The phrase-level
    geologic gate must reject it upstream, and _is_valid_species must
    reject it downstream."""

    PROSE_CAPTION = (
        "Fig. 2. Asselian and Sakmarian radiolarians of the Lower Permian "
        "South Urals in the Kondurovka (1\u20133)"
    )

    def test_entity_scan_rejects_stage_phrase(self):
        from rlpe.association import extract_taxa_from_caption

        taxa = extract_taxa_from_caption(self.PROSE_CAPTION)
        assert "Sakmarian radiolarians" not in taxa
        assert "Asselian and" not in taxa

    def test_is_valid_species_rejects_stage_phrase(self):
        from rlpe.taxon import _is_valid_species

        assert _is_valid_species("Sakmarian radiolarians") is False

    def test_real_binomial_still_extracted(self):
        from rlpe.association import extract_taxa_from_caption

        taxa = extract_taxa_from_caption(
            "Plate 3. Spinodeflandrella tetraspinosa (pl. 3, figs. 1)"
        )
        assert any("tetraspinosa" in t for t in taxa)


class TestCleanLlmSpecies:
    """2026-09-12: deepseek-flash answers the literal string "None" for
    panels it cannot identify; bare str() kept it as a truthy species
    and it propagated into matches.jsonl and panel image file names."""

    def test_none_string_collapses_to_none(self):
        from rlpe.semantic_engine import _clean_llm_species

        assert _clean_llm_species("None") is None
        assert _clean_llm_species("null") is None
        assert _clean_llm_species("unknown") is None
        assert _clean_llm_species("N/A") is None
        assert _clean_llm_species("") is None
        assert _clean_llm_species(None) is None

    def test_real_species_kept(self):
        from rlpe.semantic_engine import _clean_llm_species

        assert _clean_llm_species("Spinodeflandrella tetraspinosa") == (
            "Spinodeflandrella tetraspinosa"
        )
