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
