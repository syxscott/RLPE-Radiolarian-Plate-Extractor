"""Regression tests for audit 2026-08-19 sweep 8 — Phase 1A data semantics.

Three BLOCKER bugs in the data semantics layer:

* **B-2** (Darwin Core violation): ``TaxonRecord.verbatim_name`` and
  ``normalized_name`` were always set to the same value.  DwC requires
  them to be distinct — verbatim preserves the raw OCR string (incl.
  ``?``, ``cf.``, OCR case errors) while normalized drives entity
  linking.

* **B-3** (ICZN 19th-century citation): ``_extract_authorship``
  required a year in the parenthesised authority block, so
  ``(Haeckel)`` / ``(Ehrenberg)`` (no year — common for ICZN
  19th-century citations) was silently dropped.

* **B-6** (captioned species missed): ``_SPECIES_RE`` required a
  full binomial (genus + epithet, both >= 4 chars), so the most
  common radiolarian systematics shape — ``Genus sp.`` / ``Genus
  n. sp.`` / ``Genus ex gr. species`` — never produced caption
  entities.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe.converters import (  # noqa: E402
    _extract_authorship,
    _normalise_species_name,
    _normalize_dwc_name,
    _verbatim_species_name,
    taxon_records_from_matches,
)
from rlpe.local_pdf_parser import (  # noqa: E402
    _SPECIES_OPEN_NOMEN_STARTS,
    _SPECIES_RE,
)
from rlpe.types import MatchResult, PaperMetadata  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_match(species: str | None = "Genus species") -> MatchResult:
    """Build a MatchResult with a controllable species string."""
    pm = PaperMetadata(
        title="Test",
        authors=["Author A"],
        year=2020,
        doi="10.1000/test",
        source="opendataloader",
        confidence=0.8,
    )
    return MatchResult(
        paper_id="abc",
        figure_id="fig_1",
        panel_id="1",
        species=species,
        panel_path="/path/to/panel.png",
        bbox=[10, 20, 100, 200],
        confidence=0.6,
        label_text="1",
        caption_snippet=f"Plate 1\n1) {species}",
        ocr_text=None,
        metadata={"extraction_method": "regex"},
    )


# ---------------------------------------------------------------------------
# B-2: verbatim_name / normalized_name separation (DwC compliance)
# ---------------------------------------------------------------------------


class TestVerbatimVsNormalizedSeparation:
    """B-2: verbatim_name and normalized_name must be distinct DwC fields."""

    def test_genus_uncertainty_marker_preserved_in_verbatim(self) -> None:
        """``Theocorys? phyzella`` — the ``?`` is the OCR's
        open-nomenclature marker.  Verbatim must keep it; normalized
        must strip it (the marker is a reviewer's flag, not part of
        the scientific name).
        """
        m = _make_match(species="Theocorys? phyzella")
        taxa = taxon_records_from_matches([m])
        assert len(taxa) == 1
        t = taxa[0]
        assert t["verbatim_name"] == "Theocorys? phyzella"
        assert "?" in t["verbatim_name"]
        assert t["normalized_name"] == "Theocorys phyzella"
        assert "?" not in t["normalized_name"]

    def test_cf_marker_preserved_in_both(self) -> None:
        """``Triactoma cf. kamoense`` — the ``cf.`` is an ICZN
        qualifier, part of the scientific name.  Both fields keep
        it.
        """
        m = _make_match(species="Triactoma cf. kamoense")
        taxa = taxon_records_from_matches([m])
        assert len(taxa) == 1
        t = taxa[0]
        assert t["verbatim_name"] == "Triactoma cf. kamoense"
        assert "cf." in t["verbatim_name"]
        assert t["normalized_name"] == "Triactoma cf. kamoense"
        assert "cf." in t["normalized_name"]

    def test_ocr_case_error_preserved_in_verbatim_fixed_in_normalized(self) -> None:
        """``pHractus sp.`` — OCR-mangled capitalisation on the
        genus.  Verbatim keeps the raw OCR error so reviewers can
        see the source; normalized corrects the leading capital to
        the canonical form.
        """
        m = _make_match(species="pHractus sp.")
        taxa = taxon_records_from_matches([m])
        assert len(taxa) == 1
        t = taxa[0]
        # Verbatim preserves the OCR string, only trimming whitespace.
        assert t["verbatim_name"] == "pHractus sp."
        # Verbatim retains the OCR-mangled mixed case.
        assert "pHractus" in t["verbatim_name"]
        # Normalized differs from verbatim (whitespace + case fix).
        assert t["normalized_name"] != t["verbatim_name"]
        # The ``sp.`` ICZN marker is preserved in normalized.
        assert "sp" in t["normalized_name"].lower()

    def test_verbatim_helper_preserves_question_mark(self) -> None:
        """Direct test of the verbatim helper."""
        assert _verbatim_species_name("Theocorys? phyzella") == "Theocorys? phyzella"
        assert _verbatim_species_name("  Triactoma  cf.  kamoense  ") == "Triactoma cf. kamoense"
        assert _verbatim_species_name(None) is None
        assert _verbatim_species_name("") is None

    def test_normalize_dwc_strips_question_mark(self) -> None:
        """Direct test of the normalized helper."""
        assert _normalize_dwc_name("Theocorys? phyzella") == "Theocorys phyzella"
        assert _normalize_dwc_name("Triactoma cf. kamoense") == "Triactoma cf. kamoense"
        assert _normalize_dwc_name("pHractus sp.") == "Phractus sp"
        assert _normalize_dwc_name(None) is None
        assert _normalize_dwc_name("") is None

    def test_normalized_differs_from_verbatim_for_messy_inputs(self) -> None:
        """Sanity check: for OCR-mangled or ``?``-marked inputs, the
        two fields MUST differ.
        """
        m = _make_match(species="Actinomma? solida")
        taxa = taxon_records_from_matches([m])
        assert len(taxa) == 1
        t = taxa[0]
        assert t["verbatim_name"] != t["normalized_name"]
        assert "?" in t["verbatim_name"]
        assert "?" not in t["normalized_name"]

    def test_clean_input_verbatim_equals_normalized(self) -> None:
        """When the input is already clean (no ``?``, no OCR error),
        the two fields naturally agree.  This is the legacy shape
        the audit B-2 fixes without breaking.
        """
        m = _make_match(species="Triactoma kamoensis")
        taxa = taxon_records_from_matches([m])
        assert len(taxa) == 1
        t = taxa[0]
        assert t["verbatim_name"] == "Triactoma kamoensis"
        assert t["normalized_name"] == "Triactoma kamoensis"

    def test_normalise_species_name_unchanged(self) -> None:
        """The legacy helper must remain unchanged so call sites
        elsewhere (entity routing, taxon_id keys) still produce the
        same IDs.
        """
        assert _normalise_species_name("Theocorys? phyzella") == "Theocorys? phyzella"
        assert _normalise_species_name("  sp.  ") == "sp"
        assert _normalise_species_name(None) is None


# ---------------------------------------------------------------------------
# B-3: (Haeckel) no-year authority parsing
# ---------------------------------------------------------------------------


class TestBareParenAuthority:
    """B-3: parenthesised authority without a year (19th-century ICZN)."""

    def test_haeckel_no_year(self) -> None:
        """``Podocyrtis amphora (Haeckel)`` — Haeckel 1887 is the
        canonical work, so the year is omitted.
        """
        _, _, authorship = _extract_authorship("Podocyrtis amphora (Haeckel)")
        assert authorship == "(Haeckel)"

    def test_ehrenberg_no_year(self) -> None:
        """``Actinomma solida (Ehrenberg)`` — another 19th-century
        author where the original work (Ehrenberg 1838) is the
        canonical reference.
        """
        _, _, authorship = _extract_authorship("Actinomma solida (Ehrenberg)")
        assert authorship == "(Ehrenberg)"

    def test_smith_no_year_pattern(self) -> None:
        """``Genus species (Smith)`` — generic bare-parenthetical
        authorship.  Confirms the regex accepts any capitalised
        surname (>= 3 alpha chars).
        """
        _, _, authorship = _extract_authorship("Genus species (Smith)")
        assert authorship == "(Smith)"

    def test_authority_with_year_still_works(self) -> None:
        """Regression: ``(Smith, 1900)`` must still be recognised by
        the original year-required branch.
        """
        _, _, authorship = _extract_authorship("Triactoma kamoense (Smith, 1900)")
        assert authorship == "Smith, 1900"

    def test_authority_with_year_and_no_year_disambiguation(self) -> None:
        """When both ``(Haeckel)`` and ``(Smith, 1900)`` shapes look
        similar, the no-year branch must NOT eat the year-bearing
        form.  Bare-paren branch fires first; the paren it requires
        has no comma, so the year branch is preserved.
        """
        _, _, authorship = _extract_authorship("Podocyrtis amphora (Haeckel, 1887)")
        # Year-bearing branch: the inner capture excludes the parens.
        assert authorship == "Haeckel, 1887"

    def test_unrelated_paren_not_authority(self) -> None:
        """``Podocyrtis amphora (Podocyrtites)`` is a subgenus in
        parens, NOT an authority.  The bare-paren branch must not
        fire here because the postfix-subgenus branch handles this
        shape (and fires only when there are ≥ 2 tokens before the
        paren).
        """
        _, subgenus, _ = _extract_authorship("Podocyrtis amphora (Podocyrtites)")
        assert subgenus == "Podocyrtites"


# ---------------------------------------------------------------------------
# B-6: _SPECIES_RE extended for open-nomenclature shapes
# ---------------------------------------------------------------------------


class TestSpeciesReExtension:
    """B-6: regex must accept open-nomenclature shapes used in radiolarian systematics."""

    def test_undetermined_species_sp(self) -> None:
        """``Actinomma sp.`` — the most common radiolarian
        systematics shape: genus + undetermined species.
        """
        matches = list(_SPECIES_RE.finditer("Actinomma sp."))
        assert len(matches) == 1
        assert matches[0].group(1) == "Actinomma sp."

    def test_new_species_abbreviated(self) -> None:
        """``Triactoma n. sp.`` — abbreviated new species marker."""
        matches = list(_SPECIES_RE.finditer("Triactoma n. sp."))
        assert len(matches) == 1
        assert matches[0].group(1) == "Triactoma n. sp."

    def test_new_species_full(self) -> None:
        """``Triactoma sp. nov.`` — full new species marker."""
        matches = list(_SPECIES_RE.finditer("Triactoma sp. nov."))
        assert len(matches) == 1
        assert matches[0].group(1) == "Triactoma sp. nov."

    def test_open_naming_group(self) -> None:
        """``Stichocapsa ex gr. convexa`` — open-naming group."""
        matches = list(_SPECIES_RE.finditer("Stichocapsa ex gr. convexa"))
        assert len(matches) == 1
        assert matches[0].group(1) == "Stichocapsa ex gr. convexa"

    def test_new_genus_new_species(self) -> None:
        """``Pessagnoa n. gen. n. sp.`` — new genus + new species."""
        matches = list(_SPECIES_RE.finditer("Pessagnoa n. gen. n. sp."))
        assert len(matches) == 1
        assert matches[0].group(1) == "Pessagnoa n. gen. n. sp."

    def test_multiple_species_spp(self) -> None:
        """``Actinomma spp.`` — multiple undetermined species."""
        matches = list(_SPECIES_RE.finditer("Actinomma spp."))
        assert len(matches) == 1
        assert matches[0].group(1) == "Actinomma spp."

    def test_comparison_species(self) -> None:
        """``Genus cf. species`` — comparison species."""
        matches = list(_SPECIES_RE.finditer("Genus cf. species"))
        assert len(matches) == 1
        assert matches[0].group(1) == "Genus cf. species"

    def test_affinity_species(self) -> None:
        """``Genus aff. species`` — affinity species."""
        matches = list(_SPECIES_RE.finditer("Genus aff. species"))
        assert len(matches) == 1
        assert matches[0].group(1) == "Genus aff. species"

    def test_full_binomial_still_matches(self) -> None:
        """Regression: the traditional full binomial ``Triactoma
        kamoensis`` must still match (no back-compat break).
        """
        matches = list(_SPECIES_RE.finditer("Triactoma kamoensis"))
        assert len(matches) == 1
        assert matches[0].group(1) == "Triactoma kamoensis"

    def test_full_binomial_in_sentence(self) -> None:
        """Match within a longer caption text."""
        text = "The fauna includes Triactoma kamoensis and Actinomma sp."
        matches = list(_SPECIES_RE.finditer(text))
        assert len(matches) == 2
        matched_strings = {m.group(1) for m in matches}
        assert "Triactoma kamoensis" in matched_strings
        assert "Actinomma sp." in matched_strings

    def test_three_binomials_open_nomen_mix(self) -> None:
        """Mixed shape in a single caption: full binomial +
        ``n. sp.`` + ``ex gr.``."""
        text = "Podocyrtis amphora, Triactoma n. sp., Stichocapsa ex gr. convexa"
        matches = [_m.group(1) for _m in _SPECIES_RE.finditer(text)]
        assert "Podocyrtis amphora" in matches
        assert "Triactoma n. sp." in matches
        assert "Stichocapsa ex gr. convexa" in matches

    def test_open_nomen_starts_set_includes_required_stems(self) -> None:
        """The ICZN open-nomenclature set must contain all the
        second-token stems that the regex now accepts.
        """
        for stem in ("sp", "spp", "nov", "n", "cf", "aff", "ex", "gr"):
            assert stem in _SPECIES_OPEN_NOMEN_STARTS, f"missing stem: {stem!r}"


# ---------------------------------------------------------------------------
# Cross-cutting: ensure the schema still accepts the new shape
# ---------------------------------------------------------------------------


class TestRegressionCompatibility:
    """Cross-cutting: pipeline downstream of the changed helpers still works."""

    def test_legacy_taxon_records_from_match_basic(self) -> None:
        """The legacy ``_make_match`` species (full binomial) still
        produces a single TaxonRecord with both fields populated.
        """
        m = _make_match(species="Genus species")
        taxa = taxon_records_from_matches([m])
        assert len(taxa) == 1
        t = taxa[0]
        assert t["genus"] == "Genus"
        assert t["specific_epithet"] == "species"
        assert t["verbatim_name"] == "Genus species"
        assert t["normalized_name"] == "Genus species"

    def test_legacy_subgenus_postfix_unchanged(self) -> None:
        """The M1 audit postfix-subgenus fix (2026-08-01) must still
        work after our B-2 changes.
        """
        m = _make_match(species="Podocyrtis amphora (Podocyrtites)")
        taxa = taxon_records_from_matches([m])
        assert len(taxa) == 1
        t = taxa[0]
        # Verbatim preserves the full string.
        assert "Podocyrtites" in t["verbatim_name"]
        # Subgenus field is populated.
        assert t["generic_name"] == "Podocyrtites"
        assert t["genus"] == "Podocyrtis"
        assert t["specific_epithet"] == "amphora"

    def test_legacy_authority_with_year_unchanged(self) -> None:
        """``Podocyrtis amphora (Haeckel, 1887)`` — the year-bearing
        branch must still produce the authority correctly.
        """
        m = _make_match(species="Podocyrtis amphora (Haeckel, 1887)")
        taxa = taxon_records_from_matches([m])
        assert len(taxa) == 1
        t = taxa[0]
        assert t["scientific_name_authorship"] == "Haeckel, 1887"
