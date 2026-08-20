"""Regression tests for audit 2026-08-19 sweep — Phase 2A.

Two MAJOR bugs in the data-semantics layer:

* **M-2** (open-naming ``ex gr.`` token merging): ``_taxon_parts``
  left the ICZN ``"Genus ex gr. species"`` shape broken because the
  pre-processing only recognised the qualifier ``"gr."`` (not the
  multi-word marker ``"ex gr."``).  The result was an epithet of
  ``"ex"`` and a qualifier of ``"gr. species"`` — neither field
  matches the ICZN convention.

* **M-3** (``TaxonRecord.source`` conflation): the ``source`` field
  was overloaded — it carried the PBDB data source ("paleodb") for
  DwC-A consumers AND the pipeline-side extraction method
  ("regex" / "llm_first" / "hybrid") for internal audits.  DwC
  expects ``source`` to be the **taxonomic data source**; the
  extraction method belongs in a separate structured field.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe.converters import _taxon_parts, taxon_records_from_matches  # noqa: E402
from rlpe.schema_models import TaxonRecord  # noqa: E402
from rlpe.types import MatchResult, PaperMetadata  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_match(
    species: str | None = "Genus species",
    *,
    extraction_method: str = "regex",
    pbdb: dict | None = None,
) -> MatchResult:
    """Build a MatchResult with controllable species + extraction metadata.

    Parameters
    ----------
    species:
        The species string parsed by ``_taxon_parts``.
    extraction_method:
        Value written to ``m.metadata["extraction_method"]``; carried
        into the new structured ``extraction_method`` field on
        ``TaxonRecord`` (M-3 fix).
    pbdb:
        Optional dict written to ``m.metadata["paleodb"]["taxonomy"]``
        so the converter can mark the source as "paleodb".
    """
    pm = PaperMetadata(
        title="Test",
        authors=["Author A"],
        year=2020,
        doi="10.1000/test",
        source="opendataloader",
        confidence=0.8,
    )
    md: dict = {"extraction_method": extraction_method}
    if pbdb is not None:
        md["paleodb"] = {"taxonomy": pbdb}
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
        metadata=md,
    )


# ---------------------------------------------------------------------------
# M-2: ex gr. token merging
# ---------------------------------------------------------------------------


class TestExGrMerging:
    """M-2: ``Genus ex gr. species`` must emit qualifier='ex gr.' and
    preserve the species in ``specific_epithet``."""

    def test_actinomma_ex_gr_boreale(self) -> None:
        """``Actinomma ex gr. boreale`` — the canonical ICZN shape.

        Pre-fix the epithet was ``"ex"`` and the qualifier was
        ``"gr. boreale"``.  Post-fix the epithet is ``"boreale"``
        and the qualifier is exactly ``"ex gr."``.
        """
        result = _taxon_parts("Actinomma ex gr. boreale")
        assert result["genus"] == "Actinomma", (
            f"genus should be 'Actinomma' (first token), got {result['genus']!r}"
        )
        assert result["specific_epithet"] == "boreale", (
            f"epithet should be 'boreale' (species after ex gr. marker), "
            f"got {result['specific_epithet']!r}"
        )
        assert result["qualifier"] == "ex gr.", (
            f"qualifier should be 'ex gr.' (multi-word ICZN marker), got {result['qualifier']!r}"
        )

    def test_triactoma_ex_gr_kamoensis(self) -> None:
        """``Triactoma ex gr. kamoensis`` — the other common shape.

        Same expected split as the Actinomma case; confirms the
        fix is not genus-specific.
        """
        result = _taxon_parts("Triactoma ex gr. kamoensis")
        assert result["genus"] == "Triactoma"
        assert result["specific_epithet"] == "kamoensis"
        assert result["qualifier"] == "ex gr."

    def test_ex_group_variant_merged(self) -> None:
        """``Genus ex group species`` — the "group" spelling of the
        ``gr.`` abbreviation is also accepted (ICZN allows both
        forms; the merged token is normalised to ``"ex gr."``).
        """
        result = _taxon_parts("Genus ex group species")
        assert result["genus"] == "Genus"
        assert result["specific_epithet"] == "species"
        assert result["qualifier"] == "ex gr.", (
            f"qualifier should be normalised to 'ex gr.', got {result['qualifier']!r}"
        )

    def test_ex_gr_case_insensitive(self) -> None:
        """The pre-processing is case-insensitive — ``ex Gr.``
        capitalised differently from the genus (OCR variability)
        must still merge.
        """
        result = _taxon_parts("Genus ex Gr. species")
        assert result["genus"] == "Genus"
        assert result["specific_epithet"] == "species"
        assert result["qualifier"] == "ex gr."

    def test_ex_gr_no_epithet(self) -> None:
        """``Genus ex gr.`` (no species assigned) — the qualifier is
        ``"ex gr."`` and the epithet is ``None``.  This is the
        morphologically-uncertain group case.
        """
        result = _taxon_parts("Genus ex gr.")
        assert result["genus"] == "Genus"
        assert result["specific_epithet"] is None, (
            f"epithet should be None when ex gr. has no species, got {result['specific_epithet']!r}"
        )
        assert result["qualifier"] == "ex gr."

    def test_ex_gr_with_authority(self) -> None:
        """``Genus ex gr. species (Smith)`` — confirms the M-2 fix
        is not disturbed by the postfix-subgenus heuristic that
        precedes the ex-gr merge: the ``(Smith)`` parenthesised
        token gets swept into ``generic_name`` (pre-existing
        behavior, see Phase 63 Plan 6.18 / Bug 6.18 audit) but the
        ex-gr splitting still works correctly on the remaining
        tokens, emitting ``ex gr.`` as the qualifier and
        ``species`` as the epithet.

        We intentionally do NOT assert ``authority == "(Smith)"``
        because the postfix-subgenus branch (Phase 63 audit) fires
        first and captures the paren as ``generic_name``.  M-2 is
        only responsible for the ex-gr token split.
        """
        result = _taxon_parts("Genus ex gr. species (Smith)")
        assert result["genus"] == "Genus"
        assert result["specific_epithet"] == "species", (
            f"epithet should be 'species' after the ex gr. marker, "
            f"got {result['specific_epithet']!r}"
        )
        assert result["qualifier"] == "ex gr.", (
            f"qualifier should be 'ex gr.' (NOT 'ex gr. species'), got {result['qualifier']!r}"
        )


class TestExGrRegression:
    """M-2 must not break the OTHER open-nomenclature shapes that
    the legacy _taxon_parts already handled."""

    def test_cf_qualifier_still_works(self) -> None:
        """``Genus cf. species`` — the epithet is dropped (cf.
        REPLACES the species), qualifier includes both tokens.
        """
        result = _taxon_parts("Genus cf. species")
        assert result["genus"] == "Genus"
        assert result["specific_epithet"] is None, (
            "cf. must REPLACE the species, not coexist with it"
        )
        assert result["qualifier"] == "cf. species"

    def test_aff_qualifier_still_works(self) -> None:
        """``Genus aff. species`` — same shape as cf."""
        result = _taxon_parts("Genus aff. species")
        assert result["genus"] == "Genus"
        assert result["specific_epithet"] is None
        assert result["qualifier"] == "aff. species"

    def test_undetermined_species_still_works(self) -> None:
        """``Genus sp.`` — epithet is None, qualifier is "sp" / "sp."."""
        result = _taxon_parts("Genus sp.")
        assert result["genus"] == "Genus"
        assert result["specific_epithet"] is None
        assert result["qualifier"] is not None
        assert result["qualifier"].lower().startswith("sp")

    def test_new_species_still_works(self) -> None:
        """``Genus n. sp.`` — the P1-2 n.sp. qualifier is preserved."""
        result = _taxon_parts("Genus n. sp.")
        assert result["genus"] == "Genus"
        assert result["specific_epithet"] is None
        assert result["qualifier"] is not None
        # n. sp. handling keeps the multi-token qualifier together
        assert "n" in result["qualifier"]
        assert "sp" in result["qualifier"]

    def test_full_binomial_still_works(self) -> None:
        """``Genus species`` — no regression on the canonical shape."""
        result = _taxon_parts("Genus species")
        assert result["genus"] == "Genus"
        assert result["specific_epithet"] == "species"
        assert result["qualifier"] is None

    def test_authority_with_year_still_works(self) -> None:
        """The M-2 pre-processing must not steal authority tokens."""
        result = _taxon_parts("Genus species (Smith, 1900)")
        assert result["genus"] == "Genus"
        assert result["specific_epithet"] == "species"
        assert result["authority"] == "(Smith, 1900)"
        assert result["qualifier"] is None


class TestExGrEndToEnd:
    """M-2 through the full ``taxon_records_from_matches`` pipeline."""

    def test_ex_gr_propagates_to_taxon_record(self) -> None:
        """The ex gr. split must reach the final ``TaxonRecord``."""
        m = _make_match(species="Actinomma ex gr. boreale")
        taxa = taxon_records_from_matches([m])
        assert len(taxa) == 1
        t = taxa[0]
        assert t["genus"] == "Actinomma"
        assert t["specific_epithet"] == "boreale"
        assert t["qualifier"] == "ex gr."


# ---------------------------------------------------------------------------
# M-3: source / extraction_method split
# ---------------------------------------------------------------------------


class TestSourceExtractionSplit:
    """M-3: ``source`` = taxonomic data source; ``extraction_method``
    is a separate structured field."""

    def test_pbdb_source_marks_paleodb(self) -> None:
        """When PBDB provides family / order / class, ``source`` is
        ``"paleodb"`` (NOT the extraction method).
        """
        m = _make_match(
            species="Genus species",
            extraction_method="llm_first",
            pbdb={
                "family": "Actinommidae",
                "order": "Spumellaria",
                "class": "Polycystina",
            },
        )
        taxa = taxon_records_from_matches([m])
        assert len(taxa) == 1
        t = taxa[0]
        assert t["source"] == "paleodb", (
            f"source should be 'paleodb' when PBDB provides taxonomy, got {t['source']!r}"
        )
        # The extraction method is now in a separate field.
        assert t["extraction_method"] == "llm_first", (
            f"extraction_method should be 'llm_first', got {t['extraction_method']!r}"
        )

    def test_no_pbdb_source_is_none_string(self) -> None:
        """Without PBDB, ``source`` is ``"none"`` (string, not None —
        distinguishes "source not used" from "source is null" for
        downstream filtering).
        """
        m = _make_match(species="Genus species", extraction_method="regex")
        taxa = taxon_records_from_matches([m])
        assert len(taxa) == 1
        t = taxa[0]
        assert t["source"] == "none", f"source should be 'none' without PBDB, got {t['source']!r}"
        assert t["extraction_method"] == "regex"

    def test_no_pbdb_llm_first_source_is_none_string(self) -> None:
        """Without PBDB but with llm_first extraction, the bug case:
        ``source`` is ``"none"`` (not "llm_first").  The extraction
        method is in the new structured field."""
        m = _make_match(species="Genus species", extraction_method="llm_first")
        taxa = taxon_records_from_matches([m])
        assert len(taxa) == 1
        t = taxa[0]
        assert t["source"] == "none", (
            f"source should NOT be 'llm_first' (M-3 fix), got {t['source']!r}"
        )
        assert t["extraction_method"] == "llm_first"

    def test_no_pbdb_hybrid_source_is_none_string(self) -> None:
        """Hybrid extraction without PBDB."""
        m = _make_match(species="Genus species", extraction_method="hybrid")
        taxa = taxon_records_from_matches([m])
        assert len(taxa) == 1
        t = taxa[0]
        assert t["source"] == "none"
        assert t["extraction_method"] == "hybrid"

    def test_taxon_remarks_still_carries_extraction_method_string(self) -> None:
        """Backwards-compat: ``taxon_remarks`` still embeds the
        ``"extraction_method=..."`` string for downstream consumers
        that already filter on it (Phase 63 Plan 6.19 contract).
        """
        m = _make_match(species="Genus species", extraction_method="llm_first")
        taxa = taxon_records_from_matches([m])
        t = taxa[0]
        assert t["taxon_remarks"] is not None
        assert "llm_first" in t["taxon_remarks"], (
            f"taxon_remarks should still embed extraction_method "
            f"for backwards-compat, got {t['taxon_remarks']!r}"
        )


class TestTaxonRecordSchema:
    """M-3: ``TaxonRecord`` accepts the new structured
    ``extraction_method`` field."""

    def test_extraction_method_field_exists(self) -> None:
        """The schema must declare the new field."""
        fields = TaxonRecord.model_fields
        assert "extraction_method" in fields, (
            "TaxonRecord is missing 'extraction_method' — M-3 fix regressed?"
        )

    def test_extraction_method_default_is_none(self) -> None:
        """The new field defaults to ``None`` (matches the schema
        being optional / additive)."""
        rec = TaxonRecord(
            taxon_id="tx_1",
            verbatim_name="Genus species",
            normalized_name="Genus species",
            genus="Genus",
            specific_epithet="species",
        )
        assert rec.extraction_method is None, (
            f"extraction_method should default to None, got {rec.extraction_method!r}"
        )

    def test_extraction_method_accepts_string(self) -> None:
        """The new field accepts a string extraction method value."""
        rec = TaxonRecord(
            taxon_id="tx_1",
            verbatim_name="Genus species",
            normalized_name="Genus species",
            genus="Genus",
            specific_epithet="species",
            extraction_method="llm_first",
        )
        assert rec.extraction_method == "llm_first"

    def test_source_field_still_present_for_backcompat(self) -> None:
        """``source`` is kept (not removed) so downstream consumers
        that already read it don't break.
        """
        fields = TaxonRecord.model_fields
        assert "source" in fields, "TaxonRecord.source removed — backwards-compat broken?"
        rec = TaxonRecord(
            taxon_id="tx_1",
            verbatim_name="Genus species",
            normalized_name="Genus species",
            genus="Genus",
            specific_epithet="species",
            source="paleodb",
        )
        assert rec.source == "paleodb"


class TestSourceVsExtractionMethod:
    """Cross-cutting: ``source`` and ``extraction_method`` are
    independent fields.  PBDB=true may co-occur with any
    extraction_method; PBDB=false forces source='none' regardless
    of extraction_method."""

    def test_pbdb_with_regex_extraction(self) -> None:
        m = _make_match(
            species="Genus species",
            extraction_method="regex",
            pbdb={"family": "Actinommidae"},
        )
        taxa = taxon_records_from_matches([m])
        t = taxa[0]
        assert t["source"] == "paleodb"
        assert t["extraction_method"] == "regex"

    def test_pbdb_with_hybrid_extraction(self) -> None:
        m = _make_match(
            species="Genus species",
            extraction_method="hybrid",
            pbdb={"family": "Actinommidae", "order": "Spumellaria"},
        )
        taxa = taxon_records_from_matches([m])
        t = taxa[0]
        assert t["source"] == "paleodb"
        assert t["extraction_method"] == "hybrid"

    def test_no_pbdb_with_empty_extraction_method(self) -> None:
        """Empty extraction_method → ``extraction_method`` is None
        (cleaned by ``or None``); source is "none"."""
        m = _make_match(species="Genus species", extraction_method="")
        taxa = taxon_records_from_matches([m])
        t = taxa[0]
        assert t["source"] == "none"
        assert t["extraction_method"] is None
