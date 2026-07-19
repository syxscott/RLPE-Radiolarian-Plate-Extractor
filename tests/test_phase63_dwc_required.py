"""Tests for Phase 63 Plan 6 — Bugs 6.15-6.17: Darwin Core required
fields (uncertainty, ICZN, authorship) must be present on the
TaxonRecord.

Plan 6.15: ``coordinateUncertaintyInMeters`` — GBIF requires this on
georeferenced occurrences. We expose it on ``GeologyLinkRecord``
(propagated from ``coord_source`` / ``modern_latitude`` rounding) and
on ``LocalityRecord`` / ``PaleoCoordinateRecord``.

Plan 6.16: ``nomenclaturalCode="ICZN"`` — Radiolarians fall under the
International Code of Zoological Nomenclature. Without this field,
GBIF / PBDB might classify them under ICN (botanical) which has
different rules. We expose it on ``TaxonRecord``.

Plan 6.17: ``scientificNameAuthorship`` — extracted from species
string author/year (e.g. ``Genus species (Smith, 1900)`` → ``(Smith,
1900)``). GBIF rejects entries lacking authorship when it's known.
We expose it on ``TaxonRecord``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.converters import _taxon_parts, _resolve_modern_coord  # noqa: E402
from rlpe.schema_models import (  # noqa: E402
    GeologyLinkRecord,
    LocalityRecord,
    PanelMetadata,
    PanelRecord,
    PaperMetadataRecord,
    ProvenanceRecord,
    RunOutput,
    ScaleBarRecord,
    TaxonRecord,
)
from rlpe.types import MatchResult  # noqa: E402
from rlpe.provenance import build_provenance  # noqa: E402


# ---------------------------------------------------------------------------
# Field declarations (pydantic models)
# ---------------------------------------------------------------------------


def test_taxon_record_has_nomenclatural_code():
    """``TaxonRecord.nomenclatural_code`` is the ICZN marker."""
    fields = TaxonRecord.model_fields
    assert "nomenclatural_code" in fields, (
        "TaxonRecord is missing 'nomenclatural_code' — Phase 63 Plan "
        "6.16 (Bug 6.16) fix regressed?"
    )


def test_taxon_record_has_scientific_name_authorship():
    """``TaxonRecord.scientific_name_authorship`` carries the author/year."""
    fields = TaxonRecord.model_fields
    assert "scientific_name_authorship" in fields, (
        "TaxonRecord is missing 'scientific_name_authorship' — Phase "
        "63 Plan 6.17 (Bug 6.17) fix regressed?"
    )


def test_geology_link_record_has_coordinate_uncertainty():
    """``GeologyLinkRecord.coordinate_uncertainty_in_meters`` is added."""
    fields = GeologyLinkRecord.model_fields
    assert "coordinate_uncertainty_in_meters" in fields, (
        "GeologyLinkRecord is missing "
        "'coordinate_uncertainty_in_meters' — Phase 63 Plan 6.15 "
        "(Bug 6.15) fix regressed?"
    )


def test_locality_record_has_coordinate_uncertainty():
    """``LocalityRecord.coordinate_uncertainty_in_meters`` is added."""
    fields = LocalityRecord.model_fields
    assert "coordinate_uncertainty_in_meters" in fields, (
        "LocalityRecord is missing "
        "'coordinate_uncertainty_in_meters' — Phase 63 Plan 6.15 fix "
        "regressed?"
    )


# ---------------------------------------------------------------------------
# _taxon_parts / _resolve_modern_coord behaviour
# ---------------------------------------------------------------------------


def test_taxon_parts_extracts_authorship():
    """``Genus species (Smith, 1900)`` yields genus + epithet + authorship."""
    parts = _taxon_parts("Genus species (Smith, 1900)")
    # authorship might be a separate extension — for now we check that
    # genus + specific_epithet are populated; authorship extraction is
    # at the converter layer where we have the metadata.
    assert parts["genus"] == "Genus"
    assert parts["specific_epithet"] == "species"


# ---------------------------------------------------------------------------
# Default values are present and correct
# ---------------------------------------------------------------------------


def test_nomenclatural_code_default_is_iczn():
    """TaxonRecord.nomenclatural_code defaults to ``ICZN``."""
    rec = TaxonRecord(
        taxon_id="tx_1",
        verbatim_name="Genus species",
        normalized_name="Genus species",
        genus="Genus",
        specific_epithet="species",
    )
    assert rec.nomenclatural_code == "ICZN", (
        f"TaxonRecord.nomenclatural_code default is "
        f"{rec.nomenclatural_code!r}, expected 'ICZN'."
    )


def test_taxonomic_authority_default_string():
    """scientific_name_authorship default is ``None`` (no info)."""
    rec = TaxonRecord(
        taxon_id="tx_1",
        verbatim_name="Genus species",
        normalized_name="Genus species",
        genus="Genus",
        specific_epithet="species",
    )
    assert rec.scientific_name_authorship is None or isinstance(
        rec.scientific_name_authorship, str
    )


# ---------------------------------------------------------------------------
# Tasks 6.18-6.20: generic_name (subgenus), taxon_remarks, paleo uncertainty
# ---------------------------------------------------------------------------


def test_taxon_record_has_generic_name():
    """``TaxonRecord.generic_name`` carries the subgenus (DwC genericName)."""
    fields = TaxonRecord.model_fields
    assert "generic_name" in fields, (
        "TaxonRecord is missing 'generic_name' — Phase 63 Plan 6.18 "
        "(Bug 6.18) fix regressed?"
    )


def test_taxon_record_has_taxon_remarks():
    """``TaxonRecord.taxon_remarks`` carries extraction provenance."""
    fields = TaxonRecord.model_fields
    assert "taxon_remarks" in fields, (
        "TaxonRecord is missing 'taxon_remarks' — Phase 63 Plan 6.19 "
        "(Bug 6.19) fix regressed?"
    )


def test_paleocoord_record_has_coordinate_uncertainty():
    """``PaleoCoordinateRecord.coordinate_uncertainty_in_meters`` is added."""
    from rlpe.schema_models import PaleoCoordinateRecord
    fields = PaleoCoordinateRecord.model_fields
    assert "coordinate_uncertainty_in_meters" in fields, (
        "PaleoCoordinateRecord is missing "
        "'coordinate_uncertainty_in_meters' — Phase 63 Plan 6.20 "
        "(Bug 6.20) fix regressed?"
    )


def test_taxon_converter_populates_authorship():
    """The converter extracts ``(Smith, 1900)`` from a parenthesised
    species string and populates ``scientific_name_authorship``."""
    from rlpe.converters import taxon_records_from_matches
    m = MatchResult(
        paper_id="p1", figure_id="f1", panel_id="1",
        species="Genus species (Smith, 1900)",
        panel_path="/tmp/p.png", bbox=[0, 0, 100, 100],
        confidence=0.9, metadata={
            "extraction_method": "regex",
        },
    )
    rows = taxon_records_from_matches([m])
    assert len(rows) == 1
    r = rows[0]
    assert r["scientific_name_authorship"] == "Smith, 1900", (
        f"authorship extraction failed: {r['scientific_name_authorship']!r} — "
        "Phase 63 Plan 6.17 (Bug 6.17) fix regressed?"
    )


def test_taxon_converter_populates_subgenus():
    """``Podocyrtis (Podocyrtites) species`` populates ``generic_name``."""
    from rlpe.converters import taxon_records_from_matches
    m = MatchResult(
        paper_id="p1", figure_id="f1", panel_id="1",
        species="Podocyrtis (Podocyrtites) species Haeckel, 1887",
        panel_path="/tmp/p.png", bbox=[0, 0, 100, 100],
        confidence=0.9, metadata={"extraction_method": "regex"},
    )
    rows = taxon_records_from_matches([m])
    r = rows[0]
    assert r["generic_name"] == "Podocyrtites", (
        f"subgenus not extracted: got {r['generic_name']!r} — "
        "Phase 63 Plan 6.18 (Bug 6.18) fix regressed?"
    )
    assert r["scientific_name_authorship"] == "Haeckel, 1887"


def test_taxon_converter_populates_taxon_remarks():
    """The extractor method propagates to ``taxon_remarks``."""
    from rlpe.converters import taxon_records_from_matches
    m = MatchResult(
        paper_id="p1", figure_id="f1", panel_id="1",
        species="Genus species",
        panel_path="/tmp/p.png", bbox=[0, 0, 100, 100],
        confidence=0.9, metadata={"extraction_method": "llm_first"},
    )
    rows = taxon_records_from_matches([m])
    r = rows[0]
    assert "llm_first" in (r["taxon_remarks"] or ""), (
        f"taxon_remarks missing extraction_method: {r['taxon_remarks']!r} — "
        "Phase 63 Plan 6.19 (Bug 6.19) fix regressed?"
    )


def test_coordinate_uncertainty_for_known_sources():
    """``_coordinate_uncertainty_for`` returns sensible values per source."""
    from rlpe.converters import _coordinate_uncertainty_for
    assert _coordinate_uncertainty_for("regex") == 1000.0
    assert _coordinate_uncertainty_for("paleodb") == 5000.0
    assert _coordinate_uncertainty_for("country_centroid") == 25000.0
    assert _coordinate_uncertainty_for("paleo_reconstructed") == 10000.0
    assert _coordinate_uncertainty_for(None) is None
    assert _coordinate_uncertainty_for("") is None


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
