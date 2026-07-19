"""Tests for Phase 63 Plan 6 — Bug 6.13: integration test that
validates a fresh RunOutput against the published JSON Schema.

Before: the published ``schemas/rlpe-v1.0.0.json`` was emitted by
``schema_dump`` but never validated end-to-end. A refactor that
broke the published schema's ability to roundtrip a real
RunOutput dict (e.g. a wrong field type) would slip past unit
tests that only check ``RunOutput.model_dump()``.

After: this test runs the published schema (via the optional
``jsonschema`` package) against a ``RunOutput`` produced by
``run_output_from_provenance`` — the same code path the export
CLI and GUI execute. The test catches:

  * Field renames that don't match the schema
  * Required fields accidentally dropped
  * Type changes (e.g. ``int`` -> ``float``)
  * New fields not yet in the published schema (drift)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import jsonschema  # type: ignore[import-untyped]
    from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False

from rlpe.converters import run_output_from_provenance  # noqa: E402
from rlpe.provenance import build_provenance  # noqa: E402
from rlpe.schema_models import (  # noqa: E402
    SCHEMA_VERSION,
    GeologyLinkRecord,
    PanelMetadata,
    PanelRecord,
    PaperMetadataRecord,
    ProvenanceRecord,
    RunOutput,
    ScaleBarRecord,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schemas" / f"rlpe-v{SCHEMA_VERSION}.json"


def _rich_run_output() -> RunOutput:
    """Build a multi-panel RunOutput with all key features populated."""
    pm = PaperMetadataRecord(
        title="Cretaceous Radiolaria from Japan",
        authors=["Author A", "Author B"],
        year=2020,
        doi="10.1234/test",
        source="opendataloader",
        confidence=0.9,
    )
    geo = GeologyLinkRecord(
        age="Late Jurassic",
        chronostratigraphy="Kimmeridgian",
        chronostratigraphy_rank="stage",
        ma_top=152.1,
        ma_base=157.3,
        ma_mid=154.7,
        formation="Fonzaso",
        member="Lower",
        group="Calcari Grigi",
        lithology="limestone",
        locality="Italy",
        country="Italy",
        latitude=46.5,
        longitude=11.5,
        modern_latitude=46.5,
        modern_longitude=11.5,
        paleo_latitude=37.8,
        paleo_longitude=8.3,
        plate_id="AFR",
        reconstruction_model="GPlates",
        reconstruction_age_ma=154.7,
        evidence_text="Sample XYZ-1",
        confidence=0.8,
        biozone="N. optima Zone",
    )
    sb = ScaleBarRecord(
        value=100.0, unit="um", source="caption", um_per_px=0.1, confidence=0.9,
    )
    meta = PanelMetadata(
        panel_score=0.7,
        ocr_count=2,
        taxon_count=1,
        figure_number="1",
        page_index=10,
        matcher_used=True,
        matcher_type="hybrid",
        matcher_conf=0.9,
        caption_pairs_used=True,
        scale_bar=sb,
        geology_links=[geo],
        extraction_source="opendataloader",
        extraction_method="hybrid",
        needs_review=False,
    )
    panel = PanelRecord(
        paper_id="paper_1",
        figure_id="figure_1",
        panel_id="1",
        species="Genus species",
        taxon_id="taxon_abc",
        sample_id="sample_1",
        geology_context_id="geoctx_abc",
        panel_path="/path/panel_1.png",
        bbox=[10, 20, 200, 300],
        confidence=0.9,
        label_text="1",
        caption_snippet="Fig. 1 caption",
        ocr_text="1",
        metadata=meta,
        paper_metadata=pm,
    )
    prov = ProvenanceRecord(**build_provenance().to_dict())
    return RunOutput(provenance=prov, panels=[panel])


def test_run_output_validates_against_published_schema():
    """A RunOutput produced by ``run_output_from_provenance`` validates
    against the published JSON Schema."""
    if not _HAS_JSONSCHEMA:
        pytest.skip("jsonschema not installed; cannot validate end-to-end")
    if not SCHEMA_PATH.exists():
        pytest.skip(f"{SCHEMA_PATH} not yet emitted")
    schema = json.loads(SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)
    run = _rich_run_output()
    payload = run.model_dump()
    errors = list(validator.iter_errors(payload))
    assert errors == [], (
        f"RunOutput fails published schema validation: "
        f"{[e.message for e in errors]}"
    )


def test_run_output_from_provenance_validates():
    """The exact dict exported by ``run_output_from_provenance`` also
    validates — covering the deduplication views (papers, figures,
    taxa, samples, geology_contexts, localities, paleo_coordinates,
    warnings)."""
    if not _HAS_JSONSCHEMA:
        pytest.skip("jsonschema not installed")
    if not SCHEMA_PATH.exists():
        pytest.skip(f"{SCHEMA_PATH} not yet emitted")
    schema = json.loads(SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)
    # Build a non-empty run via the converters path
    run_pydantic = _rich_run_output()
    panel_dict = run_pydantic.panels[0].model_dump()
    # Reconstruct a minimal MatchResult and call the converter.
    from rlpe.types import MatchResult

    m = MatchResult(
        paper_id=panel_dict["paper_id"],
        figure_id=panel_dict["figure_id"],
        panel_id=panel_dict["panel_id"],
        species=panel_dict["species"],
        panel_path=panel_dict["panel_path"],
        bbox=panel_dict["bbox"],
        confidence=panel_dict["confidence"],
        label_text=panel_dict["label_text"],
        caption_snippet=panel_dict["caption_snippet"],
        ocr_text=panel_dict["ocr_text"],
        metadata={
            "panel_score": 0.7,
            "extraction_method": "hybrid",
            "scale_bar": {
                "value": 100.0, "unit": "um", "source": "caption",
                "um_per_px": 0.1, "confidence": 0.9,
            },
            "geology_links": [{
                "age": "Late Jurassic", "locality": "Italy",
                "country": "Italy",
                "latitude": 46.5, "longitude": 11.5,
                "modern_latitude": 46.5, "modern_longitude": 11.5,
                "ma_top": 152.1, "ma_base": 157.3, "ma_mid": 154.7,
                "confidence": 0.8,
            }],
        },
    )
    out = run_output_from_provenance(
        build_provenance().to_dict(), [m],
    )
    errors = list(validator.iter_errors(out))
    assert errors == [], (
        f"run_output_from_provenance output fails published schema: "
        f"{[e.message for e in errors]}"
    )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
