"""Tests for rlpe.schema_models and rlpe.converters."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rlpe.converters import (
    panel_metadata_from_match,
    panel_record_from_match,
    paper_metadata_from_internal,
    run_output_from_provenance,
)
from rlpe.provenance.stamp import build_provenance
from rlpe.schema_models import (
    PanelRecord,
    ProvenanceRecord,
    RunOutput,
    SCHEMA_VERSION,
    emit_json_schema,
    validate_run_output,
)
from rlpe.types import MatchResult, PaperMetadata


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPanelRecord:
    def test_minimal_valid(self):
        r = PanelRecord(
            paper_id="p1",
            figure_id="f1",
            panel_id="1",
            species="Genus species",
            panel_path=None,
            bbox=None,
            confidence=0.5,
        )
        d = r.model_dump()
        assert d["paper_id"] == "p1"
        assert d["confidence"] == 0.5
        assert d["metadata"]["ocr_count"] == 0  # default

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            PanelRecord(
                paper_id="p1",
                figure_id="f1",
                panel_id="1",
                species=None,
                panel_path=None,
                bbox=None,
                confidence=0.0,
                bogus_extra_field="oops",  # type: ignore[call-arg]
            )

    def test_bbox_must_have_four_ints(self):
        with pytest.raises(ValidationError):
            PanelRecord(
                paper_id="p1",
                figure_id="f1",
                panel_id=None,
                species=None,
                panel_path=None,
                bbox=[1, 2, 3],  # only 3 elements
                confidence=0.0,
            )


class TestRunOutput:
    def _provenance(self) -> ProvenanceRecord:
        p = build_provenance()
        return ProvenanceRecord(**p.to_dict())

    def test_valid_run_output(self):
        prov = self._provenance()
        out = RunOutput(provenance=prov, panels=[])
        assert out.schema_version == SCHEMA_VERSION
        assert out.provenance.pipeline_version == prov.pipeline_version

    def test_provenance_required(self):
        with pytest.raises(ValidationError):
            RunOutput(panels=[])  # type: ignore[call-arg]

    def test_round_trip_through_json(self):
        prov = self._provenance()
        out = RunOutput(
            provenance=prov,
            panels=[PanelRecord(
                paper_id="p1", figure_id="f1", panel_id="1",
                species="Genus sp", panel_path=None, bbox=[0,0,1,1],
                confidence=0.7,
            )],
        )
        s = json.dumps(out.model_dump())
        reloaded = json.loads(s)
        assert reloaded["panels"][0]["species"] == "Genus sp"

    def test_validate_run_output_function(self):
        prov = self._provenance()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "provenance": prov.model_dump(),
            "panels": [],
        }
        loaded = validate_run_output(payload)
        assert loaded.provenance.git_commit == prov.git_commit


class TestSchemaDump:
    def test_emit_creates_file(self, tmp_path):
        target = tmp_path / "schema.json"
        out = emit_json_schema(target)
        assert out == target
        assert out.exists()
        data = json.loads(out.read_text())
        assert "properties" in data
        assert "panels" in data["properties"]
        assert "provenance" in data["properties"]
        assert data["title"] == "RLPE RunOutput"

    def test_emit_includes_nested_definitions(self, tmp_path):
        target = tmp_path / "schema.json"
        emit_json_schema(target)
        data = json.loads(target.read_text())
        defs = data.get("$defs", {})
        assert "PanelRecord" in defs
        assert "ProvenanceRecord" in defs
        assert "ScaleBarRecord" in defs
        assert "GeologyLinkRecord" in defs

    def test_emit_idempotent(self, tmp_path):
        target = tmp_path / "schema.json"
        emit_json_schema(target)
        first = target.read_text()
        emit_json_schema(target)
        second = target.read_text()
        assert first == second


def _make_match() -> MatchResult:
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
        species="Genus species",
        panel_path="/path/to/panel.png",
        bbox=[10, 20, 100, 200],
        confidence=0.6,
        label_text="1",
        caption_snippet="Plate 1\n1) Genus species",
        ocr_text=None,
        metadata={
            "panel_score": 0.6,
            "ocr_count": 5,
            "taxon_count": 1,
            "figure_number": "1",
            "page_index": 11,
            "matcher_used": False,
            "matcher_type": "heuristic",
            "matcher_conf": 0.0,
            "caption_pairs_used": True,
            "scale_bar": {"value": 100.0, "unit": "um", "source": "caption",
                          "pixel_length": 962.0, "um_per_px": 0.104,
                          "confidence": 0.8},
            "geology_links": [
                {"age": "Late Jurassic", "chronostratigraphy": "Kimmeridgian",
                 "formation": "Fonzaso", "locality": "Italy", "confidence": 0.7},
            ],
            "m3_diagnostic": {},
            "extraction_source": "opendataloader",
        },
        paper_metadata=pm,
    )


class TestConverters:
    def test_panel_record_from_match(self):
        m = _make_match()
        rec = panel_record_from_match(m)
        d = rec.model_dump()
        assert d["paper_id"] == "abc"
        assert d["bbox"] == [10, 20, 100, 200]
        assert d["metadata"]["scale_bar"]["unit"] == "um"
        assert len(d["metadata"]["geology_links"]) == 1
        assert d["paper_metadata"]["year"] == 2020

    def test_panel_metadata_handles_empty_meta(self):
        m = MatchResult(
            paper_id="p", figure_id="f", panel_id=None, species=None,
            panel_path=None, bbox=None, confidence=0.0, metadata={},
        )
        pm = panel_metadata_from_match(m)
        assert pm.ocr_count == 0
        assert pm.taxon_count == 0
        assert pm.scale_bar is None
        assert pm.geology_links == []

    def test_paper_metadata_none(self):
        assert paper_metadata_from_internal(None) is None

    def test_run_output_from_provenance(self):
        prov = ProvenanceRecord(**build_provenance().to_dict())
        out = run_output_from_provenance(prov, [_make_match()])
        assert out["schema_version"] == SCHEMA_VERSION
        assert len(out["panels"]) == 1
        # Re-validate the produced dict through Pydantic
        validated = validate_run_output(out)
        assert len(validated.panels) == 1
