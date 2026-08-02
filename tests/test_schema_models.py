"""Tests for rlpe.schema_models and rlpe.converters."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from rlpe.converters import (
    _resolve_modern_coord,
    _taxon_parts,
    figure_records_from_matches,
    geology_contexts_from_matches,
    locality_records_from_geology,
    panel_metadata_from_match,
    panel_record_from_match,
    paper_metadata_from_internal,
    paper_records_from_matches,
    run_output_from_provenance,
    sample_records_from_matches,
    taxon_records_from_matches,
    warnings_from_matches,
)
from rlpe.provenance.stamp import build_provenance
from rlpe.schema_models import (
    SCHEMA_VERSION,
    PanelRecord,
    PaperMetadataRecord,
    ProvenanceRecord,
    RunOutput,
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

    def test_reassignment_metadata_fields_round_trip(self):
        """Runtime cross-figure reassignment adds reassigned_from_figure and
        reassigned_reason to metadata. These must be part of the published
        schema or the export step rejects the record."""
        r = PanelRecord(
            paper_id="p1",
            figure_id="f1",
            panel_id="1",
            species=None,
            panel_path=None,
            bbox=None,
            confidence=0.5,
            metadata={
                "reassigned_from_figure": "od_fig_X_p001_01",
                "reassigned_reason": "neighbor caption match",
            },
        )
        assert r.metadata.reassigned_from_figure == "od_fig_X_p001_01"
        assert r.metadata.reassigned_reason == "neighbor caption match"
        # round-trip through JSON
        d = json.loads(r.model_dump_json())
        assert d["metadata"]["reassigned_from_figure"] == "od_fig_X_p001_01"


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
            panels=[
                PanelRecord(
                    paper_id="p1",
                    figure_id="f1",
                    panel_id="1",
                    species="Genus sp",
                    panel_path=None,
                    bbox=[0, 0, 1, 1],
                    confidence=0.7,
                )
            ],
        )
        s = json.dumps(out.model_dump())
        reloaded = json.loads(s)
        assert reloaded["panels"][0]["species"] == "Genus sp"

    def test_round_trip_via_pydantic_json(self):
        """Full Pydantic round-trip: model_dump_json() → model_validate_json()
        → reloaded == original. The previous round-trip test only used
        json.dumps on the dict, which loses Pydantic coercion (None vs
        missing-key, list[int] vs tuple, etc.). This test is the canonical
        "can I write a record to disk and read it back" guard.
        """
        prov = self._provenance()
        # Build a fully-populated record: nested ScaleBarRecord +
        # GeologyLinkRecord + PaperMetadataRecord + reassigned metadata
        pm = PaperMetadataRecord(
            title="Round-trip test",
            authors=["Author A", "Author B"],
            year=2020,
            doi="10.1000/test",
            source="opendataloader",
            confidence=0.8,
        )
        rec = PanelRecord(
            paper_id="p1",
            figure_id="f1",
            panel_id="1",
            species="Genus sp",
            panel_path="/path/to/panel.png",
            bbox=[10, 20, 100, 200],
            confidence=0.7,
            label_text="1",
            caption_snippet="Plate 1\n1) Genus sp.",
            ocr_text="1 Genus sp",
            metadata={
                "panel_score": 0.6,
                "ocr_count": 5,
                "taxon_count": 1,
                "figure_number": "1",
                "page_index": 11,
                "matcher_used": True,
                "matcher_type": "taxon-recogniser",
                "matcher_conf": 0.85,
                "caption_pairs_used": True,
                "scale_bar": {
                    "value": 100.0,
                    "unit": "um",
                    "source": "caption",
                    "pixel_length": 962.0,
                    "um_per_px": 0.104,
                    "confidence": 0.8,
                },
                "geology_links": [
                    {
                        "age": "Late Jurassic",
                        "chronostratigraphy": "Kimmeridgian",
                        "formation": "Fonzaso",
                        "locality": "Italy",
                        "confidence": 0.7,
                    },
                    {
                        "age": "Late Jurassic",
                        "chronostratigraphy": "Kimmeridgian",
                        "formation": "Fonzaso",
                        "locality": "Italy",
                        "confidence": 0.6,
                    },
                ],
                "m3_diagnostic": {"regex_groups": 3, "fallback_used": False},
                "extraction_source": "opendataloader",
                "reassigned_from_figure": "od_fig_X_p001_01",
                "reassigned_reason": "neighbor caption match",
            },
            paper_metadata=pm,
        )
        out = RunOutput(provenance=prov, panels=[rec])

        # Round-trip through Pydantic's JSON serialiser
        json_str = out.model_dump_json()
        reloaded = RunOutput.model_validate_json(json_str)

        # Top-level equality
        assert reloaded.schema_version == out.schema_version
        assert reloaded.provenance.git_commit == out.provenance.git_commit
        assert reloaded.provenance.input_sha256 == out.provenance.input_sha256
        # Panel-level equality
        rp = reloaded.panels[0]
        assert rp.paper_id == "p1"
        assert rp.bbox == [10, 20, 100, 200]
        assert rp.confidence == 0.7
        assert rp.metadata.scale_bar is not None
        assert rp.metadata.scale_bar.unit == "um"
        assert rp.metadata.scale_bar.pixel_length == 962.0
        assert len(rp.metadata.geology_links) == 2
        assert rp.metadata.geology_links[0].locality == "Italy"
        assert rp.metadata.reassigned_from_figure == "od_fig_X_p001_01"
        assert rp.metadata.m3_diagnostic == {"regex_groups": 3, "fallback_used": False}
        assert rp.paper_metadata is not None
        assert rp.paper_metadata.authors == ["Author A", "Author B"]
        # Full equality (Pydantic __eq__ compares all fields)
        assert reloaded == out

    def test_round_trip_via_dict_then_json(self):
        """The conversion path the export pipeline uses: Pydantic →
        dict → json.dumps → json.loads → Pydantic again. Each step
        must preserve all data; if any field is silently dropped on
        one step, this catches it."""
        prov = self._provenance()
        out = run_output_from_provenance(prov, [_make_match()])
        # out is a dict, not a RunOutput
        assert isinstance(out, dict)
        # Re-validate (Pydantic coerce dict → RunOutput)
        reloaded = validate_run_output(out)
        assert len(reloaded.panels) == 1
        # Drop the dict to JSON and back: simulates disk round-trip
        s = json.dumps(out, default=str)
        d2 = json.loads(s)
        reloaded2 = validate_run_output(d2)
        assert reloaded2.panels[0].paper_id == reloaded.panels[0].paper_id
        assert reloaded2.panels[0].paper_metadata.authors == ["Author A"]

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
            "scale_bar": {
                "value": 100.0,
                "unit": "um",
                "source": "caption",
                "pixel_length": 962.0,
                "um_per_px": 0.104,
                "confidence": 0.8,
            },
            "geology_links": [
                {
                    "age": "Late Jurassic",
                    "chronostratigraphy": "Kimmeridgian",
                    "formation": "Fonzaso",
                    "locality": "Italy",
                    "confidence": 0.7,
                },
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
            paper_id="p",
            figure_id="f",
            panel_id=None,
            species=None,
            panel_path=None,
            bbox=None,
            confidence=0.0,
            metadata={},
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


class TestProductDataPackage:
    """The first-stage product/data-package exports: papers, figures,
    taxa, samples, geology contexts, localities, warnings.

    These tests guard the contract of ``run_output_from_provenance`` so
    that any extractions from internal pipeline dictionaries surface
    into the published RunOutput without losing information.
    """

    def test_run_output_includes_all_entity_collections(self):
        prov = ProvenanceRecord(**build_provenance().to_dict())
        out = run_output_from_provenance(prov, [_make_match()])
        # New collections default to empty list and are part of the
        # published RunOutput. Validate through Pydantic to catch
        # schema drift.
        for key in (
            "papers",
            "figures",
            "taxa",
            "samples",
            "geology_contexts",
            "localities",
            "paleo_coordinates",
            "warnings",
        ):
            assert key in out, f"missing {key} in converter output"
            assert isinstance(out[key], list)
        validated = validate_run_output(out)
        # Validate schema_version is exactly the published version
        # (v1.1.0 as of audit 2026-08-02). Any drift here breaks
        # downstream consumers. The source of truth is
        # ``rlpe.schema_models.SCHEMA_VERSION`` — we read it through
        # the helper rather than hard-coding the literal so a
        # future minor bump only requires updating schema_models.py.
        from rlpe.schema_models import SCHEMA_VERSION

        assert validated.schema_version == SCHEMA_VERSION

    def test_paper_records_deduped_by_paper_id(self):
        m1 = _make_match()
        m2 = replace(_make_match(), figure_id="fig_2")
        # Round 23 audit: paper_records_from_matches now returns
        # ``(records, warnings)`` so the converter can surface
        # backend failures as WarningRecord entries. Tests must
        # unpack the tuple.
        papers, _warnings = paper_records_from_matches([m1, m2])
        assert len(papers) == 1
        assert papers[0]["paper_id"] == "abc"
        assert papers[0]["title"] == "Test"
        assert "Author A" in papers[0]["authors"]

    def test_figure_records_group_by_paper_and_figure(self):
        m1 = _make_match()
        m2 = replace(_make_match(), figure_id="fig_2")
        figs = figure_records_from_matches([m1, m2])
        assert len(figs) == 2
        keys = sorted((f["paper_id"], f["figure_id"]) for f in figs)
        assert keys == [("abc", "fig_1"), ("abc", "fig_2")]

    def test_taxon_records_carry_genus_epithet_qualifier(self):
        """``Genus cf. species`` is a binomial with the epithet
        undetermined — the parser conservatively emits the qualifier
        as the whole suffix and leaves the epithet empty. The full
        string is preserved in ``verbatim_name`` so the research
        output loses no information.
        """
        m = replace(_make_match(), species="Genus cf. species")
        taxa = taxon_records_from_matches([m])
        assert len(taxa) == 1
        t = taxa[0]
        assert t["genus"] == "Genus"
        assert t["specific_epithet"] is None
        assert t["qualifier"] == "cf. species"
        assert t["verbatim_name"] == "Genus cf. species"

    def test_taxon_records_handles_nested_author_citation(self):
        """bandini2011 pl08 / pl09 use the 'Genus species cf. S. excelsa'
        shape. The earlier dual-loop parser emitted qualifier="S. excelsa"
        and overwrote the epithet — this regression test guards the
        single-pass parser.
        """
        m = replace(_make_match(), species="Genus species cf. S. excelsa")
        taxa = taxon_records_from_matches([m])
        t = taxa[0]
        assert t["genus"] == "Genus"
        assert t["specific_epithet"] == "species"
        assert t["qualifier"] == "cf. S. excelsa"
        assert t["verbatim_name"] == "Genus species cf. S. excelsa"

    def test_taxon_records_handles_genus_uncertainty_marker(self):
        """'Theocorys? phyzella' has the open-nomen '?' on the genus,
        not on the epithet. Genus 'Theocorys', epithet 'phyzella',
        qualifier '?'.
        """
        m = replace(_make_match(), species="Theocorys? phyzella")
        taxa = taxon_records_from_matches([m])
        t = taxa[0]
        assert t["genus"] == "Theocorys"
        assert t["specific_epithet"] == "phyzella"
        assert t["qualifier"] == "?"

    def test_taxon_records_rejects_qualifier_only(self):
        """'cf. species' (no genus) must not produce genus='cf.'; the
        parser must refuse to invent a genus from a qualifier token.
        """
        assert _taxon_parts("cf. species") == {
            "genus": None,
            "specific_epithet": None,
            "qualifier": None,
        }

    def test_sample_records_namespaced_by_paper_id(self):
        """Two papers both mentioning 'Sample PR-SB26' must produce two
        distinct SampleRecord entries. The earlier code deduped on
        sample_id alone, which silently dropped the second paper.

        Round 20: sample_ids now carry a prefix tag (``S_`` for
        legacy ``Sample X``, ``B_`` for Boughdiri short codes,
        ``R_`` for ``specimen N``) so the operator can tell which
        detector fired. The dedup is still by ``(paper_id, sample_id)``
        so two papers sharing a sample_id do NOT collide.
        """
        snippet = "Sample PR-SB26 (early Cretaceous)."
        m1 = replace(_make_match(), caption_snippet=snippet, metadata={})
        m2 = replace(_make_match(), paper_id="paper-2", caption_snippet=snippet, metadata={})
        samples = sample_records_from_matches([m1, m2])
        assert len(samples) == 2
        paper_ids = {s["paper_id"] for s in samples}
        assert paper_ids == {"abc", "paper-2"}
        # Round 20: sample_id is now prefixed with "S_" (legacy pattern)
        assert all(s["sample_id"] == "S_PR-SB26" for s in samples)

    def test_locality_records_namespaced_by_paper_id(self):
        """Two papers both reporting 'Italy' at (45.0, 10.0) must
        produce two distinct LocalityRecord entries.
        """
        meta = {"geology_links": [{"locality": "Italy", "latitude": 45.0, "longitude": 10.0}]}
        m1 = replace(_make_match(), metadata=meta)
        m2 = replace(_make_match(), paper_id="paper-2", metadata=meta)
        locs = locality_records_from_geology([m1, m2])
        assert len(locs) == 2
        assert {l["locality_id"] for l in locs} == {locs[0]["locality_id"]} ^ {
            locs[1]["locality_id"]
        }

    def test_modern_coord_prefers_modern_over_legacy(self):
        """When both modern_latitude and legacy latitude are present,
        the modern value wins. When only legacy is present, it is
        promoted to modern_*. When only modern is present, it is
        used directly. The dedup key still uses the legacy value when
        present so the key and the record agree.
        """
        assert _resolve_modern_coord(1.0, 2.0) == 1.0  # modern wins
        assert _resolve_modern_coord(None, 2.0) == 2.0  # legacy falls through
        assert _resolve_modern_coord(1.0, None) == 1.0  # modern only
        assert _resolve_modern_coord(None, None) is None
        # Critical: legacy=0.0 must NOT short-circuit to None; it is a
        # valid coordinate. The previous 'or' chain corrupted this case.
        assert _resolve_modern_coord(None, 0.0) == 0.0

    def test_warning_id_stable_across_match_reordering(self):
        """The warning_id for a given (paper, figure, panel, code) must
        be content-derived only. Re-ordering the matches must not
        change the set of warning_ids.
        """
        m1 = replace(
            _make_match(),
            figure_id="fig_1",
            panel_id="1",
            species=None,
            panel_path=None,
            bbox=None,
        )
        m2 = replace(
            _make_match(),
            figure_id="fig_2",
            panel_id="2",
            species=None,
            panel_path=None,
            bbox=None,
        )
        forward = warnings_from_matches([m1, m2])
        reversed_ = warnings_from_matches([m2, m1])
        ids_f = sorted(w["warning_id"] for w in forward)
        ids_r = sorted(w["warning_id"] for w in reversed_)
        assert ids_f == ids_r
        assert (
            len(ids_f) >= 4
        )  # missing_species + missing_panel_image + missing_bbox + missing_printed_panel_id per panel

    def test_figure_records_coerce_blank_strings_to_none(self):
        """Blank / whitespace figure_number / caption / caption_source
        are coerced to None so downstream consumers do not have to
        distinguish between "" and missing.
        """
        meta = {
            "figure_number": "   ",
            "figure_type": "",
            "caption_source": "",
            "image_path": " ",
            "bbox": None,
        }
        m = replace(_make_match(), metadata=meta)
        figs = figure_records_from_matches([m])
        assert len(figs) == 1
        f = figs[0]
        assert f["figure_number"] is None
        assert f["figure_type"] is None
        assert f["caption_source"] is None
        assert f["image_path"] is None

    def test_paleocoord_missing_warning_emitted_when_locality_nonempty(self):
        """Round 20: the GPlates-style paleocoord backend is now wired
        in ``paleo_coordinates_from_localities``, so the
        ``paleocoord_backend_missing`` warning is no longer emitted.
        Instead, when a locality has coords + an age, the run
        populates ``paleo_coordinates`` via the live backend. This
        test was updated to assert the new (correct) behavior."""
        meta = {
            "geology_links": [
                {
                    "locality": "Italy",
                    "latitude": 45.0,
                    "longitude": 10.0,
                    "ma_mid": 50.0,  # Eocene, for stable plate_id inference
                    "ma_top": 50.0,
                    "ma_base": 56.0,
                }
            ]
        }
        m = replace(_make_match(), metadata=meta)
        prov = ProvenanceRecord(**build_provenance().to_dict())
        out = run_output_from_provenance(prov, [m])
        codes = [w["code"] for w in out["warnings"]]
        # The deprecated warning must NOT appear — the backend is live.
        assert "paleocoord_backend_missing" not in codes
        # And the paleo_coordinates view should be populated.
        assert out["paleo_coordinates"], (
            "Round 20: paleo_coordinates expected non-empty when locality "
            "has both coords and ma_mid (GPlates backend wired)."
        )
        pc = out["paleo_coordinates"][0]
        assert pc["modern_latitude"] == 45.0
        assert pc["modern_longitude"] == 10.0
        assert pc["reconstruction_age_ma"] == 50.0

    def test_paleocoord_missing_warning_not_emitted_when_no_locality(self):
        """When no localities exist the empty paleo_coordinates list is
        expected and we do not emit a noisy warning.
        """
        m = replace(_make_match(), metadata={})
        prov = ProvenanceRecord(**build_provenance().to_dict())
        out = run_output_from_provenance(prov, [m])
        codes = [w["code"] for w in out["warnings"]]
        assert "paleocoord_backend_missing" not in codes

    def test_geology_contexts_deduped_with_ma_top_base(self):
        meta = dict(_make_match().metadata or {})
        meta["geology_links"] = [
            {
                "age": "Late Jurassic",
                "chronostratigraphy": "Kimmeridgian",
                "chronostratigraphy_rank": "age",
                "ma_top": 152.1,
                "ma_base": 149.2,
                "formation": "Fonzaso",
                "locality": "Italy",
                "evidence_text": "...",
                "confidence": 0.7,
            }
        ]
        m = replace(_make_match(), metadata=meta)
        geos = geology_contexts_from_matches([m])
        assert len(geos) == 1
        assert geos[0]["ma_top"] == 152.1
        assert geos[0]["ma_base"] == 149.2

    def test_deterministic_ma_propagates_from_stratigraphy_to_run_output(self):
        """End-to-end: extract_geology_from_sections picks up Ma bounds
        from the matched ICS row (via stratigraphy.classify_age_string),
        and geology_contexts_from_matches surfaces them in the exported
        GeologyContextRecord. Task 5 wires the previously-dropped Ma
        values through the chain so the converter output is no longer
        forced to None for these fields.
        """
        from rlpe.geology_extraction import extract_geology_from_sections

        sections = [
            {
                "section_type": "geological_setting",
                "title": "Section A",
                "text": "The Changhsingian limestone is exposed at the type locality.",
            }
        ]
        records = extract_geology_from_sections(sections)
        assert records, "expected at least one GeologyRecord"
        rec = records[0]
        assert rec.chronostratigraphy == "Changhsingian"
        assert rec.ma_top is not None
        assert rec.ma_base is not None
        assert rec.ma_mid is not None

        # Build a MatchResult carrying the record's Ma values into
        # geology_links, then verify the converter outputs them.
        meta = {"geology_links": [rec.to_dict()]}
        m = replace(_make_match(), metadata=meta)
        geos = geology_contexts_from_matches([m])
        assert len(geos) == 1
        assert geos[0]["ma_top"] == rec.ma_top
        assert geos[0]["ma_base"] == rec.ma_base
        assert geos[0]["ma_mid"] == rec.ma_mid

    def test_locality_records_deduped(self):
        meta = dict(_make_match().metadata or {})
        meta["geology_links"] = [
            {"locality": "Italy", "latitude": 45.0, "longitude": 10.0},
            {"locality": "Italy", "latitude": 45.0, "longitude": 10.0},
        ]
        m = replace(_make_match(), metadata=meta)
        locs = locality_records_from_geology([m])
        assert len(locs) == 1
        assert locs[0]["modern_latitude"] == 45.0
        assert locs[0]["modern_longitude"] == 10.0

    def test_sample_records_extracted_from_caption(self):
        m = MatchResult(
            paper_id="p",
            figure_id="f",
            panel_id="1",
            species="X",
            panel_path=None,
            bbox=None,
            confidence=0.0,
            caption_snippet="Sample PR-SB28 (latest Barremian). Figs 1-6 ... Sample PR-SB30. Fig 7 ...",
        )
        samples = sample_records_from_matches([m])
        assert len(samples) == 2
        # Round 20: sample_ids are prefixed with ``S_`` (legacy
        # ``Sample X`` detector) so the operator can tell which
        # detector fired.
        assert {s["sample_id"] for s in samples} == {"S_PR-SB28", "S_PR-SB30"}

    def test_warnings_emitted_for_missing_panel_image(self):
        m = MatchResult(
            paper_id="p",
            figure_id="f",
            panel_id="1",
            species=None,  # missing species
            panel_path=None,  # missing panel image
            bbox=None,  # missing bbox
            confidence=0.0,
            metadata={"extraction_method": "llm_first"},
        )
        warns = warnings_from_matches([m])
        codes = {w["code"] for w in warns}
        assert "missing_species" in codes
        assert "missing_panel_image" in codes
        assert "missing_bbox" in codes
        assert "llm_first_without_visual_evidence" in codes

    def test_legacy_payload_without_new_fields_validates(self):
        """A minimal RunOutput (no papers/figures/... keys) must still
        validate against the v1.0.0 schema. This is the backwards-
        compatibility guard for the published contract.
        """
        prov = ProvenanceRecord(**build_provenance().to_dict())
        legacy = {
            "schema_version": "1.0.0",
            "provenance": prov.model_dump(),
            "panels": [],
        }
        loaded = validate_run_output(legacy)
        assert loaded.schema_version == "1.0.0"
        assert loaded.papers == []
        assert loaded.figures == []
        assert loaded.panels == []

    def test_extra_unknown_fields_still_rejected(self):
        """The strict ``extra=forbid`` policy must continue to reject
        unknown fields. Adding new optional fields must not weaken this
        guard.
        """
        prov = ProvenanceRecord(**build_provenance().to_dict())
        bad = {
            "schema_version": "1.0.0",
            "provenance": prov.model_dump(),
            "panels": [],
            "unknown_field": "should fail",
        }
        with pytest.raises(ValidationError):
            validate_run_output(bad)

    def test_schema_version_pinned_to_current(self):
        """The external data-contract version is published via
        ``SCHEMA_VERSION`` (currently ``1.2.0`` after audit 2026-08-02;
        bumped from 1.1.0 for Stage-6 morphology records).

        The test reads the version through the module constant rather
        than hard-coding the literal so a future minor bump only
        requires updating ``schema_models.py`` and emitting the new
        JSON schema. Any drift between ``SCHEMA_VERSION`` and the
        ``RunOutput.schema_version`` echoed by ``validate_run_output``
        is a contract break and must fail this test.
        """
        assert SCHEMA_VERSION == "1.2.0"
        prov = ProvenanceRecord(**build_provenance().to_dict())
        ro = validate_run_output(
            {
                "schema_version": SCHEMA_VERSION,
                "provenance": prov.model_dump(),
                "panels": [],
            }
        )
        assert ro.schema_version == "1.2.0"
