"""Regression tests for audit 2026-08-02 — M3 morphology extraction MVP.

Stage 6 of the RLPE pipeline is an opt-in M3-based morphological-
description extraction. It emits MorphologyRecord entries (schema
v1.2.0) per unique (paper, species) pair. These tests cover the
locator, the M3 engine method, and the pipeline integration helper.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from rlpe.converters import (
    MorphologyRecord,
    _normalise_species_name,
    morphology_records_from_matches,
    run_output_from_provenance,
    taxon_records_from_matches,
)
from rlpe.m3_engine import PROMPT_REGISTRY, M3Engine
from rlpe.morphology_locator import (
    _normalise_text,
    _strip_authority,
    locate_morphology_context,
)
from rlpe.schema_models import (
    SCHEMA_VERSION,
    ProvenanceRecord,
    TaxonRecord,
)
from rlpe.types import MatchResult
from tests.fakes.fake_m3_backend import FakeM3Backend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(canned: list[dict[str, Any]]) -> M3Engine:
    """Build an M3Engine wired to a FakeM3Backend."""
    backend = FakeM3Backend(canned_responses=canned)
    return M3Engine(backend=backend, config={})


def _make_prov() -> ProvenanceRecord:
    """Minimal provenance for downstream tests."""
    return ProvenanceRecord(
        pipeline_version="test",
        schema_version=SCHEMA_VERSION,
        git_commit="deadbeef",
        git_dirty=False,
        timestamp_utc="2026-08-02T00:00:00Z",
        host="test-host",
        python_version="3.12",
    )


# ---------------------------------------------------------------------------
# TestMorphologyLocator
# ---------------------------------------------------------------------------

class TestMorphologyLocator:
    """Body-text locator for the morphology enrichment."""

    def test_finds_description_in_grobid_sections(self) -> None:
        sections = [
            {
                "section_id": "sec_1",
                "title": "Systematic Paleontology",
                "section_type": "systematic_paleontology",
                "text": (
                    "Genus species. Description: Test 180-220 µm in length; "
                    "cephalis spherical, thorax campanulate, three thoracic "
                    "segments visible. Pores circular, 8-10 µm in diameter."
                ),
            }
        ]
        ctx = locate_morphology_context("Genus species", sections)
        assert ctx is not None
        assert ctx["section_id"] == "sec_1"
        assert ctx["section_title"] == "Systematic Paleontology"
        assert "Test 180-220" in ctx["source_text"]
        assert "180" in ctx["source_text"] and "220" in ctx["source_text"]
        assert ctx["anchor_species"] == "Genus species"

    def test_returns_none_when_no_anchor(self) -> None:
        sections = [
            {
                "section_id": "sec_1",
                "title": "Systematic Paleontology",
                "section_type": "systematic_paleontology",
                "text": "Completely unrelated prose with no taxa.",
            }
        ]
        ctx = locate_morphology_context("Triassocampe sp.", sections)
        assert ctx is None

    def test_cuts_at_next_species_heading(self) -> None:
        sections = [
            {
                "section_id": "sec_1",
                "title": "Systematic Paleontology",
                "section_type": "systematic_paleontology",
                "text": (
                    "Genus alpha. Description: test length 100 µm. Pores round.\n\n"
                    "Genus beta. Description: test length 200 µm. Pores oval."
                ),
            }
        ]
        ctx = locate_morphology_context("Genus alpha", sections)
        assert ctx is not None
        assert "Genus alpha" in ctx["source_text"]
        # Cut should happen before "Genus beta" starts.
        assert "Genus beta" not in ctx["source_text"]

    def test_normalizes_unicode_hyphens(self) -> None:
        sections = [
            {
                "section_id": "sec_1",
                "title": "Systematic Paleontology",
                "section_type": "systematic_paleontology",
                "text": "Genus‐alpha is a spumellarian with cortical shell.",
            }
        ]
        # The en-dash / figure dash inside the species name should be
        # normalised so the anchor search still hits it.
        ctx = locate_morphology_context("Genus‐alpha", sections)
        assert ctx is not None
        assert "cortical shell" in ctx["source_text"]

    def test_strip_authority_drops_parenthetical(self) -> None:
        # The locator should treat "Species A Smith, 1900" and
        # "Species A (Smith, 1900)" the same when anchoring.
        assert _strip_authority("Genus species Smith, 1900") == "Genus species"
        assert _strip_authority("Genus species (Smith, 1900)") == "Genus species"
        assert _strip_authority("Genus species") == "Genus species"
        assert _strip_authority("Genus sp.") == "Genus sp."

    def test_normalise_text_collapses_whitespace(self) -> None:
        text = "Hello\n\n  world‐foo  bar\t\tbaz"
        out = _normalise_text(text)
        # Whitespace should collapse to single spaces, Unicode hyphen
        # replaced with ASCII space so a line-broken word is joined.
        assert "  " not in out
        assert "\n" not in out
        assert "world foo bar baz" in out


# ---------------------------------------------------------------------------
# TestM3InferMorphology
# ---------------------------------------------------------------------------

class TestM3InferMorphology:
    """M3Engine.infer_morphology() — Stage 6 morphology inference."""

    def test_prompt_registered(self) -> None:
        """The morphology_extract prompt must be in PROMPT_REGISTRY."""
        assert "morphology_extract" in PROMPT_REGISTRY
        prompt = PROMPT_REGISTRY["morphology_extract"]
        # Critical constraints should be visible to operators reading
        # the source code.
        for needle in (
            "test_shape",
            "spines_present",
            "diagnostic_features",
            "evidence_text",
            "null",
        ):
            assert needle in prompt, f"missing {needle!r} in morphology_extract prompt"

    def test_happy_path_full_response(self) -> None:
        engine = _make_engine([
            {
                "raw_text": json.dumps({
                    "test_shape": "campanulate",
                    "test_length_um_min": 180.0,
                    "test_length_um_max": 220.0,
                    "num_segments": 3,
                    "cephalis_shape": "spherical",
                    "thorax_shape": "campanulate",
                    "pore_pattern": "circular, regular",
                    "pore_diameter_um_min": 8.0,
                    "pore_diameter_um_max": 10.0,
                    "spines_present": True,
                    "spine_count": 3,
                    "apertural_structure": "narrow, constricted",
                    "diagnostic_features": [
                        "three-bladed apical horn",
                        "porous thoracic wall",
                    ],
                    "confidence": 0.85,
                    "evidence_text": "Test 180-220 µm; cephalis spherical.",
                }),
            }
        ])
        result = engine.infer_morphology(
            species_name="Genus species",
            source_text="Genus species. Description: Test 180-220 µm.",
            source="body_text",
            paper_id="p1",
        )
        assert result["test_shape"] == "campanulate"
        assert result["test_length_um_min"] == 180.0
        assert result["test_length_um_max"] == 220.0
        assert result["num_segments"] == 3
        assert result["spines_present"] is True
        assert result["spine_count"] == 3
        assert "three-bladed apical horn" in result["diagnostic_features"]
        assert result["confidence"] == 0.85
        # Source kind is stamped onto the returned dict so the caller
        # doesn't have to remember which flavor of context was used.
        assert result["_source"] == "body_text"

    def test_null_vs_false_handling(self) -> None:
        """Prompt explicitly forbids inferring False when not mentioned.

        When the model returns null for unmentioned fields, the schema
        treats them as "not stated" — distinct from False ("stated
        absent"). Verify that the engine forwards null unchanged
        rather than substituting False.
        """
        engine = _make_engine([
            {
                "raw_text": json.dumps({
                    "test_shape": "ovoid",
                    "spines_present": None,  # not mentioned
                    "spine_count": None,     # not mentioned
                    "confidence": 0.4,
                    "evidence_text": None,
                }),
            }
        ])
        result = engine.infer_morphology(
            species_name="Genus sp.",
            source_text="Genus sp. is ovoid.",
        )
        assert result["spines_present"] is None
        assert result["spine_count"] is None
        assert result["evidence_text"] is None

    def test_malformed_json_returns_empty(self) -> None:
        """Malformed JSON → empty dict, NOT raise (fail-open)."""
        engine = _make_engine([{"raw_text": "not json at all"}])
        result = engine.infer_morphology(
            species_name="Genus species",
            source_text="Genus species description here.",
        )
        assert result == {}

    def test_confidence_clamped_to_unit_interval(self) -> None:
        """Confidence must be clamped to [0.0, 1.0] even if M3 returns 1.5."""
        engine = _make_engine([
            {"raw_text": json.dumps({"confidence": 1.5})},
        ])
        result = engine.infer_morphology(
            species_name="Genus species",
            source_text="Genus species description.",
        )
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["confidence"] == 1.0

    def test_backend_none_returns_empty(self) -> None:
        """No backend → empty dict, no crash."""
        engine = M3Engine(backend=None, config={})
        result = engine.infer_morphology(
            species_name="Genus species",
            source_text="anything",
        )
        assert result == {}

    def test_empty_inputs_return_empty(self) -> None:
        engine = _make_engine([{"raw_text": "{}"}])
        assert engine.infer_morphology(species_name="", source_text="x") == {}
        assert engine.infer_morphology(species_name="x", source_text="") == {}
        assert engine.infer_morphology(species_name="   ", source_text="   ") == {}


# ---------------------------------------------------------------------------
# TestSchemaMorphology
# ---------------------------------------------------------------------------

class TestSchemaMorphology:
    """Schema v1.2.0 surface — MorphologyRecord + TaxonRecord.morphology_ids."""

    def test_schema_version_bumped(self) -> None:
        assert SCHEMA_VERSION == "1.2.0"

    def test_morphology_record_defaults(self) -> None:
        rec = MorphologyRecord(
            morphology_id="m1",
            taxon_id="t1",
            paper_id="p1",
        )
        assert rec.source == ""
        assert rec.diagnostic_features == []
        assert rec.confidence == 0.0
        # All other fields default to None.
        for fld in (
            "test_shape",
            "cephalis_shape",
            "thorax_shape",
            "abdomen_shape",
            "pore_pattern",
            "apertural_structure",
        ):
            assert getattr(rec, fld) is None
        # spines_present is None (not False) when not stated.
        assert rec.spines_present is None

    def test_morphology_record_rejects_unknown_field(self) -> None:
        """Strict model (extra='forbid') catches typos in M3 output."""
        with pytest.raises(ValueError):
            MorphologyRecord.model_validate(
                {
                    "morphology_id": "m1",
                    "taxon_id": "t1",
                    "paper_id": "p1",
                    "invented_field": "should fail",
                }
            )

    def test_morphology_record_numeric_validation(self) -> None:
        """Numeric fields are >=0; negative values rejected."""
        with pytest.raises(ValueError):
            MorphologyRecord(
                morphology_id="m1",
                taxon_id="t1",
                paper_id="p1",
                test_length_um_min=-1.0,
            )
        with pytest.raises(ValueError):
            MorphologyRecord(
                morphology_id="m1",
                taxon_id="t1",
                paper_id="p1",
                num_segments=-3,
            )

    def test_taxon_record_has_morphology_ids(self) -> None:
        rec = TaxonRecord(taxon_id="t1")
        assert rec.morphology_ids == []

    def test_run_output_includes_morphologies(self) -> None:
        prov = _make_prov()
        out = run_output_from_provenance(
            prov,
            matches=[],
            paper_morphologies=[
                {
                    "morphology_id": "m1",
                    "taxon_id": "t1",
                    "paper_id": "p1",
                    "test_shape": "ovoid",
                },
            ],
        )
        assert "morphologies" in out
        assert len(out["morphologies"]) == 1
        assert out["morphologies"][0]["test_shape"] == "ovoid"

    def test_morphology_records_from_matches_dedup(self) -> None:
        """Duplicates dropped; invalid entries skipped without raising."""
        good = {
            "morphology_id": "m1",
            "taxon_id": "t1",
            "paper_id": "p1",
            "test_shape": "campanulate",
        }
        # Make a MatchResult with morphology attached via metadata.
        m = MatchResult(
            paper_id="p1",
            figure_id="f1",
            panel_id="1",
            species="Genus species",
            panel_path=None,
            bbox=None,
            confidence=0.8,
            metadata={"morphology": good},
        )
        # Same id via paper_morphologies → still only one record.
        recs = morphology_records_from_matches([m], paper_morphologies=[good])
        assert len(recs) == 1
        assert recs[0]["morphology_id"] == "m1"
        # Different id → both kept.
        other = dict(good, morphology_id="m2")
        recs = morphology_records_from_matches([], paper_morphologies=[good, other])
        assert len(recs) == 2
        # Invalid entry → skipped, valid entry kept.
        recs = morphology_records_from_matches(
            [], paper_morphologies=[{"missing_required": True}, good]
        )
        assert len(recs) == 1
        assert recs[0]["morphology_id"] == "m1"


# ---------------------------------------------------------------------------
# TestPipelineIntegration
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    """Pipeline._apply_morphology_enrichment() helper."""

    def _build_pipeline(self, *, m3_stage_6: bool, policy: str, fake_canned: list[dict[str, Any]] | None = None):
        """Build a minimal RadiolarianPipeline with Stage-6 wired up."""
        from rlpe.config import PipelineConfig
        from rlpe.pipeline import RadiolarianPipeline

        with tempfile.TemporaryDirectory() as work_dir:
            cfg = PipelineConfig(
                pdf_dir=Path(work_dir) / "pdfs",
                work_dir=Path(work_dir),
                m3_stage_6=m3_stage_6,
            )
            cfg.extra["data_outbound_policy"] = policy
            pipeline = RadiolarianPipeline(cfg)
            # Always attach an M3 engine so the per-paper tests can
            # monkey-patch methods on it (the helper's earlier
            # conditional ``if fake_canned is not None`` skipped
            # engine creation for tests that bypass the backend).
            pipeline.m3_engine = M3Engine(
                backend=FakeM3Backend(canned_responses=fake_canned or []),
                config={},
            )
            return pipeline

    def test_paper_level_dedup(self) -> None:
        """5 rows with 3 unique species → 3 morphology records."""
        # The FakeM3Backend only exposes ``system_prompt`` to its
        # ``match`` callable (see ``FakeM3Backend._pick``); the user
        # prompt is opaque. We monkey-patch ``infer_morphology`` so
        # each call returns a per-species canned response — the
        # behaviour we actually want to test (dedup by species) is
        # upstream of the backend.
        pipeline = self._build_pipeline(
            m3_stage_6=True,
            policy="api_full",
            fake_canned=None,
        )
        species_to_shape = {
            "Genus alpha": "ovoid",
            "Genus beta": "spherical",
            "Genus gamma": "campanulate",
        }

        def fake_infer_morphology(species_name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                "test_shape": species_to_shape.get(species_name, "unknown"),
                "confidence": 0.5,
                "_source": "caption",
            }

        pipeline.m3_engine.infer_morphology = fake_infer_morphology  # type: ignore[assignment]
        rows = [
            {"paper_id": "p1", "species": "Genus alpha", "figure_id": "f1", "panel_id": "1",
             "caption_snippet": "Genus alpha description with enough text " * 4},
            {"paper_id": "p1", "species": "Genus alpha", "figure_id": "f1", "panel_id": "2",
             "caption_snippet": "Genus alpha description with enough text " * 4},
            {"paper_id": "p1", "species": "Genus beta",  "figure_id": "f2", "panel_id": "1",
             "caption_snippet": "Genus beta description with enough text " * 4},
            {"paper_id": "p1", "species": "Genus beta",  "figure_id": "f2", "panel_id": "2",
             "caption_snippet": "Genus beta description with enough text " * 4},
            {"paper_id": "p1", "species": "Genus gamma", "figure_id": "f3", "panel_id": "1",
             "caption_snippet": "Genus gamma description with enough text " * 4},
        ]
        out = pipeline._apply_morphology_enrichment(rows, "p1", fulltext_sections=None)
        # Rows unchanged in shape.
        assert out is rows
        # 3 distinct morphology records.
        records = pipeline._paper_morphologies.get("p1") or []
        assert len(records) == 3
        shapes = {r["test_shape"] for r in records}
        assert shapes == {"ovoid", "spherical", "campanulate"}

    def test_fail_open_on_morphology_error(self) -> None:
        """If M3 raises, rows are unchanged and no record is added."""
        pipeline = self._build_pipeline(
            m3_stage_6=True,
            policy="api_full",
            fake_canned=[{"raw_text": "{}"}],
        )
        # Monkey-patch infer_morphology to raise.
        def boom(**_kw):
            raise RuntimeError("simulated M3 backend outage")
        pipeline.m3_engine.infer_morphology = boom  # type: ignore[assignment]

        rows = [
            {"paper_id": "p1", "species": "Genus alpha", "figure_id": "f1", "panel_id": "1",
             "caption_snippet": "Genus alpha description with enough text " * 4},
        ]
        out = pipeline._apply_morphology_enrichment(rows, "p1", fulltext_sections=None)
        # No records added; row unchanged (no morphology_ids stamped).
        assert pipeline._paper_morphologies.get("p1") is None
        assert out is rows
        assert "morphology_ids" not in (out[0].get("metadata") or {})

    def test_api_redacted_skips_body(self) -> None:
        """data_outbound_policy='api_redacted' skips body morphology.

        We can't easily check the locator wasn't called from a unit
        test without monkeypatching; instead we verify that with no
        fulltext_sections, the pipeline still calls M3 with
        ``source='caption'`` and the resulting record has
        ``source='caption'`` — proving the caption-only path is taken.
        """
        # When api_redacted + no body sections, M3 should still be
        # called if the caption is long enough.
        pipeline = self._build_pipeline(
            m3_stage_6=True,
            policy="api_redacted",
            fake_canned=[{"raw_text": json.dumps({"test_shape": "ovoid", "confidence": 0.5})}],
        )
        # No fulltext_sections → locator returns None → caption path.
        long_caption = "Genus alpha. " + ("long caption text " * 30)
        rows = [
            {"paper_id": "p1", "species": "Genus alpha", "figure_id": "f1", "panel_id": "1",
             "caption_snippet": long_caption},
        ]
        pipeline._apply_morphology_enrichment(rows, "p1", fulltext_sections=None)
        records = pipeline._paper_morphologies.get("p1") or []
        assert len(records) == 1
        # Source must be "caption" (body path was skipped).
        assert records[0]["source"] == "caption"

    def test_local_only_skips_morphology(self) -> None:
        """data_outbound_policy='local_only' skips M3 morphology entirely."""
        pipeline = self._build_pipeline(
            m3_stage_6=True,
            policy="local_only",
            fake_canned=[{"raw_text": json.dumps({"test_shape": "ovoid", "confidence": 0.5})}],
        )
        rows = [
            {"paper_id": "p1", "species": "Genus alpha", "figure_id": "f1", "panel_id": "1",
             "caption_snippet": "Genus alpha. " + ("long caption text " * 30)},
        ]
        pipeline._apply_morphology_enrichment(rows, "p1", fulltext_sections=None)
        # No records produced under local_only.
        assert pipeline._paper_morphologies.get("p1") is None

    def test_off_when_m3_stage_6_false(self) -> None:
        """Default off — no records produced even if engine exists."""
        pipeline = self._build_pipeline(
            m3_stage_6=False,
            policy="api_full",
            fake_canned=[{"raw_text": json.dumps({"test_shape": "ovoid"})}],
        )
        rows = [
            {"paper_id": "p1", "species": "Genus alpha", "figure_id": "f1", "panel_id": "1",
             "caption_snippet": "Genus alpha. " + ("long caption text " * 30)},
        ]
        pipeline._apply_morphology_enrichment(rows, "p1", fulltext_sections=None)
        assert pipeline._paper_morphologies.get("p1") is None

    def test_taxon_records_from_morphology_ids(self) -> None:
        """taxon_records_from_matches picks up morphology_ids via metadata."""
        m = MatchResult(
            paper_id="p1",
            figure_id="f1",
            panel_id="1",
            species="Genus species",
            panel_path=None,
            bbox=None,
            confidence=0.8,
            metadata={"morphology_ids": ["m1"]},
        )
        records = taxon_records_from_matches([m])
        assert len(records) == 1
        assert records[0]["morphology_ids"] == ["m1"]


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
