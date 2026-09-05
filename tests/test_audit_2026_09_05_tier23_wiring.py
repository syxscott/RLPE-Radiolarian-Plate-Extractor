"""Audit 2026-09-05 — tier-2 / tier-3 wiring fixes.

Covers the fixes for the extractor-coverage review:

* tier3-A1/A2/A3 — converters export the ``scale_bar.warning`` flag,
  the per-entry geology ``link_source`` / ``figure_id`` provenance,
  and stop mislabelling LLM-first rows as ``matcher_type="heuristic"``.
* tier3-A4/A5 — ``_finalize_rows`` stamps ``sample_id`` /
  ``sample_ids`` and ``canonical_panel_id`` (printed > caption rule).
* tier3-B1/B2 — the cross-figure linker runs BEFORE ``_finalize_rows``
  on both extraction paths and exactly once per paper (source guards).
* tier3-B3 — "unlinked" linker results stamp nothing.
* tier3-B4 — ``link_visual_coordinates`` receives real image paths
  (the pipeline stamps ``image_path`` onto its figure views).
* tier3-B5 — the map→range-chart bridge accepts geo-vision map rows
  (``figure_type`` map / paleogeographic_map), reviving the dead
  ``matched_location`` machinery.
* tier3-B6 — ``PanelMetadata.matched_location`` export channel.
* tier3-B7/D5 — run-level warnings for a key-less range chart and an
  engine-less Stage 6.
* tier3-C — ``RunOutput.knowledge_graphs`` / ``range_charts`` drained
  from the paper-level captures (schema 1.3.0).
* tier3-D1/D2 — ``--m3-stage-6`` / ``--use-geo-vision`` implicitly
  enable ``m3_enhanced_mode`` on the CLI and the web extra builder.
* tier3-D3 — paper-level PBDB attach works on plain dict rows.
* tier3-D4 — ``enrich_geology_record`` is idempotent and wired into
  ``_finalize_rows`` (both paths).

Naming convention follows the ``test_audit_*`` family: behavioural
tests plus source-guard tripwires for the wiring that is hard to
exercise without a full pipeline run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.config import PipelineConfig  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src" / "rlpe"


def _read(rel_path: str) -> str:
    return (SRC / rel_path).read_text(encoding="utf-8")


@pytest.fixture
def pipe(tmp_path):
    from rlpe.pipeline import RadiolarianPipeline

    cfg = PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "work")
    p = RadiolarianPipeline(cfg)
    p.m3_engine = None
    return p


def _plate_row(panel_id: str = "1", species: str | None = "Genus species") -> dict[str, Any]:
    return {
        "paper_id": "p1",
        "figure_id": "fig1",
        "panel_id": panel_id,
        "species": species,
        "caption_snippet": "Plate 1. 1) Genus species, Sample B_DP2.",
        "metadata": {
            "figure_type": "plate",
            "caption_panel_id": panel_id,
        },
    }


def _match_result(metadata: dict[str, Any]) -> Any:
    from rlpe.types import MatchResult

    return MatchResult(
        paper_id="p1",
        figure_id="fig1",
        panel_id="1",
        species="Genus species",
        panel_path=None,
        bbox=None,
        confidence=0.8,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# tier3-A — converter mapping
# ---------------------------------------------------------------------------


class TestConverterMapping:
    def test_scale_bar_warning_mapped(self):
        from rlpe.converters import _scale_bar_from_meta

        rec = _scale_bar_from_meta(
            {"scale_bar": {"value": 100.0, "unit": "um", "warning": "scale_bar_disagreement"}}
        )
        assert rec is not None
        assert rec.warning == "scale_bar_disagreement"

    def test_geology_link_source_and_figure_id_mapped(self):
        from rlpe.converters import _geology_links_from_meta

        recs = _geology_links_from_meta(
            {
                "geology_links": [
                    {
                        "age": "Early Jurassic",
                        "link_source": "sample_match",
                        "figure_id": "od_plate_x_p003_pl01",
                    }
                ]
            }
        )
        assert len(recs) == 1
        assert recs[0].link_source == "sample_match"
        assert recs[0].figure_id == "od_plate_x_p003_pl01"

    def test_llm_first_row_not_labelled_heuristic(self):
        from rlpe.converters import panel_metadata_from_match

        m = _match_result({"extraction_method": "llm_first"})
        md = panel_metadata_from_match(m)
        assert md.matcher_type == "llm_first"

    def test_explicit_matcher_type_wins(self):
        from rlpe.converters import panel_metadata_from_match

        m = _match_result({"extraction_method": "llm_first", "matcher_type": "heuristic"})
        assert panel_metadata_from_match(m).matcher_type == "heuristic"


# ---------------------------------------------------------------------------
# tier3-A4/A5 — _finalize_rows stamping
# ---------------------------------------------------------------------------


class TestFinalizeStamping:
    def test_canonical_and_sample_ids_stamped(self, pipe):
        rows = [_plate_row(panel_id="1")]
        out = pipe._finalize_rows(rows)
        assert len(out) == 1
        md = out[0]["metadata"]
        # canonical: no printed_panel_id → falls back to caption_panel_id.
        assert md["canonical_panel_id"] == "1"
        # sample: the caption snippet carries "Sample B_DP2".
        assert md["sample_id"] == "B_DP2"
        assert "B_DP2" in md["sample_ids"]

    def test_printed_panel_id_wins_for_canonical(self, pipe):
        row = _plate_row(panel_id="2")
        row["metadata"]["printed_panel_id"] = "3"
        row["metadata"]["sample_id"] = "already_set"
        out = pipe._finalize_rows([row])
        md = out[0]["metadata"]
        assert md["canonical_panel_id"] == "3"
        # pre-existing sample stamp is preserved, not overwritten
        assert md["sample_id"] == "already_set"


# ---------------------------------------------------------------------------
# tier3-B1/B2/B4 — linker wiring source guards
# ---------------------------------------------------------------------------


class TestLinkerWiringGuards:
    def test_od_path_links_before_finalize(self):
        src = _read("pipeline.py")
        inner = src[src.find("def _process_one_pdf_od_inner") :]
        inner = inner[: inner.find("def ", 10)]
        finalize_idx = inner.find("return self._finalize_rows(results)")
        linker_idx = inner.find("self._apply_cross_figure_linker(results, paper_id)")
        assert finalize_idx > 0, "OD inner must still call _finalize_rows"
        assert 0 < linker_idx < finalize_idx, (
            "The cross-figure linker must run BEFORE _finalize_rows on the "
            "OD path (after it, the stub rows feeding the figure index are "
            "already stripped)."
        )

    def test_no_outer_linker_call_in_process_one_pdf(self):
        src = _read("pipeline.py")
        start = src.find("def _process_one_pdf(")
        end = src.find("def _process_one_pdf_od(")
        body = src[start:end]
        assert "_apply_cross_figure_linker" not in body, (
            "_process_one_pdf must not call the linker itself — both inner "
            "paths already do (pre-finalize); an outer call would run it a "
            "second time and overwrite link_source with 'unlinked'."
        )

    def test_figure_views_carry_image_path(self):
        src = _read("pipeline.py")
        assert '"image_path": row.get("panel_path")' in src, (
            "figure_views must carry the row image path for link_visual_coordinates (tier3-B4)."
        )


# ---------------------------------------------------------------------------
# tier3-B3 — unlinked results stamp nothing
# ---------------------------------------------------------------------------


class TestUnlinkedNoise:
    def test_unlinked_leaves_no_junk_entry(self, pipe):
        plate = {
            "paper_id": "p1",
            "figure_id": "fig1",
            "panel_id": "1",
            "species": "Genus species",
            "caption_snippet": "no useful info",
            "metadata": {"figure_type": "plate"},
        }
        strat = {
            "paper_id": "p1",
            "figure_id": "strat1",
            "panel_id": None,
            "metadata": {"figure_type": "strat_column", "caption": "Sample S1"},
        }
        out = pipe._apply_cross_figure_linker([plate, strat], paper_id="p1")
        md = out[0]["metadata"]
        assert not md.get("link_source")
        assert md.get("geology_links") in (None, [])

    def test_idempotent_against_double_invocation(self, pipe):
        row = {
            "paper_id": "p1",
            "figure_id": "fig1",
            "panel_id": "1",
            "species": "Genus species",
            "caption_snippet": "All from Sample S1",
            "metadata": {
                "figure_type": "plate",
                "link_source": "sample_match",
                "geology_links": [{"age": "Early Jurassic"}],
            },
        }
        strat = {
            "paper_id": "p1",
            "figure_id": "strat1",
            "panel_id": None,
            "metadata": {"figure_type": "strat_column", "caption": "Sample S1"},
        }
        out = pipe._apply_cross_figure_linker([row, strat], paper_id="p1")
        md = out[0]["metadata"]
        # The existing sample_match link survives untouched: no second
        # entry, no overwrite.
        assert md["link_source"] == "sample_match"
        assert len(md["geology_links"]) == 1


# ---------------------------------------------------------------------------
# tier3-B5/B6 — revived map→range-chart bridge + export channel
# ---------------------------------------------------------------------------


class TestMapBridgeRevived:
    def test_bridge_consumes_geo_vision_map_rows(self, pipe):
        map_row = {
            "paper_id": "p1",
            "figure_id": "mapfig",
            "panel_id": "GEO_VISION_PALEOGEOGRAPHIC_MAP",
            "species": None,
            "metadata": {
                "figure_type": "paleogeographic_map",
                "geology_links": [
                    {"locality": "Sikhote-Alin", "evidence_text": "map vision"},
                ],
            },
        }
        panel_row = {
            "paper_id": "p1",
            "figure_id": "platefig",
            "panel_id": "1",
            "species": "Genus species",
            "metadata": {
                "figure_type": "plate",
                "geology_links": [
                    {
                        "locality": "SA",
                        "evidence_text": "range_chart_vision[fig9] section SA",
                    }
                ],
            },
        }
        out = pipe._cross_link_map_and_range_chart([map_row, panel_row])
        matched = out[1]["metadata"].get("matched_location") or []
        assert any(m["location"] == "Sikhote-Alin" for m in matched)
        assert any(m["match_type"] == "acronym" for m in matched)

    def test_matched_location_exported_on_panel_metadata(self):
        from rlpe.converters import panel_metadata_from_match

        m = _match_result(
            {
                "matched_location": [
                    {"section": "SK-01", "location": "Sikhote-Alin", "match_type": "prefix2"}
                ]
            }
        )
        md = panel_metadata_from_match(m)
        assert md.matched_location == [
            {"section": "SK-01", "location": "Sikhote-Alin", "match_type": "prefix2"}
        ]


# ---------------------------------------------------------------------------
# tier3-B7 / D5 — run-level warnings
# ---------------------------------------------------------------------------


class TestRunLevelWarnings:
    def test_range_chart_no_key_records_warning(self, pipe, tmp_path, monkeypatch):
        from rlpe.utils import drain_warnings

        drain_warnings()  # clear slate
        # Force the no-key branch on BOTH sources: the pipeline
        # constructor may have already pulled the real ANTHROPIC_API_KEY
        # from .env into ``config.extra["MiniMax_api_key"]`` (audit-fixes
        # tests load .env into os.environ globally), so deleting the env
        # var alone is not enough — forgetting this made the test fire a
        # REAL MiniMax API call during the full-suite run.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setitem(pipe.config.extra, "MiniMax_api_key", None)
        stubs = pipe._process_range_chart(
            paper_id="p1",
            figure_id="figRC",
            caption_text="Distribution of Genus species",
            image_path=None,
        )
        assert stubs and stubs[0]["panel_id"] == "_RANGE_CHART_SKIPPED_NO_API_KEY"
        labels = [w["label"] for w in drain_warnings()]
        assert "range_chart_skipped_no_api_key" in labels

    def test_morphology_without_engine_records_warning(self, pipe):
        from rlpe.utils import drain_warnings

        drain_warnings()  # clear slate
        pipe.config.m3_stage_6 = True
        pipe.m3_engine = None
        out = pipe._apply_morphology_enrichment([_plate_row()], "p1", None)
        assert out  # rows untouched
        labels = [w["label"] for w in drain_warnings()]
        assert "m3_stage6_no_engine" in labels


# ---------------------------------------------------------------------------
# tier3-C — RunOutput knowledge_graphs / range_charts
# ---------------------------------------------------------------------------


class TestRunOutputPaperLevelViews:
    def test_run_output_carries_knowledge_graphs_and_range_charts(self):
        from rlpe.converters import run_output_from_provenance
        from rlpe.provenance import build_provenance
        from rlpe.types import MatchResult

        prov = build_provenance(PipelineConfig(pdf_dir=Path("."), work_dir=Path(".")), pdf_paths=[])
        out = run_output_from_provenance(
            prov.to_dict(),
            [
                MatchResult(
                    paper_id="p1",
                    figure_id="fig1",
                    panel_id="1",
                    species="Genus species",
                    panel_path=None,
                    bbox=None,
                    confidence=0.8,
                )
            ],
            paper_knowledge_graphs=[{"nodes": [], "edges": []}],
            paper_range_charts=[{"figure_id": "figRC", "species_ranges": []}],
        )
        assert out["knowledge_graphs"] == [{"nodes": [], "edges": []}]
        assert out["range_charts"] == [{"figure_id": "figRC", "species_ranges": []}]

    def test_run_output_schema_accepts_new_fields(self):
        from rlpe.schema_models import SCHEMA_VERSION, RunOutput

        assert SCHEMA_VERSION == "1.3.0"
        # knowledge_graphs / range_charts default to empty lists.
        assert "knowledge_graphs" in RunOutput.model_fields
        assert "range_charts" in RunOutput.model_fields

    def test_range_chart_capture_in_process_range_chart(self, pipe, tmp_path, monkeypatch):
        """A successful extraction lands in ``_paper_range_charts``."""
        from rlpe import pipeline as pl

        captured: dict[str, Any] = {}

        class _FakeChart:
            confidence = 0.9

            species_ranges: list[Any] = []

            def to_dict(self) -> dict[str, Any]:
                return {"confidence": 0.9, "species_ranges": [], "sections": []}

        def _fake_extract(**kwargs: Any) -> _FakeChart:
            captured.update(kwargs)
            return _FakeChart()

        monkeypatch.setattr("rlpe.pipeline.extract_range_chart", _fake_extract)
        pipe.config.extra["MiniMax_api_key"] = "test-key"
        out = pipe._process_range_chart(
            paper_id="p1",
            figure_id="figRC",
            caption_text="Distribution of Genus species",
            image_path=None,
        )
        assert isinstance(out, list)
        assert "p1" in pipe._paper_range_charts
        assert pipe._paper_range_charts["p1"][0]["figure_id"] == "figRC"


# ---------------------------------------------------------------------------
# tier3-D1/D2 — implicit m3_enhanced_mode
# ---------------------------------------------------------------------------


class TestImplicitEnhancedMode:
    def test_cli_auto_enable_includes_stage6_and_geo_vision(self):
        src = _read("cli.py")
        assert "or args.m3_stage_6" in src, (
            "CLI must implicitly enable m3_enhanced_mode for --m3-stage-6."
        )
        assert "or args.use_geo_vision" in src, (
            "CLI must implicitly enable m3_enhanced_mode for --use-geo-vision."
        )

    def test_web_extra_builder_auto_enables(self):
        src = _read("api/app.py")
        assert 'extra.setdefault("m3_enhanced_mode", True)' in src, (
            "The web extra builder must mirror the CLI's implicit m3_enhanced_mode opt-in."
        )
        assert 'options.get("use_geo_vision") or options.get("m3_stage_6")' in src


# ---------------------------------------------------------------------------
# tier3-D3 — paper-level PBDB on dict rows
# ---------------------------------------------------------------------------


class TestPaperLevelPbdb:
    def test_attach_paleodb_accepts_dict_rows(self, pipe, monkeypatch):
        from rlpe import paleodb as pbdb_mod
        from rlpe.types import TaxonomyMatch

        class _FakeClient:
            def __init__(self, **kwargs: Any) -> None:
                pass

            def lookup_species(self, name: str) -> TaxonomyMatch | None:
                return TaxonomyMatch(
                    name=name,
                    rank="species",
                    source="paleodb",
                    family="FakeFam",
                    order="FakeOrd",
                    class_="FakeCls",
                )

            def lookup_occurrences(self, name: str, max_n: int = 25) -> list:
                return []

        monkeypatch.setattr(pbdb_mod, "PaleoDB", _FakeClient)
        rows = [
            _plate_row(panel_id="1", species="Genus species"),
            _plate_row(panel_id="2", species="Genus other"),
        ]
        pipe._attach_paleodb_metadata(rows)
        for r in rows:
            payload = r["metadata"]["paleodb"]
            assert payload["looked_up"] is True
            tax = payload["taxonomy"]
            assert tax["family"] == "FakeFam"
            assert tax["order"] == "FakeOrd"
            # TaxonomyMatch.to_dict() maps the ``class_`` field to "class".
            assert tax["class"] == "FakeCls"

    def test_paper_level_call_site_guard(self):
        src = _read("pipeline.py")
        start = src.find("def _process_one_pdf(")
        end = src.find("def _process_one_pdf_od(")
        body = src[start:end]
        assert "self._attach_paleodb_metadata(rows)" in body, (
            "PBDB enrichment must run once per paper on the finalized rows "
            "in _process_one_pdf (tier3-D3)."
        )


# ---------------------------------------------------------------------------
# tier3-D4 — paleo enrichment idempotency
# ---------------------------------------------------------------------------


class TestPaleoEnrichIdempotent:
    def test_second_pass_does_not_overwrite(self):
        from rlpe.paleo_reconstruction import enrich_geology_record

        rec: dict[str, Any] = {
            "modern_latitude": 46.0,
            "modern_longitude": 11.0,
            "chronostratigraphy": "Early Jurassic",
            "country": "Italy",
        }
        enrich_geology_record(rec)
        first = rec.get("paleo_latitude")
        if first is None:
            pytest.skip("plate inference found no plate for this fixture")
        enrich_geology_record(rec)
        assert rec["paleo_latitude"] == first

    def test_finalize_enrichment_guard(self):
        src = _read("pipeline.py")
        assert "for gl in geo_links:" in src, (
            "_finalize_rows must enrich every geology_links entry (tier3-D4)."
        )
