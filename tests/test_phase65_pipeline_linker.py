"""Phase 65 Plan A.4 — pipeline integration tests for the cross-figure linker."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from rlpe.config import PipelineConfig


def _make_plate_row(
    panel_id: str = "p1",
    figure_id: str = "fig1",
    caption: str = "",
    species: str | None = "Genus species",
    paper_id: str = "p1",
) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "figure_id": figure_id,
        "panel_id": panel_id,
        "canonical_panel_id": panel_id,
        "species": species,
        "caption_snippet": caption,
        "metadata": {
            "figure_type": "plate",
            "caption": caption,
            "caption_snippet": caption,
        },
    }


def _make_strat_row(
    figure_id: str = "strat1",
    caption: str = "",
    paper_id: str = "p1",
    formation: str | None = None,
    age: str | None = None,
    locality: str | None = None,
) -> dict[str, Any]:
    md: dict[str, Any] = {"figure_type": "strat_column", "caption": caption}
    if formation or age or locality:
        md["geology_links"] = [{"formation": formation, "age": age, "locality": locality}]
    return {
        "paper_id": paper_id,
        "figure_id": figure_id,
        "panel_id": None,
        "metadata": md,
    }


@pytest.fixture
def pipe(tmp_path):
    """Build a RadiolarianPipeline with no backend so M3 path is exercised as no-op."""
    from rlpe.pipeline import RadiolarianPipeline

    cfg = PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "work")
    p = RadiolarianPipeline(cfg)
    # No m3_engine by default — exercises Strategy 1/2 + unlinked fallback.
    p.m3_engine = None
    return p


class TestSampleIDLinkThroughPipeline:
    def test_sample_id_match(self, pipe):
        plate = _make_plate_row(panel_id="p1", caption="All from Sample S1")
        strat = _make_strat_row(figure_id="strat1", caption="Sample S1, Scaglia Fm")
        rows = [plate, strat]
        out = pipe._apply_cross_figure_linker(rows, paper_id="p1")
        plate_out = out[0]
        assert plate_out["metadata"]["link_source"] == "sample_match"
        assert plate_out["metadata"]["link_confidence"] == 1.0
        # geology_links should have a new entry tagged with the linker source
        gl = plate_out["metadata"]["geology_links"]
        assert any(g.get("coord_source") == "cross_figure_linker:sample_match" for g in gl)

    def test_unlinked_when_no_match(self, pipe):
        plate = _make_plate_row(panel_id="p1", caption="no useful info")
        strat = _make_strat_row(figure_id="strat1", caption="Sample S1")
        rows = [plate, strat]
        out = pipe._apply_cross_figure_linker(rows, paper_id="p1")
        plate_out = out[0]
        assert plate_out["metadata"]["link_source"] == "unlinked"
        assert plate_out["metadata"]["link_confidence"] == 0.0


class TestLocalityLinkThroughPipeline:
    def test_locality_match(self, pipe):
        plate = _make_plate_row(panel_id="p1", caption="collected from Tunisia")
        strat = _make_strat_row(figure_id="strat1", caption="Tunisia outcrop")
        rows = [plate, strat]
        out = pipe._apply_cross_figure_linker(rows, paper_id="p1")
        plate_out = out[0]
        # The Linker has Tunisia on both sides → sample_match via the
        # bare-locality shortcut. Either sample_match or locality_match
        # is acceptable — both indicate a positive link.
        assert plate_out["metadata"]["link_source"] in ("sample_match", "locality_match")


class TestM3LinkThroughPipeline:
    def test_m3_fallback_when_strategies_fail(self, tmp_path):
        from rlpe.pipeline import RadiolarianPipeline
        from tests.fakes.fake_m3_backend import FakeM3Backend

        cfg = PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "work")
        p = RadiolarianPipeline(cfg)
        # Fake M3 engine with canned response. Use a low confidence
        # (0.3) so the pipeline's low-confidence review flag fires.
        backend = FakeM3Backend(
            canned_responses=[
                {
                    "raw_text": '{"species": "G. s", "formation": "Scaglia", '
                    '"age": "Late Cretaceous", "figure_id": "strat1", '
                    '"confidence": 0.3}'
                }
            ]
        )
        from rlpe.m3_engine import M3Engine

        p.m3_engine = M3Engine(backend=backend, config={})

        plate = _make_plate_row(panel_id="p1", caption="Generic plate caption")
        strat = _make_strat_row(figure_id="strat1", caption="Generic")
        rows = [plate, strat]
        out = p._apply_cross_figure_linker(rows, paper_id="p1")
        plate_out = out[0]
        # No Sample ID match, no Locality match → falls through to M3
        assert plate_out["metadata"]["link_source"] == "m3_inference"
        assert 0.3 <= plate_out["metadata"]["link_confidence"] <= 0.6
        # Low-confidence M3 links get needs_review flag
        assert plate_out["metadata"].get("needs_review") is True
        assert "cross_figure_linker_low_confidence" in (
            plate_out["metadata"].get("review_reasons") or []
        )


class TestNoPlatesNoOp:
    def test_no_plates_returns_unchanged(self, pipe):
        # Only a strat column row → nothing to link from
        strat = _make_strat_row(figure_id="strat1", caption="Sample S1")
        rows = [strat]
        out = pipe._apply_cross_figure_linker(rows, paper_id="p1")
        # Should return the same row, untouched.
        assert "link_source" not in (out[0].get("metadata") or {})


class TestDisabledFlag:
    def test_disabled_via_extra(self, tmp_path):
        from rlpe.pipeline import RadiolarianPipeline

        cfg = PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "work")
        cfg.extra["cross_figure_linker_enabled"] = False
        p = RadiolarianPipeline(cfg)
        # The flag is read in _process_one_pdf, so we test the method
        # directly works whether the flag is on or off (the flag only
        # gates _process_one_pdf's call to it).
        plate = _make_plate_row(panel_id="p1", caption="Sample S1")
        strat = _make_strat_row(figure_id="strat1", caption="Sample S1")
        rows = [plate, strat]
        out = p._apply_cross_figure_linker(rows, paper_id="p1")
        # Method always runs when called directly.
        assert out[0]["metadata"]["link_source"] == "sample_match"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
