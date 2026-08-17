"""Phase 66 Plan C.4 — pipeline integration tests for visual-coordinate linker."""

from __future__ import annotations

from typing import Any

import pytest

from rlpe.config import PipelineConfig


def _make_plate_row(
    panel_id: str = "p1",
    figure_id: str = "fig1",
    caption: str = "",
    species: str | None = "Genus species",
    paper_id: str = "p1",
    link_source: str | None = None,
) -> dict[str, Any]:
    md: dict[str, Any] = {
        "figure_type": "plate",
        "caption": caption,
        "caption_snippet": caption,
    }
    if link_source is not None:
        md["link_source"] = link_source
        md["link_confidence"] = 1.0 if link_source == "sample_match" else 0.7
        md["link_figure_id"] = "strat1"
    return {
        "paper_id": paper_id,
        "figure_id": figure_id,
        "panel_id": panel_id,
        "canonical_panel_id": panel_id,
        "species": species,
        "caption_snippet": caption,
        "metadata": md,
    }


def _make_strat_row(
    figure_id: str = "strat1",
    caption: str = "Scaglia Fm, Late Triassic",
    paper_id: str = "p1",
) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "figure_id": figure_id,
        "panel_id": None,
        "metadata": {
            "figure_type": "strat_column",
            "caption": caption,
            "geology_links": [{"formation": "Scaglia Fm", "age": "Late Triassic"}],
        },
    }


def _make_map_row(
    figure_id: str = "map1",
    caption: str = "Paleogeographic map",
    paper_id: str = "p1",
) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "figure_id": figure_id,
        "panel_id": None,
        "metadata": {
            "figure_type": "paleogeographic_map",
            "caption": caption,
            "geology_links": [],
        },
    }


class _FakeM3Visual:
    """Stand-in M3 engine exposing only cross_figure_visual_inference."""

    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.response = response or {
            "plate_panels": [
                {
                    "cell_label": "1",
                    "species": "Genus species",
                    "links_to_strat_layer": 3,
                    "links_to_age": "Late Triassic",
                    "links_to_formation": "Scaglia Fm",
                    "confidence": 0.9,
                }
            ]
        }
        self.calls = 0

    def cross_figure_visual_inference(
        self, plate_image, strat_image, plate_caption, strat_caption
    ) -> dict[str, Any]:
        self.calls += 1
        return self.response


@pytest.fixture
def pipe(tmp_path):
    from rlpe.pipeline import RadiolarianPipeline

    cfg = PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "work")
    p = RadiolarianPipeline(cfg)
    p.m3_engine = None
    return p


class TestVisualLinkerIntegration:
    def test_visual_linker_writes_to_metadata(self, pipe):
        """When Strategy 1 didn't fire and the paper has plate + strat,
        the visual linker's output lands in metadata.cross_figure_visual_links."""
        plate = _make_plate_row(
            panel_id="p1",
            caption="Specimen from Italy",
            link_source="locality_match",
        )
        strat = _make_strat_row(figure_id="strat1", caption="Scaglia Fm")
        rows = [plate, strat]
        pipe.m3_engine = _FakeM3Visual()
        out = pipe._apply_cross_figure_linker(rows, paper_id="p1")
        plate_out = out[0]
        links = plate_out["metadata"].get("cross_figure_visual_links") or []
        assert len(links) == 1
        link = links[0]
        assert link["target_figure_id"] == "strat1"
        assert link["source"] == "m3_visual"
        assert link["target_layer"] == 3
        assert link["target_age"] == "Late Triassic"
        assert link["target_formation"] == "Scaglia Fm"
        assert link["confidence"] == 0.9

    def test_visual_linker_skipped_for_sample_match(self, pipe):
        """Phase A Strategy 1 nailed it — Phase C should NOT add links."""
        plate = _make_plate_row(
            panel_id="p1",
            caption="All from Sample S1",
            link_source="sample_match",
        )
        strat = _make_strat_row(figure_id="strat1", caption="Sample S1, Scaglia Fm")
        rows = [plate, strat]
        m3 = _FakeM3Visual()
        pipe.m3_engine = m3
        out = pipe._apply_cross_figure_linker(rows, paper_id="p1")
        plate_out = out[0]
        # No visual links should have been added
        links = plate_out["metadata"].get("cross_figure_visual_links") or []
        assert links == []
        assert m3.calls == 0

    def test_visual_linker_skipped_when_no_anchor_figure(self, pipe):
        """No strat column / map → Phase C silently skips."""
        plate = _make_plate_row(
            panel_id="p1",
            caption="Italy",
            link_source="locality_match",
        )
        rows = [plate]  # only a plate, no anchor figure
        m3 = _FakeM3Visual()
        pipe.m3_engine = m3
        out = pipe._apply_cross_figure_linker(rows, paper_id="p1")
        plate_out = out[0]
        links = plate_out["metadata"].get("cross_figure_visual_links") or []
        assert links == []
        assert m3.calls == 0

    def test_visual_linker_supports_paleogeographic_map(self, pipe):
        """Paleogeographic map counts as an anchor figure."""
        plate = _make_plate_row(
            panel_id="p1",
            caption="Italy specimen",
            link_source="locality_match",
        )
        m3 = _FakeM3Visual()
        pipe.m3_engine = m3
        rows = [plate, _make_map_row(figure_id="map1")]
        out = pipe._apply_cross_figure_linker(rows, paper_id="p1")
        plate_out = out[0]
        links = plate_out["metadata"].get("cross_figure_visual_links") or []
        assert len(links) == 1
        assert links[0]["target_figure_id"] == "map1"

    def test_visual_linker_empty_response(self, pipe):
        """When M3 returns empty plate_panels, panel gets empty list."""
        plate = _make_plate_row(
            panel_id="p1",
            caption="Italy",
            link_source="locality_match",
        )
        strat = _make_strat_row(figure_id="strat1", caption="Scaglia Fm")
        rows = [plate, strat]
        pipe.m3_engine = _FakeM3Visual(response={"plate_panels": []})
        out = pipe._apply_cross_figure_linker(rows, paper_id="p1")
        plate_out = out[0]
        links = plate_out["metadata"].get("cross_figure_visual_links") or []
        assert links == []


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
