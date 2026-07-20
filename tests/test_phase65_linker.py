"""Phase 65 Plan A.2 — cross-figure linker tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from rlpe.cross_figure_linker import (
    LINK_SOURCE_LOCALITY,
    LINK_SOURCE_M3,
    LINK_SOURCE_SAMPLE,
    LINK_SOURCE_UNLINKED,
    LinkResult,
    link_species_to_geology,
)


@dataclass
class FakePanel:
    """Minimal panel object the linker accepts."""
    paper_id: str = "p1"
    figure_id: str = "fig1"
    panel_id: str | None = "panel1"
    species: str | None = "Genus species"
    caption_snippet: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _strat_column(figure_id: str = "fig2", **kwargs: Any) -> dict[str, Any]:
    """Build a strat-column figure dict."""
    base = {
        "figure_id": figure_id,
        "figure_type": "strat_column",
        "caption": "",
        "paper_id": "p1",
    }
    base.update(kwargs)
    return base


def _plate(figure_id: str = "fig1", **kwargs: Any) -> dict[str, Any]:
    """Build a plate figure dict (should NOT be indexed for matching)."""
    base = {
        "figure_id": figure_id,
        "figure_type": "plate",
        "caption": "",
        "paper_id": "p1",
    }
    base.update(kwargs)
    return base


class TestStrategy1SampleMatch:
    def test_simple_sample_hit(self):
        panel = FakePanel(caption_snippet="All specimens from Sample S1")
        fig = _strat_column(figure_id="fig2", caption="Strat column at Sample S1, Late Cretaceous")
        results = link_species_to_geology([panel], [fig])
        assert len(results) == 1
        r = results[0]
        assert r.source == LINK_SOURCE_SAMPLE
        assert r.confidence == 1.0
        assert r.figure_id == "fig2"

    def test_loc_match(self):
        panel = FakePanel(caption_snippet="Loc. Tunisia")
        fig = _strat_column(figure_id="fig2", caption="Tunisia, Scaglia Fm")
        results = link_species_to_geology([panel], [fig])
        assert results[0].source == LINK_SOURCE_SAMPLE
        assert results[0].confidence == 1.0

    def test_id_match(self):
        panel = FakePanel(caption_snippet="ID-203 described here")
        fig = _strat_column(figure_id="fig2", caption="Sample ID-203, Late Cretaceous")
        # SampleID extractor strips the "ID-" prefix, so the value is "203"
        # but the strat column caption has "Sample ID-203" — extractor
        # returns "203" too. Match works.
        results = link_species_to_geology([panel], [fig])
        # Either sample_match or locality_match — both are valid here
        assert results[0].source in (LINK_SOURCE_SAMPLE, LINK_SOURCE_LOCALITY)

    def test_no_sample_no_match(self):
        panel = FakePanel(caption_snippet="Random caption with no IDs")
        fig = _strat_column(figure_id="fig2", caption="Strat column at Sample S1")
        results = link_species_to_geology([panel], [fig])
        assert results[0].source != LINK_SOURCE_SAMPLE


class TestStrategy2LocalityMatch:
    def test_locality_hit(self):
        panel = FakePanel(caption_snippet="collected from Tunisia")
        fig = _strat_column(figure_id="fig2", caption="Tunisia outcrop, Scaglia Fm")
        results = link_species_to_geology([panel], [fig])
        assert results[0].source == LINK_SOURCE_LOCALITY
        assert results[0].confidence == 0.7
        assert results[0].locality == "Tunisia"

    def test_locality_priority(self):
        # Sample match wins over locality when both match.
        panel = FakePanel(caption_snippet="from Tunisia, Sample S1")
        fig = _strat_column(figure_id="fig2", caption="Tunisia, Sample S1, Scaglia Fm")
        results = link_species_to_geology([panel], [fig])
        assert results[0].source == LINK_SOURCE_SAMPLE
        assert results[0].confidence == 1.0

    def test_no_locality_no_match(self):
        panel = FakePanel(caption_snippet="Generic plate caption")
        fig = _strat_column(figure_id="fig2", caption="Tunisia")
        results = link_species_to_geology([panel], [fig])
        assert results[0].source != LINK_SOURCE_LOCALITY


class TestStrategy3M3Inference:
    def test_m3_called_when_no_other_match(self):
        panel = FakePanel(caption_snippet="Plate with no sample or locality")
        fig = _strat_column(figure_id="fig2", caption="Scaglia Fm, Italy")

        called = []

        def fake_m3(panel_caption: str, paper_context):
            called.append((panel_caption, paper_context))
            return {
                "formation": "Scaglia",
                "age": "Late Cretaceous",
                "locality": "Italy",
                "figure_id": "fig2",
                "confidence": 0.5,
            }

        results = link_species_to_geology([panel], [fig], m3_inference_callable=fake_m3)
        assert len(called) == 1
        assert results[0].source == LINK_SOURCE_M3
        assert results[0].confidence == 0.5
        assert results[0].formation == "Scaglia"

    def test_m3_not_called_when_sample_matches(self):
        panel = FakePanel(caption_snippet="Sample S1")
        fig = _strat_column(figure_id="fig2", caption="Sample S1, Scaglia Fm")

        def fake_m3(panel_caption: str, paper_context):
            raise AssertionError("M3 should NOT be called when sample matches")

        results = link_species_to_geology([panel], [fig], m3_inference_callable=fake_m3)
        assert results[0].source == LINK_SOURCE_SAMPLE

    def test_m3_confidence_clamped(self):
        panel = FakePanel(caption_snippet="Plate with no info")
        fig = _strat_column(figure_id="fig2", caption="Italy")

        def fake_m3(panel_caption: str, paper_context):
            return {"confidence": 0.95, "formation": "F", "age": "A"}

        results = link_species_to_geology([panel], [fig], m3_inference_callable=fake_m3)
        assert results[0].source == LINK_SOURCE_M3
        # Confidence clamped to <= 0.6 per spec
        assert results[0].confidence <= 0.6

    def test_m3_returns_garbage(self):
        panel = FakePanel(caption_snippet="Plate with no info")
        fig = _strat_column(figure_id="fig2", caption="Italy")

        def fake_m3(panel_caption: str, paper_context):
            return "not a dict"

        results = link_species_to_geology([panel], [fig], m3_inference_callable=fake_m3)
        assert results[0].source == LINK_SOURCE_UNLINKED


class TestUnlinkedFallback:
    def test_unlinked_when_nothing_matches(self):
        panel = FakePanel(caption_snippet="No useful info")
        fig = _strat_column(figure_id="fig2", caption="Italy")
        results = link_species_to_geology([panel], [fig])
        assert results[0].source == LINK_SOURCE_UNLINKED
        assert results[0].confidence == 0.0

    def test_unlinked_when_no_figures(self):
        panel = FakePanel(caption_snippet="Sample S1")
        results = link_species_to_geology([panel], [])
        assert results[0].source == LINK_SOURCE_UNLINKED

    def test_plate_figures_not_indexed(self):
        # A plate figure with sample info should NOT be indexed.
        panel = FakePanel(caption_snippet="Sample S1")
        plate = _plate(figure_id="fig1", caption="Plate with Sample S1")
        results = link_species_to_geology([panel], [plate])
        assert results[0].source == LINK_SOURCE_UNLINKED


class TestMultiPaperIsolation:
    def test_paper_id_boundary(self):
        # Panel belongs to paper p1, but figures are for p2.
        panel = FakePanel(paper_id="p1", caption_snippet="Sample S1")
        fig = _strat_column(figure_id="fig2", caption="Sample S1", paper_id="p2")
        results = link_species_to_geology([panel], [fig])
        # Different paper → cannot link → unlinked.
        assert results[0].source == LINK_SOURCE_UNLINKED

    def test_multiple_papers_link_correctly(self):
        # Two papers, each with their own panel + strat column. Each
        # panel should link to its own paper's strat column only.
        panel_a = FakePanel(paper_id="pA", panel_id="pa1", caption_snippet="Sample S1")
        panel_b = FakePanel(paper_id="pB", panel_id="pb1", caption_snippet="Sample X9")
        fig_a = _strat_column(figure_id="figA2", caption="Sample S1, Italy", paper_id="pA")
        fig_b = _strat_column(figure_id="figB2", caption="Sample X9, Greece", paper_id="pB")
        results = link_species_to_geology([panel_a, panel_b], [fig_a, fig_b])
        assert results[0].panel_id == "pa1"
        assert results[0].figure_id == "figA2"
        assert results[1].panel_id == "pb1"
        assert results[1].figure_id == "figB2"


class TestEndToEnd:
    def test_full_workflow(self):
        panel = FakePanel(
            paper_id="p1",
            panel_id="p1",
            species="Triassocampe sp.",
            caption_snippet="All from Sample S1, Late Triassic, from Italy",
        )
        fig_strat = _strat_column(
            figure_id="strat1",
            caption="Strat column: Sample S1, Italy, Scaglia Fm, Late Triassic",
        )
        results = link_species_to_geology([panel], [fig_strat])
        assert len(results) == 1
        r = results[0]
        assert r.source == LINK_SOURCE_SAMPLE
        assert r.confidence == 1.0
        assert r.species == "Triassocampe sp."

    def test_dict_panels_accepted(self):
        panel = {
            "paper_id": "p1",
            "figure_id": "fig1",
            "panel_id": "p1",
            "species": "G. s",
            "caption_snippet": "Sample S1",
        }
        fig = _strat_column(figure_id="strat1", caption="Sample S1")
        results = link_species_to_geology([panel], [fig])
        assert results[0].source == LINK_SOURCE_SAMPLE

    def test_metadata_caption_fallback(self):
        # No caption_snippet, but metadata.caption has the info.
        panel = FakePanel(caption_snippet="", metadata={"caption": "Sample S1"})
        fig = _strat_column(figure_id="strat1", caption="Sample S1")
        results = link_species_to_geology([panel], [fig])
        assert results[0].source == LINK_SOURCE_SAMPLE


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
