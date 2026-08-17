"""Phase 66 Plan C.3 — visual coordinate trigger logic tests.

The ``link_visual_coordinates`` function fires ONLY when:
  1. The panel's Phase A Strategy 1 (sample_match) didn't reach
     confidence 1.0 — i.e. ``link_source != "sample_match"`` or the
     panel has no link at all.
  2. The paper has BOTH a plate figure AND a strat column /
     paleogeographic map figure.
  3. The two figures haven't already been linked by locality share.

Otherwise it returns an empty list and the panel's existing Phase A
linkage stands. This makes Phase C a precision refinement rather than
a recall boost.
"""

from __future__ import annotations

from typing import Any

import pytest

from rlpe.cross_figure_linker import (
    LINK_SOURCE_LOCALITY,
    LINK_SOURCE_M3,
    LINK_SOURCE_SAMPLE,
    LINK_SOURCE_UNLINKED,
    LinkResult,
    link_visual_coordinates,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plate_panel(
    *,
    paper_id: str = "p1",
    panel_id: str = "fig1_1",
    species: str = "Genus species",
    caption_snippet: str = "Plate 1, figs 1-4",
    link_source: str | None = None,
    link_confidence: float = 0.0,
    link_figure_id: str | None = None,
) -> dict[str, Any]:
    """Build a panel dict shaped like the Phase A pipeline passes to the
    cross_figure_linker."""
    return {
        "paper_id": paper_id,
        "panel_id": panel_id,
        "species": species,
        "caption_snippet": caption_snippet,
        "metadata": {
            "link_source": link_source,
            "link_confidence": link_confidence,
            "link_figure_id": link_figure_id,
            "cross_figure_visual_links": [],
        },
    }


def _paper_figure(
    *,
    figure_id: str,
    figure_type: str,
    caption: str = "",
    formation: str | None = None,
    age: str | None = None,
    paper_id: str = "p1",
) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "figure_id": figure_id,
        "figure_type": figure_type,
        "caption": caption,
        "formation": formation,
        "age": age,
    }


class _FakeM3Engine:
    """Fake m3_engine for trigger-logic tests.

    Only implements ``cross_figure_visual_inference`` because that's
    the only method the trigger should call. Records each call so we
    can assert the gate logic skipped us when it should have.
    """

    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.response = response or {
            "plate_panels": [
                {
                    "cell_label": "1",
                    "species": "Genus species",
                    "links_to_strat_layer": 3,
                    "links_to_age": "Late Triassic",
                    "links_to_formation": "Scaglia",
                    "confidence": 0.9,
                }
            ]
        }
        self.calls: list[dict[str, Any]] = []

    def cross_figure_visual_inference(
        self,
        plate_image: Any,
        strat_image: Any,
        plate_caption: str,
        strat_caption: str,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "plate_caption": plate_caption,
                "strat_caption": strat_caption,
            }
        )
        return self.response


# ---------------------------------------------------------------------------
# Trigger condition tests
# ---------------------------------------------------------------------------


class TestTriggerSkipsWhenStrategy1Matched:
    def test_skipped_when_sample_match_full_confidence(self):
        """Phase A Strategy 1 already nailed it — Phase C should NOT
        run a vision call."""
        panels = [
            _plate_panel(
                link_source=LINK_SOURCE_SAMPLE,
                link_confidence=1.0,
                link_figure_id="fig_strat_2",
            )
        ]
        figures = [
            _paper_figure(figure_id="fig1", figure_type="plate"),
            _paper_figure(figure_id="fig_strat_2", figure_type="strat_column"),
        ]
        m3 = _FakeM3Engine()
        out = link_visual_coordinates(panels, figures, m3)
        assert out == [[]]  # no visual link
        assert m3.calls == []  # M3 never called

    def test_skipped_when_ambiguous_sample_match(self):
        """Even at confidence 0.9 (ambiguous sample_match), Phase C
        skips — sample_match is still the strongest Phase A signal."""
        panels = [
            _plate_panel(
                link_source=LINK_SOURCE_SAMPLE,
                link_confidence=0.9,
                link_figure_id="fig_strat_2",
            )
        ]
        figures = [
            _paper_figure(figure_id="fig1", figure_type="plate"),
            _paper_figure(figure_id="fig_strat_2", figure_type="strat_column"),
        ]
        m3 = _FakeM3Engine()
        out = link_visual_coordinates(panels, figures, m3)
        assert out == [[]]
        assert m3.calls == []


class TestTriggerFiresWhenStrategy1Missed:
    def test_fires_when_locality_match(self):
        """Strategy 1 missed → locality took over → Phase C SHOULD fire."""
        panels = [
            _plate_panel(
                link_source=LINK_SOURCE_LOCALITY,
                link_confidence=0.7,
                link_figure_id="fig_strat_2",
            )
        ]
        figures = [
            _paper_figure(figure_id="fig1", figure_type="plate"),
            _paper_figure(figure_id="fig_strat_2", figure_type="strat_column"),
        ]
        m3 = _FakeM3Engine()
        out = link_visual_coordinates(panels, figures, m3)
        assert len(out) == 1
        assert len(out[0]) == 1  # one visual link
        assert len(m3.calls) == 1  # M3 WAS called
        # The visual link should reference fig_strat_2
        link = out[0][0]
        assert link["target_figure_id"] == "fig_strat_2"
        assert link["source"] == "m3_visual"

    def test_fires_when_m3_inference_only(self):
        """No Strategy 1 hit, M3 was used → Phase C SHOULD still fire
        because Strategy 1 didn't reach confidence 1.0."""
        panels = [
            _plate_panel(
                link_source=LINK_SOURCE_M3,
                link_confidence=0.4,
                link_figure_id="fig_strat_2",
            )
        ]
        figures = [
            _paper_figure(figure_id="fig1", figure_type="plate"),
            _paper_figure(figure_id="fig_strat_2", figure_type="strat_column"),
        ]
        m3 = _FakeM3Engine()
        out = link_visual_coordinates(panels, figures, m3)
        assert len(out[0]) == 1

    def test_fires_when_unlinked(self):
        """Strategy 1, 2, 3 all missed → still fire Phase C (precision
        refinement can either confirm unlinked or find a link)."""
        panels = [
            _plate_panel(
                link_source=LINK_SOURCE_UNLINKED,
                link_confidence=0.0,
            )
        ]
        figures = [
            _paper_figure(figure_id="fig1", figure_type="plate"),
            _paper_figure(figure_id="fig_strat_2", figure_type="strat_column"),
        ]
        m3 = _FakeM3Engine()
        out = link_visual_coordinates(panels, figures, m3)
        assert len(out[0]) == 1


class TestTriggerRequiresPlateAndStrat:
    def test_skipped_when_no_strat_column(self):
        panels = [
            _plate_panel(
                link_source=LINK_SOURCE_LOCALITY,
                link_confidence=0.7,
            )
        ]
        figures = [
            _paper_figure(figure_id="fig1", figure_type="plate"),
            _paper_figure(figure_id="fig2", figure_type="range_chart"),
        ]
        m3 = _FakeM3Engine()
        out = link_visual_coordinates(panels, figures, m3)
        assert out == [[]]
        assert m3.calls == []

    def test_skipped_when_no_plate_figure(self):
        panels = [
            _plate_panel(
                link_source=LINK_SOURCE_LOCALITY,
                link_confidence=0.7,
            )
        ]
        figures = [
            _paper_figure(figure_id="fig_strat_1", figure_type="strat_column"),
            _paper_figure(figure_id="fig_map_1", figure_type="paleogeographic_map"),
        ]
        m3 = _FakeM3Engine()
        out = link_visual_coordinates(panels, figures, m3)
        assert out == [[]]
        assert m3.calls == []

    def test_skipped_when_no_figures(self):
        panels = [
            _plate_panel(
                link_source=LINK_SOURCE_LOCALITY,
                link_confidence=0.7,
            )
        ]
        m3 = _FakeM3Engine()
        out = link_visual_coordinates(panels, [], m3)
        assert out == [[]]
        assert m3.calls == []

    def test_fires_for_paleogeographic_map_too(self):
        """The trigger condition accepts ANY strat column / litholog /
        paleogeographic_map / range_chart as the target figure."""
        panels = [
            _plate_panel(
                link_source=LINK_SOURCE_LOCALITY,
                link_confidence=0.7,
            )
        ]
        figures = [
            _paper_figure(figure_id="fig1", figure_type="plate"),
            _paper_figure(figure_id="fig_map_1", figure_type="paleogeographic_map"),
        ]
        m3 = _FakeM3Engine()
        out = link_visual_coordinates(panels, figures, m3)
        assert len(out[0]) == 1


class TestEngineNone:
    def test_engine_none_returns_empty(self):
        """No M3 engine available — Phase C silently skips."""
        panels = [
            _plate_panel(
                link_source=LINK_SOURCE_LOCALITY,
                link_confidence=0.7,
            )
        ]
        figures = [
            _paper_figure(figure_id="fig1", figure_type="plate"),
            _paper_figure(figure_id="fig_strat_2", figure_type="strat_column"),
        ]
        out = link_visual_coordinates(panels, figures, None)
        assert out == [[]]

    def test_engine_without_visual_method_skipped(self):
        """Engine present but no cross_figure_visual_inference — skip."""
        panels = [
            _plate_panel(
                link_source=LINK_SOURCE_LOCALITY,
                link_confidence=0.7,
            )
        ]
        figures = [
            _paper_figure(figure_id="fig1", figure_type="plate"),
            _paper_figure(figure_id="fig_strat_2", figure_type="strat_column"),
        ]
        m3 = object()  # no method
        out = link_visual_coordinates(panels, figures, m3)
        assert out == [[]]


class TestOutputShape:
    def test_returns_list_per_panel(self):
        """Output is a list-of-lists — outer indexed by panel, inner
        is the list of visual links for that panel."""
        panels = [
            _plate_panel(panel_id="p1", link_source=LINK_SOURCE_LOCALITY, link_confidence=0.7),
            _plate_panel(panel_id="p2", link_source=LINK_SOURCE_LOCALITY, link_confidence=0.7),
        ]
        figures = [
            _paper_figure(figure_id="fig1", figure_type="plate"),
            _paper_figure(figure_id="fig_strat_2", figure_type="strat_column"),
        ]
        m3 = _FakeM3Engine()
        out = link_visual_coordinates(panels, figures, m3)
        assert len(out) == 2
        assert len(out[0]) == 1
        assert len(out[1]) == 1

    def test_empty_response_returns_empty_inner_list(self):
        """When M3 says nothing, the panel gets [] rather than a fake link."""
        panels = [
            _plate_panel(
                link_source=LINK_SOURCE_LOCALITY,
                link_confidence=0.7,
            )
        ]
        figures = [
            _paper_figure(figure_id="fig1", figure_type="plate"),
            _paper_figure(figure_id="fig_strat_2", figure_type="strat_column"),
        ]
        m3 = _FakeM3Engine(response={"plate_panels": []})
        out = link_visual_coordinates(panels, figures, m3)
        assert out == [[]]


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
