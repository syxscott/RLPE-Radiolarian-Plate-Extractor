"""Phase 66 Plan C.1 — M3 cross-figure visual inference tests.

The cross_figure_visual prompt asks M3 to look at a plate image AND a
strat-column / paleogeographic-map image together and emit per-panel
mappings:

  {
    "plate_panels": [
      {"cell_label": "5", "species": "Genus species",
       "links_to_strat_layer": 3, "links_to_age": "Late Cretaceous",
       "links_to_formation": "Scaglia Rossa", "confidence": 0.92}
    ]
  }

This is the VISION counterpart to ``cross_figure_inference`` (text-only).
It is the precision-refinement path that fires when Strategy 1 (sample
match) didn't reach confidence 1.0 — see Plan C.3 for the trigger logic.
"""

from __future__ import annotations

from typing import Any

import pytest

from rlpe.m3_engine import PROMPT_REGISTRY, M3Engine
from tests.fakes.fake_m3_backend import FakeM3Backend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(canned: list[dict[str, Any]]) -> M3Engine:
    backend = FakeM3Backend(canned_responses=canned)
    return M3Engine(backend=backend, config={})


class _DummyImage:
    """Stand-in for PIL.Image.Image — exposes .width and .height only."""

    def __init__(self, width: int = 256, height: int = 256) -> None:
        self.width = width
        self.height = height


# ---------------------------------------------------------------------------
# Prompt registry tests
# ---------------------------------------------------------------------------


class TestCrossFigureVisualPrompt:
    def test_prompt_registered(self):
        assert "cross_figure_visual" in PROMPT_REGISTRY
        prompt = PROMPT_REGISTRY["cross_figure_visual"]
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_prompt_mentions_plate_and_target(self):
        prompt = PROMPT_REGISTRY["cross_figure_visual"].lower()
        assert "plate" in prompt
        # Mentions at least one of: strat column, paleogeographic map, litholog
        assert any(
            kw in prompt
            for kw in ("strat", "paleogeographic", "litholog", "map")
        )

    def test_prompt_describes_output_shape(self):
        prompt = PROMPT_REGISTRY["cross_figure_visual"]
        # The prompt must describe the plate_panels array and the link fields
        assert "plate_panels" in prompt
        assert "links_to_strat_layer" in prompt
        assert "links_to_age" in prompt
        assert "links_to_formation" in prompt
        assert "confidence" in prompt


# ---------------------------------------------------------------------------
# Method happy path
# ---------------------------------------------------------------------------


class TestCrossFigureVisualInferenceHappyPath:
    def test_basic_visual_inference(self):
        engine = _make_engine([
            {
                "raw_text": (
                    '{"plate_panels": ['
                    '  {"cell_label": "5", "species": "Triassocampe sp.", '
                    '   "links_to_strat_layer": 3, "links_to_age": "Late Triassic", '
                    '   "links_to_formation": "Scaglia Rossa", "confidence": 0.92}'
                    ']}'
                ),
            }
        ])
        result = engine.cross_figure_visual_inference(
            plate_image=_DummyImage(),
            strat_image=_DummyImage(),
            plate_caption="All from Sample S1, Plate 1",
            strat_caption="Scaglia Rossa, Late Triassic",
        )
        assert isinstance(result, dict)
        assert "plate_panels" in result
        assert len(result["plate_panels"]) == 1
        panel = result["plate_panels"][0]
        assert panel["cell_label"] == "5"
        assert panel["species"] == "Triassocampe sp."
        assert panel["links_to_strat_layer"] == 3
        assert panel["links_to_age"] == "Late Triassic"
        assert panel["links_to_formation"] == "Scaglia Rossa"
        assert 0.0 <= panel["confidence"] <= 1.0

    def test_multiple_panels(self):
        engine = _make_engine([
            {
                "raw_text": (
                    '{"plate_panels": ['
                    '  {"cell_label": "1", "species": "Genus a", '
                    '   "links_to_strat_layer": 1, "links_to_age": "Late Triassic", '
                    '   "links_to_formation": "Scaglia", "confidence": 0.88},'
                    '  {"cell_label": "2", "species": "Genus b", '
                    '   "links_to_strat_layer": 2, "links_to_age": "Late Triassic", '
                    '   "links_to_formation": "Scaglia", "confidence": 0.81}'
                    ']}'
                ),
            }
        ])
        result = engine.cross_figure_visual_inference(
            plate_image=_DummyImage(),
            strat_image=_DummyImage(),
            plate_caption="Plate 1",
            strat_caption="Strat",
        )
        assert len(result["plate_panels"]) == 2
        assert result["plate_panels"][0]["cell_label"] == "1"
        assert result["plate_panels"][1]["cell_label"] == "2"


# ---------------------------------------------------------------------------
# Fallback paths
# ---------------------------------------------------------------------------


class TestCrossFigureVisualInferenceFallback:
    def test_backend_none_returns_empty(self):
        engine = M3Engine(backend=None, config={})
        result = engine.cross_figure_visual_inference(
            plate_image=_DummyImage(),
            strat_image=_DummyImage(),
            plate_caption="Plate 1",
            strat_caption="Strat",
        )
        assert result == {"plate_panels": []}

    def test_backend_fallback_returns_empty(self):
        backend = FakeM3Backend(canned_responses=[{"fallback_used": True, "raw_text": ""}])
        engine = M3Engine(backend=backend, config={})
        result = engine.cross_figure_visual_inference(
            plate_image=_DummyImage(),
            strat_image=_DummyImage(),
            plate_caption="Plate 1",
            strat_caption="Strat",
        )
        assert result == {"plate_panels": []}

    def test_malformed_json_returns_empty(self):
        backend = FakeM3Backend(canned_responses=[{"raw_text": "not json"}])
        engine = M3Engine(backend=backend, config={})
        result = engine.cross_figure_visual_inference(
            plate_image=_DummyImage(),
            strat_image=_DummyImage(),
            plate_caption="Plate 1",
            strat_caption="Strat",
        )
        assert result == {"plate_panels": []}

    def test_non_dict_json_returns_empty(self):
        backend = FakeM3Backend(canned_responses=[{"raw_text": "[1,2,3]"}])
        engine = M3Engine(backend=backend, config={})
        result = engine.cross_figure_visual_inference(
            plate_image=_DummyImage(),
            strat_image=_DummyImage(),
            plate_caption="Plate 1",
            strat_caption="Strat",
        )
        assert result == {"plate_panels": []}

    def test_missing_plate_panels_key_returns_empty(self):
        backend = FakeM3Backend(canned_responses=[{"raw_text": '{"other": "shape"}'}])
        engine = M3Engine(backend=backend, config={})
        result = engine.cross_figure_visual_inference(
            plate_image=_DummyImage(),
            strat_image=_DummyImage(),
            plate_caption="Plate 1",
            strat_caption="Strat",
        )
        assert result == {"plate_panels": []}

    def test_tiny_image_returns_empty(self):
        engine = _make_engine([
            {"raw_text": '{"plate_panels": [{"cell_label": "1"}]}'}
        ])
        result = engine.cross_figure_visual_inference(
            plate_image=_DummyImage(8, 8),
            strat_image=_DummyImage(8, 8),
            plate_caption="Plate 1",
            strat_caption="Strat",
        )
        assert result == {"plate_panels": []}

    def test_panel_entry_without_required_fields_is_dropped(self):
        """Per Plan C.1, panel entries that lack cell_label OR species are
        noise — drop them so the downstream linker only sees clean rows."""
        engine = _make_engine([
            {
                "raw_text": (
                    '{"plate_panels": ['
                    '  {"cell_label": "1", "species": "Genus a", "confidence": 0.9},'
                    '  {"cell_label": "2"},'  # missing species
                    '  {"species": "Genus c"}'  # missing cell_label
                    ']}'
                ),
            }
        ])
        result = engine.cross_figure_visual_inference(
            plate_image=_DummyImage(),
            strat_image=_DummyImage(),
            plate_caption="Plate 1",
            strat_caption="Strat",
        )
        assert len(result["plate_panels"]) == 1
        assert result["plate_panels"][0]["cell_label"] == "1"

    def test_panel_confidence_clamped_to_unit_interval(self):
        engine = _make_engine([
            {
                "raw_text": (
                    '{"plate_panels": [{"cell_label": "1", "species": "G s", '
                    '"confidence": 1.5}]}'
                ),
            }
        ])
        result = engine.cross_figure_visual_inference(
            plate_image=_DummyImage(),
            strat_image=_DummyImage(),
            plate_caption="Plate 1",
            strat_caption="Strat",
        )
        assert result["plate_panels"][0]["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# Image / prompt plumbing
# ---------------------------------------------------------------------------


class TestCrossFigureVisualPromptPlumbing:
    def test_includes_plate_caption_in_prompt(self):
        engine = _make_engine([
            {"raw_text": '{"plate_panels": []}'}
        ])
        engine.cross_figure_visual_inference(
            plate_image=_DummyImage(),
            strat_image=_DummyImage(),
            plate_caption="UNIQUE_PLATE_CAPTION_42",
            strat_caption="Strat caption",
        )
        # Find the infer_panel call (vision call) and check it includes our text
        calls = engine.backend.calls
        assert len(calls) >= 1
        user_prompt = calls[0].user_prompt
        assert "UNIQUE_PLATE_CAPTION_42" in user_prompt

    def test_includes_strat_caption_in_prompt(self):
        engine = _make_engine([
            {"raw_text": '{"plate_panels": []}'}
        ])
        engine.cross_figure_visual_inference(
            plate_image=_DummyImage(),
            strat_image=_DummyImage(),
            plate_caption="Plate caption",
            strat_caption="UNIQUE_STRAT_CAPTION_99",
        )
        user_prompt = engine.backend.calls[0].user_prompt
        assert "UNIQUE_STRAT_CAPTION_99" in user_prompt


if __name__ == "__main__":
    pytest.main([__file__, "-q"])