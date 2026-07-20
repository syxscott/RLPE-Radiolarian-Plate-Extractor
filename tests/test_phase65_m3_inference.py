"""Phase 65 Plan A.3 — M3 cross-figure inference tests."""

from __future__ import annotations

from typing import Any

import pytest

from rlpe.m3_engine import PROMPT_REGISTRY, M3Engine
from tests.fakes.fake_m3_backend import FakeM3Backend


def _make_engine(canned: list[dict[str, Any]]) -> M3Engine:
    backend = FakeM3Backend(canned_responses=canned)
    return M3Engine(backend=backend, config={})


class TestPromptRegistered:
    def test_prompt_in_registry(self):
        assert "cross_figure_inference" in PROMPT_REGISTRY
        prompt = PROMPT_REGISTRY["cross_figure_inference"]
        assert "species" in prompt
        assert "formation" in prompt
        assert "age" in prompt
        assert "locality" in prompt
        assert "figure_id" in prompt
        assert "confidence" in prompt

    def test_prompt_mentions_plate_caption_and_paper_summary(self):
        prompt = PROMPT_REGISTRY["cross_figure_inference"].lower()
        assert "plate" in prompt
        assert "paper" in prompt
        assert "figure" in prompt


class TestInferenceHappyPath:
    def test_basic_inference(self):
        engine = _make_engine([
            {
                "raw_text": '{"species": "Triassocampe sp.", "age": "Late Triassic", '
                            '"formation": "Scaglia", "locality": "Italy", '
                            '"figure_id": "fig2", "confidence": 0.55}',
            }
        ])
        result = engine.infer_species_age_formation(
            "All specimens from Plate 1",
            paper_context={"figures": [{"figure_id": "fig2", "figure_type": "strat_column",
                                         "caption": "Late Triassic, Italy"}]},
        )
        assert result["species"] == "Triassocampe sp."
        assert result["age"] == "Late Triassic"
        assert result["formation"] == "Scaglia"
        assert result["locality"] == "Italy"
        assert result["figure_id"] == "fig2"
        # Confidence clamped to [0.3, 0.6]
        assert 0.3 <= result["confidence"] <= 0.6

    def test_confidence_clamped_high(self):
        engine = _make_engine([
            {
                "raw_text": '{"species": "X", "confidence": 0.95}',
            }
        ])
        result = engine.infer_species_age_formation("Plate 1", {})
        assert result["confidence"] <= 0.6

    def test_confidence_clamped_low(self):
        engine = _make_engine([
            {
                "raw_text": '{"species": "X", "confidence": 0.1}',
            }
        ])
        result = engine.infer_species_age_formation("Plate 1", {})
        assert result["confidence"] >= 0.3


class TestInferenceFallback:
    def test_backend_none(self):
        engine = M3Engine(backend=None, config={})
        result = engine.infer_species_age_formation("Plate 1", {})
        assert result["confidence"] == 0.0
        assert result["species"] is None

    def test_backend_fallback(self):
        backend = FakeM3Backend(canned_responses=[{"fallback_used": True, "raw_text": ""}])
        engine = M3Engine(backend=backend, config={})
        result = engine.infer_species_age_formation("Plate 1", {})
        assert result["confidence"] == 0.0

    def test_malformed_json(self):
        backend = FakeM3Backend(canned_responses=[{"raw_text": "not json at all"}])
        engine = M3Engine(backend=backend, config={})
        result = engine.infer_species_age_formation("Plate 1", {})
        assert result["confidence"] == 0.0

    def test_non_dict_json(self):
        backend = FakeM3Backend(canned_responses=[{"raw_text": "[1, 2, 3]"}])
        engine = M3Engine(backend=backend, config={})
        result = engine.infer_species_age_formation("Plate 1", {})
        assert result["confidence"] == 0.0


class TestInferencePromptShape:
    def test_prompt_truncates_long_captions(self):
        engine = _make_engine([
            {
                "raw_text": '{"species": "X", "confidence": 0.5}',
            }
        ])
        long_caption = "A" * 1000
        long_fig_cap = "B" * 500
        result = engine.infer_species_age_formation(
            long_caption,
            paper_context={
                "figures": [{
                    "figure_id": "f1", "figure_type": "strat_column",
                    "caption": long_fig_cap, "formation": "F", "age": "A",
                }]
            },
        )
        # Should still parse — prompt assembly must not crash on long input.
        assert "species" in result

    def test_empty_context(self):
        engine = _make_engine([
            {"raw_text": '{"species": "X", "confidence": 0.5}'}
        ])
        result = engine.infer_species_age_formation("Plate 1", {})
        assert "species" in result

    def test_no_context(self):
        engine = _make_engine([
            {"raw_text": '{"species": "X", "confidence": 0.5}'}
        ])
        result = engine.infer_species_age_formation("Plate 1")
        assert "species" in result


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
