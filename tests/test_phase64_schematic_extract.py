"""Phase 64 Plan B Task 3: M3 ``extract_schematic`` method + prompt.

The new method on ``M3Engine`` runs the ``schematic_geo`` vision
prompt on schematic / diagram / reconstruction / phylogenetic
figures and returns the JSON matching the prompt contract:

  {
    "figure_type": "schematic" | "diagram" | "reconstruction" | "phylogenetic",
    "text_elements": [{"text", "type", "confidence"}, ...],
    "relationships": [{"from", "to", "label"}, ...],
    "extracted_facts": {
      "ages_mentioned": [str, ...],
      "geographic_names": [str, ...],
      "taxa_mentioned": [str, ...],
    },
    "confidence": float,
  }

Tests use ``FakeM3Backend`` (tests/fakes/fake_m3_backend.py) to
avoid any outbound HTTP traffic. The fake accepts canned responses
keyed by ``match`` callable on the system prompt so we can route
the new ``schematic_geo`` prompt to its own canned answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from PIL import Image

from rlpe.m3_engine import M3Engine, PROMPT_REGISTRY, SECTION_TYPE_BY_FIGURE
from tests.fakes.fake_m3_backend import FakeM3Backend


def _make_image(w: int = 64, h: int = 64) -> Image.Image:
    """Create a tiny solid-color image large enough to pass the
    32x32 size guard in extract_schematic."""
    return Image.new("RGB", (w, h), color=(255, 255, 255))


def _schematic_canned() -> dict:
    """Representative M3 response for a schematic figure."""
    return {
        "label": None,
        "species": None,
        "confidence": 0.96,
        "reasoning": "fake-schematic",
        "fallback_used": False,
        "raw_text": (
            "{"
            '"figure_type": "schematic",'
            '"text_elements": ['
            '{"text": "Late Triassic", "type": "age", "confidence": 0.98},'
            '{"text": "Tethys Ocean", "type": "geographic", "confidence": 0.95}'
            "],"
            '"relationships": ['
            '{"from": "box1", "to": "box2", "label": "evolved into"}'
            "],"
            '"extracted_facts": {'
            '"ages_mentioned": ["Late Triassic", "Carnian"],'
            '"geographic_names": ["Tethys"],'
            '"taxa_mentioned": ["Genus species"]'
            "},"
            '"confidence": 0.96'
            "}"
        ),
        "request_id": "fake-schematic-req-0",
        "model_version": "MiniMax-M3-fake",
        "usage": {"input_tokens": 200, "output_tokens": 120},
        "cost_cny": 0.0015,
    }


def _phylogenetic_canned() -> dict:
    """Representative M3 response for a phylogenetic tree."""
    return {
        "label": None,
        "species": None,
        "confidence": 0.92,
        "reasoning": "fake-phylogenetic",
        "fallback_used": False,
        "raw_text": (
            "{"
            '"figure_type": "phylogenetic",'
            '"text_elements": ['
            '{"text": "Nassellaria", "type": "taxon", "confidence": 0.99}'
            "],"
            '"relationships": ['
            '{"from": "node_a", "to": "node_b", "label": "sister to"}'
            "],"
            '"extracted_facts": {'
            '"ages_mentioned": [],'
            '"geographic_names": [],'
            '"taxa_mentioned": ["Nassellaria"]'
            "},"
            '"confidence": 0.92'
            "}"
        ),
        "request_id": "fake-phyl-req-0",
        "model_version": "MiniMax-M3-fake",
        "usage": {"input_tokens": 200, "output_tokens": 100},
        "cost_cny": 0.0014,
    }


def _make_engine_with_canned(canned: list[dict]) -> M3Engine:
    backend = FakeM3Backend(canned_responses=canned)
    return M3Engine(backend=backend)


class TestSchematicGeoPrompt:
    """PROMPT_REGISTRY['schematic_extract'] exists with the contract
    documented in the design spec."""

    def test_prompt_registered(self) -> None:
        assert "schematic_extract" in PROMPT_REGISTRY
        prompt = PROMPT_REGISTRY["schematic_extract"]
        assert isinstance(prompt, str)
        # Sanity-check the prompt mentions the JSON shape.
        assert "text_elements" in prompt
        assert "relationships" in prompt
        assert "extracted_facts" in prompt
        assert "figure_type" in prompt

    def test_section_type_mapping_for_all_four_types(self) -> None:
        """All 4 new figure types map to a single section_type value
        so downstream filters can group them together."""
        for fig_type in ("schematic", "diagram", "reconstruction", "phylogenetic"):
            assert SECTION_TYPE_BY_FIGURE.get(fig_type) == "schematic_figure"


class TestExtractSchematic:
    """M3Engine.extract_schematic dispatches to schematic_extract prompt."""

    def test_returns_prompt_contract_shape(self) -> None:
        engine = _make_engine_with_canned(
            [
                {
                    "match": lambda sp: "schematic_extract" in sp,
                    **_schematic_canned(),
                }
            ]
        )
        result = engine.extract_schematic(
            image=_make_image(),
            caption="Figure 5. Schematic of the paleoceanographic model.",
            figure_type="schematic",
            paper_id="pouille2014",
            figure_id="fig5",
        )
        assert result is not None
        assert result["figure_type"] == "schematic"
        assert isinstance(result["text_elements"], list)
        assert len(result["text_elements"]) == 2
        # Text element shape
        elem = result["text_elements"][0]
        assert set(elem.keys()) == {"text", "type", "confidence"}
        assert elem["text"] == "Late Triassic"
        assert elem["type"] == "age"
        assert 0.0 <= elem["confidence"] <= 1.0
        # Relationships shape
        rel = result["relationships"][0]
        assert set(rel.keys()) == {"from", "to", "label"}
        # Extracted facts shape
        facts = result["extracted_facts"]
        assert "ages_mentioned" in facts
        assert "geographic_names" in facts
        assert "taxa_mentioned" in facts
        # Overall confidence
        assert 0.0 <= result["confidence"] <= 1.0

    def test_returns_none_for_unsupported_figure_type(self) -> None:
        """Calling extract_schematic with figure_type='plate' (which
        has its own extract_geology path) must return None — the
        caller falls through to the geology extraction path."""
        engine = _make_engine_with_canned(
            [
                {
                    "match": lambda sp: "schematic_extract" in sp,
                    **_schematic_canned(),
                }
            ]
        )
        result = engine.extract_schematic(
            image=_make_image(),
            caption="Plate 1.",
            figure_type="plate",
            paper_id="any",
            figure_id="any",
        )
        assert result is None

    def test_returns_none_for_tiny_image(self) -> None:
        """An image below 32x32 returns None without making the
        M3 call — vision on a 16x16 thumbnail is pure noise."""
        engine = _make_engine_with_canned(
            [
                {
                    "match": lambda sp: "schematic_extract" in sp,
                    **_schematic_canned(),
                }
            ]
        )
        tiny = Image.new("RGB", (16, 16), color=(255, 255, 255))
        result = engine.extract_schematic(
            image=tiny,
            caption="Caption",
            figure_type="schematic",
            paper_id="any",
            figure_id="any",
        )
        assert result is None

    def test_returns_none_for_missing_image(self) -> None:
        """A non-PIL image (no .width) returns None without raising."""
        engine = _make_engine_with_canned([])
        result = engine.extract_schematic(
            image=None,  # type: ignore[arg-type]
            caption="Caption",
            figure_type="schematic",
            paper_id="any",
            figure_id="any",
        )
        assert result is None

    def test_provenance_fields_stamped(self) -> None:
        """The returned dict carries _paper_id / _figure_id /
        _source for downstream audit. These are leading-underscore
        keys so the JSONL export can strip them."""
        engine = _make_engine_with_canned(
            [
                {
                    "match": lambda sp: "schematic_extract" in sp,
                    **_schematic_canned(),
                }
            ]
        )
        result = engine.extract_schematic(
            image=_make_image(),
            caption="Schematic diagram",
            figure_type="schematic",
            paper_id="bandini2011",
            figure_id="fig-sch-1",
        )
        assert result is not None
        assert result["_paper_id"] == "bandini2011"
        assert result["_figure_id"] == "fig-sch-1"
        assert result["_source"] == "schematic_extract"

    def test_overrides_model_figure_type_with_classifier(self) -> None:
        """If the LLM emits figure_type='diagram' but the classifier
        (figure_type arg) says 'schematic', prefer the classifier's
        value — caption-based classification is more reliable."""
        bad_response = _schematic_canned()
        # Replace the figure_type field in the JSON literal.
        bad_response["raw_text"] = bad_response["raw_text"].replace(
            '"figure_type": "schematic"',
            '"figure_type": "diagram"',
        )
        engine = _make_engine_with_canned(
            [
                {
                    "match": lambda sp: "schematic_extract" in sp,
                    **bad_response,
                }
            ]
        )
        result = engine.extract_schematic(
            image=_make_image(),
            caption="Schematic caption",
            figure_type="schematic",
            paper_id="any",
            figure_id="any",
        )
        assert result is not None
        assert result["figure_type"] == "schematic"

    def test_phylogenetic_routing(self) -> None:
        """extract_schematic works for phylogenetic figures too —
        the four types share a single prompt."""
        engine = _make_engine_with_canned(
            [
                {
                    "match": lambda sp: "schematic_extract" in sp,
                    **_phylogenetic_canned(),
                }
            ]
        )
        result = engine.extract_schematic(
            image=_make_image(),
            caption="Phylogenetic tree of Nassellaria",
            figure_type="phylogenetic",
            paper_id="pouille2014",
            figure_id="fig-phyl-1",
        )
        assert result is not None
        assert result["figure_type"] == "phylogenetic"
        assert result["extracted_facts"]["taxa_mentioned"] == ["Nassellaria"]

    def test_handles_malformed_json_gracefully(self) -> None:
        """When the LLM returns malformed JSON, the method returns
        None instead of raising — same contract as extract_geology."""
        bad_response = _schematic_canned()
        bad_response["raw_text"] = "{ this is not valid JSON }"
        engine = _make_engine_with_canned(
            [
                {
                    "match": lambda sp: "schematic_extract" in sp,
                    **bad_response,
                }
            ]
        )
        result = engine.extract_schematic(
            image=_make_image(),
            caption="Schematic caption",
            figure_type="schematic",
            paper_id="any",
            figure_id="any",
        )
        assert result is None
