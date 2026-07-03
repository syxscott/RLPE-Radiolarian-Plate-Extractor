"""Tests for M3Engine.extract_geology() — multi-modal geology vision.

extract_geology() is the sibling of match_panel(): it sends a figure
image + caption to the MiniMax-M3 backend and asks for structured
geology fields (lithology, formation, member, group, country, biozone,
Ma range, coordinates). Output is a list of dicts, each shaped like a
GeologyLinkRecord so callers can append straight into
``panel.metadata.geology_links``.

These tests use ``FakeM3Backend`` from tests.fakes to avoid any real
MiniMax API call.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

# rlpe.m3_engine pulls in cv2 transitively when M3Engine() is *instantiated*
# (not when the module is imported). The PROMPT_REGISTRY constant is
# importable in any env because it's a plain module-level dict. We guard
# only the M3Engine-backed tests with cv2 skip.
import rlpe.m3_engine as _m3_mod

try:
    PROMPT_REGISTRY = _m3_mod.PROMPT_REGISTRY
    SECTION_TYPE_BY_FIGURE = _m3_mod.SECTION_TYPE_BY_FIGURE
except AttributeError:
    # Tests run before the implementation lands; the constants exist
    # once extract_geology() is implemented in Commit 2.
    PROMPT_REGISTRY = {}
    SECTION_TYPE_BY_FIGURE = {}


def _has_cv2() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except Exception:
        return False


_HAS_CV2 = _has_cv2()


from tests.fakes.fake_m3_backend import FakeM3Backend  # noqa: E402

# M3Engine-backed tests need cv2; PROMPT_REGISTRY tests above don't.
requires_cv2 = pytest.mark.skipif(not _HAS_CV2, reason="M3Engine requires cv2")


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def fake_backend():
    """Build a FakeM3Backend with one canned response per figure type."""
    return FakeM3Backend(
        canned_responses=[
            {
                "match": lambda s: "range_chart_geo" in s,
                "raw_text": (
                    '{"geo":[{"age":"Late Permian","chronostratigraphy":"Changhsingian",'
                    '"chronostratigraphy_rank":"age","ma_top":251.9,"ma_base":254.14,'
                    '"ma_mid":253.02,"formation":"Talung Formation","member":null,'
                    '"group":null,"lithology":"chert","locality":"Pingdingshan",'
                    '"country":"China","latitude":31.0,"longitude":117.0,'
                    '"biozone":"N. optima Zone","confidence":0.85}]}'
                ),
                "fallback_used": False,
                "label": None,
                "species": None,
                "confidence": 0.85,
                "reasoning": "from range chart",
                "request_id": "fake-rc-1",
                "model_version": "MiniMax-M3-fake",
                "usage": {"input_tokens": 1000, "output_tokens": 200},
                "cost_cny": 0.005,
            },
            {
                "match": lambda s: "map_geo" in s,
                "raw_text": (
                    '{"geo":[{"age":null,"chronostratigraphy":null,'
                    '"chronostratigraphy_rank":null,"ma_top":null,"ma_base":null,'
                    '"ma_mid":null,"formation":null,"member":null,"group":null,'
                    '"lithology":null,"locality":"Karnezeika","country":"Greece",'
                    '"latitude":37.5,"longitude":23.0,"biozone":null,"confidence":0.7}]}'
                ),
                "fallback_used": False,
                "label": None,
                "species": None,
                "confidence": 0.7,
                "reasoning": "from map",
                "request_id": "fake-map-1",
                "model_version": "MiniMax-M3-fake",
                "usage": {"input_tokens": 800, "output_tokens": 150},
                "cost_cny": 0.004,
            },
            {
                "match": lambda s: "litholog_geo" in s,
                "raw_text": (
                    '{"geo":[{"age":null,"chronostratigraphy":null,'
                    '"chronostratigraphy_rank":null,"ma_top":null,"ma_base":null,'
                    '"ma_mid":null,"formation":"Dalong","member":"Upper",'
                    '"group":null,"lithology":"siliceous mudstone","locality":null,'
                    '"country":null,"latitude":null,"longitude":null,'
                    '"biozone":null,"confidence":0.6}]}'
                ),
                "fallback_used": False,
                "label": None,
                "species": None,
                "confidence": 0.6,
                "reasoning": "from litholog",
                "request_id": "fake-lith-1",
                "model_version": "MiniMax-M3-fake",
                "usage": {"input_tokens": 700, "output_tokens": 130},
                "cost_cny": 0.003,
            },
        ]
    )


def _engine_with(backend):
    """Build a M3Engine backed by ``backend`` and minimal config.

    Lazy-imports ``M3Engine`` so a missing cv2 only fails when this is
    actually called (each test that needs it is decorated with
    ``@requires_cv2``). Constant-only tests in TestPromptRegistry don't
    go through here.
    """
    from rlpe.m3_engine import M3Engine

    return M3Engine(config={"m3_match_samples": 1}, backend=backend)


# --------------------------------------------------------------------------- tests


class TestPromptRegistry:
    """Lock the figure-type -> system-prompt mapping."""

    def test_six_prompts_registered(self):
        for key in (
            "plate_geo",
            "range_chart_geo",
            "map_geo",
            "strat_column_geo",
            "litholog_geo",
            "paleogeo_map_geo",
        ):
            assert key in PROMPT_REGISTRY, f"missing prompt: {key}"
            assert isinstance(PROMPT_REGISTRY[key], str)
            assert len(PROMPT_REGISTRY[key]) > 50, "each prompt should be substantive (>50 chars)"

    def test_each_prompt_returns_json_shape(self):
        for key, prompt in PROMPT_REGISTRY.items():
            assert "geo" in prompt, f"{key} prompt missing JSON 'geo' key"
            assert "age" in prompt
            assert "formation" in prompt
            assert "confidence" in prompt

    def test_section_type_table_covers_known_figures(self):
        assert SECTION_TYPE_BY_FIGURE["range_chart"] == "range_chart"
        assert SECTION_TYPE_BY_FIGURE["map"] == "location_map"
        assert SECTION_TYPE_BY_FIGURE["paleogeographic_map"] == "paleogeographic_map"
        assert SECTION_TYPE_BY_FIGURE["stratigraphic_column"] == "stratigraphic_column"
        assert SECTION_TYPE_BY_FIGURE["litholog_column"] == "litholog_column"


class TestExtractGeology:
    """Lock the M3Engine.extract_geology() method's contract.

    Each test that actually instantiates ``M3Engine`` is marked
    ``@pytest.mark.skipif(not _HAS_CV2, ...)`` because the constructor
    transitively imports cv2. PROMPT_REGISTRY / SECTION_TYPE_BY_FIGURE
    are module-level constants that don't need cv2.
    """

    def _image(self):
        from PIL import Image

        return Image.new("RGB", (96, 96), color=(255, 255, 255))

    @requires_cv2
    def test_unknown_figure_type_returns_empty_list(self, fake_backend):
        engine = _engine_with(fake_backend)
        result = engine.extract_geology(
            image=self._image(),
            caption="",
            figure_type="unknown_xyz",
            paper_id="p1",
            figure_id="f1",
        )
        assert result == []
        assert len(fake_backend.calls) == 0, "unknown figure_type must not call backend"

    @requires_cv2
    def test_image_too_small_short_circuits(self, fake_backend):
        from PIL import Image

        engine = _engine_with(fake_backend)
        result = engine.extract_geology(
            image=Image.new("RGB", (8, 8), color=(255, 255, 255)),
            caption="",
            figure_type="range_chart",
            paper_id="p1",
            figure_id="f1",
        )
        assert result == []
        assert len(fake_backend.calls) == 0

    @requires_cv2
    def test_range_chart_extracts_full_record(self, fake_backend):
        engine = _engine_with(fake_backend)
        result = engine.extract_geology(
            image=self._image(),
            caption="Range chart of late Permian radiolarians from South China.",
            figure_type="range_chart",
            paper_id="Feng_2007",
            figure_id="fig3",
        )
        assert len(result) == 1
        rec = result[0]
        assert rec["age"] == "Late Permian"
        assert rec["chronostratigraphy"] == "Changhsingian"
        assert rec["ma_top"] == 251.9
        assert rec["ma_base"] == 254.14
        assert rec["formation"] == "Talung Formation"
        assert rec["lithology"] == "chert"
        assert rec["locality"] == "Pingdingshan"
        assert rec["country"] == "China"
        assert rec["biozone"] == "N. optima Zone"
        assert rec["section_type"] == "range_chart"
        assert rec["confidence"] == 0.85

    @requires_cv2
    def test_map_extracts_country_and_coordinates(self, fake_backend):
        engine = _engine_with(fake_backend)
        result = engine.extract_geology(
            image=self._image(),
            caption="Geographic map of Argolis Peninsula.",
            figure_type="map",
            paper_id="Bandini_2006",
            figure_id="fig1",
        )
        assert len(result) == 1
        rec = result[0]
        assert rec["country"] == "Greece"
        assert rec["locality"] == "Karnezeika"
        assert rec["latitude"] == 37.5
        assert rec["longitude"] == 23.0
        assert rec["section_type"] == "location_map"

    @requires_cv2
    def test_litholog_extracts_lithology_member(self, fake_backend):
        engine = _engine_with(fake_backend)
        result = engine.extract_geology(
            image=self._image(),
            caption="Lithological column of Dalong Formation.",
            figure_type="litholog_column",
            paper_id="Feng_2007",
            figure_id="fig5",
        )
        assert len(result) == 1
        rec = result[0]
        assert rec["formation"] == "Dalong"
        assert rec["member"] == "Upper"
        assert rec["lithology"] == "siliceous mudstone"
        assert rec["section_type"] == "litholog_column"

    @requires_cv2
    def test_backend_called_exactly_once(self, fake_backend):
        engine = _engine_with(fake_backend)
        engine.extract_geology(
            image=self._image(),
            caption="",
            figure_type="range_chart",
            paper_id="p1",
            figure_id="f1",
        )
        assert len(fake_backend.calls) == 1
        assert fake_backend.calls[0].method == "infer_panel"

    @requires_cv2
    def test_map_prompt_mentions_country(self, fake_backend):
        engine = _engine_with(fake_backend)
        engine.extract_geology(
            image=self._image(),
            caption="",
            figure_type="map",
            paper_id="p1",
            figure_id="f1",
        )
        sys_prompt = fake_backend.calls[0].system_prompt
        assert "country" in sys_prompt.lower()
        assert "locality" in sys_prompt.lower()

    @requires_cv2
    def test_range_chart_prompt_mentions_ma_top(self, fake_backend):
        engine = _engine_with(fake_backend)
        engine.extract_geology(
            image=self._image(),
            caption="",
            figure_type="range_chart",
            paper_id="p1",
            figure_id="f1",
        )
        sys_prompt = fake_backend.calls[0].system_prompt
        assert "ma_top" in sys_prompt
        assert "biozone" in sys_prompt.lower()

    @requires_cv2
    def test_cost_summary_increments_after_extract(self, fake_backend):
        engine = _engine_with(fake_backend)
        before = fake_backend.cost_summary()["calls"]
        engine.extract_geology(
            image=self._image(),
            caption="",
            figure_type="range_chart",
            paper_id="p1",
            figure_id="f1",
        )
        after = fake_backend.cost_summary()["calls"]
        assert after == before + 1

    @requires_cv2
    def test_bad_json_shape_returns_empty_list(self):
        backend = FakeM3Backend(
            canned_responses=[
                {
                    "match": lambda s: "range_chart_geo" in s,
                    "raw_text": "this is not json at all",
                    "fallback_used": False,
                }
            ]
        )
        engine = _engine_with(backend)
        result = engine.extract_geology(
            image=self._image(),
            caption="",
            figure_type="range_chart",
            paper_id="p1",
            figure_id="f1",
        )
        assert result == []

    @requires_cv2
    def test_missing_required_field_returns_empty_list(self):
        backend = FakeM3Backend(
            canned_responses=[
                {
                    "match": lambda s: "range_chart_geo" in s,
                    "raw_text": '{"geo": "not-an-object"}',
                    "fallback_used": False,
                }
            ]
        )
        engine = _engine_with(backend)
        result = engine.extract_geology(
            image=self._image(),
            caption="",
            figure_type="range_chart",
            paper_id="p1",
            figure_id="f1",
        )
        assert result == []

    @requires_cv2
    def test_each_record_carries_paper_id_and_figure_id(self, fake_backend):
        engine = _engine_with(fake_backend)
        result = engine.extract_geology(
            image=self._image(),
            caption="",
            figure_type="range_chart",
            paper_id="Feng_2007",
            figure_id="fig3",
        )
        # paper_id / figure_id flow into evidence_text or section_title
        # so downstream audit can trace each link back to source.
        rec = result[0]
        assert rec is not None
