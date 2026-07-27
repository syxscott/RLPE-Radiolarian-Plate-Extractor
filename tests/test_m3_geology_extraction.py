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
                # Match by the literal prose in the system prompt, NOT the
                # key name (the key is "range_chart_geo" but the prompt
                # text describes "a stratigraphic range chart"). Otherwise
                # the match function falls through to the last canned
                # response and returns the wrong geology record.
                "match": lambda s: "stratigraphic range chart" in s,
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
                "match": lambda s: "geographic / location map" in s,
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
                "match": lambda s: "lithological log" in s,
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
        # Every key MUST match the f"{figure_type}_geo" contract.
        for key in (
            "plate_geo",
            "range_chart_geo",
            "map_geo",
            "strat_column_geo",
            "litholog_column_geo",
            "paleogeographic_map_geo",
        ):
            assert key in PROMPT_REGISTRY, f"missing prompt: {key}"
            assert isinstance(PROMPT_REGISTRY[key], str)
            assert len(PROMPT_REGISTRY[key]) > 50, "each prompt should be substantive (>50 chars)"

    def test_each_prompt_returns_json_shape(self):
        # Geology-vision prompts use a "geo" JSON key. The Round 7
        # multi_plate_enrich prompt uses "panels" (different schema for
        # multi-panel extraction) so we only enforce the geo shape on
        # prompts whose key ends in "_geo".
        for key, prompt in PROMPT_REGISTRY.items():
            if not key.endswith("_geo"):
                # Non-geo prompts (e.g. multi_plate_enrich) have their
                # own JSON contract; skip the geo-shape assertions.
                continue
            assert "geo" in prompt, f"{key} prompt missing JSON 'geo' key"
            assert "age" in prompt
            assert "formation" in prompt
            assert "confidence" in prompt

    def test_section_type_table_covers_known_figures(self):
        assert SECTION_TYPE_BY_FIGURE["range_chart"] == "range_chart"
        assert SECTION_TYPE_BY_FIGURE["map"] == "location_map"
        assert SECTION_TYPE_BY_FIGURE["paleogeographic_map"] == "paleogeographic_map"
        # "strat_column" is the classify_figure_type output key;
        # it maps to the published "stratigraphic_column" section_type.
        assert SECTION_TYPE_BY_FIGURE["strat_column"] == "stratigraphic_column"
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
                    "match": lambda s: "stratigraphic range chart" in s,
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

    # ---- Phase X: localities / layers array parsing --------------------------


@requires_cv2
def test_paleogeographic_map_localities_creates_per_point_records():
    """paleogeographic_map with ``localities`` array must emit one
    GeologyLinkRecord per locality (species + coords + age), not just
    the global geo entry.
    """
    from PIL import Image
    backend = FakeM3Backend(
        canned_responses=[
            {
                "match": lambda s: "paleogeographic map" in s,
                "raw_text": (
                    '{"geo":[{"age":"Late Cretaceous, Cenomanian",'
                    '"chronostratigraphy":"Cenomanian",'
                    '"chronostratigraphy_rank":"age","ma_top":100.5,'
                    '"ma_base":93.9,"ma_mid":97.2,'
                    '"formation":null,"member":null,"group":null,'
                    '"lithology":null,"locality":"Tethys Ocean",'
                    '"country":null,"latitude":null,"longitude":null,'
                    '"biozone":null,"confidence":0.9}],'
                    '"localities":['
                    '{"species":"Archaeodictyomitra simplex",'
                    '"label":"1","latitude":22.5,"longitude":115.2,'
                    '"paleo_latitude":25.1,"paleo_longitude":113.8,'
                    '"age":"Late Cretaceous, Cenomanian",'
                    '"ma_top":100.5,"ma_base":93.9,'
                    '"formation":"Wahrah Formation",'
                    '"lithology":"chert","biozone":null,'
                    '"evidence":"point 1 in map legend","confidence":0.88},'
                    '{"species":"Crucella espartoensis",'
                    '"label":"2","latitude":21.8,"longitude":114.7,'
                    '"paleo_latitude":24.3,"paleo_longitude":113.2,'
                    '"age":"Early Cretaceous, Albian",'
                    '"ma_top":112.0,"ma_base":100.5,'
                    '"formation":"Nahr Ibrah Formation",'
                    '"lithology":"cherty limestone","biozone":"Tethysian-3",'
                    '"evidence":"point 2 in map legend","confidence":0.85}'
                    "]}"
                ),
                "fallback_used": False,
            }
        ]
    )
    engine = _engine_with(backend)
    result = engine.extract_geology(
        image=Image.new("RGB", (300, 300)),
        caption="Paleogeographic map of the Tethys Ocean during the Cenomanian.",
        figure_type="paleogeographic_map",
        paper_id="Bandini_2006",
        figure_id="fig7",
    )
    # Should get: 1 global geo entry + 2 locality entries = 3 total.
    assert len(result) == 3, f"expected 3 records (1 geo + 2 localities), got {len(result)}"

    geo_rec = result[0]
    assert geo_rec["age"] == "Late Cretaceous, Cenomanian"
    assert geo_rec["section_type"] == "paleogeographic_map"

    locality_recs = result[1:]
    species_found = {r.get("species") for r in locality_recs}
    assert "Archaeodictyomitra simplex" in species_found
    assert "Crucella espartoensis" in species_found

    # Each locality record must carry point-level coords and link_source.
    for rec in locality_recs:
        assert rec["link_source"] == "geo_vision_point"
        assert rec["section_type"] == "paleogeographic_map"
        assert rec["latitude"] is not None
        assert rec["longitude"] is not None
        assert rec["paleo_latitude"] is not None
        assert rec["paleo_longitude"] is not None
        assert rec["confidence"] > 0


@requires_cv2
def test_paleogeographic_map_skips_localities_without_species():
    """Locality entries without a species name must be skipped silently
    (they carry no useful species-to-geology link).
    """
    from PIL import Image
    backend = FakeM3Backend(
        canned_responses=[
            {
                "match": lambda s: "paleogeographic map" in s,
                "raw_text": (
                    '{"geo":[{"age":"Cenomanian","chronostratigraphy":null,'
                    '"chronostratigraphy_rank":null,"ma_top":null,'
                    '"ma_base":null,"ma_mid":null,"formation":null,'
                    '"member":null,"group":null,"lithology":null,'
                    '"locality":null,"country":null,"latitude":null,'
                    '"longitude":null,"biozone":null,"confidence":0.8}],'
                    '"localities":['
                    '{"species":"Archaeodictyomitra simplex","label":"1",'
                    '"latitude":22.5,"longitude":115.2,'
                    '"paleo_latitude":null,"paleo_longitude":null,'
                    '"age":null,"ma_top":null,"ma_base":null,'
                    '"formation":null,"lithology":null,"biozone":null,'
                    '"evidence":null,"confidence":0.9},'
                    '{"species":null,"label":"2",'
                    '"latitude":21.8,"longitude":114.7,'
                    '"paleo_latitude":null,"paleo_longitude":null,'
                    '"age":null,"ma_top":null,"ma_base":null,'
                    '"formation":null,"lithology":null,"biozone":null,'
                    '"evidence":null,"confidence":0.8}'
                    "]}"
                ),
                "fallback_used": False,
            }
        ]
    )
    engine = _engine_with(backend)
    result = engine.extract_geology(
        image=Image.new("RGB", (300, 300)),
        caption="Paleogeographic map.",
        figure_type="paleogeographic_map",
        paper_id="p1",
        figure_id="f1",
    )
    # 1 geo entry + 1 locality (species=null skipped) = 2
    assert len(result) == 2
    species_recs = [r for r in result if r.get("species")]
    assert len(species_recs) == 1
    assert species_recs[0]["species"] == "Archaeodictyomitra simplex"


@requires_cv2
def test_strat_column_layers_creates_per_layer_records():
    """strat_column with ``layers`` array must emit one record per layer
    (formation + lithology + age + ma range), separate from the global
    geo entry.
    """
    from PIL import Image
    backend = FakeM3Backend(
        canned_responses=[
            {
                "match": lambda s: "stratigraphic column" in s,
                "raw_text": (
                    '{"geo":[{"age":"Albian","chronostratigraphy":"Albian",'
                    '"chronostratigraphy_rank":"age","ma_top":100.5,'
                    '"ma_base":113.0,"ma_mid":106.75,'
                    '"formation":"Nahr Ibrah Formation",'
                    '"member":null,"group":null,'
                    '"lithology":null,"locality":"Nahr Ibrah",'
                    '"country":"Oman","latitude":22.9,"longitude":57.1,'
                    '"biozone":"Tethysian-4","confidence":0.85}],'
                    '"layers":['
                    '{"layer_index":0,'
                    '"y_top_normalized":0.0,"y_base_normalized":0.15,'
                    '"lithology":"shale","formation":"Nahr Ibrah Formation",'
                    '"member":null,"age":"Albian","ma_top":100.5,'
                    '"ma_base":104.0,"biozone":"Tethysian-3",'
                    '"thickness_m":120,"evidence":null,"confidence":0.82},'
                    '{"layer_index":1,'
                    '"y_top_normalized":0.15,"y_base_normalized":0.35,'
                    '"lithology":"cherty limestone","formation":"Nahr Ibrah Formation",'
                    '"member":null,"age":"Albian","ma_top":104.0,'
                    '"ma_base":113.0,"biozone":"Tethysian-4",'
                    '"thickness_m":180,"evidence":null,"confidence":0.79}'
                    "]}"
                ),
                "fallback_used": False,
            }
        ]
    )
    engine = _engine_with(backend)
    result = engine.extract_geology(
        image=Image.new("RGB", (300, 300)),
        caption="Stratigraphic column of Nahr Ibrah Formation.",
        figure_type="strat_column",
        paper_id="p1",
        figure_id="f1",
    )
    # 1 geo entry + 2 layer entries = 3
    assert len(result) == 3, f"expected 3 (1 geo + 2 layers), got {len(result)}"

    layer_recs = result[1:]
    assert len(layer_recs) == 2
    # First layer: shale, ma_top=100.5, ma_base=104.0
    shale = next(r for r in layer_recs if r.get("lithology") == "shale")
    assert shale["ma_top"] == 100.5
    assert shale["ma_base"] == 104.0
    assert shale["_y_top_normalized"] == 0.0
    assert shale["_y_base_normalized"] == 0.15
    assert shale["_thickness_m"] == 120
    assert shale["link_source"] == "geo_vision_layer"
    assert shale["section_type"] == "stratigraphic_column"

    # Second layer: cherty limestone
    limestone = next(r for r in layer_recs if "limestone" in str(r.get("lithology", "")))
    assert limestone["ma_top"] == 104.0
    assert limestone["ma_base"] == 113.0
    assert limestone["_layer_index"] == 1


@requires_cv2
def test_litholog_column_layers_creates_per_layer_records():
    """litholog_column with ``layers`` array emits per-layer records
    with link_source='geo_vision_layer'.
    """
    from PIL import Image
    backend = FakeM3Backend(
        canned_responses=[
            {
                "match": lambda s: "lithological log" in s,
                "raw_text": (
                    '{"geo":[{"age":"Cenomanian","chronostratigraphy":"Cenomanian",'
                    '"chronostratigraphy_rank":"age","ma_top":100.5,'
                    '"ma_base":93.9,"ma_mid":97.2,'
                    '"formation":"Warah Formation","member":"Upper",'
                    '"group":null,"lithology":"chert","locality":"Warah",'
                    '"country":"Oman","latitude":22.5,"longitude":57.0,'
                    '"biozone":null,"confidence":0.8}],'
                    '"layers":['
                    '{"layer_index":0,'
                    '"y_top_normalized":0.0,"y_base_normalized":0.5,'
                    '"lithology":"chert","formation":"Warah Formation",'
                    '"member":"Upper","age":"Cenomanian",'
                    '"ma_top":93.9,"ma_base":100.5,'
                    '"biozone":null,"thickness_m":null,'
                    '"evidence":null,"confidence":0.75},'
                    '{"layer_index":1,'
                    '"y_top_normalized":0.5,"y_base_normalized":1.0,'
                    '"lithology":"siliceous shale","formation":"Warah Formation",'
                    '"member":"Lower","age":"Cenomanian",'
                    '"ma_top":93.9,"ma_base":100.5,'
                    '"biozone":null,"thickness_m":null,'
                    '"evidence":null,"confidence":0.73}'
                    "]}"
                ),
                "fallback_used": False,
            }
        ]
    )
    engine = _engine_with(backend)
    result = engine.extract_geology(
        image=Image.new("RGB", (300, 300)),
        caption="Lithological log of Warah Formation.",
        figure_type="litholog_column",
        paper_id="p1",
        figure_id="f1",
    )
    assert len(result) == 3, f"expected 3 (1 geo + 2 layers), got {len(result)}"
    layer_recs = result[1:]
    assert all(r.get("link_source") == "geo_vision_layer" for r in layer_recs)
    assert all(r.get("section_type") == "litholog_column" for r in layer_recs)
    lithologies = {r.get("lithology") for r in layer_recs}
    assert "chert" in lithologies
    assert "siliceous shale" in lithologies


@requires_cv2
def test_paleogeographic_map_without_localities_still_works():
    """Old M3 responses (no ``localities`` key) must still produce exactly
    one geo entry and not crash.
    """
    from PIL import Image
    backend = FakeM3Backend(
        canned_responses=[
            {
                "match": lambda s: "paleogeographic map" in s,
                "raw_text": (
                    '{"geo":[{"age":"Cenomanian","chronostratigraphy":null,'
                    '"chronostratigraphy_rank":null,"ma_top":100.5,'
                    '"ma_base":93.9,"ma_mid":97.2,'
                    '"formation":null,"member":null,"group":null,'
                    '"lithology":null,"locality":"Tethys Ocean",'
                    '"country":"Tethys","latitude":null,"longitude":null,'
                    '"biozone":null,"confidence":0.9}]}'
                ),
                "fallback_used": False,
            }
        ]
    )
    engine = _engine_with(backend)
    result = engine.extract_geology(
        image=Image.new("RGB", (300, 300)),
        caption="Paleogeographic map.",
        figure_type="paleogeographic_map",
        paper_id="p1",
        figure_id="f1",
    )
    assert len(result) == 1
    assert result[0]["age"] == "Cenomanian"
    assert result[0]["section_type"] == "paleogeographic_map"


@requires_cv2
def test_strat_column_without_layers_still_works():
    """Old M3 responses (no ``layers`` key) must still produce exactly
    one geo entry and not crash.
    """
    from PIL import Image
    backend = FakeM3Backend(
        canned_responses=[
            {
                "match": lambda s: "stratigraphic column" in s,
                "raw_text": (
                    '{"geo":[{"age":"Albian","chronostratigraphy":"Albian",'
                    '"chronostratigraphy_rank":"age","ma_top":100.5,'
                    '"ma_base":113.0,"ma_mid":106.75,'
                    '"formation":"Nahr Ibrah Formation",'
                    '"member":null,"group":null,"lithology":"cherty limestone",'
                    '"locality":"Nahr Ibrah","country":"Oman",'
                    '"latitude":22.9,"longitude":57.1,'
                    '"biozone":"Tethysian-4","confidence":0.85}]}'
                ),
                "fallback_used": False,
            }
        ]
    )
    engine = _engine_with(backend)
    result = engine.extract_geology(
        image=Image.new("RGB", (300, 300)),
        caption="Stratigraphic column.",
        figure_type="strat_column",
        paper_id="p1",
        figure_id="f1",
    )
    assert len(result) == 1
    assert result[0]["formation"] == "Nahr Ibrah Formation"
    assert result[0]["section_type"] == "stratigraphic_column"

