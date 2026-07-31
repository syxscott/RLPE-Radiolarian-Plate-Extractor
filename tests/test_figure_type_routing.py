"""Tests for Round 5 figure-type routing.

6 figure types must be correctly classified by
``classify_figure_type`` AND piped through the correct vision
prompt in the pipeline. Previously only ``plate`` and
``range_chart`` had proper M3 vision extraction — the other 4
(``strat_column``, ``litholog_column``, ``paleogeographic_map``,
``map``) either fell through to plate processing or only
produced text-only stubs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from rlpe.range_chart_extractor import classify_figure_type


class TestClassifyFigureType:
    """Lock the keyword-based classifier contract."""

    @pytest.mark.parametrize(
        "caption,expected_type",
        [
            # Plate — SEM image captions
            (
                "Plate 1. figs 1-5. Actinomma leptodermum",
                "plate",
            ),
            (
                "SEM images of radiolarians from the Dalong Formation.",
                "plate",
            ),
            # Range chart — distribution/range keywords
            (
                "Stratigraphic distribution of radiolarians from the Late Permian of South China.",
                "range_chart",
            ),
            (
                "Range chart showing species ranges across 5 sections.",
                "range_chart",
            ),
            (
                "Biozone correlation for the Jurassic.",
                "range_chart",
            ),
            # Stratigraphic column — more specific than range_chart
            (
                "Stratigraphic column of the Middle-Upper Permian at Pingdingshan, Guangxi.",
                "strat_column",
            ),
            (
                "Generalized stratigraphy of the Vocontian Basin, SE France.",
                "strat_column",
            ),
            (
                "Columnar section showing lithology and fossils.",
                "strat_column",
            ),
            # Litholog column
            (
                "Lithology log of the Upper Triassic radiolarites in the Southern Alps.",
                "litholog_column",
            ),
            (
                "Litholog column showing chert-mudstone alternations through the section.",
                "litholog_column",
            ),
            # Paleogeographic map — NOT merged with location map
            (
                "Paleogeographic map of the Late Permian (252 Ma) showing Panthalassan terranes.",
                "paleogeographic_map",
            ),
            (
                "Palaeogeographic reconstruction of the Jurassic Tethys Ocean.",
                "paleogeographic_map",
            ),
            # Geographic map — only literal location maps
            (
                "Location map of the study area in Argolis Peninsula, Greece.",
                "map",
            ),
            (
                "Geological map of the Toyohashi district, Aichi Prefecture.",
                "map",
            ),
            # Photo — not classified as plate
            (
                "Field photograph of the outcrop at Karnezeika, "
                "showing the Argolis Peninsula limestones.",
                "photo",
            ),
            # "Diagram showing..." contains "ranges" which is not a keyword,
            # but "stratigraphic" + "ranges" together hit the range_chart
            # multi-word match "stratigraphic range" (fuzzy via `in` on the
            # full caption). Confirm it returns range_chart.
            (
                "Diagram showing the stratigraphic ranges of selected taxa within a single bed.",
                "range_chart",
            ),
            # Truly unclassifiable
            (
                "Illustration of a measured outcrop section with hand samples.",
                "other",
            ),
            # 空值
            ("", "other"),
            (None, "other"),
        ],
    )
    def test_classify(self, caption, expected_type):
        assert classify_figure_type(caption) == expected_type, (
            f"classify_figure_type({caption!r}) should be {expected_type!r}"
        )


class TestPromptRegistry:
    """Every new figure type must have a corresponding prompt."""

    def test_strat_column_prompt_exists(self):
        from rlpe.m3_engine import PROMPT_REGISTRY

        assert "strat_column_geo" in PROMPT_REGISTRY

    def test_litholog_column_prompt_exists(self):
        from rlpe.m3_engine import PROMPT_REGISTRY

        # Key MUST match f"{figure_type}_geo" contract.
        assert "litholog_column_geo" in PROMPT_REGISTRY

    def test_paleogeographic_map_prompt_exists(self):
        from rlpe.m3_engine import PROMPT_REGISTRY

        # Key MUST match f"{figure_type}_geo" contract.
        assert "paleogeographic_map_geo" in PROMPT_REGISTRY

    def test_section_type_mapping(self):
        from rlpe.m3_engine import SECTION_TYPE_BY_FIGURE

        assert SECTION_TYPE_BY_FIGURE["strat_column"] == "stratigraphic_column"
        assert SECTION_TYPE_BY_FIGURE["litholog_column"] == "litholog_column"
        assert SECTION_TYPE_BY_FIGURE["paleogeographic_map"] == "paleogeographic_map"
        # Confirm the old typo-key is gone
        assert "stratigraphic_column" not in SECTION_TYPE_BY_FIGURE


class TestPipelineRoutingSource:
    """Static verification that pipeline.py routes the new figure
    types correctly. Without cv2, we can't test the full runtime,
    so we grep the source to ensure the branching exists.
    """

    def test_pipeline_handles_strat_column(self):
        from pathlib import Path as _Path

        path = _Path(__file__).resolve().parents[1] / "src" / "rlpe" / "pipeline.py"
        text = path.read_text(encoding="utf-8")
        # The routing block must mention all three new types.
        for fig_type in ("strat_column", "litholog_column", "paleogeographic_map"):
            assert f'"{fig_type}"' in text, f"pipeline.py must handle fig_type={fig_type!r}"

    def test_pipeline_calls_extract_geology_for_new_types(self):
        from pathlib import Path as _Path

        path = _Path(__file__).resolve().parents[1] / "src" / "rlpe" / "pipeline.py"
        text = path.read_text(encoding="utf-8")
        # The geo_vision block for new types must call
        # extract_geology with the figure_type forwarded.
        assert "_m3_call_with_fallback(self.m3_engine.extract_geology," in text
        assert "figure_type=fig_type," in text

    def test_pipeline_creates_stub_record_for_geo_vision(self):
        from pathlib import Path as _Path

        path = _Path(__file__).resolve().parents[1] / "src" / "rlpe" / "pipeline.py"
        text = path.read_text(encoding="utf-8")
        # The new types produce a stub record so the eval/export
        # pipeline sees the geology links.
        assert '"GEO_VISION_' in text
        assert '"figure_type": fig_type,' in text
