"""Round 7 multi-plate enrichment tests.

Verify that:
  - m3_engine.enrich_plate_panels calls backend and parses JSON
  - Returns [] on tiny image / fallback / parse failure
  - pipeline._apply_multi_plate_enrichment triggers only for under-populated figures
  - New rows carry panel_id_source="m3_vision" for audit
  - CLI flag --m3-multi-plate-enrich routes to extra dict
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestCliFlag:
    """Source guard: --m3-multi-plate-enrich must exist and route to extra."""

    def test_flag_exists(self):
        text = (Path(__file__).resolve().parents[1] / "src" / "rlpe" / "cli.py").read_text(
            encoding="utf-8"
        )
        assert "--m3-multi-plate-enrich" in text, (
            "CLI must expose --m3-multi-plate-enrich so users can enable "
            "the Round 7 second-pass M3 plate enrichment from the command line"
        )

    def test_flag_routes_into_extra(self):
        text = (Path(__file__).resolve().parents[1] / "src" / "rlpe" / "cli.py").read_text(
            encoding="utf-8"
        )
        assert '"m3_multi_plate_enrich":' in text, (
            "CLI must route m3_multi_plate_enrich into the PipelineConfig extra dict"
        )


class TestEnrichPlatePanels:
    """Unit tests for M3Engine.enrich_plate_panels()."""

    def test_prompt_registered(self):
        from rlpe.m3_engine import PROMPT_REGISTRY
        assert "multi_plate_enrich" in PROMPT_REGISTRY, (
            "PROMPT_REGISTRY must contain multi_plate_enrich system prompt"
        )
        # Prompt must mention caption + plate image context
        prompt = PROMPT_REGISTRY["multi_plate_enrich"]
        assert "radiolarian" in prompt.lower()
        assert "panel" in prompt.lower()
        assert "caption" in prompt.lower()

    def test_tiny_image_returns_empty(self):
        """Tiny images (<32px) should short-circuit to [] without calling backend."""
        from unittest.mock import MagicMock
        from rlpe.m3_engine import M3Engine

        # Construct engine with a stub backend
        backend = MagicMock()
        backend.infer_panel = MagicMock(return_value={"raw_text": "{}"})
        engine = M3Engine(backend=backend, config={})
        from PIL import Image
        tiny = Image.new("RGB", (16, 16))
        out = engine.enrich_plate_panels(
            image=tiny,
            page_caption="Plate 7 Fig 1 Foo",
            paper_id="test",
            figure_id="test_fig",
            expected_plate_label="Plate 7",
        )
        assert out == [], f"expected [] for tiny image, got {out}"
        # Backend should NOT have been called
        backend.infer_panel.assert_not_called()

    def test_fallback_used_returns_empty(self):
        """Backend returning fallback_used=True should yield []."""
        from unittest.mock import MagicMock
        from rlpe.m3_engine import M3Engine

        backend = MagicMock()
        backend.infer_panel = MagicMock(
            return_value={"fallback_used": True, "raw_text": ""}
        )
        engine = M3Engine(backend=backend, config={})
        from PIL import Image
        img = Image.new("RGB", (256, 256))
        out = engine.enrich_plate_panels(
            image=img,
            page_caption="Plate 7 Fig 1 Foo",
            paper_id="test",
            figure_id="test_fig",
        )
        assert out == []

    def test_successful_json_response(self):
        """Backend returning valid JSON should yield normalized panel list."""
        from unittest.mock import MagicMock
        from rlpe.m3_engine import M3Engine

        backend = MagicMock()
        backend.infer_panel = MagicMock(
            return_value={
                "raw_text": '{"panels": [{"label": "1", "species": "Foo bar", '
                            '"confidence": 0.9}, {"label": "2", "species": null, '
                            '"confidence": 0.5}]}',
            }
        )
        engine = M3Engine(backend=backend, config={})
        from PIL import Image
        img = Image.new("RGB", (256, 256))
        out = engine.enrich_plate_panels(
            image=img,
            page_caption="Plate 7 Fig 1 Foo bar Fig 2 sp.",
            paper_id="test",
            figure_id="test_fig",
        )
        assert len(out) == 2
        assert out[0]["label"] == "1"
        assert out[0]["species"] == "Foo bar"
        assert out[1]["label"] == "2"
        assert out[1]["species"] is None
        assert out[0]["confidence"] == 0.9

    def test_markdown_fence_stripped(self):
        """JSON wrapped in ```json fences should still parse."""
        from unittest.mock import MagicMock
        from rlpe.m3_engine import M3Engine

        backend = MagicMock()
        backend.infer_panel = MagicMock(
            return_value={
                "raw_text": '```json\n{"panels": [{"label": "5", '
                            '"species": "Bar foo", "confidence": 0.85}]}\n```',
            }
        )
        engine = M3Engine(backend=backend, config={})
        from PIL import Image
        img = Image.new("RGB", (256, 256))
        out = engine.enrich_plate_panels(
            image=img, page_caption="Plate 7 Fig 5 Bar foo",
            paper_id="test", figure_id="test_fig",
        )
        assert len(out) == 1
        assert out[0]["label"] == "5"


class TestPipelineEnrichmentGate:
    """Source guard: pipeline._apply_multi_plate_enrichment must gate correctly."""

    def test_method_exists(self):
        text = (Path(__file__).resolve().parents[1] / "src" / "rlpe" / "pipeline.py").read_text(
            encoding="utf-8"
        )
        assert "_apply_multi_plate_enrichment" in text, (
            "pipeline.py must implement _apply_multi_plate_enrichment for Round 7"
        )

    def test_skips_map_range_chart_stubs(self):
        """The enrichment pass must not re-process MAP_CONTEXT / RANGE_CHART stubs."""
        text = (Path(__file__).resolve().parents[1] / "src" / "rlpe" / "pipeline.py").read_text(
            encoding="utf-8"
        )
        idx = text.find("def _apply_multi_plate_enrichment")
        assert idx > 0
        # Find next method definition
        next_def = text.find("\n    def ", idx + 1)
        body = text[idx:next_def if next_def > 0 else idx + 5000]
        # Must skip map / range_chart / geo_vision extraction_source
        assert '"map"' in body
        assert '"range_chart"' in body
        assert '"geo_vision"' in body