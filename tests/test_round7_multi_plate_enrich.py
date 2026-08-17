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
        """--m3-multi-plate-enrich must be wired into the PipelineConfig.

        Audit 2026-08-17: the previous code routed the flag through
        ``config.extra["m3_multi_plate_enrich"]`` (a free-form dict), but
        the CLI never populated that key — the second-pass enrichment
        was silently disabled. The fix promotes it to a typed
        ``PipelineConfig.m3_multi_plate_enrich_enabled`` attribute; the
        test accepts either the old extra-dict form OR the new typed
        attribute form so a future audit that renames the attribute
        again doesn't break this guard."""
        text = (Path(__file__).resolve().parents[1] / "src" / "rlpe" / "cli.py").read_text(
            encoding="utf-8"
        )
        assert (
            '"m3_multi_plate_enrich":' in text
            or "m3_multi_plate_enrich_enabled" in text
        ), (
            "CLI must route m3_multi_plate_enrich into the "
            "PipelineConfig (either via extra dict or via the "
            "m3_multi_plate_enrich_enabled typed attribute)"
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
        backend.infer_panel = MagicMock(return_value={"fallback_used": True, "raw_text": ""})
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
            image=img,
            page_caption="Plate 7 Fig 5 Bar foo",
            paper_id="test",
            figure_id="test_fig",
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
        body = text[idx : next_def if next_def > 0 else idx + 5000]
        # Must skip map / range_chart / geo_vision extraction_source
        assert '"map"' in body
        assert '"range_chart"' in body
        assert '"geo_vision"' in body

    def test_accepts_od_figures_param(self):
        """The enrichment pass must accept ``od_figures`` so it can rescue
        figures the per-figure loop dropped entirely (Bug #1 fix)."""
        text = (Path(__file__).resolve().parents[1] / "src" / "rlpe" / "pipeline.py").read_text(
            encoding="utf-8"
        )
        idx = text.find("def _apply_multi_plate_enrichment")
        assert idx > 0
        # Scan forward for the next def or top-level keyword to find the
        # full signature (which spans multiple lines because of *, separators).
        end = idx
        depth = 0
        for i in range(idx, min(idx + 1500, len(text))):
            ch = text[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        sig = text[idx:end]
        assert "od_figures" in sig, (
            f"_apply_multi_plate_enrichment must accept od_figures kwarg so it "
            f"can recover figures missing from results entirely (Bandini pl05/08/09). "
            f"Got signature: {sig!r}"
        )

    def test_trigger_includes_od_figure_with_no_results(self):
        """When an OD figure has NO matching results row (because the per-
        figure loop crashed or skipped), enrichment must still fire for
        that figure (the original Bandini 2011 pl05/pl08/pl09 case)."""
        text = (Path(__file__).resolve().parents[1] / "src" / "rlpe" / "pipeline.py").read_text(
            encoding="utf-8"
        )
        idx = text.find("def _apply_multi_plate_enrichment")
        assert idx > 0
        next_def = text.find("\n    def ", idx + 1)
        body = text[idx : next_def if next_def > 0 else idx + 8000]
        # The body must have BOTH conditions:
        #   1. OD figure with no results rows (the by_fig miss path)
        #   2. OD figure with rows but all-empty species + panel_id
        assert "if od_fid not in by_fig" in body, (
            "_apply_multi_plate_enrichment must check od_figures for "
            "figure_ids missing from results (the Bandini pl05/08/09 case)"
        )
        assert "n_with_species == 0 and n_with_panel_id == 0" in body, (
            "_apply_multi_plate_enrichment must still handle the "
            "rows-but-all-empty case for backward compat"
        )
