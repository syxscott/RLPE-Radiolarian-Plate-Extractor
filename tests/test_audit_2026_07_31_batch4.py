"""Regression tests for audit 2026-07-31 batch 4 (LLM/M3/OCR chain).

Covers:
  - LLM-first consumption of backend-parsed results (no strict
    re-parse of raw_text that discarded paid results with preambles)
  - confidence float() safety (LLM "high" strings)
  - period-separated discrete labels ("Figs 1-3. 5. 8. 10. 12:")
  - string "false" booleans
  - _safe_json_loads with multiple objects
  - m3_temperature / m3_thinking_budget config knobs reach the backend
  - FallbackRecommendedError wiring (not swallowed, backend switched)
  - OCR zh → ch_sim for EasyOCR
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class TestSafeBool:
    def test_string_false_is_false(self):
        from rlpe.m3_engine import _safe_bool

        assert _safe_bool("false") is False
        assert _safe_bool("False") is False
        assert _safe_bool("no") is False
        assert _safe_bool("0") is False
        assert _safe_bool(False) is False

    def test_true_spellings(self):
        from rlpe.m3_engine import _safe_bool

        assert _safe_bool("true") is True
        assert _safe_bool(True) is True
        assert _safe_bool("yes") is True
        assert _safe_bool("1") is True

    def test_default_on_garbage(self):
        from rlpe.m3_engine import _safe_bool

        assert _safe_bool("maybe", default=True) is True
        assert _safe_bool(None, default=True) is True


class TestSafeFloat:
    def test_non_numeric_confidence(self):
        from rlpe.m3_engine import _safe_float

        assert _safe_float("high") == 0.0
        assert _safe_float("0.8") == 0.8
        assert _safe_float(None) == 0.0
        assert _safe_float(0.5) == 0.5


class TestSafeJsonLoads:
    def test_multiple_objects(self):
        from rlpe.m3_engine import _safe_json_loads

        assert _safe_json_loads('{"a": 1} {"b": 2}') == {"a": 1}

    def test_preamble(self):
        from rlpe.m3_engine import _safe_json_loads

        assert _safe_json_loads('Here are the panels: {"label": "1"}') == {"label": "1"}

    def test_array_and_nested(self):
        from rlpe.m3_engine import _safe_json_loads

        assert _safe_json_loads('[{"a": 1}]') == [{"a": 1}]
        assert _safe_json_loads('{"a": {"b": 2}}') == {"a": {"b": 2}}


class TestPeriodSeparatedLabels:
    def test_discrete_labels_parse(self):
        from rlpe.m3_engine import _regex_parse_caption

        pairs = _regex_parse_caption("Figs 1-3. 5. 8. 10. 12: Archaespongoprunum sp.")
        assert len(pairs) == 1
        assert pairs[0].labels == ["1", "2", "3", "5", "8", "10", "12"]

    def test_normal_caption_untouched(self):
        from rlpe.m3_engine import _regex_parse_caption

        pairs = _regex_parse_caption("figs 1-2. Entactinia itsukichiensis")
        assert pairs[0].labels == ["1", "2"]


class TestLlmFirstContract:
    def test_backend_parsed_result_consumed_directly(self, tmp_path):
        """The backend returns the parsed dict (possibly with a
        preamble-tainted raw_text). The pipeline must consume the
        parsed result without re-parsing raw_text."""
        from unittest.mock import patch

        from rlpe.config import PipelineConfig

        with (
            patch("rlpe.pipeline.GrobidClient"),
            patch("rlpe.pipeline.OCRBackend"),
            patch("rlpe.pipeline.TaxonRecognizer"),
            patch("rlpe.pipeline.PanelSegmenter"),
        ):
            cfg = PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "work")
            cfg.extra["use_llm_first"] = True
            from rlpe.pipeline import RadiolarianPipeline

            pipe = RadiolarianPipeline(cfg)

            class FakeBackend:
                backend_name = "fake"

                def infer_panel(self, **kwargs):
                    # backend already parsed; raw_text has a preamble
                    # that would fail strict json.loads
                    return {
                        "panels": [
                            {"label": "1", "species": "Actinomma leptodermum", "confidence": 0.9}
                        ],
                        "raw_text": 'Here are the panels: [{"label": "1"}]',
                        "fallback_used": False,
                    }

            pipe.gemma_runtime = FakeBackend()
            from rlpe.types import CaptionRecord, FigureRegion

            cap = CaptionRecord(
                paper_id="t",
                figure_id="f1",
                caption="Fig. 1. 1. Actinomma leptodermum",
                entities=[],
                figure_number="1",
                page_index=1,
            )
            import numpy as np

            region = FigureRegion(
                page_index=1,
                bbox=(0, 0, 100, 100),
                crop_path="t.png",
                score=0.9,
                region_id="r1",
            )
            rows = pipe._llm_first_extract(
                paper_id="t",
                figure_id="f1",
                caption=cap,
                region_img=np.zeros((10, 10, 3), dtype=np.uint8),
                region=region,
                figure_index=1,
            )
            assert rows, "backend-parsed result must be consumed"
            assert rows[0]["species"] == "Actinomma leptodermum"

    def test_single_panel_dict_consumed(self, tmp_path):
        from unittest.mock import patch

        from rlpe.config import PipelineConfig

        with (
            patch("rlpe.pipeline.GrobidClient"),
            patch("rlpe.pipeline.OCRBackend"),
            patch("rlpe.pipeline.TaxonRecognizer"),
            patch("rlpe.pipeline.PanelSegmenter"),
        ):
            cfg = PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "work")
            cfg.extra["use_llm_first"] = True
            from rlpe.pipeline import RadiolarianPipeline

            pipe = RadiolarianPipeline(cfg)

            class FakeBackend:
                backend_name = "fake"

                def infer_panel(self, **kwargs):
                    # model ignored the array instruction → single dict
                    return {
                        "label": "1",
                        "species": "Unuma echinatus",
                        "confidence": 0.95,
                        "raw_text": "",
                        "fallback_used": False,
                    }

            pipe.gemma_runtime = FakeBackend()
            from rlpe.types import CaptionRecord, FigureRegion

            cap = CaptionRecord(
                paper_id="t",
                figure_id="f1",
                caption="Fig. 1. 1. Unuma echinatus",
                entities=[],
                figure_number="1",
                page_index=1,
            )
            import numpy as np

            region = FigureRegion(
                page_index=1,
                bbox=(0, 0, 100, 100),
                crop_path="t.png",
                score=0.9,
                region_id="r1",
            )
            rows = pipe._llm_first_extract(
                paper_id="t",
                figure_id="f1",
                caption=cap,
                region_img=np.zeros((10, 10, 3), dtype=np.uint8),
                region=region,
                figure_index=1,
            )
            assert rows and rows[0]["species"] == "Unuma echinatus"


class TestConfigKnobs:
    def test_temperature_and_thinking_reach_backend(self):
        from rlpe.m3_engine import M3Engine

        class FakeBackend:
            temperature = 0.1
            thinking_budget_tokens = 1024
            max_output_tokens = 2048

        eng = M3Engine(
            FakeBackend(),
            {"m3_temperature": 0.7, "m3_thinking_budget": 512, "m3_max_output_tokens": 8192},
        )
        assert eng.backend.temperature == 0.7
        assert eng.backend.thinking_budget_tokens == 512
        assert eng.backend.max_output_tokens == 8192

    def test_defaults_preserved(self):
        from rlpe.m3_engine import M3Engine

        class FakeBackend:
            temperature = 0.1
            thinking_budget_tokens = 1024

        eng = M3Engine(FakeBackend(), {})
        assert eng.backend.temperature == 0.1
        assert eng.backend.thinking_budget_tokens == 1024


class TestFallbackWiring:
    def test_fallback_recommended_error_not_swallowed_in_text(self):
        """_infer_text must re-raise FallbackRecommendedError (it used
        to be swallowed by except Exception)."""
        from rlpe.llm_backends import FallbackRecommendedError
        from rlpe.m3_engine import M3Engine

        class FakeBackend:
            enable_thinking = True

            def infer_text(self, **kwargs):
                raise FallbackRecommendedError("4xx", "ollama")

        eng = M3Engine(FakeBackend(), {})
        with pytest.raises(FallbackRecommendedError):
            eng._infer_text("sys", "user")

    def test_fallback_retry_wrapper_switches_backend(self, tmp_path):
        from unittest.mock import patch

        from rlpe.config import PipelineConfig
        from rlpe.llm_backends import FallbackRecommendedError

        with (
            patch("rlpe.pipeline.GrobidClient"),
            patch("rlpe.pipeline.OCRBackend"),
            patch("rlpe.pipeline.TaxonRecognizer"),
            patch("rlpe.pipeline.PanelSegmenter"),
        ):
            cfg = PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "work")
            from rlpe.pipeline import RadiolarianPipeline

            pipe = RadiolarianPipeline(cfg)
            calls = {"n": 0}

            def boom():
                calls["n"] += 1
                if calls["n"] == 1:
                    raise FallbackRecommendedError("4xx", "ollama")
                return "ok"

            class FakeRuntime:
                backend = object()
                backend_name = "fake"

            class FakeM3:
                backend = object()

            pipe.gemma_runtime = FakeRuntime()
            pipe.m3_engine = FakeM3()
            pipe._build_local_gemma_fallback = lambda: FakeRuntime()

            result = pipe._m3_call_with_fallback(boom)
            assert result == "ok"
            assert calls["n"] == 2
            assert pipe.m3_engine.backend is pipe.gemma_runtime.backend


class TestOcrLangMapping:
    def test_zh_maps_to_ch_sim_for_easyocr(self):
        import rlpe.ocr as ocr_mod

        # verify the mapping exists in the EasyOCR branch source
        src = Path(ocr_mod.__file__).read_text(encoding="utf-8")
        assert 'ch_sim" if l == "zh"' in src, "zh → ch_sim mapping missing"
