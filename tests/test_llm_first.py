"""Tests for the LLM-first extraction path in pipeline.py."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# The pipeline module imports cv2 at module scope. Skip the test module
# on minimal environments so the suite can still collect cleanly.
pytest.importorskip("cv2")


class TestLLMFirstExtract:
    """Tests for RadiolarianPipeline._llm_first_extract."""

    @pytest.fixture(autouse=True)
    def _patch_pipeline(self, tmp_path):
        """Create a minimal RadiolarianPipeline with mocked dependencies."""
        from rlpe.config import PipelineConfig

        with (
            patch("rlpe.pipeline.GrobidClient"),
            patch("rlpe.pipeline.OCRBackend"),
            patch("rlpe.pipeline.TaxonRecognizer"),
            patch("rlpe.pipeline.PanelSegmenter"),
        ):
            cfg = PipelineConfig(
                pdf_dir=tmp_path,
                work_dir=tmp_path / "work",
            )
            cfg.extra["use_llm_first"] = True
            from rlpe.pipeline import RadiolarianPipeline

            self.pipe = RadiolarianPipeline(cfg)

    def _make_caption(self, text="Fig. 1. 1-4. Actinomma leptodermum"):
        from rlpe.types import CaptionRecord

        return CaptionRecord(
            paper_id="test",
            figure_id="fig1",
            caption=text,
            entities=[],
            figure_number="1",
            page_index=1,
        )

    def _make_region(self):
        from rlpe.types import FigureRegion

        return FigureRegion(
            page_index=1,
            bbox=(0, 0, 100, 100),
            crop_path="test.png",
            score=0.9,
            region_id="test",
        )

    def test_returns_none_when_no_backend(self):
        """Without a backend, _llm_first_extract returns None (triggers fallback)."""
        self.pipe.gemma_runtime = None
        result = self.pipe._llm_first_extract(
            paper_id="p",
            figure_id="f",
            caption=self._make_caption(),
            region_img=MagicMock(),
            region=self._make_region(),
            figure_index=1,
        )
        assert result is None

    def test_returns_none_for_placeholder_caption(self):
        """Placeholder captions should be skipped."""
        mock_backend = MagicMock()
        self.pipe.gemma_runtime = mock_backend
        result = self.pipe._llm_first_extract(
            paper_id="p",
            figure_id="f",
            caption=self._make_caption("Auto-generated figure for page 5"),
            region_img=MagicMock(),
            region=self._make_region(),
            figure_index=1,
        )
        assert result is None
        mock_backend.infer_panel.assert_not_called()

    def _run_with_mock_backend(self, backend):
        """Helper: set backend and run with PIL/cv2 mocked."""
        self.pipe.gemma_runtime = backend
        mock_img = MagicMock()
        with patch("rlpe.pipeline.cv2") as mock_cv2, patch("PIL.Image") as mock_pil:
            mock_cv2.cvtColor.return_value = MagicMock()
            mock_pil.fromarray.return_value = mock_img
            return self.pipe._llm_first_extract(
                paper_id="p",
                figure_id="f",
                caption=self._make_caption(),
                region_img=MagicMock(shape=(100, 100, 3)),
                region=self._make_region(),
                figure_index=1,
            )

    def test_returns_none_on_fallback_used(self):
        """When the backend returns fallback_used=True, should return None."""
        mock_backend = MagicMock()
        mock_backend.infer_panel.return_value = {
            "fallback_used": True,
            "error": "network error",
        }
        result = self._run_with_mock_backend(mock_backend)
        assert result is None

    def test_returns_single_panel_when_no_caption_additions(self):
        """A single-panel LLM result is returned as-is (1 panel, no
        fallback to None) when the caption parser does not find more
        panels than the LLM returned. Pre-fix behaviour was to return
        None on len(llm_results) < 2; the v21 hybrid gate replaced
        that strict "needs ≥2 panels" requirement with a more lenient
        "needs fallback if the LLM truncated its output" rule."""
        mock_backend = MagicMock()
        mock_backend.infer_panel.return_value = {
            "fallback_used": False,
            "panels": [{"label": "A", "species": "Test sp.", "confidence": 0.9}],
        }
        result = self._run_with_mock_backend(mock_backend)
        assert result is not None
        assert len(result) == 1
        assert result[0]["panel_id"] == "A"
        assert result[0]["species"] == "Test sp."

    def test_returns_match_results_on_success(self):
        """Successful LLM extraction returns MatchResult dicts."""
        mock_backend = MagicMock()
        mock_backend.backend_name = "test_llm"
        mock_backend.infer_panel.return_value = {
            "fallback_used": False,
            "panels": [
                {"label": "1", "species": "Actinomma leptodermum", "confidence": 0.95},
                {"label": "2", "species": "Actinomma holtedahli", "confidence": 0.88},
            ],
        }
        result = self._run_with_mock_backend(mock_backend)
        assert result is not None
        assert len(result) == 2
        assert result[0]["panel_id"] == "1"
        assert result[0]["species"] == "Actinomma leptodermum"
        assert result[1]["panel_id"] == "2"
        assert result[1]["metadata"]["extraction_method"] == "llm_first"

    def test_handles_raw_text_json(self):
        """When result has raw_text but no panels key, parse from raw text."""
        import json

        mock_backend = MagicMock()
        mock_backend.backend_name = "test_llm"
        mock_backend.infer_panel.return_value = {
            "fallback_used": False,
            "raw_text": json.dumps(
                {
                    "panels": [
                        {"label": "A", "species": "Test sp. 1", "confidence": 0.8},
                        {"label": "B", "species": "Test sp. 2", "confidence": 0.7},
                    ]
                }
            ),
        }
        result = self._run_with_mock_backend(mock_backend)
        assert result is not None
        assert len(result) == 2

    def test_sets_caption_panel_id_and_panel_id_source(self):
        """LLM-first rows must stamp label provenance (caption_panel_id,
        panel_id_source='llm_first') so downstream consumers can
        distinguish caption-derived ids from true image-OCR / bbox
        evidence. printed_panel_id is intentionally NOT set here —
        the visual-evidence path is reserved for the classical OCR
        and Stage-3 bbox/crop work.
        """
        mock_backend = MagicMock()
        mock_backend.backend_name = "MiniMax"
        mock_backend.infer_panel.return_value = {
            "fallback_used": False,
            "panels": [
                {"label": "1", "species": "Actinomma leptodermum", "confidence": 0.95},
                {"label": "2a", "species": "Actinomma holtedahli", "confidence": 0.88},
            ],
        }
        result = self._run_with_mock_backend(mock_backend)
        assert result is not None
        assert len(result) == 2
        for row, expected in zip(result, ("1", "2a"), strict=True):
            assert row["metadata"]["caption_panel_id"] == expected
            assert row["metadata"]["panel_id_source"] == "llm_first"
            # printed_panel_id must NOT be set by the LLM-first path
            # — that is reserved for true image-OCR / Stage-3 evidence.
            assert "printed_panel_id" not in row["metadata"]
