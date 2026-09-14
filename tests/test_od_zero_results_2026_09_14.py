"""2026-09-14 — Ark 12s-failure regression tests (OD zero-results path).

The Volcano-Ark llm_compare runs produced a 12 s "extraction" that was
actually: OD pass 1 → zero results → GROBID re-entry (a no-op under
``disable_grobid`` that just re-ran OD) → OD pass 2 → zero results →
depth-4 guard → a single misleading ``_ingestion_grobid_cycle`` stub,
with no warning explaining what happened. These tests lock in:

1. ``disable_grobid=true``: neither zero-results re-entry point may
   call back into ``_process_one_pdf_grobid`` (no wasted second OD
   pass, no cycle stub).
2. The zero-results branch emits a structured ``od_zero_results``
   warning with the per-reason figure-skip breakdown.
3. The pairing-flake shape (kids tree HAS figure/image elements but
   extraction yielded 0 figures) emits the third-shape warning.
4. The cycle stub's ingestion_error names the disabled GROBID instead
   of claiming an OD↔GROBID cycle.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def pipe(tmp_path):
    from rlpe.config import PipelineConfig
    from rlpe.pipeline import RadiolarianPipeline

    with (
        patch("rlpe.pipeline.GrobidClient"),
        patch("rlpe.pipeline.OCRBackend"),
        patch("rlpe.pipeline.TaxonRecognizer"),
        patch("rlpe.pipeline.PanelSegmenter"),
    ):
        cfg = PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "w")
        cfg.extra["disable_grobid"] = True
        p = RadiolarianPipeline(cfg)
        p._od_extractor = MagicMock()
        yield p


def _od_result(figures, json_data, success=True):
    from rlpe.opendataloader_extractor import OpenDataLoaderResult

    return OpenDataLoaderResult(
        paper_id="p1",
        json_data=json_data,
        output_dir=Path("od"),
        figures=figures,
        fulltext_sections=[],
        success=success,
    )


def _drain_warnings():
    from rlpe.utils import drain_warnings

    return drain_warnings()


class TestDisableGrobidNoReentry:
    def test_zero_results_returns_empty_without_grobid_call(self, pipe, tmp_path):
        """Figures present but unreadable → loop skips all → zero
        results: must NOT re-enter GROBID; must return [] and emit the
        od_zero_results warning with the skip breakdown."""
        from rlpe.opendataloader_extractor import FigureCaptionPair

        fig = FigureCaptionPair(
            figure_id="f1",
            page_number=1,
            image_paths=[str(tmp_path / "missing_image.png")],  # imread fails
            caption_text="Fig. 1. Test plate",
            merged_bbox=None,
        )
        pipe.od_extractor.extract.return_value = _od_result(
            figures=[fig], json_data={"kids": [{"type": "image"}]}
        )
        grobid_spy = MagicMock(return_value=[{"sentinel": True}])
        with patch.object(pipe, "_process_one_pdf_grobid", grobid_spy):
            _drain_warnings()
            rows = pipe._process_one_pdf_od_inner("p1", tmp_path / "paper.pdf")
        assert rows == []
        grobid_spy.assert_not_called()
        labels = {w["label"] for w in _drain_warnings()}
        assert "od_zero_results" in labels

    def test_zero_results_warning_carries_skip_breakdown(self, pipe, tmp_path):
        from rlpe.opendataloader_extractor import FigureCaptionPair

        fig = FigureCaptionPair(
            figure_id="f1",
            page_number=1,
            image_paths=[str(tmp_path / "nope.png")],
            caption_text="Fig. 1",
            merged_bbox=None,
        )
        pipe.od_extractor.extract.return_value = _od_result(
            figures=[fig], json_data={"kids": [{"type": "image"}]}
        )
        with patch.object(pipe, "_process_one_pdf_grobid"):
            _drain_warnings()
            pipe._process_one_pdf_od_inner("p1", tmp_path / "paper.pdf")
        zero = [w for w in _drain_warnings() if w["label"] == "od_zero_results"]
        assert zero, "expected an od_zero_results warning"
        assert "unreadable_images=1" in zero[0]["message"]
        assert "extracted 1 figure(s)" in zero[0]["message"]

    def test_no_figures_no_json_returns_empty_without_grobid_call(self, pipe, tmp_path):
        """success=True with neither figures nor JSON: the second
        re-entry point must also stay quiet under disable_grobid."""
        pipe.od_extractor.extract.return_value = _od_result(
            figures=[], json_data=None, success=True
        )
        grobid_spy = MagicMock(return_value=[{"sentinel": True}])
        with patch.object(pipe, "_process_one_pdf_grobid", grobid_spy):
            rows = pipe._process_one_pdf_od_inner("p1", tmp_path / "paper.pdf")
        assert rows == []
        grobid_spy.assert_not_called()


class TestGrobidReentryPreserved:
    def test_without_disable_grobid_reentry_still_happens(self, tmp_path):
        """Machines WITH a working GROBID keep the layout fallback."""
        from rlpe.config import PipelineConfig
        from rlpe.opendataloader_extractor import FigureCaptionPair
        from rlpe.pipeline import RadiolarianPipeline

        with (
            patch("rlpe.pipeline.GrobidClient"),
            patch("rlpe.pipeline.OCRBackend"),
            patch("rlpe.pipeline.TaxonRecognizer"),
            patch("rlpe.pipeline.PanelSegmenter"),
        ):
            cfg = PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "w")
            assert not cfg.extra.get("disable_grobid", False)
            p = RadiolarianPipeline(cfg)
            p._od_extractor = MagicMock()
            fig = FigureCaptionPair(
                figure_id="f1",
                page_number=1,
                image_paths=[str(tmp_path / "nope.png")],
                caption_text="Fig. 1",
                merged_bbox=None,
            )
            p.od_extractor.extract.return_value = _od_result(
                figures=[fig], json_data={"kids": [{"type": "image"}]}
            )
            grobid_spy = MagicMock(return_value=[{"sentinel": True}])
            with patch.object(p, "_process_one_pdf_grobid", grobid_spy):
                rows = p._process_one_pdf_od_inner("p1", tmp_path / "paper.pdf")
            grobid_spy.assert_called_once()
            assert rows == [{"sentinel": True}]


class TestStubHonesty:
    def test_cycle_stub_names_disabled_grobid(self, pipe, tmp_path):
        pipe.config.extra["disable_grobid"] = True
        stub = pipe._make_od_grobid_cycle_stub("p1", tmp_path / "paper.pdf", "grobid")
        assert "GROBID is disabled" in stub["metadata"]["ingestion_error"]
        assert "disable_grobid=true" in stub["metadata"]["ingestion_error"]

    def test_cycle_stub_keeps_cycle_message_with_grobid(self, tmp_path):
        from rlpe.config import PipelineConfig
        from rlpe.pipeline import RadiolarianPipeline

        with (
            patch("rlpe.pipeline.GrobidClient"),
            patch("rlpe.pipeline.OCRBackend"),
            patch("rlpe.pipeline.TaxonRecognizer"),
            patch("rlpe.pipeline.PanelSegmenter"),
        ):
            p = RadiolarianPipeline(PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "w"))
            stub = p._make_od_grobid_cycle_stub("p1", tmp_path / "paper.pdf", "grobid")
            assert "cycle detected" in stub["metadata"]["ingestion_error"]


class TestPairingFlakeWarning:
    def test_figure_type_kids_but_zero_figures_warns(self, pipe, tmp_path):
        """The third silent shape: kids tree HAS figure/image elements
        but extraction returned 0 figures — must emit a warning."""
        pipe.od_extractor.extract.return_value = _od_result(
            figures=[],
            json_data={"kids": [{"type": "image"}, {"type": "caption"}, {"type": "paragraph"}]},
        )
        with patch.object(pipe, "_process_one_pdf_grobid"):
            _drain_warnings()
            pipe._process_one_pdf_od_inner("p1", tmp_path / "paper.pdf")
        msgs = [w["message"] for w in _drain_warnings()]
        assert any("pairing flake" in m for m in msgs), msgs


class TestSampleIdPrecedence:
    def test_specimen_number_demoted_below_sample_code(self, pipe):
        """Boughdiri 2007: caption items carry BOTH the sample code
        (CH4) and the plate-internal specimen number (7). sample_id
        must resolve to the sample code; the specimen number stays in
        sample_ids as a trailing, clearly-demoted entry."""
        rows = [
            {
                "paper_id": "p",
                "figure_id": "f",
                "panel_id": "1",
                "species": "Ristola altissima altissima",
                "caption_snippet": "1) Ristola altissima altissima (RUST), CH4, specimen 7, 550 um",
                "metadata": {},
            }
        ]
        out = pipe._finalize_rows(rows)
        md = out[0].get("metadata") or {}
        assert md.get("sample_id") == "CH4", md.get("sample_id")
        assert md.get("sample_ids", [])[-1].endswith("7")
