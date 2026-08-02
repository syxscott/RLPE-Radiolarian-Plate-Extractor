"""Regression tests for audit 2026-08-02 — max_regions_per_caption safety cap."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe.config import PipelineConfig  # noqa: E402
from rlpe.types import CaptionRecord, FigureRegion, PageRecord  # noqa: E402


def _make_regions(tmp_path: Path) -> list[FigureRegion]:
    return [
        FigureRegion(
            page_index=1,
            bbox=(0, index * 100, 100, index * 100 + 90),
            crop_path=str(tmp_path / f"r{index}.png"),
            score=1.0 - index / 10,
            region_id=f"r{index}",
        )
        for index in range(5)
    ]


def _run_grobid_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_regions: int,
) -> list[FigureRegion]:
    """Drive one caption through the GROBID region-processing loop."""
    import rlpe.pipeline as pipeline_mod
    from rlpe.grobid import GrobidResult

    caption = CaptionRecord(
        paper_id="paper",
        figure_id="fig_1",
        caption="Plate 1. Radiolarians.",
        entities=[],
        figure_number="1",
        page_index=1,
    )
    page = PageRecord(
        page_index=1,
        image_path=str(tmp_path / "p1.png"),
        text="Plate 1",
    )

    with (
        mock.patch("rlpe.pipeline.GrobidClient"),
        mock.patch("rlpe.pipeline.OCRBackend"),
        mock.patch("rlpe.pipeline.TaxonRecognizer"),
        mock.patch("rlpe.pipeline.PanelSegmenter"),
    ):
        config = PipelineConfig(
            pdf_dir=tmp_path,
            work_dir=tmp_path / "work",
            max_regions_per_caption=max_regions,
        )
        from rlpe.pipeline import RadiolarianPipeline

        pipe = RadiolarianPipeline(config)

    pipe.grobid.process_pdf.return_value = GrobidResult(
        paper_id="paper",
        pdf_path=tmp_path / "paper.pdf",
        tei_path=None,
        tei_xml="",
        captions=[caption],
        fulltext_sections=[],
        success=True,
    )

    regions = _make_regions(tmp_path)
    monkeypatch.setattr(pipeline_mod, "render_pdf_pages", lambda *a, **k: [page])
    monkeypatch.setattr(pipeline_mod, "choose_best_page", lambda *a, **k: page)
    monkeypatch.setattr(pipeline_mod, "find_plate_pages", lambda *a, **k: [])
    monkeypatch.setattr(pipeline_mod, "detect_figure_regions", lambda *a, **k: list(regions))
    monkeypatch.setattr(
        pipeline_mod.cv2,
        "imread",
        lambda *a, **k: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    for name in (
        "_cross_figure_reassign",
        "_link_range_chart_geology",
        "_cross_link_map_and_range_chart",
        "_finalize_rows",
    ):
        monkeypatch.setattr(pipe, name, lambda rows: rows)

    seen: list[FigureRegion] = []

    def fake_process_region(**kwargs):
        region = kwargs["region"]
        seen.append(region)
        return [{"figure_id": "fig_1", "panel_id": region.region_id}]

    monkeypatch.setattr(pipe, "_process_region", fake_process_region)
    pipe._process_one_pdf_grobid_inner("paper", tmp_path / "paper.pdf")
    return seen


class TestMaxRegionsPerCaption:
    def test_caps_at_max_regions(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = _run_grobid_path(tmp_path, monkeypatch, max_regions=3)

        assert [region.region_id for region in seen] == ["r0", "r1", "r2"]

    def test_default_is_3(self, tmp_path: Path) -> None:
        config = PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "work")

        assert config.max_regions_per_caption == 3

    def test_max_zero_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="max_regions_per_caption"):
            PipelineConfig(
                pdf_dir=tmp_path,
                work_dir=tmp_path / "work",
                max_regions_per_caption=0,
            )

    def test_max_51_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="max_regions_per_caption"):
            PipelineConfig(
                pdf_dir=tmp_path,
                work_dir=tmp_path / "work",
                max_regions_per_caption=51,
            )

    def test_dropped_logged(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO, logger="rlpe.pipeline")

        _run_grobid_path(tmp_path, monkeypatch, max_regions=3)

        assert (
            "max_regions_per_caption=3 cap dropped 2 lower-scored regions "
            "for fig=fig_1" in caplog.text
        )
