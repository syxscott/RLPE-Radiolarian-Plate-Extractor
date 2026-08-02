"""Regression tests for audit 2026-08-02 — multi-region fallback for GROBID path.

The GROBID path in ``RadiolarianPipeline._process_one_pdf_grobid_inner``
used to sort the detected figure regions by score and then process only
``chosen_regions[0]``. Multi-plate papers put each plate in its own
detected region, so a paper like Bandini 2011 (9 plates / 215 panels)
had 8 of its 9 plates silently discarded.

Post-fix the loop iterates *every* chosen region, processes each one,
and merges the rows — deduplicating by ``(figure_id, panel_id)`` so two
regions that both detect panel "1" contribute a single row (the one
from the higher-scoring region, since the list is sorted first).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe.types import CaptionRecord, FigureRegion, PageRecord  # noqa: E402


def _make_pipeline(tmp_path: Path):
    """Build a pipeline with all heavy collaborators mocked out."""
    from rlpe.config import PipelineConfig

    with (
        mock.patch("rlpe.pipeline.GrobidClient"),
        mock.patch("rlpe.pipeline.OCRBackend"),
        mock.patch("rlpe.pipeline.TaxonRecognizer"),
        mock.patch("rlpe.pipeline.PanelSegmenter"),
    ):
        cfg = PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "work")
        cfg.extra["cross_figure_linker_enabled"] = False
        from rlpe.pipeline import RadiolarianPipeline

        return RadiolarianPipeline(cfg)


def _grobid_result(tmp_path: Path, caption: CaptionRecord):
    from rlpe.grobid import GrobidResult

    return GrobidResult(
        paper_id=caption.paper_id,
        pdf_path=tmp_path / "paper.pdf",
        tei_path=None,
        tei_xml="",
        captions=[caption],
        fulltext_sections=[],
        success=True,
    )


def _run_grobid_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    regions: list[FigureRegion],
    region_rows,
) -> tuple[list[dict[str, Any]], list[FigureRegion]]:
    """Drive the GROBID path over ``regions``.

    ``region_rows(region)`` supplies the rows that the (stubbed)
    ``_process_region`` returns for that region. Returns
    ``(rows, regions_seen_in_call_order)``.
    """
    import rlpe.pipeline as pipeline_mod

    caption = CaptionRecord(
        paper_id="bandini2011",
        figure_id="fig_1",
        caption="Plate 1. Radiolarians. 1, Unuma echinatus; 2, Archaeodictyomitra sp.",
        entities=[],
        figure_number="1",
        page_index=1,
    )
    page = PageRecord(page_index=1, image_path=str(tmp_path / "p1.png"), text="Plate 1")

    pipe = _make_pipeline(tmp_path)
    pipe.grobid.process_pdf.return_value = _grobid_result(tmp_path, caption)

    monkeypatch.setattr(pipeline_mod, "render_pdf_pages", lambda *a, **k: [page])
    monkeypatch.setattr(pipeline_mod, "choose_best_page", lambda *a, **k: page)
    monkeypatch.setattr(pipeline_mod, "find_plate_pages", lambda *a, **k: [])
    monkeypatch.setattr(pipeline_mod, "detect_figure_regions", lambda p, **k: list(regions))
    monkeypatch.setattr(
        pipeline_mod.cv2, "imread", lambda *a, **k: np.zeros((8, 8, 3), dtype=np.uint8)
    )
    # Isolate the loop under test from the post-processing chain.
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
        return region_rows(region)

    monkeypatch.setattr(pipe, "_process_region", fake_process_region)

    rows = pipe._process_one_pdf_grobid_inner("bandini2011", tmp_path / "paper.pdf")
    return rows, seen


class TestMultiRegionFallback:
    """Audit 2026-08-02: every detected region must be processed."""

    def test_multiple_regions_per_caption_all_processed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """3 regions on one caption → 3 panels, not 1 (pre-fix behaviour)."""
        regions = [
            FigureRegion(
                page_index=1,
                bbox=(0, 0, 100, 100),
                crop_path=str(tmp_path / "r0.png"),
                score=0.9,
                region_id="r0",
            ),
            FigureRegion(
                page_index=1,
                bbox=(0, 200, 100, 300),
                crop_path=str(tmp_path / "r1.png"),
                score=0.8,
                region_id="r1",
            ),
            FigureRegion(
                page_index=1,
                bbox=(0, 400, 100, 500),
                crop_path=str(tmp_path / "r2.png"),
                score=0.7,
                region_id="r2",
            ),
        ]

        def region_rows(region: FigureRegion):
            return [
                {
                    "paper_id": "bandini2011",
                    "figure_id": "fig_1",
                    "panel_id": region.region_id,
                    "species": "Unuma echinatus",
                }
            ]

        rows, seen = _run_grobid_path(tmp_path, monkeypatch, regions, region_rows)

        assert len(seen) == 3, "every chosen region must be handed to _process_region"
        assert len(rows) == 3, f"expected one panel per region, got {len(rows)}"
        assert {r["panel_id"] for r in rows} == {"r0", "r1", "r2"}

    def test_regions_sorted_by_score_desc(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Processing order stays ``(-score, page_index, bbox_y, bbox_x)``."""
        regions = [
            # Deliberately shuffled; every sort key is exercised.
            FigureRegion(page_index=2, bbox=(10, 50, 60, 90), score=0.5, region_id="c"),
            FigureRegion(page_index=1, bbox=(0, 10, 50, 60), score=0.9, region_id="a"),
            FigureRegion(page_index=1, bbox=(5, 10, 55, 60), score=0.5, region_id="b"),
            FigureRegion(page_index=1, bbox=(0, 200, 50, 250), score=0.9, region_id="a2"),
        ]

        def region_rows(region: FigureRegion):
            return [{"figure_id": "fig_1", "panel_id": region.region_id}]

        rows, seen = _run_grobid_path(tmp_path, monkeypatch, regions, region_rows)

        expected = sorted(regions, key=lambda r: (-r.score, r.page_index, r.bbox[1], r.bbox[0]))
        assert [r.region_id for r in seen] == [r.region_id for r in expected]
        # Highest-scoring region is still first — the pre-fix "best
        # region" is never demoted, the others are merely added after it.
        assert seen[0].region_id == "a"
        assert [r["panel_id"] for r in rows] == [r.region_id for r in expected]

    def test_dedup_after_multi_region(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two regions both detecting panel "1" yield a single row."""
        regions = [
            FigureRegion(page_index=1, bbox=(0, 0, 100, 100), score=0.9, region_id="r0"),
            FigureRegion(page_index=1, bbox=(0, 200, 100, 300), score=0.8, region_id="r1"),
        ]

        def region_rows(region: FigureRegion):
            # Both regions detect panel "1"; only the second also has "2".
            rows = [
                {
                    "figure_id": "fig_1",
                    "panel_id": "1",
                    "species": f"from-{region.region_id}",
                }
            ]
            if region.region_id == "r1":
                rows.append({"figure_id": "fig_1", "panel_id": "2", "species": "sp2"})
            return rows

        rows, seen = _run_grobid_path(tmp_path, monkeypatch, regions, region_rows)

        assert len(seen) == 2, "both regions still processed"
        panel_ids = [r["panel_id"] for r in rows]
        assert panel_ids.count("1") == 1, f"panel 1 duplicated: {panel_ids}"
        assert sorted(panel_ids) == ["1", "2"]
        # The surviving duplicate comes from the higher-scoring region.
        panel_1 = next(r for r in rows if r["panel_id"] == "1")
        assert panel_1["species"] == "from-r0"

    def test_stub_rows_without_panel_id_are_not_deduped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``panel_id=None`` stub rows are context carriers, not panels.

        Collapsing them on ``(figure_id, None)`` would lose per-region
        geology / ingestion context, so the dedup skips falsy panel_ids.
        """
        regions = [
            FigureRegion(page_index=1, bbox=(0, 0, 100, 100), score=0.9, region_id="r0"),
            FigureRegion(page_index=1, bbox=(0, 200, 100, 300), score=0.8, region_id="r1"),
        ]

        def region_rows(region: FigureRegion):
            return [{"figure_id": "fig_1", "panel_id": None, "src": region.region_id}]

        rows, _ = _run_grobid_path(tmp_path, monkeypatch, regions, region_rows)

        assert len(rows) == 2, "stub rows must survive both regions"
        assert {r["src"] for r in rows} == {"r0", "r1"}


def test_grobid_path_no_longer_indexes_chosen_regions_zero() -> None:
    """Source guard: the single-region ``chosen_regions[0]`` must stay gone."""
    from rlpe.pipeline import RadiolarianPipeline

    src = inspect.getsource(RadiolarianPipeline._process_one_pdf_grobid_inner)
    assert "chosen_regions[0]" not in src, (
        "GROBID path regressed to processing only the best-scoring region; "
        "it must iterate all chosen_regions (audit 2026-08-02)."
    )
    assert "for region in chosen_regions:" in src
