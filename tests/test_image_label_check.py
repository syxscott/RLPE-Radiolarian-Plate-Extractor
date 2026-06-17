"""Tests for the image_label_check module.

The module's core logic is pure-Python (path resolution, metric
aggregation). We don't run real EasyOCR in unit tests — we mock the
reader to keep CI fast and avoid depending on OCR packages.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.evaluation.image_label_check import (  # noqa: E402
    ImageLabelStats,
    _resolve_panel_path,
    run_image_label_check,
)


class _FakeReader:
    """Pretends to be an EasyOCR reader. Returns the same numbers for
    every image, in the order given."""

    def __init__(self, numbers: list[str]) -> None:
        self.numbers = numbers

    def readtext(self, arr):  # noqa: ARG002
        return [([[0, 0], [10, 0], [10, 10], [0, 10]], self.numbers[0], 0.99)]


def test_resolve_panel_path_exact():
    p = Path(__file__).resolve()
    assert _resolve_panel_path(str(p), Path("/")) == p


def test_resolve_panel_path_empty_returns_none(tmp_path):
    assert _resolve_panel_path("", tmp_path) is None


def test_resolve_panel_path_fallback_search(tmp_path):
    # The exact path doesn't exist; we should find it by tail under
    # work/*/panels/<paper_id>/<fig>/panel_NN.png
    paper_id = "abc123"
    fig = "fig_01"
    panel = "panel_03.png"
    work = tmp_path / "work" / "run_x" / "panels" / paper_id / fig
    work.mkdir(parents=True)
    (work / panel).write_bytes(b"")
    # Pred file wrote a path that points at a different run dir
    fake_pred_path = str(tmp_path / "work" / "run_y_DELETED" / "panels" / paper_id / fig / panel)
    resolved = _resolve_panel_path(fake_pred_path, tmp_path)
    assert resolved is not None
    assert resolved.name == panel
    assert paper_id in str(resolved)


def test_resolve_panel_path_fallback_output_panels_layout(tmp_path):
    """Some refresh runs (e.g. work/beccaro_only_out) put panels under
    an extra ``output/`` segment, so the glob pattern must also try
    ``work/*/output/panels/<paper_id>/<fig>/panel_NN.png``. This was
    a real bug: beccaro2006's 35 v18 preds pointed at
    ``work/beccaro2006_only_out/output/panels/...`` (a typo'd path)
    and the v18 panels actually live at
    ``work/beccaro_only_out/output/panels/...`` — the path-mismatch
    glob previously missed them and the OCR coverage reported 0 for
    beccaro."""
    paper_id = "abc123"
    fig = "fig_01"
    panel = "panel_03.png"
    work = tmp_path / "work" / "run_x" / "output" / "panels" / paper_id / fig
    work.mkdir(parents=True)
    (work / panel).write_bytes(b"")
    # Pred file wrote a path with a typo'd run dir name
    fake_pred_path = str(
        tmp_path / "work" / "run_y_TYPO" / "output" / "panels" / paper_id / fig / panel
    )
    resolved = _resolve_panel_path(fake_pred_path, tmp_path)
    assert resolved is not None, "fallback glob should find panels under work/*/output/panels/"
    assert resolved.name == panel
    assert paper_id in str(resolved)


def test_image_label_stats_rate_property():
    s = ImageLabelStats(paper_id="p1", n_checked=10, n_ocr_has_label=5, n_image_label_match=4)
    assert s.image_label_match_rate == 0.4
    assert s.ocr_coverage == 0.5


def test_run_image_label_check_with_fake_reader(tmp_path):
    import numpy as np
    from PIL import Image

    paper_id = "paper1"
    work = tmp_path / "work" / "r" / "panels" / paper_id / "fig_01"
    work.mkdir(parents=True)
    for n in ("panel_05.png", "panel_06.png"):
        Image.new("RGB", (10, 10), color=(255, 255, 255)).save(work / n)
    preds = [
        {
            "paper_id": paper_id,
            "figure_id": "fig_01",
            "panel_id": "5",
            "panel_path": str(work / "panel_05.png"),
        },
        {
            "paper_id": paper_id,
            "figure_id": "fig_01",
            "panel_id": "6",
            "panel_path": str(work / "panel_06.png"),
        },
    ]

    # Reader returns "5" for the first image (match) and "9" for the
    # second (mismatch). To make the fake return different things per
    # image we need a stateful reader.
    class _Stateful(_FakeReader):
        def __init__(self):
            self.calls = 0

        def readtext(self, arr):
            self.calls += 1
            if self.calls == 1:
                return [([[0, 0], [10, 0], [10, 10], [0, 10]], "5", 0.99)]
            return [([[0, 0], [10, 0], [10, 10], [0, 10]], "9", 0.99)]

    out = run_image_label_check(preds, tmp_path, reader=_Stateful())
    p = out["papers"][paper_id]
    assert p["n_checked"] == 2
    assert p["n_ocr_has_label"] == 2
    assert p["n_image_label_match"] == 1
    assert p["image_label_match_rate"] == 0.5
    assert out["aggregate"]["n_image_label_match"] == 1
    # Mismatch should be recorded
    assert len(p["mismatches"]) == 1
    assert p["mismatches"][0]["pred_panel_id"] == "6"
    assert p["mismatches"][0]["ocr_label"] == "9"
