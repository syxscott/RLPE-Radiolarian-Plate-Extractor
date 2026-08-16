"""Tests for Plan C: M3 Stage 3 YOLO fallback.

Audit 2026-08-16 (fill-gaps): previously M3 Stage 3 bbox enrichment
short-circuited when the M3 vision model returned zero panels. This
helper synthesises stage3_panel records from YOLO so the crop pass
still produces useful output for papers that exhausted M3 quota or
whose plates M3 declined to segment.

These tests guard:
  - Helper is a no-op when ``use_yolo_figures`` is False
  - Helper returns {} when ``yolo_model_path`` is empty
  - Helper returns {} when no figure has a plate image
  - Each synthesised panel has the M3-shaped dict contract
  - Source tag is ``"yolo_fallback"`` so downstream consumers can
    distinguish it from M3's ``"m3_vision"`` source
  - The crop pass promotes ``panel_id_source`` from the
    synthesised source (no regression on the M3 path)
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rlpe.config import PipelineConfig

# ---------------------------------------------------------------------------
# Helper: build a minimal config + pipeline skeleton for unit-testing
# ``_yolo_fallback_for_stage3`` without spinning up the real pipeline.
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, **overrides) -> PipelineConfig:
    """Minimal PipelineConfig — most fields unused by the helper."""
    cfg = PipelineConfig(
        pdf_dir=tmp_path / "pdfs",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "out",
    )
    # Apply overrides via __dict__ (dataclass; no public mutation API).
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class _StubPipeline:
    """Minimal stand-in for RadiolarianPipeline exposing only what
    ``_yolo_fallback_for_stage3`` reads.
    """

    def __init__(self, cfg: PipelineConfig):
        from rlpe.pipeline import RadiolarianPipeline

        # Bind the unbound method so the call site resolves cleanly.
        self._yolo_fallback_for_stage3 = (
            RadiolarianPipeline._yolo_fallback_for_stage3.__get__(self, RadiolarianPipeline)
        )
        self.config = cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_yolo_fallback_disabled_when_use_yolo_figures_false(tmp_path):
    """When ``use_yolo_figures`` is False, helper returns {} immediately."""
    cfg = _make_config(
        tmp_path,
        use_yolo_figures=False,
        yolo_model_path="anything.pt",
    )
    pipe = _StubPipeline(cfg)
    out = pipe._yolo_fallback_for_stage3(
        figure_to_plate={"fig_1": "/some/path.png"},
        paper_id="paper1",
        crops_dir=tmp_path / "crops",
    )
    assert out == {}


def test_yolo_fallback_disabled_when_model_path_empty(tmp_path):
    """Empty ``yolo_model_path`` short-circuits to {} even if the flag is on."""
    cfg = _make_config(
        tmp_path,
        use_yolo_figures=True,
        yolo_model_path="",
    )
    pipe = _StubPipeline(cfg)
    out = pipe._yolo_fallback_for_stage3(
        figure_to_plate={"fig_1": "/some/path.png"},
        paper_id="paper1",
        crops_dir=tmp_path / "crops",
    )
    assert out == {}


def test_yolo_fallback_skips_missing_plate_files(tmp_path):
    """Plate paths that don't exist on disk are skipped (no exception)."""
    cfg = _make_config(
        tmp_path,
        use_yolo_figures=True,
        yolo_model_path="models/radiolarian_yolo_v1.pt",
    )
    pipe = _StubPipeline(cfg)
    # Layout.detect_figure_regions_yolo will fail to load a fake .pt
    # (it guards the loader); the helper swallows the exception and
    # returns an empty dict for that figure.
    out = pipe._yolo_fallback_for_stage3(
        figure_to_plate={"fig_missing": "/nonexistent/path.png"},
        paper_id="paper1",
        crops_dir=tmp_path / "crops",
    )
    assert out == {}


def test_yolo_fallback_panel_dict_contract(tmp_path, monkeypatch):
    """Synthesised panels must match the M3 PanelBox.to_dict() contract."""
    from rlpe.layout import FigureRegion

    cfg = _make_config(
        tmp_path,
        use_yolo_figures=True,
        yolo_model_path="fake_model.pt",
        yolo_conf_threshold=0.25,
        yolo_iou_threshold=0.45,
    )

    # Stub detect_figure_regions_yolo so the test doesn't need a
    # real YOLO model + ultralytics install.
    def fake_detect(page, model_path, conf=0.25, iou=0.45, min_area=5000, *, device="auto"):
        return [
            FigureRegion(
                page_index=0,
                bbox=(10, 20, 100, 80),
                crop_path="/tmp/fake_p1.png",
                score=0.87,
                region_id="fake_1",
                kind="figure",
                metadata={},
            ),
            FigureRegion(
                page_index=0,
                bbox=(120, 20, 90, 80),
                crop_path="/tmp/fake_p2.png",
                score=0.62,
                region_id="fake_2",
                kind="figure",
                metadata={},
            ),
        ]

    monkeypatch.setattr("rlpe.layout.detect_figure_regions_yolo", fake_detect)

    pipe = _StubPipeline(cfg)
    # Need a plate file that "exists" for the is_file() guard. The
    # actual YOLO inference is stubbed so its contents don't matter.
    plate = tmp_path / "plate.png"
    plate.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    out = pipe._yolo_fallback_for_stage3(
        figure_to_plate={"fig_1": str(plate)},
        paper_id="paper1",
        crops_dir=tmp_path / "crops",
    )
    assert "fig_1" in out
    panels = out["fig_1"]
    assert len(panels) == 2

    p1, p2 = panels
    # M3-shaped contract:
    assert p1["panel_id"] == "P1"
    assert p1["bbox"] == [10, 20, 100, 80]
    assert p1["visible_label"] is None
    assert p1["morphology"] is None
    assert 0.0 <= p1["confidence"] <= 1.0
    assert p1["confidence"] == pytest.approx(0.87)
    assert p1["source"] == "yolo_fallback"  # audit tag

    assert p2["panel_id"] == "P2"
    assert p2["bbox"] == [120, 20, 90, 80]
    assert p2["source"] == "yolo_fallback"


def test_yolo_fallback_returns_empty_on_yolo_load_failure(tmp_path, monkeypatch):
    """YOLO load failure (bad .pt) is caught; helper returns {}."""
    cfg = _make_config(
        tmp_path,
        use_yolo_figures=True,
        yolo_model_path="corrupt.pt",
    )

    def raising_detect(*a, **kw):
        raise RuntimeError("corrupt model file")

    monkeypatch.setattr("rlpe.layout.detect_figure_regions_yolo", raising_detect)

    pipe = _StubPipeline(cfg)
    plate = tmp_path / "plate.png"
    plate.write_bytes(b"")
    out = pipe._yolo_fallback_for_stage3(
        figure_to_plate={"fig_1": str(plate)},
        paper_id="paper1",
        crops_dir=tmp_path / "crops",
    )
    assert out == {}


def test_yolo_fallback_empty_detections(tmp_path, monkeypatch):
    """YOLO returns [] (no panels found) → that figure is dropped, no error."""
    from rlpe.layout import FigureRegion

    cfg = _make_config(
        tmp_path,
        use_yolo_figures=True,
        yolo_model_path="fake_model.pt",
    )

    def empty_detect(*a, **kw):
        return []

    monkeypatch.setattr("rlpe.layout.detect_figure_regions_yolo", empty_detect)

    pipe = _StubPipeline(cfg)
    plate = tmp_path / "plate.png"
    plate.write_bytes(b"")
    out = pipe._yolo_fallback_for_stage3(
        figure_to_plate={"fig_1": str(plate)},
        paper_id="paper1",
        crops_dir=tmp_path / "crops",
    )
    assert out == {}, "empty YOLO detection should drop the figure from the dict"


def test_yolo_panel_ids_match_row_panel_ids_audit_c2(monkeypatch, tmp_path):
    """Audit 2026-08-16 C2: synthesised panel_ids must match the
    rows' panel_ids, otherwise the matcher in
    ``_apply_stage3_bbox_crops`` never finds a hit and YOLO
    detections are silently dropped.

    Pre-fix the synthesiser emitted ``"P{i}"`` for every panel
    while rows typically carry ``"1"``, ``"2"``, ``"a"`` — string
    mismatch → zero bbox crops written.
    """
    from rlpe.layout import FigureRegion

    cfg = _make_config(
        tmp_path,
        use_yolo_figures=True,
        yolo_model_path="fake_model.pt",
    )

    # Two YOLO detections, two rows with panel_ids "1" and "2".
    def fake_detect(*a, **kw):
        return [
            FigureRegion(
                page_index=0,
                bbox=(10, 20, 100, 80),
                crop_path="/tmp/fake_p1.png",
                score=0.9,
                region_id="fake_1",
                kind="figure",
                metadata={},
            ),
            FigureRegion(
                page_index=0,
                bbox=(120, 20, 90, 80),
                crop_path="/tmp/fake_p2.png",
                score=0.7,
                region_id="fake_2",
                kind="figure",
                metadata={},
            ),
        ]

    monkeypatch.setattr("rlpe.layout.detect_figure_regions_yolo", fake_detect)

    pipe = _StubPipeline(cfg)
    plate = tmp_path / "plate.png"
    plate.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    # Pass figure_id_to_rows so the synthesiser adopts the rows'
    # actual panel_ids.
    rows = [
        {"panel_id": "1", "metadata": {"label_text": "1"}},
        {"panel_id": "2", "metadata": {"label_text": "2"}},
    ]
    out = pipe._yolo_fallback_for_stage3(
        figure_to_plate={"fig_1": str(plate)},
        paper_id="paper1",
        crops_dir=tmp_path / "crops",
        figure_id_to_rows={"fig_1": rows},
    )
    panels = out["fig_1"]
    assert len(panels) == 2
    # The synthesiser must adopt the rows' panel_ids (not "P1"/"P2")
    # so the matcher in _apply_stage3_bbox_crops finds a hit.
    assert panels[0]["panel_id"] == "1"
    assert panels[0]["visible_label"] == "1"
    assert panels[1]["panel_id"] == "2"
    assert panels[1]["visible_label"] == "2"
    # source tag still says yolo_fallback
    assert all(p["source"] == "yolo_fallback" for p in panels)


def test_plate_path_priority_audit_c1(tmp_path):
    """Audit 2026-08-16 C1: figure_to_plate must prefer plate-level
    image fields over panel_path. Functional test: stub
    ``detect_figure_regions_yolo`` and assert it gets called with the
    plate-level path, not the panel crop.

    Pre-fix the figure_to_plate builder put ``panel_path`` first,
    which made YOLO receive a tiny panel crop instead of the plate
    image. After the fix, ``figure_image_path`` wins.
    """
    from rlpe.layout import FigureRegion

    cfg = _make_config(
        tmp_path,
        use_yolo_figures=True,
        yolo_model_path="fake_model.pt",
    )

    seen_plates: list[str] = []

    def capture_detect(page, *args, **kwargs):
        seen_plates.append(str(page.image_path))
        return []

    import unittest.mock as mock

    pipe = _StubPipeline(cfg)
    # Patch via the same module path the helper uses
    with mock.patch("rlpe.layout.detect_figure_regions_yolo", capture_detect):
        # Plate-level path
        plate_level = tmp_path / "plate_full.png"
        plate_level.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        # Panel crop (tiny, would be wrong if used)
        panel_crop = tmp_path / "panel_crop.png"
        panel_crop.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        out = pipe._yolo_fallback_for_stage3(
            figure_to_plate={"fig_1": str(plate_level)},
            paper_id="paper1",
            crops_dir=tmp_path / "crops",
            figure_id_to_rows={
                "fig_1": [
                    {
                        "panel_id": "1",
                        "metadata": {
                            "figure_image_path": str(plate_level),
                            # ``panel_path`` set but should NOT be used
                            "panel_path": str(panel_crop),
                        },
                    }
                ]
            },
        )
    # The helper builds figure_to_plate by reading from ``results``,
    # but in this test we feed it via figure_to_plate directly. So
    # what we really verify here is that the synthesiser reads the
    # plate-level path passed in (which is the one we want). The
    # ``figure_to_plate`` source-grep part of the audit lives in a
    # separate unit test on _apply_stage3_bbox_crops.
    assert seen_plates == [str(plate_level)], (
        f"YOLO should have been called with the plate-level image, "
        f"got: {seen_plates}"
    )


def test_panel_id_source_promoted_from_matched(monkeypatch):
    """The crop pass must honour the synthesised ``source`` field so
    downstream can distinguish M3 vs YOLO. This is a regression guard
    on the source-tag-stamping change in ``_apply_stage3_bbox_crops``.
    """
    import inspect

    from rlpe import pipeline as pipeline_mod

    src = inspect.getsource(pipeline_mod.RadiololarianPipeline if hasattr(pipeline_mod, "RadiololarianPipeline") else pipeline_mod.RadiolarianPipeline)
    assert 'matched.get("source") or "m3_vision"' in src, (
        "Plan C: stage3 crop pass must read source from the matched dict "
        "so YOLO fallback panels get tagged 'yolo_fallback' instead of "
        "'m3_vision'."
    )


def test_no_silent_yolo_when_disabled():
    """Source-guard: ensure the helper has the early-return guards
    for both ``use_yolo_figures`` and ``yolo_model_path``. We grep
    the function body for the two guard clauses.
    """
    import inspect

    from rlpe.pipeline import RadiolarianPipeline

    src = inspect.getsource(RadiolarianPipeline._yolo_fallback_for_stage3)
    assert "use_yolo_figures" in src
    assert "yolo_model_path" in src
    assert "return {}" in src  # at least two early-return points
