"""Regression tests for audit 2026-08-02 — YOLO COCO model filename warning (B2)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.config import PipelineConfig  # noqa: E402


class TestYoloModelValidation:
    def test_coco_filename_triggers_warning(self, tmp_path, caplog):
        model_path = tmp_path / "yolo11x.pt"
        model_path.write_bytes(b"not-a-real-model")
        with caplog.at_level(logging.WARNING, logger="rlpe.config"):
            PipelineConfig(
                pdf_dir=tmp_path,
                work_dir=tmp_path,
                use_yolo_figures=True,
                yolo_model_path=str(model_path),
            )
        assert any("COCO-pretrained" in rec.getMessage() for rec in caplog.records), caplog.text

    def test_radiolarian_filename_no_warning(self, tmp_path, caplog):
        model_path = tmp_path / "yolo_radiolaria_v1.pt"
        model_path.write_bytes(b"not-a-real-model")
        with caplog.at_level(logging.WARNING, logger="rlpe.config"):
            PipelineConfig(
                pdf_dir=tmp_path,
                work_dir=tmp_path,
                use_yolo_figures=True,
                yolo_model_path=str(model_path),
            )
        assert not any("COCO-pretrained" in rec.getMessage() for rec in caplog.records), caplog.text

    def test_no_path_no_warning(self, tmp_path):
        with pytest.raises(ValueError, match="yolo_model_path"):
            PipelineConfig(
                pdf_dir=tmp_path,
                work_dir=tmp_path,
                use_yolo_figures=True,
                yolo_model_path="",
            )
