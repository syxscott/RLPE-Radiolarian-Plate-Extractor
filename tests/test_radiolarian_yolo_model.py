"""Regression tests for the trained radiolarian YOLO detector."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "radiolarian_yolo_v1.pt"
VAL_IMAGES = REPO_ROOT / "data" / "yolo_dataset" / "images" / "val"


def _require_model() -> Path:
    if not MODEL_PATH.is_file():
        pytest.skip("needs training: models/radiolarian_yolo_v1.pt is missing")
    return MODEL_PATH


class TestRadiolarianYoloModel:
    def test_model_file_exists(self):
        model_path = _require_model()
        assert model_path.is_file()

    def test_model_has_single_class(self):
        pytest.importorskip("ultralytics")
        from ultralytics import YOLO

        model = YOLO(str(_require_model()))
        assert model.names == {0: "radiolarian_panel"}

    def test_model_inference_runs(self):
        pytest.importorskip("ultralytics")
        from ultralytics import YOLO

        sample_images = sorted(VAL_IMAGES.glob("*.png")) + sorted(VAL_IMAGES.glob("*.jpg"))
        if not sample_images:
            pytest.skip("needs training data: no validation image is available")
        model = YOLO(str(_require_model()))
        # Force CPU inference so the test doesn't OOM on shared/limited
        # GPU memory in dev environments. CI skips this branch entirely
        # via ``pytest.importorskip("ultralytics")`` (ultralytics isn't
        # in the test extras), so the device choice only affects local
        # runs where ultralytics is installed.
        results = model.predict(
            source=str(sample_images[0]), save=False, verbose=False, device="cpu"
        )
        assert results
        assert len(results) == 1
