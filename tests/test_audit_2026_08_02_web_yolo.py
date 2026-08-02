"""Regression tests for audit 2026-08-02 — YOLO web API surface (B1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("pydantic")


@pytest.fixture
def app_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Import the API with isolated upload/work directories."""
    monkeypatch.setenv("RLPE_API_TEST_TMP", str(tmp_path))
    for module_name in list(sys.modules):
        if module_name == "rlpe.api.app" or module_name.startswith("rlpe.api."):
            sys.modules.pop(module_name, None)
    import rlpe.api.app as module  # noqa: E402

    return module


@pytest.fixture
def client(app_module):
    from fastapi.testclient import TestClient

    return TestClient(app_module.app)


def _upload(client, options: dict | None = None):
    """Upload a minimal PDF payload through the multipart API."""
    data = {}
    if options is not None:
        data["options"] = json.dumps(options)
    return client.post(
        "/jobs/upload",
        files={"file": ("yolo-surface-test.pdf", b"%PDF-1.4\n", "application/pdf")},
        data=data,
    )


class TestYoloWebSurface:
    def test_job_options_accept_yolo_fields(self, app_module, client, monkeypatch):
        """The multipart JSON options field accepts the YOLO settings."""
        received: list[dict] = []

        def fake_run_job(job_id, pdf_path, options):
            received.append(options)

        monkeypatch.setattr(app_module, "_run_job", fake_run_job)
        response = _upload(
            client,
            {"use_yolo_figures": True, "yolo_model_path": "/tmp/fake.pt"},
        )

        assert response.status_code in (200, 202), response.text
        assert received and received[0]["use_yolo_figures"] is True
        assert received[0]["yolo_model_path"] == "/tmp/fake.pt"

    def test_pipeline_kwargs_forward_yolo(self, app_module, client, monkeypatch):
        """An uploaded job forwards all YOLO fields into PipelineConfig."""
        captured: dict = {}

        class FakeConfig:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.extra = kwargs["extra"]

        class FakePipeline:
            def __init__(self, config, progress_callback=None):
                self.config = config

            def run(self):
                return []

        monkeypatch.setattr(app_module, "PipelineConfig", FakeConfig)
        monkeypatch.setattr(app_module, "RadiolarianPipeline", FakePipeline)
        response = _upload(
            client,
            {
                "use_yolo_figures": True,
                "yolo_model_path": "/tmp/fake.pt",
                "yolo_conf_threshold": 0.31,
                "yolo_iou_threshold": 0.52,
                "yolo_device": "0",
            },
        )

        assert response.status_code in (200, 202), response.text
        assert captured["use_yolo_figures"] is True
        assert captured["yolo_model_path"] == "/tmp/fake.pt"
        assert captured["yolo_conf_threshold"] == 0.31
        assert captured["yolo_iou_threshold"] == 0.52
        assert captured["yolo_device"] == "0"

    def test_job_options_yolo_defaults(self, app_module):
        """YOLO defaults remain disabled and match PipelineConfig defaults."""
        options = app_module.JobOptions()

        assert options.use_yolo_figures is False
        assert options.yolo_model_path == ""
        assert options.yolo_conf_threshold == 0.25
        assert options.yolo_iou_threshold == 0.45
        assert options.yolo_device == "auto"
