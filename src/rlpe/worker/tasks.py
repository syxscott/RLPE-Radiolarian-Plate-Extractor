from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..config import PipelineConfig
from ..pipeline import RadiolarianPipeline

try:
    from celery import Celery
except Exception:  # pragma: no cover
    Celery = None


def _build_celery() -> Celery | None:
    if Celery is None:
        return None
    broker = os.environ.get("RLPE_CELERY_BROKER", "redis://localhost:6379/0")
    backend = os.environ.get("RLPE_CELERY_BACKEND", broker)
    app = Celery("rlpe", broker=broker, backend=backend)
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_track_started=True,
    )
    return app


celery_app = _build_celery()


if celery_app is not None:

    @celery_app.task(name="rlpe.process_pdf_batch")
    def process_pdf_batch(pdf_dir: str, work_dir: str, config_extra: dict[str, Any] | None = None):
        cfg = PipelineConfig(
            pdf_dir=Path(pdf_dir),
            work_dir=Path(work_dir),
            extra=config_extra or {},
        )
        pipeline = RadiolarianPipeline(cfg)
        return pipeline.run()

    @celery_app.task(name="rlpe.process_gpu_gemma")
    def process_gpu_gemma(pdf_dir: str, work_dir: str, gpu_id: int = 0, config_extra: dict[str, Any] | None = None):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        extra = dict(config_extra or {})
        extra["use_gemma4"] = True
        cfg = PipelineConfig(
            pdf_dir=Path(pdf_dir),
            work_dir=Path(work_dir),
            extra=extra,
        )
        pipeline = RadiolarianPipeline(cfg)
        return pipeline.run()
