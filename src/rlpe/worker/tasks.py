from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import _KNOWN_EXTRA_KEYS, PipelineConfig
from ..pipeline import RadiolarianPipeline

try:
    from celery import Celery

    _HAS_CELERY = True
except Exception:  # pragma: no cover
    _HAS_CELERY = False

if TYPE_CHECKING:
    from celery import Celery as _CeleryT  # noqa: F401


def _build_celery():
    """Construct a Celery app, returning None when celery isn't installed.

    The return type intentionally lacks an explicit annotation: the
    previous ``Celery | None`` annotation evaluated at runtime when
    celery wasn't installed (because ``from __future__ import
    annotations`` is in effect, but the annotation was used by tools
    that materialise it). Skipping the annotation keeps the function
    importable in environments without celery.
    """
    if not _HAS_CELERY:
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
        # Phase 55 audit: validate config_extra against known keys to prevent
        # injection of arbitrary config from untrusted Celery task messages.
        if config_extra:
            unknown = set(config_extra.keys()) - _KNOWN_EXTRA_KEYS
            if unknown:
                raise ValueError(f"Unknown config_extra keys: {sorted(unknown)}")
        cfg = PipelineConfig(
            pdf_dir=Path(pdf_dir),
            work_dir=Path(work_dir),
            extra=config_extra or {},
        )
        pipeline = RadiolarianPipeline(cfg)
        return pipeline.run()

    @celery_app.task(name="rlpe.process_gpu_gemma")
    def process_gpu_gemma(
        pdf_dir: str, work_dir: str, gpu_id: int = 0, config_extra: dict[str, Any] | None = None
    ):
        # Phase 55 audit: validate config_extra against known keys.
        if config_extra:
            unknown = set(config_extra.keys()) - _KNOWN_EXTRA_KEYS
            if unknown:
                raise ValueError(f"Unknown config_extra keys: {sorted(unknown)}")
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
