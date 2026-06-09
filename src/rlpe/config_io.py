from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def save_config(config: PipelineConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pdf_dir": str(config.pdf_dir),
        "work_dir": str(config.work_dir),
        "output_dir": str(config.output_dir) if config.output_dir else None,
        "grobid_url": config.grobid_url,
        "use_gpu": config.use_gpu,
        "ocr_backend": config.ocr_backend,
        "taxon_model": config.taxon_model,
        "min_panel_score": config.min_panel_score,
        "caption_window": config.caption_window,
        "num_workers": config.num_workers,
        "render_dpi": config.render_dpi,
        "save_intermediate": config.save_intermediate,
        "extra": config.extra,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# Coercion defaults — match ``PipelineConfig``'s field types so a JSON
# file written by an older version of the tool (where the field
# default may have changed) still loads cleanly.
def _coerce(name: str, value: Any, default: Any) -> Any:
    """Coerce a loaded JSON value to the expected Python type.

    JSON itself doesn't enforce types (a string ``"0.8"`` is valid
    JSON), and previous versions of the saved-config format used
    different defaults for some fields. Passing a string where the
    field is typed as ``float`` would otherwise raise deep inside
    the pipeline with a confusing traceback; we coerce here so the
    error message — if any — points at the file and field.
    """
    if value is None:
        return default
    target_type = type(default)
    if target_type is bool:
        # ``bool`` is a subclass of ``int`` so check it first.
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float, str)):
            if isinstance(value, str) and value.lower() in {"true", "false"}:
                return value.lower() == "true"
            return bool(value)
        return default
    if target_type is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    if target_type is float and isinstance(value, (int, float)):
        return float(value)
    if target_type is str and isinstance(value, str):
        return value
    try:
        return target_type(value)
    except (TypeError, ValueError):
        logger.warning(
            "config_io: could not coerce %s=%r to %s; using default %r",
            name, value, target_type.__name__, default,
        )
        return default


def load_config(path: Path) -> PipelineConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"config_io: could not read config from {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"config_io: top-level JSON in {path} must be an object, "
            f"got {type(payload).__name__}"
        )
    return PipelineConfig(
        pdf_dir=Path(payload["pdf_dir"]),
        work_dir=Path(payload["work_dir"]),
        output_dir=Path(payload["output_dir"]) if payload.get("output_dir") else None,
        grobid_url=_coerce("grobid_url", payload.get("grobid_url", "http://localhost:8070"), "http://localhost:8070"),
        use_gpu=_coerce("use_gpu", payload.get("use_gpu", True), True),
        ocr_backend=_coerce("ocr_backend", payload.get("ocr_backend", "paddleocr"), "paddleocr"),
        taxon_model=_coerce("taxon_model", payload.get("taxon_model", "en_eco"), "en_eco"),
        min_panel_score=_coerce("min_panel_score", payload.get("min_panel_score", 0.8), 0.8),
        caption_window=_coerce("caption_window", payload.get("caption_window", 2), 2),
        num_workers=_coerce("num_workers", payload.get("num_workers", 4), 4),
        render_dpi=_coerce("render_dpi", payload.get("render_dpi", 200), 200),
        save_intermediate=_coerce("save_intermediate", payload.get("save_intermediate", True), True),
        extra=payload.get("extra", {}) or {},
    )
