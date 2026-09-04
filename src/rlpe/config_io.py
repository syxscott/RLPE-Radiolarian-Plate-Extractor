from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def save_config(config: PipelineConfig, path: Path) -> None:
    """Atomically persist a sanitised config to ``path``.

    Audit 2026-09-01 CR-3 / BL-34: the previous implementation called
    ``path.write_text(...)`` directly — a mid-write SIGKILL / OOM left
    a truncated config that ``json.loads`` couldn't parse, breaking
    the next CLI startup with a confusing ``JSONDecodeError``. Now we
    write to a sibling temp file and ``os.replace`` atomically so the
    destination is always either the previous version or the new one.

    Also (CR-10 follow-up): the previous ``return bool(value)`` fallback
    in ``_coerce`` is gone, and secrets are removed rather than
    replaced with the literal ``"***REDACTED***"`` — the previous
    behaviour wrote a string-shaped ``***REDACTED***`` value that the
    loader accepted as a valid API key, leading to a confusing
    auth-failed error instead of a clear "missing key" message.
    """
    import os as _os
    import tempfile as _tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    # Redact secrets before persisting. The previous version wrote the
    # full ``extra`` dict (which contains ``MiniMax_api_key`` if the
    # user supplied one inline) to disk, which meant the API key
    # silently leaked into any backup / sync / file-sharing path that
    # picked up the config JSON. Strip the recognised secret fields
    # entirely (rather than substituting ``"***REDACTED***"`` which
    # the loader would have accepted as a valid API-key string).
    secret_keys = {
        "MiniMax_api_key",
        "ANTHROPIC_API_KEY",
        "MiniMax_API_KEY",
        "_MiniMax_external_handler",  # injected by web/API layer, may carry tokens
        # Audit 2026-09-01 CR-17 follow-up: also redact cloud-provider
        # keys that may be carried via ``extra`` when the operator
        # routes MiniMax through AWS Bedrock / Vertex / Azure.
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "AZURE_OPENAI_API_KEY",
        "VERTEX_AI_API_KEY",
    }
    sanitized_extra = {k: v for k, v in (config.extra or {}).items() if k not in secret_keys}
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
        "od_caption_window": config.od_caption_window,
        # audit 2026-07-26: persist YOLO fields so a saved config
        # round-trips (previously load_config read them but save_config
        # never wrote them, so they silently reset to defaults).
        "use_yolo_figures": config.use_yolo_figures,
        "yolo_model_path": config.yolo_model_path,
        "yolo_conf_threshold": config.yolo_conf_threshold,
        "yolo_iou_threshold": config.yolo_iou_threshold,
        "extra": sanitized_extra,
    }
    fd, tmp_path = _tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with _os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            try:
                _os.fsync(f.fileno())
            except OSError:
                # fsync may fail on network filesystems; rename is
                # still atomic so we don't lose data.
                pass
        _os.replace(tmp_path, path)
    except Exception:
        try:
            _os.unlink(tmp_path)
        except OSError:
            pass
        raise


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
            if isinstance(value, str):
                # audit 2026-07-26: bool("0") is True (non-empty str),
                # which would flip a "0"/"false"-meaning flag on. Handle
                # the common string spellings explicitly.
                lv = value.strip().lower()
                if lv in {"true", "1"}:
                    return True
                if lv in {"false", "0", ""}:
                    return False
            return bool(value)
        # Audit 2026-09-01 CR-10: the previous ``return default`` for
        # all unrecognised types silently coerced a JSON ``"save_intermediate": "no"``
        # into the default ``True`` (since ``bool("no")`` is True and
        # the only string branch already returned). Now we never reach
        # this point with an unrecognised bool input — but if we do,
        # fall through to the structured ``return default`` rather
        # than the previous ``return bool(value)`` which would
        # silently flip the user's intent (e.g. a future
        # ``datetime.isoformat()`` would become ``True``).
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
            name,
            value,
            target_type.__name__,
            default,
        )
        return default


def load_config(path: Path) -> PipelineConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"config_io: could not read config from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"config_io: top-level JSON in {path} must be an object, got {type(payload).__name__}"
        )
    return PipelineConfig(
        pdf_dir=Path(payload["pdf_dir"]),
        work_dir=Path(payload["work_dir"]),
        output_dir=Path(payload["output_dir"]) if payload.get("output_dir") else None,
        grobid_url=_coerce(
            "grobid_url",
            payload.get("grobid_url", "http://localhost:8070"),
            "http://localhost:8070",
        ),
        use_gpu=_coerce("use_gpu", payload.get("use_gpu", True), True),
        ocr_backend=_coerce("ocr_backend", payload.get("ocr_backend", "paddleocr"), "paddleocr"),
        taxon_model=_coerce("taxon_model", payload.get("taxon_model", "en_eco"), "en_eco"),
        min_panel_score=_coerce("min_panel_score", payload.get("min_panel_score", 0.8), 0.8),
        caption_window=_coerce("caption_window", payload.get("caption_window", 2), 2),
        num_workers=_coerce("num_workers", payload.get("num_workers", 4), 4),
        render_dpi=_coerce("render_dpi", payload.get("render_dpi", 200), 200),
        save_intermediate=_coerce(
            "save_intermediate", payload.get("save_intermediate", False), False
        ),
        od_caption_window=_coerce("od_caption_window", payload.get("od_caption_window", 5), 5),
        use_yolo_figures=_coerce("use_yolo_figures", payload.get("use_yolo_figures", False), False),
        yolo_model_path=_coerce("yolo_model_path", payload.get("yolo_model_path", ""), ""),
        yolo_conf_threshold=_coerce(
            "yolo_conf_threshold", payload.get("yolo_conf_threshold", 0.25), 0.25
        ),
        yolo_iou_threshold=_coerce(
            "yolo_iou_threshold", payload.get("yolo_iou_threshold", 0.45), 0.45
        ),
        yolo_device=_coerce("yolo_device", payload.get("yolo_device", "auto"), "auto"),
        extra=payload.get("extra", {}) or {},
    )
