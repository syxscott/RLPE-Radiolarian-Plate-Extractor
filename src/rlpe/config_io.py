from __future__ import annotations

import json
from pathlib import Path

from .config import PipelineConfig


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


def load_config(path: Path) -> PipelineConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PipelineConfig(
        pdf_dir=Path(payload["pdf_dir"]),
        work_dir=Path(payload["work_dir"]),
        output_dir=Path(payload["output_dir"]) if payload.get("output_dir") else None,
        grobid_url=payload.get("grobid_url", "http://localhost:8070"),
        use_gpu=payload.get("use_gpu", True),
        ocr_backend=payload.get("ocr_backend", "paddleocr"),
        taxon_model=payload.get("taxon_model", "en_eco"),
        min_panel_score=payload.get("min_panel_score", 0.8),
        caption_window=payload.get("caption_window", 2),
        num_workers=payload.get("num_workers", 4),
        render_dpi=payload.get("render_dpi", 200),
        save_intermediate=payload.get("save_intermediate", True),
        extra=payload.get("extra", {}),
    )
