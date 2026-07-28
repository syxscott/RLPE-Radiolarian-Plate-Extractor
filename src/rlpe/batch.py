from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .pipeline import RadiolarianPipeline
from .utils import ensure_dir, write_jsonl


def run_batch(config: PipelineConfig) -> list[dict[str, Any]]:
    pipeline = RadiolarianPipeline(config)
    return pipeline.run()


def run_batch_parallel(
    config: PipelineConfig, max_workers: int | None = None
) -> list[dict[str, Any]]:
    pdf_files = sorted(config.pdf_dir.glob("*.pdf"))
    if not pdf_files:
        return []

    ensure_dir(config.resolved_output_dir())
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers or config.num_workers) as pool:
        futures = []
        for pdf in pdf_files:
            futures.append(pool.submit(_run_single, config, pdf))
        for fut in as_completed(futures):
            try:
                rows.extend(fut.result())
            except Exception as exc:
                # Log and continue so one crashed worker doesn't kill the whole batch.
                import logging
                logging.getLogger(__name__).error(
                    "Batch worker failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
    write_jsonl(config.manifests_dir() / "matches.jsonl", rows)
    # Produce the same run_output.json bundle as pipeline.run() so downstream
    # consumers (eval scripts, web UI) get a consistent data package.
    from .utils import write_json
    run_output = {
        "rows": rows,
        "n_rows": len(rows),
        "n_papers": len(pdf_files),
        "config": {
            "use_gpu": config.use_gpu,
            "ocr_backend": config.ocr_backend,
            "caption_window": config.caption_window,
            "od_caption_window": config.od_caption_window,
            "use_yolo_figures": config.use_yolo_figures,
            "yolo_model_path": config.yolo_model_path,
        },
    }
    try:
        write_json(config.manifests_dir() / "run_output.json", run_output)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Failed to write run_output.json")
    return rows


def _run_single(config: PipelineConfig, pdf_path: Path) -> list[dict[str, Any]]:
    local_config = PipelineConfig(
        pdf_dir=config.pdf_dir,
        work_dir=config.work_dir,
        output_dir=config.output_dir,
        grobid_url=config.grobid_url,
        use_gpu=config.use_gpu,
        ocr_backend=config.ocr_backend,
        taxon_model=config.taxon_model,
        min_panel_score=config.min_panel_score,
        caption_window=config.caption_window,
        num_workers=1,
        render_dpi=config.render_dpi,
        save_intermediate=config.save_intermediate,
        # audit 2026-07-26: forward od_caption_window + YOLO fields
        # (previously dropped, silently reverting to defaults).
        od_caption_window=config.od_caption_window,
        use_yolo_figures=config.use_yolo_figures,
        yolo_model_path=config.yolo_model_path,
        yolo_conf_threshold=config.yolo_conf_threshold,
        yolo_iou_threshold=config.yolo_iou_threshold,
        yolo_device=config.yolo_device,
        extra=copy.deepcopy(config.extra),
    )
    pipeline = RadiolarianPipeline(local_config)
    return pipeline._process_one_pdf(pdf_path)
