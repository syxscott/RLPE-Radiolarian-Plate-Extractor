from __future__ import annotations

import copy
import dataclasses
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .pipeline import RadiolarianPipeline
from .utils import ensure_dir, write_json, write_jsonl


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
    # Audit 2026-09-01 CR-5 / BL-39: track which PDFs failed so the
    # run_output.json downstream consumers (eval scripts, web UI) can
    # distinguish "n_rows == 0 because all workers crashed" from
    # "n_rows == 0 because the PDFs had no species". Previously the
    # batch loop logged the exception and moved on with no record;
    # the eval script then reported F1 on the empty row set as if it
    # were a real result.
    failed_pdfs: list[dict[str, str]] = []
    # Audit 2026-09-01 (mj-bench 23): ``as_completed`` returns in
    # completion order, which is non-deterministic across runs and
    # pollutes cross-paper comparison. Keep the original (alphabetical)
    # PDF order so the eval script can stably join on paper_id.
    pdf_to_future: dict[Path, Any] = {}
    with ProcessPoolExecutor(max_workers=max_workers or config.num_workers) as pool:
        for pdf in pdf_files:
            pdf_to_future[pdf] = pool.submit(_run_single, config, pdf)
        for pdf in pdf_files:
            fut = pdf_to_future[pdf]
            try:
                rows.extend(fut.result())
            except Exception as exc:
                # Log and continue so one crashed worker doesn't kill the whole batch.
                logging.getLogger(__name__).error(
                    "Batch worker failed for %s: %s: %s",
                    pdf.name,
                    type(exc).__name__,
                    exc,
                )
                failed_pdfs.append(
                    {"pdf": pdf.name, "error_type": type(exc).__name__, "error": str(exc)}
                )
    write_jsonl(config.manifests_dir() / "matches.jsonl", rows)
    # Produce the same run_output.json bundle as pipeline.run() so downstream
    # consumers (eval scripts, web UI) get a consistent data package.
    run_output = {
        "rows": rows,
        "n_rows": len(rows),
        "n_papers": len(pdf_files),
        "n_failed": len(failed_pdfs),
        "failed_pdfs": failed_pdfs,
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
        logging.getLogger(__name__).exception("Failed to write run_output.json")
    return rows


def _run_single(config: PipelineConfig, pdf_path: Path) -> list[dict[str, Any]]:
    # Audit 2026-09-01 BL-35 / CR-4: the previous implementation
    # explicitly listed every ``PipelineConfig`` field in the
    # constructor — but the dataclass gained 7+ new fields
    # (``m3_per_panel`` / ``m3_stage_6`` / ``use_m3_stage3`` /
    # ``m3_multi_plate_enrich`` / ``m3_temperature`` /
    # ``m3_thinking_budget`` / ...) and each silently fell back to
    # its default when invoked through the batch path. Users who set
    # ``--m3-per-panel`` on the CLI saw the per-panel M3 path *only*
    # when they used the single-PDF ``run()``; the batch mode ignored
    # the flag entirely. Replace the manual field list with
    # ``dataclasses.replace`` so future field additions propagate
    # automatically.
    local_config = dataclasses.replace(
        config,
        # Per-worker overrides: force single-threaded SAM2/YOLO (the
        # SAM2 lock we just added in segmentation.py makes this safe
        # at >1, but ``ProcessPoolExecutor`` already isolates workers
        # in their own processes so we save the per-worker pool
        # spin-up cost by pinning num_workers=1).
        num_workers=1,
        # Avoid accidental cross-process mutation of the extra dict.
        extra=copy.deepcopy(config.extra),
    )
    pipeline = RadiolarianPipeline(local_config)
    return pipeline._process_one_pdf(pdf_path)
