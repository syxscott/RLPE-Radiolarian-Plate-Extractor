from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .pipeline import RadiolarianPipeline
from .utils import ensure_dir, write_jsonl


def run_batch(config: PipelineConfig) -> list[dict[str, Any]]:
    pipeline = RadiolarianPipeline(config)
    return pipeline.run()


def run_batch_parallel(config: PipelineConfig, max_workers: int | None = None) -> list[dict[str, Any]]:
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
            rows.extend(fut.result())
    write_jsonl(config.manifests_dir() / "matches.jsonl", rows)
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
        extra=dict(config.extra),
    )
    pipeline = RadiolarianPipeline(local_config)
    return pipeline._process_one_pdf(pdf_path)
