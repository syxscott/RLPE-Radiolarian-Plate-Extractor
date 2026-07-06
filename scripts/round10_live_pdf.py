"""Single-PDF driver that runs the full RadiolarianPipeline with
MiniMax M3 as the LLM backend, bypassing the multi-file batch scan.

Used by Round 10 live smoke tests: we don't want to re-scan the
whole 100+ PDF corpus; we want to point the pipeline at ONE specific
file and report what it extracted.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from dotenv import load_dotenv

load_dotenv(_REPO / ".env", override=False)

from rlpe.config import PipelineConfig
from rlpe.pipeline import RadiolarianPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("round10_live")


def _prepare_workspace(pdf: Path, pdf_dir: Path) -> Path:
    """Copy the PDF into ``pdf_dir`` so PipelineConfig's non-recursive
    ``pdf_dir.glob('*.pdf')`` picks it up exactly once.
    """
    pdf_dir.mkdir(parents=True, exist_ok=True)
    target = pdf_dir / pdf.name
    if target.resolve() != pdf.resolve():
        shutil.copy2(pdf, target)
    return target


def run_one(pdf: Path, work_root: Path, *, with_llm: bool) -> dict:
    pdf_dir = work_root / "_input"
    work_dir = work_root / "run"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    _prepare_workspace(pdf, pdf_dir)

    cfg = PipelineConfig(
        pdf_dir=pdf_dir,
        work_dir=work_dir,
        grobid_url="http://localhost:8070",
        ocr_backend="paddleocr",
        use_gpu=False,
        save_intermediate=True,
        extra={
            "use_opendataloader": True,
            "use_gemma4": with_llm,
            "llm_backend": "minimax" if with_llm else "transformers",
            "gemma_conf_threshold": 0.7,
            "MiniMax_enable_thinking": True,
            "MiniMax_thinking_budget_tokens": 1024,
            "MiniMax_fallback_default": "rules",
        },
    )

    pipeline = RadiolarianPipeline(cfg, progress_callback=lambda *a: None)
    t0 = time.time()
    rows = pipeline.run()
    elapsed = time.time() - t0
    summary = {
        "pdf": pdf.name,
        "rows": rows,
        "elapsed_s": round(elapsed, 1),
        "manifest": str(work_dir / "output" / "manifests" / "matches.jsonl"),
        "run_output": str(work_dir / "output" / "manifests" / "run_output.json"),
    }
    return summary


def summarise(rows: list[dict]) -> dict:
    n_total = len(rows)
    n_with_species = sum(1 for r in rows if r.get("species"))
    n_with_panel_id = sum(1 for r in rows if r.get("panel_id"))
    # Source breakdown
    src_counter: dict[str, int] = {}
    for r in rows:
        md = r.get("metadata") or {}
        src = md.get("extraction_method") or "unknown"
        src_counter[src] = src_counter.get(src, 0) + 1
    return {
        "n_total": n_total,
        "n_with_species": n_with_species,
        "n_with_panel_id": n_with_panel_id,
        "by_method": src_counter,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--work-root", type=Path, default=_REPO / "work" / "round10_live")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM (rules-only path)")
    args = parser.parse_args()

    pdf = args.pdf.resolve()
    if not pdf.exists():
        print(f"PDF not found: {pdf}")
        return 2
    args.work_root.mkdir(parents=True, exist_ok=True)

    print(f"PDF: {pdf.name}")
    print(f"Work root: {args.work_root}")
    print(f"LLM enabled: {not args.no_llm}")
    print("---")

    summary = run_one(pdf, args.work_root, with_llm=not args.no_llm)
    stats = summarise(summary["rows"])
    print("---")
    print(f"Rows emitted:        {stats['n_total']}")
    print(f"With panel_id:       {stats['n_with_panel_id']}")
    print(f"With species:        {stats['n_with_species']}")
    print("By extraction_method:")
    for k, v in stats["by_method"].items():
        print(f"  {k}: {v}")
    print(f"Elapsed:             {summary['elapsed_s']}s")
    print(f"Manifest:            {summary['manifest']}")
    print(f"run_output.json:     {summary['run_output']}")

    # Save summary alongside
    out_json = args.work_root / "summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pdf": pdf.name,
                "stats": stats,
                "elapsed_s": summary["elapsed_s"],
                "rows_sample": [
                    {
                        k: r.get(k)
                        for k in (
                            "paper_id",
                            "figure_id",
                            "panel_id",
                            "species",
                            "confidence",
                            "label_text",
                        )
                    }
                    for r in summary["rows"][:20]
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    print(f"\nSummary saved to: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
