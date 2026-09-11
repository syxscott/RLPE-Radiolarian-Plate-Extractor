"""Single-paper batch worker — subprocess isolation for the batch runner.

The batch runner (``RadiolarianPipeline.run``) historically processed
every PDF inside the parent process. That means a hard crash in a
native layer (the recurring PaddleOCR ``SIGSEGV`` on odd caption-band
shapes is the documented example) killed the ENTIRE batch, and every
``--resume`` attempt rewrote ``matches.jsonl`` / ``run_output.json``
with only its own papers — losing earlier papers' rows from the
aggregate.

With ``batch_isolation="subprocess"`` the parent instead spawns one
``python -m rlpe.worker`` process per PDF (see
``RadiolarianPipeline._process_one_pdf_in_subprocess``):

* a native crash takes down at most that ONE paper (the parent records
  an ``_ingestion_worker_crash`` stub row and keeps going);
* the parent collects every paper's rows in memory and writes the
  aggregate once, so ``--resume`` chains stay complete.

The full pipeline configuration is handed over through a 0600 config
JSON inside the work dir (written by the parent via
:func:`dump_worker_config`), including the resolved LLM credentials
that ``save_config`` would normally strip.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rlpe.worker",
        description="Process exactly ONE PDF with a fresh pipeline "
        "(batch subprocess isolation, F19). Internal entry point.",
    )
    parser.add_argument("--config", required=True, help="Path to the worker config JSON")
    parser.add_argument("--pdf", required=True, help="Path to the PDF to process")
    parser.add_argument("--out", required=True, help="Path to write the rows JSON array")
    args = parser.parse_args(argv)

    from rlpe.config_io import load_worker_config
    from rlpe.llm_usage import collect_llm_usage, merge_llm_usage
    from rlpe.pipeline import RadiolarianPipeline

    config = load_worker_config(Path(args.config))
    pipe = RadiolarianPipeline(config)
    rows = pipe._process_one_pdf(Path(args.pdf))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    # 2026-09-11 usage-reporting fix: with batch_isolation="subprocess"
    # every LLM call happens inside THIS worker process, and the
    # counters die with it — the parent's own backend stays untouched
    # and used to write an all-zeros llm_usage.json. Persist this
    # paper's usage to a sidecar next to the rows file; the parent
    # aggregates the sidecars (see
    # RadiolarianPipeline._process_one_pdf_in_subprocess).
    try:
        summaries = [collect_llm_usage(getattr(pipe, "gemma_runtime", None))]
        engine = getattr(pipe, "semantic_engine", None)
        if engine is not None:
            summaries.append(collect_llm_usage(engine))
        usage = merge_llm_usage(summaries)
        if usage:
            usage_path = out_path.with_suffix(".llm_usage.json")
            usage_path.write_text(json.dumps(usage, ensure_ascii=False), encoding="utf-8")
    except Exception:  # usage reporting must never fail the worker
        import logging

        logging.getLogger(__name__).debug(
            "worker llm-usage sidecar write failed", exc_info=True
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
