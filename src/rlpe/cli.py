from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig
from .pipeline import RadiolarianPipeline
from .utils import ensure_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Radiolarian plate extraction pipeline")
    p.add_argument("--pdf-dir", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--grobid-url", type=str, default="http://localhost:8070")
    p.add_argument("--ocr-backend", type=str, default="paddleocr", choices=["paddleocr", "easyocr"])
    p.add_argument("--taxon-model", type=str, default="en_eco")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--min-panel-score", type=float, default=0.8)
    p.add_argument("--render-dpi", type=int, default=200)
    p.add_argument("--save-intermediate", action="store_true")
    p.add_argument("--sam2-checkpoint", type=str, default=None)
    p.add_argument("--sam2-model-cfg", type=str, default=None)
    p.add_argument("--sam2-grid-size", type=int, default=6)
    p.add_argument("--sam2-max-point-prompts", type=int, default=48)
    p.add_argument("--sam2-max-box-prompts", type=int, default=24)
    p.add_argument("--use-neural-matcher", action="store_true")
    p.add_argument("--matcher-checkpoint-path", type=str, default=None)
    p.add_argument("--taxon-hf-model-path", type=str, default=None)
    p.add_argument("--taxon-lexicon-path", type=str, default=None)
    p.add_argument("--use-gemma4", action="store_true")
    p.add_argument("--llm-backend", type=str, default="llamacpp", choices=["transformers", "ollama", "llamacpp", "llama.cpp", "llama_cpp"])
    p.add_argument("--gemma-model-path", type=str, default=None)
    p.add_argument("--llama-model", type=str, default=None)
    p.add_argument("--llama-host", type=str, default="http://127.0.0.1:8080")
    p.add_argument("--llama-timeout-sec", type=int, default=120)
    p.add_argument("--ollama-model", type=str, default=None)
    p.add_argument("--ollama-host", type=str, default="http://127.0.0.1:11434")
    p.add_argument("--gemma-timeout-sec", type=int, default=120)
    p.add_argument("--gemma-conf-threshold", type=float, default=0.70)
    p.add_argument("--gemma-prompt-lang", type=str, default="zh", choices=["zh", "en"])
    p.add_argument("--gemma-no-4bit", action="store_true")
    p.add_argument("--gemma-no-bfloat16", action="store_true")
    p.add_argument("--use-geology-llm", action="store_true")
    p.add_argument("--export-csv", type=Path, default=None)
    p.add_argument("--export-json", type=Path, default=None)
    p.add_argument("--export-jsonl", type=Path, default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = PipelineConfig(
        pdf_dir=args.pdf_dir,
        work_dir=args.work_dir,
        output_dir=args.output_dir,
        grobid_url=args.grobid_url,
        ocr_backend=args.ocr_backend,
        taxon_model=args.taxon_model,
        num_workers=args.num_workers,
        min_panel_score=args.min_panel_score,
        render_dpi=args.render_dpi,
        save_intermediate=args.save_intermediate,
        extra={
            "sam2_checkpoint": args.sam2_checkpoint,
            "sam2_model_cfg": args.sam2_model_cfg,
            "sam2_grid_size": args.sam2_grid_size,
            "sam2_max_point_prompts": args.sam2_max_point_prompts,
            "sam2_max_box_prompts": args.sam2_max_box_prompts,
            "use_neural_matcher": args.use_neural_matcher,
            "matcher_checkpoint_path": args.matcher_checkpoint_path,
            "taxon_hf_model_path": args.taxon_hf_model_path,
            "taxon_lexicon_path": args.taxon_lexicon_path,
            "use_gemma4": args.use_gemma4,
            "llm_backend": args.llm_backend,
            "gemma_model_path": args.gemma_model_path,
            "llama_model": args.llama_model,
            "llama_host": args.llama_host,
            "llama_timeout_sec": args.llama_timeout_sec,
            "ollama_model": args.ollama_model,
            "ollama_host": args.ollama_host,
            "gemma_timeout_sec": args.gemma_timeout_sec,
            "gemma_conf_threshold": args.gemma_conf_threshold,
            "gemma_prompt_lang": args.gemma_prompt_lang,
            "gemma_use_4bit": not args.gemma_no_4bit,
            "gemma_bfloat16": not args.gemma_no_bfloat16,
            "gemma_device_map": "auto",
            "use_geology_llm": args.use_geology_llm,
        },
    )
    ensure_dir(cfg.work_dir)
    pipeline = RadiolarianPipeline(cfg)
    rows = pipeline.run()

    if args.export_csv:
        from .export import export_csv

        export_csv(rows, args.export_csv)
    if args.export_json:
        from .export import export_json

        export_json(rows, args.export_json)
    if args.export_jsonl:
        from .export import export_jsonl

        export_jsonl(rows, args.export_jsonl)

    print(f"processed={len(list(cfg.pdf_dir.glob('*.pdf')))} rows={len(rows)} output={cfg.resolved_output_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
