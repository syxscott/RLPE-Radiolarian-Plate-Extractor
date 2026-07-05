from __future__ import annotations

import argparse

# Load .env from the project root so MiniMax API keys, model names, etc.
# are available without exporting manually.  No-op if python-dotenv is
# not installed or the file is missing.
#
# Precedence policy: for the project's MiniMax-related keys
# (ANTHROPIC_*, MiniMax_*) the .env file wins over any pre-existing OS
# env var. This matters because tools like Claude Code set
# ``ANTHROPIC_BASE_URL`` globally for their own backend (e.g.
# ``ark.cn-beijing.volces.com``); without the project-level override,
# RLPE would silently connect to the wrong endpoint. For all other
# keys (PATH, HTTP_PROXY, ...) the OS env remains authoritative.
# ``RLPE_FORCE_ENV_OVERRIDE=1`` flips the behaviour to "always
# override" as an escape hatch for unusual setups.
import os as _os
from pathlib import Path

try:
    from dotenv import dotenv_values, find_dotenv, load_dotenv

    _env_path = find_dotenv(usecwd=True) or str(Path(__file__).resolve().parents[2] / ".env")
    if _env_path and Path(_env_path).exists():
        # First, do the standard non-override load so unset keys come in.
        load_dotenv(_env_path, override=False)
        # Then selectively override the project's reserved keys.
        _force_override = _os.environ.get("RLPE_FORCE_ENV_OVERRIDE") == "1"
        _project_keys = {
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_MODEL",
            "MiniMax_API_KEY",
            "MiniMax_MODEL",
            "MiniMax_BASE_URL",
        }
        for k, v in (dotenv_values(_env_path) or {}).items():
            if v is None:
                continue
            if _force_override or k in _project_keys:
                _os.environ[k] = v
except ImportError:
    pass

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
    p.add_argument(
        "--use-gpu",
        action="store_true",
        default=None,
        help="Enable GPU for OCR and neural modules. "
        "Default: auto-detect (True if CUDA available, else False).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of PDFs to process in parallel. Clamped to [1, 32]: "
        "values above 32 saturate GROBID / OCR / CUDA long before "
        "they help throughput, and 0 would crash ThreadPoolExecutor.",
    )
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
    p.add_argument(
        "--llm-backend",
        type=str,
        default="llamacpp",
        choices=[
            "transformers",
            "ollama",
            "llamacpp",
            "llama.cpp",
            "llama_cpp",
            "MiniMax",
            "MiniMax-m3",
            "minimax",
        ],
    )
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
    # MiniMax M3 API parameters
    p.add_argument(
        "--MiniMax-api-key",
        type=str,
        default=None,
        help="MiniMax subscription key (or set ANTHROPIC_API_KEY env)",
    )
    p.add_argument("--MiniMax-endpoint", type=str, default="https://api.minimaxi.com/anthropic")
    p.add_argument("--MiniMax-model", type=str, default="MiniMax-M3")
    p.add_argument("--MiniMax-max-concurrent", type=int, default=8)
    p.add_argument("--MiniMax-timeout-sec", type=int, default=120)
    p.add_argument("--MiniMax-max-retries", type=int, default=3)
    p.add_argument(
        "--MiniMax-no-thinking", action="store_true", help="Disable extended thinking (default: ON)"
    )
    p.add_argument("--MiniMax-thinking-budget", type=int, default=1024)
    p.add_argument("--MiniMax-max-output-tokens", type=int, default=2048)
    p.add_argument(
        "--MiniMax-fallback-default",
        type=str,
        default="rules",
        choices=["gemma4", "rules", "stop", "retry"],
        help="Headless fallback when --no-interactive",
    )
    p.add_argument(
        "--MiniMax-interactive",
        action="store_true",
        help="Enable interactive popup prompt on API error (CLI)",
    )
    p.add_argument(
        "--data-outbound-policy",
        type=str,
        default="api_redacted",
        choices=["api_full", "api_redacted", "local_only"],
        help="What data is sent to the LLM backend. Defaults to "
        "api_redacted (caption text + plate region; sensitive fields "
        "like raw PDF bytes are stripped before sending). Override "
        "with api_full to send the full PDF text, or local_only to "
        "skip remote LLM calls entirely.",
    )
    p.add_argument("--use-geology-llm", action="store_true")
    p.add_argument(
        "--use-geo-vision",
        action="store_true",
        help="Enable multi-modal MiniMax-M3 vision extraction of "
        "geology fields (lithology, formation, country, Ma, biozone) "
        "from stratigraphic column / litholog / paleogeographic-map "
        "/ range-chart figures. Off by default (avoids M3 API cost).",
    )
    p.add_argument(
        "--geo-vision-figure-types",
        default=None,
        help="Comma-separated figure-type allowlist for geo-vision. "
        "Default: strat_column,litholog_column,paleogeographic_map,range_chart. "
        "Use e.g. 'range_chart' alone to focus on species distribution.",
    )
    p.add_argument(
        "--use-m3-stage3",
        action="store_true",
        help="Enable M3 Stage 3 panel bbox detection + crop enrichment. "
        "Off by default; requires MiniMax API access.",
    )
    p.add_argument(
        "--m3-multi-plate-enrich",
        action="store_true",
        help="Round 7 second-pass M3 multi-plate enrichment. Fires when "
        "OD dropped a plate's caption-image pairing (e.g. Bandini 2011 "
        "Plate 7-9): asks M3 to extract the panel list from the plate "
        "image + page-level caption. Off by default (avoids M3 API cost).",
    )
    # ---- OpenDataLoader PDF parser (replaces GROBID) -----------------------
    p.add_argument(
        "--use-opendataloader",
        action="store_true",
        help="Use OpenDataLoader-pdf for figure/caption extraction "
        "(no GROBID server needed). Default off.",
    )
    # ---- M3 5-stage engine -------------------------------------------------
    p.add_argument(
        "--m3-enhanced-mode",
        action="store_true",
        default=None,
        help="Enable M3 5-stage semantic engine (default: ON for MiniMax backend)",
    )
    p.add_argument(
        "--m3-disable-stage",
        type=int,
        action="append",
        default=[],
        choices=[1, 2, 3, 4, 5],
        help="Disable a specific M3 stage (1=caption, 2=classify, 3=segment, 4=match, 5=critique). "
        "Can be passed multiple times.",
    )
    p.add_argument(
        "--m3-match-samples",
        type=int,
        default=1,
        help="Number of self-consistency samples for stage 4 (default 1)",
    )
    p.add_argument(
        "--m3-diagnostic-dir",
        type=str,
        default=None,
        help="Dump every M3 call (system prompt + image + result) to this directory for debugging.",
    )
    p.add_argument(
        "--m3-retry-without-thinking",
        dest="m3_retry_without_thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If a M3 call returns empty, retry once with extended "
        "thinking disabled (default: ON). Disable on slow or "
        "constrained backends where the second call is too "
        "expensive to be worth the chance of recovery.",
    )
    p.add_argument(
        "--m3-skip-match-on-empty-caption",
        dest="m3_skip_match_on_empty_caption",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip M3 stage 4 (panel matching) when the caption "
        "parser returned no (label->species) pairs (default: "
        "ON). Disable if you want M3 to attempt visual-only "
        "matching for figures with no caption parseable "
        "structure.",
    )
    # ---- Paleobiology Database (opt-in) -------------------------------------
    p.add_argument(
        "--use-paleodb",
        action="store_true",
        help="Look up matched species against the Paleobiology Database "
        "(taxonomy + occurrence records). Off by default.",
    )
    p.add_argument(
        "--paleodb-max-occurrences",
        type=int,
        default=25,
        help="Max occurrence records per species (default 25).",
    )
    p.add_argument(
        "--paleodb-endpoint",
        type=str,
        default=None,
        help="PBDB API base URL (default https://paleobiodb.org/data1.2).",
    )
    p.add_argument(
        "--paleodb-cache-dir",
        type=str,
        default=None,
        help="Directory for PBDB JSON cache (default ~/.cache/rlpe/paleodb).",
    )
    p.add_argument(
        "--paleodb-offline",
        action="store_true",
        help="Never make network calls to PBDB (cache-only).",
    )
    p.add_argument("--export-csv", type=Path, default=None)
    p.add_argument("--export-json", type=Path, default=None)
    p.add_argument("--export-jsonl", type=Path, default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    # Clamp --num-workers to a sane range. ThreadPoolExecutor requires
    # ``max_workers >= 1``; values above ~32 saturate the OCR / SAM2 /
    # GROBID stack long before they help throughput. A user typo (e.g.
    # ``--num-workers 0``) would otherwise crash the pool at submit
    # time. Silently clamping matches what most CLI tools do.
    args.num_workers = max(1, min(32, int(args.num_workers)))
    # Resolve --use-gpu: explicit flag wins, else auto-detect CUDA.
    if args.use_gpu is None:
        try:
            import torch

            use_gpu_flag = bool(torch.cuda.is_available())
        except ImportError:
            use_gpu_flag = False
    else:
        use_gpu_flag = bool(args.use_gpu)

    cfg = PipelineConfig(
        pdf_dir=args.pdf_dir,
        work_dir=args.work_dir,
        output_dir=args.output_dir,
        grobid_url=args.grobid_url,
        ocr_backend=args.ocr_backend,
        taxon_model=args.taxon_model,
        use_gpu=use_gpu_flag,
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
            "MiniMax_api_key": args.MiniMax_api_key,
            "MiniMax_endpoint": args.MiniMax_endpoint,
            "MiniMax_model": args.MiniMax_model,
            "MiniMax_max_concurrent": args.MiniMax_max_concurrent,
            "MiniMax_timeout_sec": args.MiniMax_timeout_sec,
            "MiniMax_max_retries": args.MiniMax_max_retries,
            "MiniMax_enable_thinking": not args.MiniMax_no_thinking,
            "MiniMax_thinking_budget_tokens": args.MiniMax_thinking_budget,
            "MiniMax_max_output_tokens": args.MiniMax_max_output_tokens,
            "MiniMax_fallback_default": args.MiniMax_fallback_default,
            "MiniMax_interactive": args.MiniMax_interactive,
            "data_outbound_policy": args.data_outbound_policy,
            "use_geology_llm": args.use_geology_llm,
            "use_geo_vision": args.use_geo_vision,
            "use_m3_stage3": args.use_m3_stage3,
            "m3_multi_plate_enrich": args.m3_multi_plate_enrich,
            "geo_vision_figure_types": (
                [t.strip() for t in args.geo_vision_figure_types.split(",") if t.strip()]
                if args.geo_vision_figure_types
                else None
            ),
            "use_opendataloader": args.use_opendataloader,
            "use_paleodb": args.use_paleodb,
            "paleodb_max_occurrences": args.paleodb_max_occurrences,
            "paleodb_endpoint": args.paleodb_endpoint,
            "paleodb_cache_dir": args.paleodb_cache_dir,
            "paleodb_offline": args.paleodb_offline,
        },
    )
    # Inject M3 engine config. We only set ``m3_enhanced_mode`` if the user
    # passed the flag explicitly; default-ON behavior lives in pipeline.py.
    if args.m3_enhanced_mode is not None:
        cfg.extra["m3_enhanced_mode"] = bool(args.m3_enhanced_mode)
    for n in args.m3_disable_stage or []:
        cfg.extra[f"m3_stage_{n}"] = False
    if args.m3_match_samples:
        cfg.extra["m3_match_samples"] = int(args.m3_match_samples)
    if args.m3_diagnostic_dir:
        cfg.extra["m3_diagnostic_dir"] = str(args.m3_diagnostic_dir)
    if args.m3_retry_without_thinking is not None:
        cfg.extra["m3_retry_without_thinking"] = bool(args.m3_retry_without_thinking)
    if args.m3_skip_match_on_empty_caption is not None:
        cfg.extra["m3_skip_match_on_empty_caption"] = bool(args.m3_skip_match_on_empty_caption)
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

    print(
        f"processed={len(list(cfg.pdf_dir.glob('*.pdf')))} rows={len(rows)} output={cfg.resolved_output_dir()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
