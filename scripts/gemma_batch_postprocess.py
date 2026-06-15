from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rlpe.gemma_postprocess import (
    GemmaRuntime,
    batch_gemma_postprocess_rows,
    load_gemma4_llamacpp,
    load_gemma4_model,
    load_gemma4_ollama,
)
from rlpe.llm_backends import (
    FallbackHandler,
    build_MiniMax_backend_from_env_or_config,
    cli_fallback_prompt,
)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply Gemma4 postprocess to RLPE rows.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--backend", type=str, default="llamacpp", choices=["transformers", "ollama", "llamacpp", "llama.cpp", "llama_cpp", "MiniMax", "minimax"])
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--llama-model", type=str, default=None)
    parser.add_argument("--llama-host", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--ollama-model", type=str, default=None)
    parser.add_argument("--ollama-host", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--conf-threshold", type=float, default=0.70)
    parser.add_argument("--prompt-lang", type=str, default="zh", choices=["zh", "en"])
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--no-bfloat16", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=120)
    # MiniMax M3 API arguments
    parser.add_argument("--MiniMax-api-key", type=str, default=None)
    parser.add_argument("--MiniMax-endpoint", type=str, default="https://api.minimaxi.com/anthropic")
    parser.add_argument("--MiniMax-model", type=str, default="MiniMax-M3")
    parser.add_argument("--MiniMax-max-concurrent", type=int, default=8)
    parser.add_argument("--MiniMax-max-retries", type=int, default=3)
    parser.add_argument("--MiniMax-thinking-budget", type=int, default=1024)
    parser.add_argument("--MiniMax-no-thinking", action="store_true")
    parser.add_argument("--MiniMax-interactive", action="store_true",
                        help="Prompt user (CLI) on API errors")
    parser.add_argument("--MiniMax-fallback-default", type=str, default="rules",
                        choices=["gemma4", "rules", "stop", "retry"])
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    if args.backend in {"MiniMax", "minimax"}:
        mini_extra = {
            "MiniMax_api_key": args.MiniMax_api_key,
            "MiniMax_endpoint": args.MiniMax_endpoint,
            "MiniMax_model": args.MiniMax_model,
            "MiniMax_max_concurrent": args.MiniMax_max_concurrent,
            "MiniMax_max_retries": args.MiniMax_max_retries,
            "MiniMax_thinking_budget_tokens": args.MiniMax_thinking_budget,
            "MiniMax_enable_thinking": not args.MiniMax_no_thinking,
        }
        mini_backend = build_MiniMax_backend_from_env_or_config(mini_extra)
        # Build the handler for cost reporting; the actual fallback decision
        # in this batch script is made by inspecting the result rows below.
        _handler = FallbackHandler(default_action=args.MiniMax_fallback_default)
        if args.MiniMax_interactive:
            _handler.on_error = cli_fallback_prompt
        runtime = GemmaRuntime(backend=mini_backend, backend_name="MiniMax")
    elif args.backend in {"llamacpp", "llama.cpp", "llama_cpp"}:
        runtime = load_gemma4_llamacpp(
            host=args.llama_host,
            model_name=args.llama_model or args.model_path or args.ollama_model,
            timeout_sec=args.timeout_sec,
        )
    elif args.backend == "ollama":
        runtime = load_gemma4_ollama(
            model_name=args.ollama_model or args.model_path or "gemma4",
            host=args.ollama_host,
            timeout_sec=args.timeout_sec,
        )
    else:
        if not args.model_path:
            raise ValueError("--model-path is required when backend=transformers")
        runtime = load_gemma4_model(
            model_path=args.model_path,
            use_4bit=args.use_4bit,
            bfloat16=not args.no_bfloat16,
            device_map="auto",
        )
    enhanced = batch_gemma_postprocess_rows(
        runtime=runtime,
        rows=rows,
        conf_threshold=args.conf_threshold,
        prompt_lang=args.prompt_lang,
    )

    # ----- MiniMax fallback detection (batch path) -----
    # batch_gemma_postprocess_rows does NOT raise on API errors; it marks the
    # row with gemma_error / gemma_fallback. Here we surface them and (if
    # --MiniMax-interactive) ask the user what to do.
    if args.backend in {"MiniMax", "minimax"} and args.MiniMax_interactive:
        err_rows = [r for r in enhanced if r.get("gemma_error") or r.get("gemma_fallback")]
        if err_rows:
            first = err_rows[0]
            err_info = {
                "error": first.get("gemma_error", "unknown"),
                "error_type": first.get("gemma_error_type", "MiniMaxAPIError"),
                "context": f"batch rows_with_errors={len(err_rows)} of {len(enhanced)}",
            }
            try:
                action = cli_fallback_prompt(err_info)
            except Exception:
                action = args.MiniMax_fallback_default
            print(f"[MiniMax fallback] action={action} for {len(err_rows)} failed rows")
            if action == "stop":
                save_jsonl(args.output_jsonl, enhanced)  # persist what we have
                print("STOP requested; partial output saved.")
                return 2

    save_jsonl(args.output_jsonl, enhanced)
    print(f"done input={len(rows)} output={len(enhanced)}")
    # MiniMax cost summary
    try:
        mini_b = getattr(runtime.backend, "cost_summary", None)
        if callable(mini_b):
            summary = mini_b()
            print(f"MiniMax usage: calls={summary['calls']} errors={summary['errors']} "
                  f"in_tok={summary['input_tokens']} out_tok={summary['output_tokens']} "
                  f"cost_cny={summary['total_cost_cny']}")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
