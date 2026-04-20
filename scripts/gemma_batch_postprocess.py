from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rlpe.gemma_postprocess import batch_gemma_postprocess_rows, load_gemma4_model, load_gemma4_ollama
from rlpe.gemma_postprocess import load_gemma4_llamacpp


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
    parser.add_argument("--backend", type=str, default="llamacpp", choices=["transformers", "ollama", "llamacpp", "llama.cpp", "llama_cpp"])
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
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    if args.backend in {"llamacpp", "llama.cpp", "llama_cpp"}:
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
    save_jsonl(args.output_jsonl, enhanced)
    print(f"done input={len(rows)} output={len(enhanced)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
