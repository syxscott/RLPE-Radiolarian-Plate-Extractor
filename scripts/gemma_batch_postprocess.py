from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rlpe.gemma_postprocess import batch_gemma_postprocess_rows, load_gemma4_model


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
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--conf-threshold", type=float, default=0.70)
    parser.add_argument("--prompt-lang", type=str, default="zh", choices=["zh", "en"])
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--no-bfloat16", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
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
