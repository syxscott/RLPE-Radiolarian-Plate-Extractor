from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rlpe.evaluation import evaluate, evaluate_run, write_markdown_report
from rlpe.evaluation.gold import GoldPanel


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RLPE predictions against gold data")
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument(
        "--gold",
        type=Path,
        required=True,
        help="A single .jsonl file or a directory of .jsonl files (concatenated).",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.gold.is_dir():
        summary = evaluate_run(args.pred, args.gold)
    else:
        pred = load_jsonl(args.pred)
        gold = [GoldPanel(**g) for g in load_jsonl(args.gold)]
        summary = evaluate(pred, gold)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_markdown_report(summary, args.output.with_suffix(".md"))
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())