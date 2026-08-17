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
from rlpe.evaluation.image_label_check import run_image_label_check


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
    parser.add_argument(
        "--image-label-check",
        action="store_true",
        help=(
            "Run an additional sanity check: re-OCR each prediction's panel "
            "image and compare to the predicted panel_id. Reports "
            "`image_label_match_rate` per paper + aggregate. Adds ~5-15 min "
            "on a 9-paper corpus because EasyOCR runs on every panel."
        ),
    )
    parser.add_argument(
        "--image-label-cache",
        type=Path,
        default=Path("work/image_label_ocr_cache.json"),
        help=(
            "Path to the on-disk cache of OCR results for "
            "--image-label-check. Reusing the same path across runs "
            "makes the second run essentially free (the OCR step is "
            "skipped for every unchanged panel). Set to an empty "
            "string to disable caching. Default: "
            "work/image_label_ocr_cache.json"
        ),
    )
    args = parser.parse_args()

    if args.gold.is_dir():
        summary = evaluate_run(args.pred, args.gold)
    else:
        pred = load_jsonl(args.pred)
        # audit 2026-07-26: gold jsonl may carry extra fields the
        # dataclass doesn't know; filter to known fields to avoid
        # TypeError on GoldPanel(**g).
        import dataclasses as _dc

        _gold_fields = {f.name for f in _dc.fields(GoldPanel)}
        gold = [
            GoldPanel(**{k: v for k, v in g.items() if k in _gold_fields})
            for g in load_jsonl(args.gold)
        ]
        summary = evaluate(pred, gold)
    summary_dict = summary.to_dict()
    if args.image_label_check:
        # audit 2026-07-26: an empty --image-label-cache '' should
        # disable the cache. Path('') becomes PosixPath('.'), whose
        # str is '.', so the old truthiness check failed and wrote
        # cache files into the CWD. Check .name instead.
        cache_path = (
            args.image_label_cache
            if (args.image_label_cache and args.image_label_cache.name)
            else None
        )
        image_label_stats = run_image_label_check(
            predictions=load_jsonl(args.pred),
            root=ROOT,
            cache_path=cache_path,
        )
        summary_dict["image_label_check"] = image_label_stats
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_markdown_report(summary, args.output.with_suffix(".md"))
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps(summary_dict, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
