"""CLI: run an evaluation of a predictions JSONL against the gold set.

Usage::

    PYTHONPATH=src python scripts/run_evaluation.py \\
        --predictions work/batch4_v2/results.jsonl \\
        --gold-dir data/gold \\
        --output-dir work/batch4_v2/eval_reports
"""
from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rlpe.evaluation import (  # noqa: E402
    evaluate_run,
    write_markdown_report,
)
from rlpe.provenance.stamp import (  # noqa: E402
    PIPELINE_VERSION,
    build_provenance,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Run an evaluation report")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--gold-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--label", type=str, default="baseline",
                   help="Suffix for report filename (e.g. 'baseline', 'pouille-fix')")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = evaluate_run(args.predictions, args.gold_dir)

    # Embed provenance so the report is self-describing.
    provenance = build_provenance(
        config={"predictions": str(args.predictions), "gold_dir": str(args.gold_dir)},
        pdf_paths=[args.predictions],
    )

    # JSON
    json_path = args.output_dir / f"eval_{args.label}.json"
    payload = {
        "provenance": provenance.to_dict(),
        "report": report.to_dict(),
    }
    json_path.write_text(
        __import__("json").dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    # Markdown
    md_path = args.output_dir / f"eval_{args.label}.md"
    notes = (
        f"_Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} "
        f"by rlpe v{PIPELINE_VERSION} (commit {provenance.git_commit})._\n"
    )
    write_markdown_report(report, md_path, title=f"RLPE Evaluation ({args.label})", notes=notes)

    # Stdout summary
    agg = report.aggregate
    print(f"== {args.label} ==")
    print(f"Papers: {agg.get('n_papers', 0)}  Gold panels: {agg.get('n_gold', 0)}")
    print(f"Species: P={agg.get('species_precision', 0):.1%} "
          f"R={agg.get('species_recall', 0):.1%} "
          f"F1={agg.get('species_f1', 0):.1%}")
    print(f"Panel match: {agg.get('panel_match_rate', 0):.1%}  "
          f"Exact: {agg.get('exact_match_rate', 0):.1%}")
    for pid, m in sorted(report.papers.items()):
        print(f"  {pid}: F1={m.species_f1:.1%}  ({m.species_tp}/{m.n_gold} panels matched species)")
    print(f"Report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
