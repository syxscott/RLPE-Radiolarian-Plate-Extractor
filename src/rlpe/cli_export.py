"""CLI: export a RunOutput JSONL into all three downstream views.

Usage::

    PYTHONPATH=src python -m rlpe.cli_export \\
        --input work/batch4_v2/results.jsonl \\
        --output-dir work/batch4_v2/exports \\
        [--include-unmatched] [--ml-seed foo]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rlpe.converters import (  # noqa: E402
    match_result_from_dict,
    panel_record_from_match,
)
from rlpe.exporters import (  # noqa: E402
    AnalysisOptions,
    DwCAOptions,
    MLOptions,
    write_csv,
    write_dwca_zip,
    write_ml_split,
    write_parquet,
)
from rlpe.provenance.stamp import build_provenance  # noqa: E402
from rlpe.schema_models import ProvenanceRecord, RunOutput  # noqa: E402
from rlpe.types import MatchResult  # noqa: E402


def _run_output_from_jsonl(input_path: Path) -> RunOutput:
    """Build a RunOutput from a JSONL of results.

    For now this reads the existing batch4_v2 JSONL shape (which doesn't
    yet have a top-level provenance block). A later revision will
    require the JSONL to be a RunOutput shape.
    """
    import json
    import logging

    matches: list[MatchResult] = []
    with open(input_path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                logging.getLogger(__name__).warning(
                    "Skipping malformed JSONL line %d in %s: %s",
                    line_no,
                    input_path,
                    exc,
                )
                continue
            # Convert dict to MatchResult via the shared helper.
            m = match_result_from_dict(d)
            matches.append(m)
    panels = [panel_record_from_match(m) for m in matches]
    prov = ProvenanceRecord(
        **build_provenance(
            config={"input": str(input_path)},
        ).to_dict()
    )
    return RunOutput(provenance=prov, panels=panels)


def main() -> int:
    p = argparse.ArgumentParser(description="Export a RunOutput to all three views")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--include-unmatched",
        action="store_true",
        help="Include rows with no species in the analysis view",
    )
    p.add_argument("--ml-seed", type=str, default="rlpe-v1")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = _run_output_from_jsonl(args.input)

    # 1. Analysis (CSV + Parquet)
    n_csv = write_csv(
        run,
        args.output_dir / "analysis.csv",
        options=AnalysisOptions(include_unmatched=args.include_unmatched),
    )
    print(f"  analysis.csv: {n_csv} rows")
    try:
        n_parquet = write_parquet(
            run,
            args.output_dir / "analysis.parquet",
            options=AnalysisOptions(include_unmatched=args.include_unmatched),
        )
        print(f"  analysis.parquet: {n_parquet} rows")
    except ImportError as e:
        print(f"  analysis.parquet: skipped ({e})")

    # 2. ML (JSONL with splits)
    counts = write_ml_split(
        run,
        args.output_dir / "ml",
        options=MLOptions(seed=args.ml_seed),
    )
    print(f"  ml/: train={counts['train']} val={counts['validation']} test={counts['test']}")

    # 3. Archive (DwC-A)
    n_dwca = write_dwca_zip(
        run,
        args.output_dir / "archive.zip",
        options=DwCAOptions(include_unmatched=args.include_unmatched),
    )
    print(f"  archive.zip: {n_dwca} occurrence rows")

    print(f"\nExports written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
