from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rlpe.grobid import process_pdf_dir
from rlpe.utils import write_jsonl
from dataclasses import asdict


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch Grobid processing")
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grobid-url", type=str, default="http://localhost:8070")
    args = parser.parse_args()

    results = process_pdf_dir(args.pdf_dir, args.output_dir, server_url=args.grobid_url)
    write_jsonl(args.output_dir / "grobid_results.jsonl", [
        {
            "paper_id": r.paper_id,
            "pdf_path": str(r.pdf_path),
            "tei_path": str(r.tei_path) if r.tei_path else None,
            "success": r.success,
            "error": r.error,
            "captions": [
                {
                    "paper_id": c.paper_id,
                    "figure_id": c.figure_id,
                    "caption": c.caption,
                    "entities": [asdict(e) for e in c.entities],
                }
                for c in r.captions
            ],
        }
        for r in results
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
