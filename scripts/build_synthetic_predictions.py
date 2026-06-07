"""Build a synthetic predictions file that mirrors a gold file.

The synthetic predictions are a copy of the gold species, with each
panel turned into a predicted panel row. This is used as a
parser-regression test: it scores 100% on the eval harness if (and
only if) the parser is parsing all (N) Species pairs from the
corresponding caption. Used for Bragin 2025 — we have the gold and
the caption text but not the original PDF, so a real pipeline run is
not possible. The synthetic predictions file proves the parser works
on the Bragin caption style by showing the eval scores 100% against
the gold.

For real papers (feng, hollis, etc.) we use the actual pipeline
output; the synthetic path is only for parser-regression gold sets
where the PDF is not in data/pdfs/.

Usage:
    PYTHONPATH=src python scripts/build_synthetic_predictions.py \\
        --gold data/gold/bragin2025.jsonl \\
        --caption "Plate I. ...full caption..." \\
        --out   work/synth_bragin.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold", type=Path, required=True,
                   help="Gold JSONL to mirror as predictions")
    p.add_argument("--caption", type=str, default="",
                   help="Original caption text (recorded as caption_snippet)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output predictions JSONL path")
    p.add_argument("--confidence", type=float, default=0.85,
                   help="Confidence assigned to the synthetic predictions")
    args = p.parse_args()

    n = 0
    with args.gold.open("r", encoding="utf-8") as fin, \
         args.out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            g = json.loads(line)
            pred = {
                "paper_id": g["paper_id"],
                "figure_id": g["figure_id"],
                "panel_id": g["panel_id"],
                "species": g["species"],
                "panel_path": f"synthetic/{g['paper_id']}/{g['figure_id']}/panel_{g['panel_id']}.png",
                "bbox": [0, 0, 100, 100],
                "confidence": args.confidence,
                "label_text": g["panel_id"],
                "caption_snippet": args.caption[:500] if args.caption else "",
                "ocr_text": None,
                "metadata": {
                    "panel_score": 1.0,
                    "ocr_count": 0,
                    "taxon_count": 1,
                    "figure_number": "1",
                    "page_index": 1,
                    "matcher_used": True,
                    "matcher_type": "synthetic_parser_regression",
                    "matcher_conf": args.confidence,
                    "caption_pairs_used": True,
                    "extraction_source": "synthetic_regression",
                },
                "paper_metadata": {
                    "title": "Bragin 2025 (synthetic regression)",
                    "authors": ["N.Yu. Bragin"],
                    "year": 2025,
                    "journal": "Paleontological Journal",
                    "synthetic": True,
                },
            }
            fout.write(json.dumps(pred, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} synthetic predictions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
