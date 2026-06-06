"""Re-apply RLPE's caption parser to existing JSONL output.

The pouille run in batch4_v2 was done before the Pouille-style caption
parser was extended. This script reads the existing JSONL, re-parses
the caption_snippet for each panel, and writes a corrected JSONL.

This is a stop-gap: the proper fix is to re-run the full pipeline, but
that requires re-rendering the PDFs and re-segmenting panels (slow).
Re-parsing the captions is O(panels) and lets us validate the fix.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rlpe.m3_engine import _regex_parse_caption  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Re-apply caption parser to existing JSONL")
    p.add_argument("--input", type=Path, required=True,
                   help="Path to existing results.jsonl")
    p.add_argument("--output", type=Path, required=True,
                   help="Where to write the corrected JSONL")
    p.add_argument("--paper-id", type=str, default="cb2011ef7be94959",
                   help="Paper ID to reprocess (default: pouille)")
    args = p.parse_args()

    # Read all rows; build (figure_id, panel_id) → row index
    rows: list[dict] = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    # Group rows by figure_id for the target paper
    by_figure: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if r.get("paper_id") == args.paper_id:
            by_figure[r["figure_id"]].append(i)

    total_updated = 0
    for fig_id, idxs in by_figure.items():
        # Get the canonical caption from the first row
        caption = rows[idxs[0]].get("caption_snippet") or ""
        if not caption:
            continue
        # Parse it
        pairs = _regex_parse_caption(caption)
        if not pairs:
            continue
        # Build label → species lookup
        label_lookup: dict[str, str] = {}
        for cp in pairs:
            sp = getattr(cp, "species", None)
            if not sp:
                continue
            for lbl in getattr(cp, "labels", None) or []:
                label_lookup[str(lbl).strip()] = sp
        if not label_lookup:
            continue
        # Apply to rows
        for i in idxs:
            plabel = (rows[i].get("panel_id") or "").strip()
            if plabel in label_lookup:
                old = rows[i].get("species")
                new = label_lookup[plabel]
                if old != new:
                    rows[i]["species"] = new
                    total_updated += 1
                    # Mark in metadata that the species was re-derived
                    md = rows[i].get("metadata") or {}
                    md["species_reprocess_source"] = "regex_post_fix"
                    md["species_reprocess_old"] = old
                    rows[i]["metadata"] = md

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
    print(f"Updated {total_updated} panels in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
