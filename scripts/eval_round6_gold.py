"""Round 6 eval: live matches.jsonl vs gold JSONL per paper.

Reports per-paper precision / recall / F1 (species match) and panel_match
rate. Used to verify Round 5/6 fixes actually recover pl07/08/09 of
bandini2011 and pl02/pl03 of pouille2014 etc.

Usage:
    PYTHONPATH=src python scripts/eval_round6_gold.py \\
        --matches work/oa_smoke_round6_bandini2011/output/manifests/matches.jsonl \\
        --gold data/gold/bandini2011.jsonl \\
        --paper-id 4f1bf415485765b8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def normalize_species(s: str | None) -> str:
    """Normalize species string for lenient matching.

    Drops:
      - "?" uncertainty markers and empty "(?:)" / "()" groups
      - Trailing author citations ("Carter", "Dumitrica", "Pessagno", etc.)
      - "sensu Author YEAR" / "Author YEAR" citations
      - "sp.", "spp.", "sp. A", "sp. cf. epithet", "sp. aff. epithet" qualifiers
        that gold may or may not include
      - "gen. et sp. indet." → "indet."
    """
    if not s:
        return ""
    import re

    s = s.strip().lower()
    # Drop "(?)" / "()" / "?" markers — these mark genus-level uncertainty.
    s = re.sub(r"\(\s*\?*\s*\)", "", s)
    s = s.replace("?", "")
    # "gen. et sp. indet." → "indet." (collapse the 3-token boilerplate).
    s = re.sub(r"\bgen\s+et\s+sp\s+indet\b", "indet", s)
    # Drop "sensu Author" / "Author YEAR" / "Author" trailing citations.
    # Examples: "Risella tledoensis Carter", "Canoptum sp. aff. C. unicum Pessagno",
    # "Livarella densiporata Kozur and Mostler", "Praecitriduma mostleri Kozur 1984".
    # Pattern: trailing "Author1 [and Author2] YEAR?" — handles multi-author.
    s = re.sub(r"\s+(?:and\s+)?[A-Z][a-z][\w-]*(?:\s+and\s+[A-Z][a-z][\w-]*)?\s+\d{4}\s*$", "", s)
    s = re.sub(r"\s+(?:and\s+)?[A-Z][a-z][\w-]*(?:\s+and\s+[A-Z][a-z][\w-]*)?\s*$", "", s)
    # Drop "sensu Author YEAR?" explicitly.
    s = re.sub(r"\s+sensu\s+[a-z][\w-]*\s*\d*\s*$", "", s)
    # Drop trailing " sp. cf. epithet" / " sp. aff. epithet" qualifiers.
    s = re.sub(r"\s+sp\s+cf\s+[a-z][\w-]*$", "", s)
    s = re.sub(r"\s+sp\s+aff\s+[a-z][\w-]*$", "", s)
    # Collapse " sp." → " sp" so "Entactinia sp." matches "Entactinia sp" in gold.
    s = re.sub(r"\bsp\.\s*$", "sp", s)
    s = re.sub(r"\bspp\.\s*$", "spp", s)
    # Collapse multiple spaces and strip trailing punctuation.
    s = " ".join(s.split())
    s = s.rstrip(".").strip()
    return s


def load_gold(gold_path: Path, paper_id: str) -> dict[str, set[str]]:
    """Return figure_id → set of (panel_id, normalized_species)."""
    fig_species: dict[str, set[str]] = {}
    with gold_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if paper_id and r.get("paper_id") != paper_id:
                continue
            pid = r.get("panel_id")
            sp = normalize_species(r.get("species"))
            if pid and sp:
                fig_species.setdefault(r.get("figure_id", ""), set()).add(f"{pid}|{sp}")
    return fig_species


def load_matches(matches_path: Path, paper_id: str | None) -> list[dict[str, Any]]:
    rows = []
    with matches_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if paper_id and r.get("paper_id") != paper_id:
                continue
            rows.append(r)
    return rows


def eval_per_paper(matches: list[dict[str, Any]], gold: dict[str, set[str]]) -> dict[str, Any]:
    # Group matches by figure_id
    pred_by_fig: dict[str, set[str]] = {}
    for r in matches:
        pid = r.get("panel_id")
        sp = normalize_species(r.get("species"))
        if pid and sp:
            pred_by_fig.setdefault(r.get("figure_id", ""), set()).add(f"{pid}|{sp}")
    per_fig = {}
    total_tp = total_fp = total_fn = 0
    for fid, gold_pairs in gold.items():
        pred_pairs = pred_by_fig.get(fid, set())
        tp = len(gold_pairs & pred_pairs)
        fp = len(pred_pairs - gold_pairs)
        fn = len(gold_pairs - pred_pairs)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_fig[fid] = {"tp": tp, "fp": fp, "fn": fn, "p": prec, "r": rec, "f1": f1,
                        "gold_n": len(gold_pairs), "pred_n": len(pred_pairs)}
        total_tp += tp
        total_fp += fp
        total_fn += fn
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    return {"per_figure": per_fig, "micro": {"p": micro_p, "r": micro_r, "f1": micro_f1,
                                              "tp": total_tp, "fp": total_fp, "fn": total_fn}}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matches", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--paper-id", type=str, default=None)
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args(argv)

    if not args.matches.exists():
        print(f"ERROR: matches file not found: {args.matches}", file=sys.stderr)
        return 1
    if not args.gold.exists():
        print(f"ERROR: gold file not found: {args.gold}", file=sys.stderr)
        return 1

    matches = load_matches(args.matches, args.paper_id)
    gold = load_gold(args.gold, args.paper_id or "")
    if not matches:
        print(f"WARNING: 0 matches in {args.matches}", file=sys.stderr)
    if not gold:
        print(f"WARNING: 0 gold rows for paper_id={args.paper_id}", file=sys.stderr)
    result = eval_per_paper(matches, gold)
    if args.summary_only:
        m = result["micro"]
        print(f"micro P={m['p']:.3f} R={m['r']:.3f} F1={m['f1']:.3f} "
              f"(tp={m['tp']} fp={m['fp']} fn={m['fn']})")
    else:
        print(f"=== Per-figure breakdown (paper_id={args.paper_id}) ===")
        for fid, m in sorted(result["per_figure"].items()):
            print(f"  {fid[-25:]:25} F1={m['f1']:.3f} P={m['p']:.3f} R={m['r']:.3f} "
                  f"gold={m['gold_n']:>3} pred={m['pred_n']:>3} tp={m['tp']}")
        m = result["micro"]
        print()
        print(f"=== Micro ===")
        print(f"P={m['p']:.3f} R={m['r']:.3f} F1={m['f1']:.3f} "
              f"tp={m['tp']} fp={m['fp']} fn={m['fn']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())