"""Eval all 9 gold papers against existing live run outputs.

Walks known live-run directories under work/, matches paper_id from
matches.jsonl against the corresponding gold JSONL, and reports
per-paper precision / recall / F1 plus a micro F1 aggregate.

Usage:
    PYTHONPATH=src python scripts/eval_all_gold.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Add src/ for the eval_round6_gold helper.
_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

# Reuse the lenient species normalize + per-fig F1.
from eval_round6_gold import (  # type: ignore[import-not-found]  # noqa: E402
    eval_per_paper,
    normalize_species,
)

# Project root for resolving relative paths.
_REPO_ROOT = Path(__file__).resolve().parents[1]


# Map (path-component substring, gold filename) so we can find the
# right matches.jsonl + gold.jsonl pair without hard-coding paper_ids.
# paper_id is also in matches.jsonl but reading from gold JSONL is simpler.
# The substring is matched against any path component.
LIVE_RUN_DIRS = [
    ("Bandini_2006", None),  # bandini2006 has no gold JSONL
    ("Beccaro_2006", "data/gold/beccaro2006.jsonl"),
    ("Boughdiri_2007", "data/gold/boughdiri2007.jsonl"),
    ("Danelian_2006", "data/gold/danelian2006.jsonl"),
    ("Pouille_2014", "data/gold/pouille2014.jsonl"),
    ("bandini2011", "data/gold/bandini2011.jsonl"),
    ("baumgartner2008", "data/gold/baumgartner2008.jsonl"),
    ("Feng_2007", "data/gold/feng2007.jsonl"),
    ("Hollis_2006", "data/gold/hollis2006.jsonl"),
    ("Bragin_2025", "data/gold/bragin2025.jsonl"),
]

WORK_BASE = _REPO_ROOT / "work"


def load_gold(gold_path: Path) -> dict[str, set[str]]:
    """Return figure_id -> set of normalized (panel_id|species)."""
    fig_pairs: dict[str, set[str]] = {}
    with gold_path.open() as f:
        for line in f:
            try:
                r = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            pid = r.get("panel_id")
            sp = normalize_species(r.get("species"))
            if pid and sp:
                fig_pairs.setdefault(r.get("figure_id", ""), set()).add(f"{pid}|{sp}")
    return fig_pairs


def load_matches(matches_path: Path) -> list[dict[str, Any]]:
    rows = []
    with matches_path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return rows


def find_live_dir(prefix: str) -> Path | None:
    """Return the best matches.jsonl under work/ whose path contains ``prefix``.

    The Round 6/7 smoke driver stores results in two layouts:

      1. ``work/oa_smoke_round6_v1/Bandini_2006 - .../output/manifests/matches.jsonl``
         (each paper gets its own sub-dir)
      2. ``work/oa_smoke_round6_bandini2011_v3/output/manifests/matches.jsonl``
         (whole run in one dir; matches.jsonl lives directly under output/)

    We search under both ``oa_smoke_round6_*`` and the older
    ``oa_smoke_v1`` / ``v2`` / ``v3`` directories (those are the
    pre-Round-6 driver runs that wrote matches.jsonl under
    <paper>/output/manifests/). The newest round6 version wins.

    We also search non-oa_smoke dirs (e.g. ``feng_fix_test``) so that
    freshly run results take precedence over stale smoke-test outputs.

    Matching strategy: extract the paper root word(s) from ``prefix``
    (e.g. 'Feng' from 'Feng_2007') and match against PDF filenames in
    the run's pdfs/ subdir AND against any path component. The paper
    root is the first underscore-separated word.
    """
    if not WORK_BASE.exists():
        return None
    # Paper root = text before the first underscore (e.g. "Feng_2007" -> "feng").
    paper_root = prefix.split("_")[0].lower() if "_" in prefix else prefix.lower()
    candidates: list[tuple[int, Path]] = []
    for top in WORK_BASE.iterdir():
        if not top.is_dir():
            continue
        is_smoke = top.name.startswith("oa_smoke")
        for mp in top.rglob("matches.jsonl"):
            if "output" not in mp.parts or "manifests" not in mp.parts:
                continue
            # Match against (a) any path component, or (b) PDF names in
            # pdfs/ subdir. (b) catches layout 2 where the run-dir name
            # itself doesn't include the paper name.
            in_path = any(paper_root in part.lower() for part in mp.parts)
            pdfs_dir = top / "pdfs"
            in_pdfs = False
            if pdfs_dir.is_dir():
                in_pdfs = any(
                    paper_root in p.name.lower()
                    for p in pdfs_dir.iterdir()
                    if p.is_file() and p.suffix.lower() == ".pdf"
                )
            if not (in_path or in_pdfs):
                continue
            is_round6 = top.name.startswith("oa_smoke_round6")
            # Version score: lower is better (wins sort).
            # Non-oa_smoke dirs get score 0 (best) so fresh runs win.
            if not is_smoke:
                version_score = 0
            else:
                version_score = (
                    1
                    if "_v3" in top.name
                    else 2
                    if "_v2" in top.name
                    else 3
                    if "_v1" in top.name
                    else 4
                )
            # Heavily penalize oa_smoke_v2/v3 (older driver) so round6
            # always wins ties.
            round6_score = 0 if is_round6 else 1000
            candidates.append((round6_score + version_score, mp))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def main() -> int:
    summary_rows: list[dict[str, Any]] = []
    for prefix, gold_rel in LIVE_RUN_DIRS:
        if gold_rel is None:
            # No gold JSONL for this paper (e.g. Bandini_2006); skip.
            continue
        matches_path = find_live_dir(prefix)
        gold_path = _REPO_ROOT / gold_rel
        if matches_path is None:
            print(f"⚠️  {prefix}: no matches.jsonl found under work/")
            continue
        if not gold_path.exists():
            print(f"⚠️  {prefix}: gold file missing ({gold_path})")
            continue
        matches = load_matches(matches_path)
        gold = load_gold(gold_path)
        # Filter gold to only this paper's paper_id (in case multiple papers
        # share a single gold file due to live-run paper_id drift).
        match_paper_id = matches[0].get("paper_id") if matches else None
        gold_filtered = {
            fid: pairs for fid, pairs in gold.items() if not match_paper_id or match_paper_id in fid
        }
        # If no gold survived the paper_id filter (e.g. the live run was
        # re-extracted with a different PDF and got a different paper_id hash),
        # fall back to suffix matching: extract the last two underscore-separated
        # tokens from each figure_id (e.g. p006_pl01) so that
        # od_plate_<hash1>_p006_pl01 matches od_plate_<hash2>_p006_pl01.
        if not gold_filtered and matches:
            pred_fids = {r.get("figure_id") for r in matches}
            gold_fids = set(gold.keys())
            # Build suffix → gold figure_ids mapping
            suffix_to_gold: dict[str, list[str]] = {}
            for gfid in gold_fids:
                parts = gfid.rsplit("_", 2)
                suffix = "_".join(parts[-2:])  # e.g. "p006_pl01"
                suffix_to_gold.setdefault(suffix, []).append(gfid)
            # Map each pred fid to its suffix and rewrite matches in-place
            fid_rewrite: dict[str, str] = {}
            for m in matches:
                pfid = m.get("figure_id", "")
                if pfid in fid_rewrite:
                    continue  # already mapped this pred fid
                parts = pfid.rsplit("_", 2)
                suffix = "_".join(parts[-2:])
                for gfid in suffix_to_gold.get(suffix, []):
                    fid_rewrite[pfid] = gfid
                    print(f"  suffix-match: pred fid={pfid} → gold fid={gfid}")
                    break
            # Rewrite matches figure_ids in-place
            if fid_rewrite:
                for m in matches:
                    pfid = m.get("figure_id", "")
                    if pfid in fid_rewrite:
                        m["figure_id"] = fid_rewrite[pfid]
                gold_filtered = {
                    gfid: pairs for gfid, pairs in gold.items() if gfid in fid_rewrite.values()
                }
        if not gold_filtered:
            print(
                f"⚠️  {prefix}: live paper_id={match_paper_id} matches no gold figure_ids; skipping"
            )
            continue
        result = eval_per_paper(matches, gold_filtered)
        m = result["micro"]
        summary_rows.append(
            {
                "paper": prefix,
                "rows": len(matches),
                "gold_pairs": sum(len(s) for s in gold_filtered.values()),
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "P": m["p"],
                "R": m["r"],
                "F1": m["f1"],
            }
        )

    if not summary_rows:
        print("No papers evaluated.")
        return 1

    # Print table
    print()
    print(
        f"{'Paper':<22} {'rows':>5} {'gold':>5} {'tp':>4} {'fp':>4} {'fn':>4} {'P':>6} {'R':>6} {'F1':>6}"
    )
    print("-" * 80)
    total_tp = total_fp = total_fn = 0
    for r in summary_rows:
        total_tp += r["tp"]
        total_fp += r["fp"]
        total_fn += r["fn"]
        print(
            f"{r['paper']:<22} {r['rows']:>5} {r['gold_pairs']:>5} "
            f"{r['tp']:>4} {r['fp']:>4} {r['fn']:>4} "
            f"{r['P']:>6.3f} {r['R']:>6.3f} {r['F1']:>6.3f}"
        )
    print("-" * 80)
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    print(
        f"{'MICRO (8 papers)':<22} {'':>5} {'':>5} "
        f"{total_tp:>4} {total_fp:>4} {total_fn:>4} "
        f"{micro_p:>6.3f} {micro_r:>6.3f} {micro_f1:>6.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
