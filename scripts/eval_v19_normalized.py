"""Eval v19 with a panel_id normalization pass.

The v19 LLM-first run produced dirty panel_ids on bandini pl07/08/09:
- duplicates (5 rows with panel_id='1' for the same figure)
- compound labels ('1,2' for the same species)
- None / null panel_ids

Before re-running eval with the bandini pl07/08/09 gold, normalize
predictions in-memory:

  1. Drop rows where panel_id is None/empty
  2. Split compound labels 'a,b' / 'a, b' / 'a-b' / 'a-b' into multiple rows
     (each carries the same species)
  3. Drop duplicate (figure_id, panel_id) rows after splitting (keep the
     first non-empty species)
  4. Numeric-only: '1' → '1', '1a' → '1a'

This is a *post-hoc* normalizer applied at eval time only. It does
not modify the underlying v19 predictions JSONL. If it shows a
meaningful F1 lift, we can promote the same logic to the matcher
in src/rlpe/evaluation/metrics.py so future runs benefit automatically.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rlpe.evaluation.gold import GoldPanel, load_gold  # noqa: E402
from rlpe.evaluation.metrics import evaluate  # noqa: E402

GOLD_DIR = REPO / "data" / "gold"
PREDS_PATH = REPO / "work" / "v19_run" / "output" / "manifests" / "matches.jsonl"

# Patterns that mean "this is one panel labeled with a range/list":
#   "1, 2"   "1,2"    "1-3"   "Figs 1, 2" (already stripped in caption parser)
#   "1a-b" (skip — too complex)
_RANGE_SEP_RE = re.compile(r"[,;]\s*")


def _split_panel_id(panel_id: str) -> list[str]:
    """Split compound panel_ids into individual ones.

    "1, 2"  → ["1", "2"]
    "1,2"   → ["1", "2"]
    "1-3"   → ["1", "2", "3"]
    "1"     → ["1"]
    "1a"    → ["1a"]   (alpha suffix preserved; no range)
    "None"  → []
    """
    if not panel_id:
        return []
    panel_id = panel_id.strip()
    if not panel_id:
        return []
    # Try comma split
    if "," in panel_id or ";" in panel_id:
        parts = [p.strip() for p in _RANGE_SEP_RE.split(panel_id) if p.strip()]
        if all(p.isdigit() for p in parts):
            return parts
        if all(re.match(r"^\d+[a-zA-Z]?$", p) for p in parts):
            return parts
    # Try hyphen range (only if both ends are pure integers)
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", panel_id)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a <= b and b - a <= 20:  # sanity bound
            return [str(i) for i in range(a, b + 1)]
    return [panel_id]


def normalize_predictions(preds: list[dict]) -> list[dict]:
    """Apply the panel_id normalization pass.

    Returns a NEW list (does not mutate input).
    """
    out: list[dict] = []
    # First pass: split + drop None/empty
    for p in preds:
        pid = p.get("panel_id")
        if pid is None or str(pid).strip() in ("", "None", "null"):
            continue
        species = (p.get("species") or "").strip()
        sub_ids = _split_panel_id(str(pid))
        for sid in sub_ids:
            out.append({
                "paper_id": p.get("paper_id"),
                "figure_id": p.get("figure_id"),
                "panel_id": sid,
                "species": species,
                "metadata": p.get("metadata") or {},
            })
    # Second pass: dedup by (figure_id, panel_id) keeping non-empty species
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for p in out:
        by_key[(p["paper_id"], p.get("figure_id") or "", p["panel_id"])].append(p)
    deduped: list[dict] = []
    for key, items in by_key.items():
        # Prefer items with non-empty species
        with_sp = [x for x in items if x["species"]]
        if with_sp:
            deduped.append(with_sp[0])
        elif items:
            deduped.append(items[0])
    return deduped


def main() -> int:
    # Load preds
    preds = []
    with open(PREDS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            preds.append({
                "paper_id": d.get("paper_id"),
                "figure_id": d.get("figure_id"),
                "panel_id": d.get("panel_id"),
                "species": d.get("species"),
                "metadata": d.get("metadata") or {},
            })
    print(f"Loaded {len(preds)} raw predictions")
    preds_norm = normalize_predictions(preds)
    print(f"After normalize: {len(preds_norm)} predictions")

    # Load gold
    gold = []
    for gp in sorted(GOLD_DIR.glob("*.jsonl")):
        if gp.name.endswith(".removed"):
            continue
        gold.extend(load_gold(gp))
    print(f"Loaded {len(gold)} gold entries")

    # Eval before / after
    NOISE = ("19cd1def9ef08554", "cf16f28a9601baf3")
    preds_clean = [p for p in preds if p["paper_id"] not in NOISE]
    preds_norm_clean = [p for p in preds_norm if p["paper_id"] not in NOISE]

    print()
    print("=" * 70)
    print(f"v19 live LLM-first — string-match F1, excluding noise papers {NOISE}")
    print("=" * 70)

    agg_raw = evaluate(preds_clean, gold).aggregate
    print(f"  RAW  (no panel_id norm):  F1={agg_raw['species_f1']:.4f}  panel_match={agg_raw['panel_match_rate']:.4f}")

    agg_norm = evaluate(preds_norm_clean, gold).aggregate
    print(f"  NORM (panel_id split):    F1={agg_norm['species_f1']:.4f}  panel_match={agg_norm['panel_match_rate']:.4f}")

    print()
    print("Per paper:")
    papers_data = evaluate(preds_norm_clean, gold)
    for pid in sorted(set(g.paper_id for g in gold)):
        if pid in NOISE:
            continue
        paper_preds = [p for p in preds_norm_clean if p["paper_id"] == pid]
        paper_gold = [g for g in gold if g.paper_id == pid]
        a = evaluate(paper_preds, paper_gold).aggregate
        print(f"  {pid[:14]:<14}  gold={a['n_gold']:>3} pred={a['n_pred_panels']:>3} "
              f"panel_match={a['panel_match_rate']:.3f} soft_F1={a['species_f1']:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())