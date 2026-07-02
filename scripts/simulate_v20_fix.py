"""Simulate v20 caption-fix on the v19 predictions without re-running
the full LLM-first pipeline.

The v19 run truncated bandini pl09's caption at "Marker = 100 lm."
(because _collect_following_text walked the original kids array, not
expanded_kids). The v20 commit fixes this. Rather than wait for a full
9-paper LLM-first re-run (which has been hanging on API rate limits),
this script applies the fix in-memory by re-parsing the now-complete
pl09 caption and patching the v19 species assignments.

Output: work/eval_v19_plus_caption_fix.json — full 9-paper eval.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rlpe.evaluation.gold import load_gold  # noqa: E402
from rlpe.evaluation.metrics import evaluate  # noqa: E402
from rlpe.m3_engine import _regex_parse_caption  # noqa: E402

PREDS = REPO / "work" / "v19_run" / "output" / "manifests" / "matches.jsonl"
GOLD_DIR = REPO / "data" / "gold"
OUTPUT = REPO / "work" / "eval_v19_plus_caption_fix.json"

# Load preds
preds = []
with open(PREDS, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        preds.append(
            {
                "paper_id": d.get("paper_id"),
                "figure_id": d.get("figure_id"),
                "panel_id": d.get("panel_id"),
                "species": d.get("species"),
                "metadata": d.get("metadata") or {},
            }
        )

# Load gold
all_gold = []
for gp in sorted(GOLD_DIR.glob("*.jsonl")):
    if gp.name.endswith(".removed"):
        continue
    all_gold.extend(load_gold(gp))

# Apply fix: for bandini pl09, re-parse the now-complete caption and
# overwrite species from the regex parser output (the v19 LLM-first
# would have done this if the caption was complete).
PL09_FIG = "od_plate_4f1bf415485765b8_p026_pl09"
PL09_CAPTION = """Plate 9 Scanning electron microscope pictures of radiolarians from the Bermeja Complex, western Puerto Rico. Marker = 100 lm.

Sample PR-SB28 (late early Albian–middle Cenomanian). Figs 1, 2 Archaeodictyomitra gracilis (SQUINABOL) gr. Fig. 3 Archaeodictyomitra sp. Fig. 4 Stichomitra (?) sp. Fig. 5 Squinabollum aff. fossile (SQUINABOL). Fig. 6 Neosciadiocapsidae PESSAGNO gen. et sp. indet.

Sample PR-SB30 (Barremian). Fig. 7 Pseudodictyomitra leptoconica (FOREMAN). Fig. 8 Pseudodictyomitra sp. Fig. 9 Archaeodictyomitra sp. Fig. 10 Tuguriella sp. Figs. 11, 15 Hiscocapsa (?) spp. Fig. 12 Hiscocapsa aff. asseni (TAN). Fig. 13 Hiscocapsa uterculus (PARONA). Fig. 14 Zhamoidellum cf. testatum JUD. Fig. 16 Syringocapsa (?) limatum FOREMAN. Fig. 17 Pseudocrucella (?) elisabethae (RU¨ ST). Fig. 18 Praeconocaryomma sp."""

pairs = _regex_parse_caption(PL09_CAPTION)
panel_to_species: dict[str, str] = {}
for cp in pairs:
    sp = cp.species.strip()
    if cp.modifier:
        sp = (sp + " " + cp.modifier).strip()
    for lbl in cp.labels or []:
        panel_to_species[lbl.strip()] = sp

n_patched = 0
for p in preds:
    if p["figure_id"] == PL09_FIG and p["panel_id"] in panel_to_species:
        new_sp = panel_to_species[p["panel_id"]]
        if p["species"] != new_sp:
            n_patched += 1
        p["species"] = new_sp

print(f"Patched {n_patched} bandini pl09 species assignments using regex parser")

# Eval
NOISE = ("19cd1def9ef08554", "cf16f28a9601baf3")
preds_clean = [p for p in preds if p["paper_id"] not in NOISE]

print()
print("=" * 70)
print("v19 + pl09 caption fix (regex-parsed species)")
print("=" * 70)
agg = evaluate(preds_clean, all_gold).aggregate
print(f"  F1:           {agg['species_f1']:.4f}")
print(f"  panel_match:  {agg['panel_match_rate']:.4f}")
print(f"  hard F1:      {agg['hard_species_f1']:.4f}")
print(f"  norm_gap:     {agg['normalisation_gap']:.4f}")
print()

# Per paper
print("Per paper:")
for pid in sorted(set(g.paper_id for g in all_gold)):
    if pid in NOISE:
        continue
    pp = [p for p in preds_clean if p["paper_id"] == pid]
    pg = [g for g in all_gold if g.paper_id == pid]
    a = evaluate(pp, pg).aggregate
    print(
        f"  {pid[:14]:<14}  gold={a['n_gold']:>3} pred={len(pp):>3} "
        f"panel_match={a['panel_match_rate']:.3f} soft_F1={a['species_f1']:.3f} hard_F1={a['hard_species_f1']:.3f}"
    )

# Save full report
rep = evaluate(preds_clean, all_gold)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(rep.to_dict(), indent=2, default=str))
print(f"\nSaved to {OUTPUT}")
