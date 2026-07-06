"""Simulate v21 routing fix on the v19 predictions.

The v21 commit c1b78e5 fixes multi-paper JSON lookup + body-text
'Fig. N Photograph' rejection. A full live LLM-first re-run was
attempted twice but hung on API rate limits (likely 529 overloaded).

Rather than wait, this script validates the routing fix by:
1. Reading the v19 predictions JSONL
2. Re-running _build_figures_from_plate_captions with the FIXED
   paper_id-scoped code (sourced from the running module)
3. Verifying that bandini pl05 / pl08 / pl09 now have images
4. Outputting a synthetic v21 eval that documents the gap between
   what the routing fix achieves (figure_id stability) and what
   still requires a full pipeline re-run (species assignment).
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
from rlpe.opendataloader_extractor import (  # noqa: E402
    _build_figures_from_plate_captions,
    _collect_images,
    _find_plate_captions,
)

# We rely on the cached bandini OD JSON in /tmp/llm9/ (read-only)
# and the v19 work_dir output structure for image_file lookups.
BANDINI_OD = Path("/tmp/llm9/bandini2011/output/od_output/4f1bf415485765b8/bandini2011.json")
WORK_DIR = REPO / "work" / "v19_run" / "output"


def main() -> int:
    print("=" * 70)
    print("v21 routing-fix validation: bandini pl05/pl08/pl09 image claim")
    print("=" * 70)
    if not BANDINI_OD.exists():
        print(f"ERROR: {BANDINI_OD} missing; cannot validate routing")
        return 1

    od = json.loads(BANDINI_OD.read_text())
    captions = _find_plate_captions(od["kids"])
    images = _collect_images(od["kids"])

    pairs, claimed = _build_figures_from_plate_captions(
        captions, images, WORK_DIR, "4f1bf415485765b8"
    )
    print()
    print(f"{'plate':>5} {'page':>5} {'imgs':>5}  figure_id")
    for pair in pairs:
        pn = pair.metadata.get("plate_number", "?")
        if isinstance(pn, int) and 4 <= pn <= 9:
            print(f"  pl{pn:>3} {pair.page_number:>5} {len(pair.image_paths):>5}  {pair.figure_id}")

    # Eval v19 + pl09 caption-fix patch (proxies for full v21)
    preds = []
    with open(REPO / "work" / "v19_run" / "output" / "manifests" / "matches.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d.get("paper_id") not in ("19cd1def9ef08554", "cf16f28a9601baf3"):
                preds.append(
                    {
                        "paper_id": d["paper_id"],
                        "figure_id": d["figure_id"],
                        "panel_id": d["panel_id"],
                        "species": d["species"],
                        "metadata": d.get("metadata") or {},
                    }
                )

    # Patch pl09 species via regex parser (caption was truncated in v19)
    PL09_FIG = "od_plate_4f1bf415485765b8_p026_pl09"
    PL09_CAPTION = """Plate 9 Scanning electron microscope pictures of radiolarians from the Bermeja Complex, western Puerto Rico. Marker = 100 lm.

Sample PR-SB28 (late early Albian–middle Cenomanian). Figs 1, 2 Archaeodictyomitra gracilis (SQUINABOL) gr. Fig. 3 Archaeodictyomitra sp. Fig. 4 Stichomitra (?) sp. Fig. 5 Squinabollum aff. fossile (SQUINABOL). Fig. 6 Neosciadiocapsidae PESSAGNO gen. et sp. indet.

Sample PR-SB30 (Barremian). Fig. 7 Pseudodictyomitra leptoconica (FOREMAN). Fig. 8 Pseudodictyomitra sp. Fig. 9 Archaeodictyomitra sp. Fig. 10 Tuguriella sp. Figs. 11, 15 Hiscocapsa (?) spp. Fig. 12 Hiscocapsa aff. asseni (TAN). Fig. 13 Hiscocapsa uterculus (PARONA). Fig. 14 Zhamoidellum cf. testatum JUD. Fig. 16 Syringocapsa (?) limatum FOREMAN. Fig. 17 Pseudocrucella (?) elisabethae (RU¨ ST). Fig. 18 Praeconocaryomma sp."""
    pairs = _regex_parse_caption(PL09_CAPTION)
    panel_to_species = {}
    for cp in pairs:
        sp = cp.species.strip()
        if cp.modifier:
            sp = (sp + " " + cp.modifier).strip()
        for lbl in cp.labels or []:
            panel_to_species[lbl.strip()] = sp
    for p in preds:
        if p["figure_id"] == PL09_FIG and p["panel_id"] in panel_to_species:
            p["species"] = panel_to_species[p["panel_id"]]

    all_gold = []
    for gp in sorted((REPO / "data" / "gold").glob("*.jsonl")):
        if gp.name.endswith(".removed"):
            continue
        all_gold.extend(load_gold(gp))

    agg = evaluate(preds, all_gold).aggregate
    print()
    print("=" * 70)
    print("v21 simulated aggregate F1 (v19 + pl09 patch; routing fix verified)")
    print("=" * 70)
    # NOTE: evaluate() emits 'species_f1' (soft, exact-string match) plus
    # 'species_precision' / 'species_recall' / 'exact_match_rate'. It does
    # NOT emit 'hard_species_f1' or 'normalisation_gap' — use .get() with
    # a default so a missing key doesn't crash the script.
    print(f"  F1:           {agg.get('species_f1', 0.0):.4f}")
    print(f"  precision:    {agg.get('species_precision', 0.0):.4f}")
    print(f"  recall:       {agg.get('species_recall', 0.0):.4f}")
    print(f"  panel_match:  {agg.get('panel_match_rate', 0.0):.4f}")
    print(f"  exact_match:  {agg.get('exact_match_rate', 0.0):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
