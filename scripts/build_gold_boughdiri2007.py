"""Build gold annotations for Boughdiri 2007 (Jurassic Radiolaria, Tunisia).

Source caption: Plate I (item 111 in the OpenDataLoader output). The
caption is on the page immediately following the "Plate I" heading;
the figure_id is taken from the actual pipeline run on this paper.

The Boughdiri caption uses the Danelian "N) Species, sample, specimen,
scale" convention. Panels 24-26 share a single "Archaeodictyomitra
spp." species entry (no per-panel split in the caption).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GOLD_DIR = REPO / "data" / "gold"


# Paper: 178d4e1e9d93136c (content-based hash of work/gold_expansion/pdfs/boughdiri2007.pdf)
PAPER_ID = "178d4e1e9d93136c"
# figure_id is the actual pipeline-produced id for Plate I on page 11.
# The pipeline (opendataloader_extractor._figure_id) names plates
# ``od_plate_<hash>_p<NNN>_pl<NN>`` and the gold must use the same
# scheme so the eval can match preds to gold entries.
FIGURE_ID = f"od_plate_{PAPER_ID}_p011_pl01"

# Plate I panels (caption: "1) Ristola altissima altissima ... 27) Archaeodictyomitra sp. aff. minoensis")
PLATE_FIGURES = {
    FIGURE_ID: {
        "1": "Ristola altissima altissima",
        "2": "Palinandromeda podbielensis",
        "3": "Orbiculiforma sp. aff. mclaughlini",
        "4": "Podobursa sp.",
        "5": "Stichocapsa decora",
        "6": "Striatojaponocapsa synconexa",
        "7": "Striatojaponocapsa synconexa",
        "8": "Gongylothorax sp.",
        "9": "Williriedellum sp.",
        "10": "Gongylothorax favosus",
        "11": "Protunuma japonicus",
        "12": "Unuma cf. darnoensis",
        "13": "Stichocapsa robusta",
        "14": "Kilinora tecta",
        "15": "Eucyrtidiellum ptyctum",
        "16": "?Sethocapsa sp.",
        "17": "?Archaeodictyomitra sp.",
        "18": "Cinguloturris carpatica",
        "19": "Spongocapsula palmerae",
        "20": "Obesacapsula morroensis",
        "21": "Mirifusus guadalupensis",
        "22": "Transhsuum sp.",
        "23": "Transhsuum brevicostatum",
        "24": "Archaeodictyomitra spp.",
        "25": "Archaeodictyomitra spp.",
        "26": "Archaeodictyomitra spp.",
        "27": "Archaeodictyomitra sp. aff. minoensis",
    },
}


def build() -> int:
    rows: list[dict[str, str]] = []
    for fid, panels in PLATE_FIGURES.items():
        for pid, species in panels.items():
            rows.append({
                "paper_id": PAPER_ID,
                "figure_id": fid,
                "panel_id": pid,
                "species": species,
            })
    out = GOLD_DIR / "boughdiri2007.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


if __name__ == "__main__":
    n = build()
    print(f"Wrote {n} gold entries to {GOLD_DIR / 'boughdiri2007.jsonl'}")
