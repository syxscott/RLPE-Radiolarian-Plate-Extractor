"""Build gold annotations for Baumgartner 2008 (IRIS, Nicaragua).

Source captions come from the paper's Plates 1-3:

  Plate 1 (Middle and Upper Jurassic Radiolaria from the Siuna serpentinite)
  Plate 2 (Upper Jurassic Radiolaria from the Siuna serpentinite)
  Plate 3 (Uppermost Triassic, Rhaetian, Sabalos quarry)

The figure_id values are taken from the actual pipeline run on this paper
(see work/gold_expansion/output/manifests/matches.jsonl).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GOLD_DIR = REPO / "data" / "gold"


PLATE_FIGURES = {
    "od_plate_5d2f7b7852911a67_p015_pl01": {
        # Plate 1: 13 panels
        "1": "Williriedellum marcucciae",
        "2": "Williriedellum marcucciae",
        "3": "Williriedellum sp. S",
        "4": "Williriedellum sp. cf. W. sp. S",
        "5": "Linaresia sp. cf. L. chrafatensis",
        "6": "Zhamoidellum sp.",
        "7": "Zhamoidellum sp.",
        "8": "Xitus spp.",
        "9": "Xitus spp.",
        "10": "Pseudodictyomitra primitiva",
        "11": "Archaeodictyomitra",
        "12": "Mirifusus dianae",
        "13": "Sethocapsa sp. cf. S. dorysphaeroides",
    },
    "od_plate_5d2f7b7852911a67_p018_pl02": {
        # Plate 2: 21 panels
        "1": "Zhamoidellum ovum",
        "2": "Zhamoidellum ovum",
        "3": "Williriedellum carpathicum",
        "4": "Williriedellum carpathicum",
        "5": "Zhamoidellum sp. 2",
        "6": "Zhamoidellum ventricosum",
        "7": "Williriedellum sp.",
        "8": "Zhamoidellum spp.",
        "9": "Zhamoidellum spp.",
        "10": "Zhamoidellum spp.",
        "11": "Protunuma japonicus",
        "12": "Tricolocapsa sp.",
        "13": "Stichomitra sp. cf. S. acuta",
        "14": "Sethocapsa sp. cf. S. zweilii",
        "15": "Zhamoidellum sp. cf. Z. calamin",
        "16": "Sethocapsa spp.",
        "17": "Sethocapsa spp.",
        "18": "Sethocapsa sp. cf. S. uterculus",
        "19": "Triactoma sp.",
        "20": "Acaeniotyle sp.",
        "21": "Hiscocapsa sp.",
    },
    "od_plate_5d2f7b7852911a67_p019_pl03": {
        # Plate 3: 27 panels
        "1": "Ferresium triquetrum",
        "2": "Ferresium sp.",
        "3": "Ferresium sp.",
        "4": "Risella tledoensis",
        "5": "Paricrioma cistella",
        "6": "Betraccium sp.",
        "7": "Livarella densiporata",
        "8": "Tetraporobracchia sp. C",
        "9": "Spumellaria indet. A",
        "10": "Spumellaria indet. B",
        "11": "Spumellaria indet. B",
        "12": "Veghycyclia austrica",
        "13": "Orbiculiformella multibrachiata",
        "14": "Kungalaria newcombi",
        "15": "Praecitriduma mostleri",
        "16": "Canoptum sp. aff. C. unicum",
        "17": "Praeparvicingula sp.",
        "18": "Canoptum triassicum",
        "19": "Laxtorum capitaneum",
        "20": "Globolaxtorum sp. A",
        "21": "Globolaxtorum sp. A",
        "22": "Globolaxtorum sp. B",
        "23": "Nassellaria indet. A",
        "24": "Proparvicingula sp.",
        "25": "Globolaxtorum spp.",
        "26": "Globolaxtorum spp.",
        "27": "Canutus beehivensis",
    },
}

PAPER_ID = "5d2f7b7852911a67"


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
    out = GOLD_DIR / "baumgartner2008.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


if __name__ == "__main__":
    n = build()
    print(f"Wrote {n} gold entries to {GOLD_DIR / 'baumgartner2008.jsonl'}")
