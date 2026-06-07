"""Build gold annotations for Beccaro 2006 (Swiss J. Geosci.,
"Radiolarian biostratigraphy of the Rosso Ammonitico Medio...").

Source: Plate 1 of the paper (the only plate; 35 species panels
arranged in a 5-row x 7-column montage of UAZ A-F index species).

Caption format: "N – Genus epithet AUTHOR, Section Code,
UAZ Letter, xMag" — a flat list separated by newlines. The parser
treats each line as a separate (label, species) clause. Author
names, sample codes, and magnification are all metadata that
follow the species name and the parser correctly drops them
at the modifier boundary.

Figure_id is taken from the actual pipeline run on this paper
(see work/beccaro_only_out/output/manifests/matches.jsonl).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GOLD_DIR = REPO / "data" / "gold"


# Paper ID is the content-based SHA1 of the PDF
PAPER_ID = "5d5264c7bf0b0a43"


PLATE_FIGURES = {
    f"od_plate_{PAPER_ID}_p013_pl01": {
        # Plate 1 (35 panels): UAZ A-F index species
        "1":  "Eucyrtidiellum unumaense dentatum",
        "2":  "Stylocapsa oblongula",
        "3":  "Unuma echinatus",
        "4":  "Stylocapsa catenarum",
        "5":  "Praewilliriedellum convexum",
        "6":  "Stichocapsa robusta",
        "7":  "Triactoma parablakei",
        "8":  "Tricolocapsa plicarum",
        "9":  "Tethysetta dhimenaensis dhimenaensis",
        "10": "Eucyrtidiellum ptyctum",
        "11": "Williriedellum carpathicum",
        "12": "Podobursa vannae",
        "13": "Ristola altissima altissima",
        "14": "Podobursa polyacantha",
        "15": "Xitus magnus",
        "16": "Eucyrtidiellum unumaense",
        "17": "Tritrabs hayi",
        "18": "Williriedellum marcucciae",
        "19": "Zhamoidellum exquisitum",
        "20": "Angulobracchia biordinalis",
        "21": "Tetratrabs bulbosa",
        "22": "Tetratrabs zealis",
        "23": "Napora lospensis",
        "24": "Emiluvia orea",
        "25": "Mirifusus dianae minor",
        "26": "Mirifusus dianae dianae",
        "27": "Pseudoeucyrtis sp.",
        "28": "Podocapsa amphitreptera",
        "29": "Eucyrtidiellum nodosum",
        "30": "Ristola altissima nodosa",
        "31": "Emiluvia ultima",
        "32": "Pseudoeucyrtis reticularis",
        "33": "Acaeniotyle umbilicata",
        "34": "Syringocapsa spinellifera",
        "35": "Napora boneti",
    },
}


def main() -> int:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GOLD_DIR / "beccaro2006.jsonl"
    rows = []
    for fid, panels in PLATE_FIGURES.items():
        for panel_id, species in panels.items():
            rows.append({
                "paper_id": PAPER_ID,
                "figure_id": fid,
                "panel_id": panel_id,
                "species": species,
            })
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
