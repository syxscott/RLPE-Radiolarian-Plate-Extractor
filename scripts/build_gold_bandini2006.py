"""Build gold annotations for Bandini 2006 ("Upper Cretaceous
radiolarians from Karnezeika, Argolis Peninsula, Greece").

We annotate only the radiolarian-bearing plates (Plates 1 and 2).
Plate 3 of this paper is a foraminifera plate (Helvetoglobotruncana,
Marginotruncana, etc.) — those are out of scope for RLPE.

Caption format: "Figures N-M Genus epithet AUTHOR YEAR AlXX_YYY
(Figs. N and M)" — multi-line paragraph with each figure range
on its own line. Labels are written as "N" (the first number)
because the gold standard treats "Figures N-M" as panel N
(higher-resolution: see Plate 1 in particular, where Figures 5-6
becomes labels 5 and 6).

This convention is consistent with the gold for hollis2006 (where
"Figure N" and "Plate 1" panels are labeled 1..N).

Figure_id values are from work/bandini2006_only_out/output/manifests/matches.jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GOLD_DIR = REPO / "data" / "gold"


# Paper ID is the content-based SHA1 of the PDF
PAPER_ID = "19cd1def9ef08554"


PLATE_FIGURES = {
    # Plate 1 (radiolarians): "Figure 3 Acaeniotyle sp. A; 4 Acaeniotyle
    # sp. B; 5-6 Archaeocenosphaera (?) mellifera; 7-8 Archaeocenosphaera
    # (?) sp.; 9-10 Triactoma cellulosa; 11-12 Triactoma hexeris; 13
    # Triactoma sp. aff. T. hexeris; 14 Pseudoacanthosphaera galeata;
    # 15 Pseudoacanthosphaera superba; 16 Pseudoacanthosphaera sp. aff.
    # P. spinosissima; 17 Pseudoacanthosphaera (?) sp.; 18
    # Tetracanthellipsis euganeus; 19-20 Praeconocaryomma universa;
    # 21-22 Praeconocaryomma californiaensis; 23-24 Praeconocaryomma
    # lipmanae; 25-26 Praeconocaryomma sp.; 27-28 Crucella messinae;
    # 29-30 Crucella cachensis; 31 Halesium triacanthum; 32-33 Halesium
    # sp.; 34 Pessagnobrachia sp."
    f"od_plate_{PAPER_ID}_p017_pl01": {
        "5":  "Archaeocenosphaera mellifera",
        "6":  "Archaeocenosphaera mellifera",
        "7":  "Archaeocenosphaera sp.",
        "8":  "Archaeocenosphaera sp.",
        "9":  "Triactoma cellulosa",
        "10": "Triactoma cellulosa",
        "11": "Triactoma hexeris",
        "12": "Triactoma hexeris",
        "13": "Triactoma sp.",
        "14": "Pseudoacanthosphaera galeata",
        "15": "Pseudoacanthosphaera superba",
        "16": "Pseudoacanthosphaera sp.",
        "17": "Pseudoacanthosphaera sp.",
        "18": "Tetracanthellipsis euganeus",
        "19": "Praeconocaryomma universa",
        "20": "Praeconocaryomma universa",
        "21": "Praeconocaryomma californiaensis",
        "22": "Praeconocaryomma californiaensis",
        "23": "Praeconocaryomma lipmanae",
        "24": "Praeconocaryomma lipmanae",
        "25": "Praeconocaryomma sp.",
        "26": "Praeconocaryomma sp.",
        "27": "Crucella messinae",
        "28": "Crucella messinae",
        "29": "Crucella cachensis",
        "30": "Crucella cachensis",
        "31": "Halesium triacanthum",
        "32": "Halesium sp.",
        "33": "Halesium sp.",
        "34": "Pessagnobrachia sp.",
    },
    # Plate 2 (radiolarians): "Figures 1-2 Dactyliodiscus sp.; 3-4
    # Pseudoaulophacus sculptus; 5-6 Pseudoaulophacus putahensis; 7-8
    # Patellula helios; 9-10 Patellula ecliptica; 11-12 Patellula heroica;
    # 13 Patellula sp.; 14-15 Acanthocircus venetus; 16-17 Acanthocircus
    # tympanum; 18-19 Acanthocircus hueyi; 20-21 Dictyomitra formosa;
    # 22 Dictyomitra sp.; 23-24 Dictyomitra montisserei; 25 Dictyomitra
    # urakawensis; 26 Torculum coronatum; 27-28 Pseudodictyomitra
    # pseudomacrocephala; 29-30 Stichomitra communis; 31-32 Stichomitra
    # stocki"
    f"od_plate_{PAPER_ID}_p019_pl02": {
        "1":  "Dactyliodiscus sp.",
        "2":  "Dactyliodiscus sp.",
        "3":  "Pseudoaulophacus sculptus",
        "4":  "Pseudoaulophacus sculptus",
        "5":  "Pseudoaulophacus putahensis",
        "6":  "Pseudoaulophacus putahensis",
        "7":  "Patellula helios",
        "8":  "Patellula helios",
        "9":  "Patellula ecliptica",
        "10": "Patellula ecliptica",
        "11": "Patellula heroica",
        "12": "Patellula heroica",
        "13": "Patellula sp.",
        "14": "Acanthocircus venetus",
        "15": "Acanthocircus venetus",
        "16": "Acanthocircus tympanum",
        "17": "Acanthocircus tympanum",
        "18": "Acanthocircus hueyi",
        "19": "Acanthocircus hueyi",
        "20": "Dictyomitra formosa",
        "21": "Dictyomitra formosa",
        "22": "Dictyomitra sp.",
        "23": "Dictyomitra montisserei",
        "24": "Dictyomitra montisserei",
        "27": "Pseudodictyomitra pseudomacrocephala",
        "28": "Pseudodictyomitra pseudomacrocephala",
        "29": "Stichomitra communis",
        "30": "Stichomitra communis",
        "31": "Stichomitra stocki",
        "32": "Stichomitra stocki",
    },
}


def main() -> int:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GOLD_DIR / "bandini2006.jsonl"
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
