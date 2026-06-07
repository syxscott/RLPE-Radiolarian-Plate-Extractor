"""Build gold annotations for Feng 2007 (JMicro, Latest Permian Entactinaria).

Source captions come from the paper's Plates 1-5 (the paper has a
6th plate but with mostly indeterminate specimens — we skip it for
now to keep the gold strict).

Plate captions in this paper use the "figs N-M. Species" convention
that the standard _CAPTION_CLAUSE_RE handles. Some captions are split
across a paragraph and a list structure in the OpenDataLoader output;
we concatenate them here.

Figure_id values are taken from the actual pipeline run on this paper
(see work/feng_only_out/output/manifests/matches.jsonl).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GOLD_DIR = REPO / "data" / "gold"


# Paper ID is the content-based SHA1 of the PDF
PAPER_ID = "e28de2b07edc8950"


PLATE_FIGURES = {
    # Plate 1 (20 panels): figs 1-2. E. itsukichiensis; 3-4. E. reticulata;
    # 5-6. E. modesta; 7-8. E. minuta; 9-20. E. wangi
    f"od_plate_{PAPER_ID}_p006_pl01": {
        "1": "Entactinia itsukichiensis",
        "2": "Entactinia itsukichiensis",
        "3": "Entactinia reticulata",
        "4": "Entactinia reticulata",
        "5": "Entactinia modesta",
        "6": "Entactinia modesta",
        "7": "Entactinia minuta",
        "8": "Entactinia minuta",
        "9": "Entactinia wangi",
        "10": "Entactinia wangi",
        "11": "Entactinia wangi",
        "12": "Entactinia wangi",
        "13": "Entactinia wangi",
        "14": "Entactinia wangi",
        "15": "Entactinia wangi",
        "16": "Entactinia wangi",
        "17": "Entactinia wangi",
        "18": "Entactinia wangi",
        "19": "Entactinia wangi",
        "20": "Entactinia wangi",
    },
    # Plate 2 (24 panels): figs 1-6. E. sashidai; 7-8. Entactinia sp. 1;
    # 9. Entactinia sp. 2; 10. E. meishanensis; 11-13. T. brevispinosa;
    # 14. T. pseudocimelia; and a few more species further in caption
    f"od_plate_{PAPER_ID}_p008_pl02": {
        "1": "Entactinia sashidai",
        "2": "Entactinia sashidai",
        "3": "Entactinia sashidai",
        "4": "Entactinia sashidai",
        "5": "Entactinia sashidai",
        "6": "Entactinia sashidai",
        "7": "Entactinia sp. 1",
        "8": "Entactinia sp. 1",
        "9": "Entactinia sp. 2",
        "10": "Entactinia meishanensis",
        "11": "Trilonche brevispinosa",
        "12": "Trilonche brevispinosa",
        "13": "Trilonche brevispinosa",
        "14": "Trilonche pseudocimelia",
        "15": "Trilonche pseudocimelia",
        "16": "Trilonche pseudocimelia",
    },
    # Plate 3 (16 panels): figs 1-5. T. crassispinosa; 6, 9-12. K. magna;
    # 7-8. Trilonche? sp. 1; plus more
    f"od_plate_{PAPER_ID}_p010_pl03": {
        "1": "Trilonche crassispinosa",
        "2": "Trilonche crassispinosa",
        "3": "Trilonche crassispinosa",
        "4": "Trilonche crassispinosa",
        "5": "Trilonche crassispinosa",
        "6": "Kashiwara magna",
        "7": "Trilonche? sp. 1",
        "8": "Trilonche? sp. 1",
        "9": "Kashiwara magna",
        "10": "Kashiwara magna",
        "11": "Kashiwara magna",
        "12": "Kashiwara magna",
    },
    # Plate 4 (30 panels): figs 1-5. T. variabilis; 6-9. T. minutus;
    # 10. Triaenosphaera sp. 4; 11. Triaenosphaera sp. 1; 12. Triaenosphaera sp. 5;
    # 13. Triaenosphaera sp. 2; 14-16. T. megacantha
    f"od_plate_{PAPER_ID}_p012_pl04": {
        "1": "Triaenosphaera variabilis",
        "2": "Triaenosphaera variabilis",
        "3": "Triaenosphaera variabilis",
        "4": "Triaenosphaera variabilis",
        "5": "Triaenosphaera variabilis",
        "6": "Triaenosphaera minutus",
        "7": "Triaenosphaera minutus",
        "8": "Triaenosphaera minutus",
        "9": "Triaenosphaera minutus",
        "10": "Triaenosphaera sp. 4",
        "11": "Triaenosphaera sp. 1",
        "12": "Triaenosphaera sp. 5",
        "13": "Triaenosphaera sp. 2",
        "14": "Triaenosphaera megacantha",
        "15": "Triaenosphaera megacantha",
        "16": "Triaenosphaera megacantha",
    },
    # Plate 5 (22 panels): figs 1-4. P. densa; 5-8, 18, 20. P. ormistoni;
    # 9-12, 19. W. dongpanica; 13, 17. U. virgispinosum; 14-16. H. mammilla
    f"od_plate_{PAPER_ID}_p014_pl05": {
        "1": "Provisocyntra densa",
        "2": "Provisocyntra densa",
        "3": "Provisocyntra densa",
        "4": "Provisocyntra densa",
        "5": "Provisocyntra ormistoni",
        "6": "Provisocyntra ormistoni",
        "7": "Provisocyntra ormistoni",
        "8": "Provisocyntra ormistoni",
        "9": "Wuyia dongpanica",
        "10": "Wuyia dongpanica",
        "11": "Wuyia dongpanica",
        "12": "Wuyia dongpanica",
        "13": "Uberinterna virgispinosum",
        "14": "Hegleria mammilla",
        "15": "Hegleria mammilla",
        "16": "Hegleria mammilla",
        "17": "Uberinterna virgispinosum",
        "18": "Provisocyntra ormistoni",
        "19": "Wuyia dongpanica",
        "20": "Provisocyntra ormistoni",
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
    out = GOLD_DIR / "feng2007.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


if __name__ == "__main__":
    n = build()
    print(f"Wrote {n} gold entries to {GOLD_DIR / 'feng2007.jsonl'}")
