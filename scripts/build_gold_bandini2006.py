"""Build gold annotations for Bandini 2006 ("Upper Cretaceous
radiolarians from Karnezeika, Argolis Peninsula, Greece").

STATUS: DEFERRED — paper_id mismatch (see CHANGELOG 2026-06-07
"bandini2006 gold removed" entry).

The gold file this script generated used paper_id
`19cd1def9ef08554` for what was supposed to be the Karnezeika
paper. The actual SHA1 of `data/pdfs/bandini2006_greece.pdf` is
`b3113f9ee26cb9f6c085105237d5621942603ee7` — a different paper.
The species in the gold (Archaeocenosphaera, Triactoma,
Pseudoacanthosphaera, Halesium, Pessagnobrachia) are from a
Mesozoic paper with similar SEM-plate layout, not the
Karnezeika paper (which has Dactyliodiscus, Pseudoaulophacus,
Patellula, Acanthocircus, Dictyomitra, Stichomitra species
on its radiolarian plates).

This script is kept as a historical record of what the gold
*was* and as a starting point for re-annotation work against
the correct paper. To rebuild against the actual Karnezeika
PDF, the species list and panel labels must be re-derived
from `work/bandini2006_only_out/output/od_output/19cd1def9ef08554/bandini2006_greece.json`
captions (Plates 1-2, the radiolarian plates; Plate 3 is
foraminifera and out of scope). Plate 1 has only 2 panels
(`Acaeniotyle rebellis`), Plate 2 has 32 panels.

The original caption format we expected:
  "Figures N-M Genus epithet AUTHOR YEAR AlXX_YYY
   (Figs. N and M)"
— multi-line paragraph with each figure range on its own
line. The actual Karnezeika captions match this shape; the
parser coverage on it should be good once the gold is fixed.

The old, mismatched gold is preserved at
`work/bandini2006.jsonl.removed` for reference.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


# Paper ID was the content-based SHA1 of the PDF — but the value
# baked into the old gold (`19cd1def9ef08554`) does not match the
# actual SHA1 of `data/pdfs/bandini2006_greece.pdf`
# (`b3113f9ee26cb9f6c085105237d5621942603ee7`). See the docstring.
PAPER_ID = "b3113f9ee26cb9f6c085105237d5621942603ee7"


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
    raise SystemExit(
        "bandini2006 gold is DEFERRED: paper_id mismatch. "
        "See the docstring at the top of this file and "
        "CHANGELOG.md (2026-06-07, 'bandini2006 gold removed' entry). "
        "The old gold is preserved at work/bandini2006.jsonl.removed. "
        "Re-annotation against the actual Karnezeika PDF is a future task."
    )


if __name__ == "__main__":
    raise SystemExit(main())
