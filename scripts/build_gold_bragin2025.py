"""Build gold annotations for Bragin 2025 (Oxfordian-Kimmeridgian
radiolarians from the Nordvik section, northern Siberia).

Source: Bragin 2025, Plate I (11 panels), family Parvicingulidae.

The Bragin 2025 caption format is unique among the radiolarian
literature: panel labels are wrapped in parenthesised form
"(1) Species" rather than the more common "1) Species" (Danelian) or
"1, 2- Species" (Baumgartner) conventions. The whole plate caption is a
single chunk with a "Plate I. ...prose..." preamble, so the
caption-parser fall-through chain had to be extended (see
``_DANELIAN_CLAUSE_RE`` in ``m3_engine.py`` and the new
``tests/test_bragin_caption_parser.py``).

Paper_id convention: the other 7 gold files use a content-based SHA1
of the PDF (the ``paper_id`` produced by ``stable_id`` in
``rlpe.utils``). For Bragin 2025 we do NOT have the actual PDF in
``data/pdfs/`` yet, so we use a human-readable identifier
``bragin2025``. The eval harness accepts any string as paper_id — it
only needs to match between gold and predictions.

This is a parser-regression gold set: it locks in the Bragin caption
format and exercises the (N) Species parenthesised parser path. To
include Bragin in a full end-to-end eval, copy the actual PDF into
``data/pdfs/`` and re-run the pipeline; the SHA1 paper_id will then
replace the placeholder.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GOLD_DIR = REPO / "data" / "gold"


PAPER_ID = "bragin2025"


# Plate I (11 panels): family Parvicingulidae from the Nordvik section
PLATE_FIGURES = {
    f"od_plate_{PAPER_ID}_p001_pl01": {
        "1": "Praeparvicingula blackhorsensis",
        "2": "Praeparvicingula donnae",
        "3": "Parvicingula khabakovi",
        "4": "Nordvikella elegans",
        "5": "Arctocapsula perforata",
        "6": "Echinocampe aliferum",
        "7": "Echinocampe modestum",
        "8": "Pantanellium tierrablancaense",
        "9": "Pantanellium sp.",  # cf. P. tierrablancaense — disambiguator
        "10": "Pantanellium moscowiense",
        "11": "Pantanellium moscowiense",  # (10, 11) range expanded
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
    out = GOLD_DIR / "bragin2025.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


if __name__ == "__main__":
    n = build()
    print(f"Wrote {n} gold entries to {GOLD_DIR / 'bragin2025.jsonl'}")
