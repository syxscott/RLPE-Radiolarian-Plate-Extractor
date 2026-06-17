"""Re-parse pouille2014 captions with the updated _SPECIES_NAME_RE
(which now captures "Genus? species" forms like "Syntagentactinia? angulata"
in body-text plate references) and refresh the pouille rows in
combined_7_v9.jsonl. The other 6 papers are copied unchanged.

Output: work/combined_7_v10.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from rlpe.m3_engine import _regex_parse_caption  # noqa: E402
from rlpe.opendataloader_extractor import _find_plate_captions  # noqa: E402

PREDICTIONS_IN = ROOT / "work" / "combined_7_v9.jsonl"
PREDICTIONS_OUT = ROOT / "work" / "combined_7_v10.jsonl"

POUILLE_PAPER_ID = "2225994d55021328"
OD_JSON = next((ROOT / "work" / "pouille_recon" / "od_output" / "4dc5b4d95e910e95").glob("*.json"))


def _build_label_to_species() -> dict[str, dict[str, str]]:
    """Map plate-number -> {label -> species} using the reconstructed
    captions produced by _find_plate_captions on the OD output. Skips
    `kind="fig"` captions (Fig. 1 charts) so labels like "Schematic map"
    don't get attached to plate panels."""
    with open(OD_JSON) as f:
        doc = json.load(f)
    captions = _find_plate_captions(doc.get("kids", []))
    out: dict[str, dict[str, str]] = {}
    for c in captions:
        # Reconstructed captions have no `kind`; real plate captions have
        # kind="plate"; fig captions (Schematic map, Pie diagram) are "fig".
        if c.get("kind") == "fig":
            continue
        plate = str(c["plate_number"])
        pairs = _regex_parse_caption(c["content"])
        bucket = out.setdefault(plate, {})
        for p in pairs:
            for lbl in p.labels:
                bucket.setdefault(lbl, p.species)
    return out


def _plate_from_figure_id(figure_id: str) -> str | None:
    """od_plate_XXX_p005_pl01 -> '1'."""
    if not figure_id:
        return None
    if "_pl0" not in figure_id:
        return None
    return figure_id.rsplit("_pl0", 1)[-1]


def main() -> int:
    plate_to_label_to_species = _build_label_to_species()
    print(
        "Loaded pouille plate labels: "
        f"{ {p: len(d) for p, d in plate_to_label_to_species.items()} }"
    )

    rows = [json.loads(l) for l in PREDICTIONS_IN.read_text().splitlines() if l]
    print(f"Loaded {len(rows)} predictions from {PREDICTIONS_IN.name}")

    new_rows: list[dict] = []
    n_updated = 0
    for r in rows:
        if r.get("paper_id") != POUILLE_PAPER_ID:
            new_rows.append(r)
            continue
        plate = _plate_from_figure_id(r.get("figure_id", ""))
        lbl = r.get("panel_id", "")
        species = None
        if plate and lbl:
            species = plate_to_label_to_species.get(plate, {}).get(lbl)
        if species:
            r2 = dict(r)
            r2["species"] = species
            new_rows.append(r2)
            n_updated += 1
        else:
            new_rows.append(r)

    # Add placeholder rows for (plate, label) combinations that have a
    # species in the new parser output but no prediction in v9.
    existing_keys: set[tuple[str, str]] = {
        (r.get("figure_id", ""), r.get("panel_id", ""))
        for r in rows
        if r.get("paper_id") == POUILLE_PAPER_ID
    }
    plate_to_fig: dict[str, str] = {}
    for r in rows:
        if r.get("paper_id") != POUILLE_PAPER_ID:
            continue
        p = _plate_from_figure_id(r.get("figure_id", ""))
        if p and p not in plate_to_fig:
            plate_to_fig[p] = r["figure_id"]

    n_added = 0
    for plate, label_to_species in plate_to_label_to_species.items():
        fig = plate_to_fig.get(plate)
        if not fig:
            continue
        for lbl, sp in label_to_species.items():
            if (fig, lbl) in existing_keys:
                continue
            n_added += 1
            new_rows.append(
                {
                    "paper_id": POUILLE_PAPER_ID,
                    "figure_id": fig,
                    "panel_id": lbl,
                    "species": sp,
                    "panel_path": "",
                    "bbox": [0, 0, 0, 0],
                    "confidence": 0.0,
                    "label_text": lbl,
                    "caption_snippet": "",
                    "ocr_text": "",
                    "metadata": {
                        "matcher_type": "pouille-reparsed-2026-06-07",
                    },
                }
            )

    PREDICTIONS_OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in new_rows) + "\n"
    )
    print(
        f"Updated {n_updated} rows, added {n_added} missing pouille rows; "
        f"wrote {len(new_rows)} rows to {PREDICTIONS_OUT.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
