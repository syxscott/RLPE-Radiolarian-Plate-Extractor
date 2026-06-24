"""Re-parse baumgartner2008 captions with the updated _BAUMGARTNER_CLAUSE_RE
(which now preserves trailing single-letter species identifiers like "S"
in "Williriedellum sp. S" and "W. sp. S") and refresh the baum rows in
combined_7_v8.jsonl. The other 6 papers are copied unchanged.

Output: work/combined_7_v9.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from rlpe.m3_engine import _regex_parse_caption  # noqa: E402

PREDICTIONS_IN = ROOT / "work" / "combined_7_v8.jsonl"
PREDICTIONS_OUT = ROOT / "work" / "combined_7_v9.jsonl"

BAUM_PAPER_ID = "58d7972c37307959"
OD_JSON = (
    ROOT
    / "work"
    / "baum_rerun_v5"
    / "output"
    / "od_output"
    / BAUM_PAPER_ID
    / "baumgartner2008.json"
)


def _load_plate_captions() -> dict[str, str]:
    with open(OD_JSON) as f:
        d = json.load(f)
    plate_to_caption: dict[str, str] = {}
    for k in d.get("kids", []):
        if k.get("type") != "caption":
            continue
        txt = k.get("content", "") or ""
        for plate in ("1", "2", "3"):
            marker = f"Plate {plate} -"
            if marker in txt and plate not in plate_to_caption:
                plate_to_caption[plate] = txt
    return plate_to_caption


def _build_label_to_species() -> dict[str, dict[str, str]]:
    """Map (plate, label) -> species. Plate is parsed from figure_id
    suffix _pl0N."""
    plate_to_caption = _load_plate_captions()
    out: dict[str, dict[str, str]] = {}
    for plate, caption in plate_to_caption.items():
        pairs = _regex_parse_caption(caption)
        out[plate] = {}
        for p in pairs:
            for lbl in p.labels:
                out[plate][lbl] = p.species
    return out


def _plate_from_figure_id(figure_id: str) -> str | None:
    """od_plate_58d7972c37307959_p015_pl01 -> '1'."""
    if not figure_id:
        return None
    if "_pl0" not in figure_id:
        return None
    return figure_id.rsplit("_pl0", 1)[-1]


def main() -> int:
    plate_to_label_to_species = _build_label_to_species()
    print(
        f"Loaded baum plate labels: { {p: len(d) for p, d in plate_to_label_to_species.items()} }"
    )

    rows = [json.loads(l) for l in PREDICTIONS_IN.read_text().splitlines() if l]
    print(f"Loaded {len(rows)} predictions from {PREDICTIONS_IN.name}")

    new_rows: list[dict] = []
    n_updated = 0
    n_added = 0
    for r in rows:
        if r.get("paper_id") != BAUM_PAPER_ID:
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

    # For each plate, add rows for any labels that have species in the
    # new parser output but no prediction in the v8 file.
    existing_keys: set[tuple[str, str]] = {
        (r.get("figure_id", ""), r.get("panel_id", ""))
        for r in rows
        if r.get("paper_id") == BAUM_PAPER_ID
    }
    # We need figure_id per plate; build a map from (plate) -> figure_id
    # by scanning existing rows.
    plate_to_fig: dict[str, str] = {}
    for r in rows:
        if r.get("paper_id") != BAUM_PAPER_ID:
            continue
        p = _plate_from_figure_id(r.get("figure_id", ""))
        if p and p not in plate_to_fig:
            plate_to_fig[p] = r["figure_id"]

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
                    "paper_id": BAUM_PAPER_ID,
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
                        "matcher_type": "baum-reparsed-2026-06-07",
                    },
                }
            )

    PREDICTIONS_OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in new_rows) + "\n"
    )
    print(
        f"Updated {n_updated} rows, added {n_added} missing baum rows; "
        f"wrote {len(new_rows)} rows to {PREDICTIONS_OUT.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
