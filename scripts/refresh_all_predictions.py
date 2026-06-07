"""Refresh ALL papers' species assignments by re-parsing OD JSON
captions with the latest m3_engine regex (which now handles
"Genus? species" forms like "Periphaena? duplus", trinomials like
"Lamptonium fabaeforme fabaeforme", and inline body-text references).

For each paper:
  1. Locate the OD JSON
  2. Call _find_plate_captions + _regex_parse_caption to get
     plate-number -> {label -> species} mapping
  3. Update species in the prediction rows where (figure_id plate, panel_id)
     matches a parsed (plate, label) pair
  4. Add missing (figure_id, label) placeholder rows for parsed pairs
     that don't have a corresponding prediction

Output: work/combined_7_v11.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from rlpe.m3_engine import _regex_parse_caption  # noqa: E402
from rlpe.opendataloader_extractor import _find_plate_captions  # noqa: E402

PREDICTIONS_IN = ROOT / "work" / "combined_7_v11.jsonl"
PREDICTIONS_OUT = ROOT / "work" / "combined_7_v12.jsonl"

OD_JSON_BY_PAPER = {
    "4f1bf415485765b8": ROOT / "work" / "all7_rerun" / "output" / "od_output"
        / "4f1bf415485765b8" / "bandini2011.json",
    "58d7972c37307959": ROOT / "work" / "all7_rerun" / "output" / "od_output"
        / "58d7972c37307959" / "baumgartner2008.json",
    "178d4e1e9d93136c": ROOT / "work" / "boughdiri_rerun" / "output" / "od_output"
        / "178d4e1e9d93136c" / "boughdiri2007.json",
    "17a129b4e9ca975a": ROOT / "work" / "all7_rerun" / "output" / "od_output"
        / "17a129b4e9ca975a" / "danelian2006.json",
    "e28de2b07edc8950": ROOT / "work" / "feng_rerun_v2" / "output" / "od_output"
        / "e28de2b07edc8950" / "feng2007.json",
    "a0f363c21b6941d7": ROOT / "work" / "all7_rerun" / "output" / "od_output"
        / "a0f363c21b6941d7" / "hollis2006.json",
    "2225994d55021328": next(
        (ROOT / "work" / "pouille_recon" / "od_output" / "4dc5b4d95e910e95").glob("*.json")
    ),
}


def _build_label_to_species_for(paper_id: str) -> dict[str, dict[str, str]]:
    """Map plate-number -> {label -> species} for a single paper.
    Skips kind='fig' captions (those are chart/diagram captions, not plates)."""
    od_path = OD_JSON_BY_PAPER[paper_id]
    if not od_path.exists():
        return {}
    with open(od_path) as f:
        doc = json.load(f)
    captions = _find_plate_captions(doc.get("kids", []))
    out: dict[str, dict[str, str]] = {}
    for c in captions:
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
    if not figure_id:
        return None
    if "_pl0" not in figure_id:
        return None
    return figure_id.rsplit("_pl0", 1)[-1]


def main() -> int:
    paper_to_mapping: dict[str, dict[str, dict[str, str]]] = {}
    for pid in OD_JSON_BY_PAPER:
        paper_to_mapping[pid] = _build_label_to_species_for(pid)
        plate_counts = {p: len(d) for p, d in paper_to_mapping[pid].items()}
        print(f"{pid}: {plate_counts}")

    # Build (paper_id, plate_number) -> caption_page from the OD JSON.
    # The "Plate N" or "Explanation of Plate N" caption text sits on a
    # specific page; the actual plate image is typically on the same
    # page, the next page, or two pages later. We use this to pick the
    # right figure_id from candidates that point to chart pages vs
    # real plate pages.
    caption_page_by_plate: dict[tuple[str, str], int] = {}
    for pid, od_path in OD_JSON_BY_PAPER.items():
        if not od_path.exists():
            continue
        with open(od_path) as f:
            doc = json.load(f)
        for k in doc.get("kids", []):
            pg = k.get("page number")
            if pg is None:
                continue
            txt = (k.get("content", "") or "").strip()
            m = re.match(r"(?:Explanation of )?Plate\s+(\d+)", txt)
            if m:
                plate = m.group(1)
                if (pid, plate) not in caption_page_by_plate:
                    caption_page_by_plate[(pid, plate)] = int(pg)

    rows = [
        json.loads(l)
        for l in PREDICTIONS_IN.read_text().splitlines()
        if l
    ]
    print(f"\nLoaded {len(rows)} predictions from {PREDICTIONS_IN.name}")

    # Compute the chosen figure_id per (paper, plate) BEFORE processing
    # any rows. Used to filter out rows that point to a chart page
    # mis-identified by OD as a plate (e.g. baum p002 vs p015).
    plate_to_fig: dict[str, dict[str, str]] = {}

    # Add missing placeholder rows for (paper, plate, label) combinations
    # that exist in the parsed mapping but not in the predictions
    existing_keys: dict[str, set[tuple[str, str]]] = {}
    for r in rows:
        pid = r.get("paper_id")
        existing_keys.setdefault(pid, set()).add(
            (r.get("figure_id", ""), r.get("panel_id", ""))
        )

    plate_to_fig: dict[str, dict[str, str]] = {}
    # Score each (paper, plate, fig) candidate. The real plate page
    # has the highest count of panels whose label is in the parsed
    # caption's label→species map. Chart pages mis-identified as
    # plates by OD (e.g. baum p002 "Fig. 1 terrane map" mis-classified
    # as pl01) have panels with labels that don't appear in the parsed
    # caption (e.g. "T", "d", "P1", "Z", "B", "N", "0", "98" for
    # text-fragment noise). Tiebreak on total panel count.
    plate_fig_matched: dict[tuple[str, str, str], int] = {}
    plate_fig_total: dict[tuple[str, str, str], int] = {}
    plate_fig_valid_bbox: dict[tuple[str, str, str], int] = {}
    for r in rows:
        pid = r.get("paper_id")
        fig = r.get("figure_id", "")
        p = _plate_from_figure_id(fig)
        if not (pid and p):
            continue
        lbl = (r.get("panel_id", "") or "").strip()
        key = (pid, p, fig)
        plate_fig_total[key] = plate_fig_total.get(key, 0) + 1
        if lbl in paper_to_mapping.get(pid, {}).get(p, {}):
            plate_fig_matched[key] = plate_fig_matched.get(key, 0) + 1
        bbox = r.get("bbox", [0, 0, 0, 0])
        # Bbox is stored as (x, y, w, h); validity = positive width/height.
        if (
            isinstance(bbox, list)
            and len(bbox) == 4
            and bbox[2] > 0
            and bbox[3] > 0
        ):
            plate_fig_valid_bbox[key] = plate_fig_valid_bbox.get(key, 0) + 1
    grouped: dict[tuple[str, str], list[tuple[str, int, int, int]]] = {}
    for (pid, plate, fig), total in plate_fig_total.items():
        grouped.setdefault((pid, plate), []).append(
            (fig, plate_fig_matched.get((pid, plate, fig), 0), total,
             plate_fig_valid_bbox.get((pid, plate, fig), 0))
        )
    for (pid, plate), candidates in grouped.items():
        # If we have a caption-page hint (the page where OD found the
        # "Plate N" caption), prefer the candidate whose page is
        # closest to caption_page + 1 (typical layout: caption on
        # left page, plate image on right page). This is the most
        # reliable signal for distinguishing real plates from chart
        # pages mis-identified as plates by OD.
        cap_page = caption_page_by_plate.get((pid, plate))
        if cap_page is not None:
            scored = []
            for fig, digit, total, vbbox in candidates:
                m = re.search(r"_p(\d+)_pl0", fig)
                page = int(m.group(1)) if m else 0
                # Try offsets 0, 1, 2 (caption can be on same page,
                # right-page, or two pages before the image). Pick
                # the smallest offset that hits an existing candidate.
                best_off = min(
                    (abs(page - (cap_page + off)), off)
                    for off in (0, 1, 2)
                )
                scored.append((best_off[0], -vbbox, -total, fig))
            scored.sort()
            best = scored[0][3]
        else:
            # No caption-page hint. Use valid-bbox count as the primary
            # signal (real plate images have well-formed bboxes);
            # chart-page text fragments have broken or zero-area bboxes.
            best = max(candidates, key=lambda x: (x[3], x[2]))[0]
        plate_to_fig.setdefault(pid, {})[plate] = best

    # Now process input rows: drop any that point to a figure_id we
    # did NOT select for their (paper, plate). Keeps panel detection
    # and bounding boxes from the kept rows. The dropped rows are
    # chart-page noise (e.g. baum p002) that the OD pass mis-classified
    # as a plate — they would otherwise persist in the output and
    # pollute the evaluation.
    kept: list[dict] = []
    n_dropped = 0
    for r in rows:
        pid = r.get("paper_id")
        fig = r.get("figure_id", "")
        plate = _plate_from_figure_id(fig)
        if pid and plate and plate in plate_to_fig.get(pid, {}):
            chosen_fig = plate_to_fig[pid][plate]
            if fig != chosen_fig:
                n_dropped += 1
                continue
        kept.append(r)
    print(f"Dropped {n_dropped} rows pointing to non-chosen figure_ids")

    new_rows: list[dict] = []
    per_paper_updated: dict[str, int] = {}
    for r in kept:
        pid = r.get("paper_id")
        plate = _plate_from_figure_id(r.get("figure_id", ""))
        lbl = r.get("panel_id", "")
        species = None
        if pid in paper_to_mapping and plate and lbl:
            species = paper_to_mapping[pid].get(plate, {}).get(lbl)
        if species:
            r2 = dict(r)
            r2["species"] = species
            new_rows.append(r2)
            per_paper_updated[pid] = per_paper_updated.get(pid, 0) + 1
        else:
            new_rows.append(r)

    per_paper_added: dict[str, int] = {}
    for pid, mapping in paper_to_mapping.items():
        for plate, label_to_species in mapping.items():
            fig = plate_to_fig.get(pid, {}).get(plate)
            if not fig:
                continue
            for lbl, sp in label_to_species.items():
                if (fig, lbl) in existing_keys.get(pid, set()):
                    continue
                per_paper_added[pid] = per_paper_added.get(pid, 0) + 1
                new_rows.append({
                    "paper_id": pid,
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
                        "matcher_type": "all-reparsed-2026-06-07",
                    },
                })

    PREDICTIONS_OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in new_rows) + "\n"
    )
    print(f"\nWrote {len(new_rows)} rows to {PREDICTIONS_OUT.name}")
    print("Updates per paper:")
    for pid in OD_JSON_BY_PAPER:
        upd = per_paper_updated.get(pid, 0)
        add = per_paper_added.get(pid, 0)
        print(f"  {pid}: updated={upd}, added={add}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
