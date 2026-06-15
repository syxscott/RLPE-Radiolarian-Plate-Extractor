"""Build gold annotations for Bandini 2006 ("Upper Cretaceous
radiolarians from Karnezeika, Argolis Peninsula, Greece").

HISTORY
=======
The original gold for this paper used paper_id `19cd1def9ef08554`,
but the actual SHA1 of `data/pdfs/bandini2006_greece.pdf` is
`b3113f9ee26cb9f6c085105237d5621942603ee7` — a different paper
(an older Mesozoic SEM-plate compendium with Archaeocenosphaera
and Triactoma, not the Karnezeika paper). The old gold was
preserved at `work/bandini2006.jsonl.removed` and removed from
`data/gold/` in 2026-06-07 (v16).

This script now generates a *scaffold* for the **actual** Karnezeika
paper. It runs the production caption parser over the OD JSON
extracted from the real Karnezeika PDF, pre-fills species from
the captions, and emits `work/bandini2006_gold_scaffold.jsonl` for
manual review against panel images.

USAGE
=====
  PYTHONPATH=src python scripts/build_gold_bandini2006.py

Reads:
  work/bandini2006_only_out/output/od_output/19cd1def9ef08554/bandini2006_greece.json
Writes:
  work/bandini2006_gold_scaffold.jsonl   (one row per (figure_id, panel_id))

The scaffold pre-fills species from the Karnezeika caption
("Figure N" / "Figures N-M" lists on Plates 1-2). Coverage:
  Plate 1: 32 panels (figures 3-34 except 1, 2 which are not
    radiolarian panels — they are overview / geological-map /
    stratigraphic-log illustrations)
  Plate 2: 33 panels (figures 1-33)
  Plate 3: excluded — it's a foraminifera plate
    (Helvetoglobotruncana, Marginotruncana, Dicarinella), not
    radiolarian.

After running this script, a domain expert must:
  1. Open each panel image in work/bandini2006_only_out/output/panels/...
  2. Verify the species string in the scaffold
  3. Re-key the rows with paper_id = b3113f9ee26cb9f6c085105237d5621942603ee7
     (NOT the 19cd1def... cache value that the OD JSON path uses)
     and the real figure_id (e.g. od_fig_<paper_id>_p017_pl01) —
     the scaffold uses TODO_PLATE_1/2 placeholders for figure_id
  4. Save as data/gold/bandini2006.jsonl

The script does NOT write data/gold/bandini2006.jsonl directly —
the expert's sign-off is required for the gold to enter the eval
corpus.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from rlpe.m3_engine import _regex_parse_caption, _normalize_species  # noqa: E402
from rlpe.opendataloader_extractor import _find_plate_captions  # noqa: E402

# Paper ID = SHA1 of bandini2006_greece.pdf. The cached OD output
# at 19cd1def9ef08554 was generated with an older hash scheme; the
# correct SHA1 of the Karnezeika PDF is the value below. The
# scaffold script writes the correct paper_id so the eval harness
# can find it after the expert signs off.
PAPER_ID = "b3113f9ee26cb9f6c085105237d5621942603ee7"

OD_JSON = REPO / "work" / "bandini2006_only_out" / "output" / "od_output" \
    / "19cd1def9ef08554" / "bandini2006_greece.json"
OUT_PATH = REPO / "work" / "bandini2006_gold_scaffold.jsonl"


def _find_karnezeika_plate_captions(doc: dict) -> list[tuple[int, str]]:
    """Return [(plate_number, content), ...] for the Karnezeika
    Plates 1-2 (the radiolarian plates). Filters out the
    'Overview' / 'Geological Map' / 'Stratigraphic log' plate-shape
    fig-captions that have "Fig. N. ..." content, AND the
    Plate 3 foraminifera plate (Helvetoglobotruncana, Marginotruncana,
    Dicarinella) — that's a different fossil group, out of scope for
    the radiolarian eval corpus."""
    caps = _find_plate_captions(doc.get("kids", []))
    out: list[tuple[int, str]] = []
    for c in caps:
        if c.get("kind") != "plate":
            continue
        content = c.get("content") or ""
        if "Figure" not in content:
            continue
        plate_number = int(c.get("plate_number", 0))
        if plate_number not in (1, 2):
            continue
        out.append((plate_number, content))
    return out


def _postprocess_pairs(pairs, plate_number: int) -> list[tuple[list[str], str]]:
    """Convert parser output into (labels, normalized_species) tuples
    suitable for the gold scaffold. Applies the v17 _normalize_species
    so the species string matches the corpus convention. Filters
    out labels that look like '190' / '320' / '060' (sample IDs that
    the parser sometimes grabs as a stray 'Figures' clause)."""
    out: list[tuple[list[str], str]] = []
    for p in pairs:
        sp = (p.species or "").strip()
        if p.modifier:
            sp = (sp + " " + p.modifier).strip()
        if not sp:
            continue
        sp = _normalize_species(sp) or sp
        # Filter stray sample-ID labels (3-digit numbers like 190/320/060
        # that come from "Al72_190" suffix in the caption).
        labels = [
            lbl for lbl in p.labels
            if not (re.fullmatch(r"\d{3,4}", lbl))
        ]
        if not labels:
            continue
        out.append((labels, sp))
    return out


def main() -> int:
    if not OD_JSON.exists():
        print(f"ERROR: OD JSON not found at {OD_JSON}")
        return 1
    with open(OD_JSON) as f:
        doc = json.load(f)
    caps = _find_karnezeika_plate_captions(doc)
    if not caps:
        print("ERROR: no Karnezeika plate captions found in OD JSON")
        return 1
    print(f"Found {len(caps)} Karnezeika plate captions:")
    for pn, content in caps:
        print(f"  Plate {pn}: {len(content)} chars")

    rows: list[dict] = []
    for plate_number, content in caps:
        # Heuristic figure_id matches the OD output's naming pattern
        # (page_index is needed to pick the right one; Karnezeika
        # Plate 1 sits on page 16, Plate 2 on page 18 in the PDF —
        # OD labels them p017 and p019 for the body, but the
        # authoritative page is the one where the caption appears).
        # The expert will need to adjust these figure_ids to match
        # the real OD panel extraction output. We mark them as
        # `TODO_PLATE_<N>` so the gap is obvious.
        figure_id = f"TODO_PLATE_{plate_number}"
        pairs = _regex_parse_caption(content)
        clean = _postprocess_pairs(pairs, plate_number)
        if not clean:
            print(f"  Plate {plate_number}: parser produced no clean pairs")
            continue
        print(f"  Plate {plate_number}: {len(clean)} clean pairs")
        for labels, sp in clean:
            for lbl in labels:
                rows.append({
                    "paper_id": PAPER_ID,
                    "figure_id": figure_id,
                    "panel_id": lbl,
                    "species": sp,
                    "metadata": {
                        "source": "build_gold_bandini2006.scaffold",
                        "needs_review": True,
                        "review_notes": (
                            "TODO: verify species against panel image. "
                            "Set figure_id to the real OD figure_id "
                            "(e.g. od_plate_<paper_id>_p017_pl01). "
                            "If the species is wrong, fix it. If the "
                            "panel is missing from this list (e.g. "
                            "Figure 4 Acaeniotyle sp. B), add it "
                            "manually with the correct species."
                        ),
                    },
                })

    OUT_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    )
    print(f"\nWrote {len(rows)} scaffold rows to {OUT_PATH.relative_to(REPO)}")
    print("Expert review required before moving to data/gold/bandini2006.jsonl.")
    print(f"Unique figure_ids (placeholders): {sorted(set(r['figure_id'] for r in rows))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
