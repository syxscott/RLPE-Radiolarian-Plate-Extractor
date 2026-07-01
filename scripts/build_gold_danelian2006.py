"""Build gold annotations for Danelian 2006 (Eclogae Geol. Helv. 99 Suppl. 1,
S21-S33, "Upper Jurassic Radiolaria from the Vocontian basin of SE France").

The paper's plate captions use:
  1. **Genus-initial abbreviation** after the full genus is established earlier
     in the same plate:
        "5-6) Archaeodictyomitra patricki KOCHER, Mg-29; 7) A. patricki, Mg-2"
     Both "Archaeodictyomitra patricki" and "A. patricki" refer to the same
     species; the gold originally kept the abbreviation form, which the LLM
     never produces (the LLM always emits the full genus). This caused 19/42
     danelian mismatches.
  2. **"sp.aff. X" / "sp.cf. X" qualifiers** (e.g. "Archaeodictyomitra
     sp.aff. A. patricki") — the original gold captured these as bare
     "Archaeodictyomitra sp", losing the "aff. X" specificity.
  3. **"sp. A" / "sp. B" morphotype identifiers** — the original gold
     captured these as bare "Archaeodictyomitra sp", losing the morphotype.

This script regenerates `data/gold/danelian2006.jsonl` by:
  1. Running the production caption parser on the real PDF
  2. Walking the parsed (label, species) pairs in caption order, expanding
     "A. epithet" using the most-recent full genus starting with "A" in
     the same plate
  3. Preserving "sp.aff.", "sp.cf.", "sp. A" qualifiers intact
  4. Writing to data/gold/danelian2006.jsonl with paper_id = SHA1 hash
     of the PDF (not the placeholder "danelian2006" used during scaffold)
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rlpe.evaluation import GoldPanel, write_gold  # noqa: E402
from rlpe.m3_engine import _regex_parse_caption  # noqa: E402

# Stable paper_id used by all existing cached predictions and the
# live LLM-first run. Kept as a string (not SHA1) for backward
# compatibility with v18_fixed and earlier cached eval baselines.
PAPER_ID = "17a129b4e9ca975a"

GOLD_DIR = REPO / "data" / "gold"
PDF_PATH = REPO / "data" / "pdfs" / "danelian2006.pdf"


def _get_paper_id(pdf_path: Path) -> str:
    """Return the stable paper_id (see PAPER_ID comment)."""
    return PAPER_ID


# The danelian paper has two plates. We re-parse the captions and expand
# genus-initial abbreviations. Plate 1 + Plate 2 are separate figure groups;
# abbreviation context is plate-local (e.g. "A." means "Archaeodictyomitra"
# within plate 1 because that's the most-recent "A" genus, not because the
# paper has a global convention).
PLATE1_CAPTION = """Plate 1

Scanning Electron Micrographs of Radiolaria extracted from samples of the Méouge section. Bar scale (upper right) is equal to 100 µm for all figures.

1) Acastea sp.cf. A. remusa HULL, Mg-100; 2-3) Archaeodictyomitra apiarium (RÜST), Mg-2; 4) Archaeodictyomitra etrusca CHIARI et al. Mg-29; 5-6) Archaeodictyomitra patricki KOCHER, Mg-29; 7) A. patricki, Mg-2; 8) Archaeodictyomitra sp.aff. A. patricki KOCHER, Mg-2 ; 9) Archaeodictyomitra shengi YANG, Mg-29; 10) A. shengi, Mg-77; 11) Archaeodictyomitra spelae CHIARI et al., Mg-77; 12) Archaeodictyomitra sp. A, Mg-2; 13) Cinguloturris fusiforma HORI, Mg-100; 14) Cinguloturris sp.cf. C. fusiforma HORI, Mg-77; 15) Emiluvia pentaporata STEIGER & STEIGER, Mg-100; 16) Eucyrtidiellum ptyctum (SANFILIPPO & RIEDEL), Morphotype A, Mg-77 ; 17) E. ptyctum, Morphotype B, Mg-2; 18) Gongylothorax favosus DUMITRICA, Mg-37; 19) Loopus doliolum DUMITRICA, Mg-37; 20) L. doliomum, Mg-29; 21) Loopus venustus (CHIARI et al.), Mg-2; 22) L. venustus, Mg-2; 23) L. venustus, Mg-133."""

PLATE2_CAPTION = """Plate 2

Scanning Electron Micrographs of Radiolaria extracted from samples of the Méouge section. Bar scale (upper right) is equal to 100 µm for all figures except of fig. 7 (=200 µm).

1) Pantanellium oligoporum (VINASSA), Mg-2; 2) Praeconocaryomma scatebra HULL, Mg-21; 3) Protunuma japonicus MATSUOKA & YAO, Mg-37; 4) P. japonicus, Mg-37; 5) Ristola altissima cf.ssp. R. a. altissima (RÜST) sensu Baumgartner et al. 1995b, Mg-37; 6) Saitoum pagei PESSAGNO, Mg-21; 7) Spongocapsula palmerae (PESSAGNO), Mg-21; 8) Stichocapsa tuscanica CHIARI, CORTESE & MARCUCCI, Mg-77; 9) Suna sp., Mg-77; 10) Tethysetta (?) sp., Mg-77; 11) Triactoma foremanae MUZAVOR; Mg-77; 12) Tripocyclia sp.cf. T.luciae JUD, Mg-100; 13) Tritrabs sp.cf. T. exotica (PESSAGNO), Mg-37; 14) Williriedellum carpathicum DUMITRICA, Mg77 ; 15) W. carpathicum, Mg-21; 16) Williriedellum crystallinum DUMITRICA, Mg-37; 17) Zhamoidellum ovum DUMITRICA, Mg-37 ; 18) Z. ovum, Mg-21; 19) Z. ovum, Mg-2."""


def _expand_abbreviations(plate: int, pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Walk (label, species) pairs in caption order. When a species is
    "X. epithet" (genus-initial + epithet), find the most-recent full
    "Genus epithet" in the same plate and expand. Preserves
    "sp.aff. X", "sp.cf. X", "sp. A" qualifiers intact.

    Returns a new list with expanded species strings.
    """
    out: list[tuple[str, str]] = []
    # recent_pairs maps initial letter -> most-recent full genus
    recent_pairs: dict[str, str] = {}

    for label, species in pairs:
        m = re.match(r"^([A-Z])\.\s+(.+)$", species)
        if m:
            initial, rest = m.group(1), m.group(2)
            # Look for the most-recent full genus starting with this initial
            # in the same plate. If we have one, expand.
            full_genus = recent_pairs.get(initial)
            if full_genus:
                expanded = f"{full_genus} {rest}"
                out.append((label, expanded))
                continue
        # Either a full genus name, or an abbreviation with no context.
        # Update recent_pairs with the leading genus if we can identify it.
        m2 = re.match(r"^([A-Z][a-z]+)\s+", species)
        if m2:
            genus = m2.group(1)
            recent_pairs[genus[0].upper()] = genus
        out.append((label, species))
    return out


def _build_gold_for_plate(plate: int, paper_id: str, caption: str) -> list[GoldPanel]:
    """Build GoldPanel list for one plate."""
    # First pass: parse with the regex parser to get (label, species, [modifier]) tuples
    raw_pairs = _regex_parse_caption(caption)
    # raw_pairs is list of CaptionPair(label=list, species=str, modifier=str)
    # Flatten to (label, species) for our expand function.
    flat: list[tuple[str, str]] = []
    for cp in raw_pairs:
        sp = cp.species.strip()
        if cp.modifier:
            sp = (sp + " " + cp.modifier).strip()
        for lbl in (cp.labels or []):
            lbl_s = lbl.strip()
            if lbl_s:
                flat.append((lbl_s, sp))

    expanded = _expand_abbreviations(plate, flat)

    # Plate 1 is on page 11 of the danelian paper, Plate 2 on page 13.
    # (Plate N sits 2 pages after plate N-1 because the in-between
    # pages contain a Fig. N location map + caption).
    plate_pages = {1: 11, 2: 13}
    figure_id = f"od_plate_{paper_id}_p{plate_pages[plate]:03d}_pl0{plate}"

    panels: list[GoldPanel] = []
    for label, species in expanded:
        panels.append(
            GoldPanel(
                paper_id=paper_id,
                figure_id=figure_id,
                panel_id=label,
                species=species,
            )
        )
    return panels


def main() -> int:
    if not PDF_PATH.exists():
        print(f"ERROR: {PDF_PATH} not found")
        return 1
    paper_id = _get_paper_id(PDF_PATH)
    print(f"Paper ID (SHA1): {paper_id}")

    all_panels: list[GoldPanel] = []
    all_panels.extend(_build_gold_for_plate(1, paper_id, PLATE1_CAPTION))
    all_panels.extend(_build_gold_for_plate(2, paper_id, PLATE2_CAPTION))

    out_path = GOLD_DIR / "danelian2006.jsonl"
    write_gold(all_panels, out_path)
    print(f"Wrote {len(all_panels)} gold entries to {out_path}")
    # Show sample
    print("\nFirst 5 entries:")
    for p in all_panels[:5]:
        print(f"  panel {p.panel_id}: {p.species}")
    print("\nAbbreviated-species rows (should be 0):")
    for p in all_panels:
        if re.match(r"^[A-Z]\.\s+", p.species):
            print(f"  panel {p.panel_id}: {p.species}  ← STILL ABBREVIATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
