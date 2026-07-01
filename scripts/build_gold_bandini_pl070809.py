"""Build verified gold for bandini2011 plates 7/8/9 from the actual PDF
caption text on pages 24-26 of `data/pdfs/bandini2011.pdf`.

This is the gold-update that closes the v19 regression: the v19
caption-parser fix recovers 22+24+18 panels for pl07/08/09 that the
06-29 pipeline silently routed to bogus `od_fig_*` IDs. With the
gold only annotating pl01-06 + 1 panel in pl07, the recovered panels
counted as FPs and dragged the 9-paper aggregate F1 from 87.45% to
83.70%.

Verified caption text (from `data/pdfs/bandini2011.pdf`, pages 24-26):

  Plate 7 (page 24, sample PR-SB26, latest Barremian-late Aptian):
    Fig. 1, 2  Archaeodictyomitra spp.
    Fig. 3      Thanarla sp.
    Fig. 4      Parvicingula sp.
    Fig. 5      Pseudoeucyrtis corpulentus DUMITRICA
    Fig. 6      Obesacapsula sp.
    Fig. 7      Pseudoeucyrtis hanni (TAN)
    Fig. 8      Obeliscoites cf. vinassai (SQUINABOL)
    Fig. 9      Nassellaria gen. et sp. indet.
    Fig. 10     Napora sp.
    Fig. 11     Nassellaria gen. et sp. indet.
    Fig. 12     Acaeniotyle cf. umbilicata (RU¨ ST)
    Fig. 13     Emiluvia sp.
    Fig. 14     Acanthocircus sp.
    Fig. 15     Crucella sp.
    Fig. 16     Pantanelliidae gen. et sp. indet.
    Fig. 17     Crucella sp. (with five rays)
    Fig. 18     Praeconocaryomma sp.

  Plate 8 (page 25, sample PR-SB27, late Aptian-middle Cenomanian):
    Figs. 1, 2  Archaeodictyomitra cf. immenhauseri DUMITRICA
    Fig. 3      Archaeodictyomitra cf. gracilis (SQUINABOL)
    Figs. 4, 5  Archaeodictyomitra montisserei (SQUINABOL)
    Figs. 6, 7  Archaeodictyomitra spp.
    Figs. 8, 9  Stichomitra (?) spp.
    Fig. 10     Thanarla aff. veneta (SQUINABOL)
    Fig. 11     Thanarla brouweri (TAN)
    Fig. 12     Thanarla sp.
    Figs. 13-15 Archaeospongoprunum sp.
    Figs. 16, 17 Stylosphaera spp.
    Fig. 18     Halesium (?) cf. palmatum DUMITRICA
    Figs. 19, 20 Pantanellium sp.
    Fig. 21     Quadrigastrum lapideum O'DOGHERTY
    Fig. 22     Spumellaria gen. et sp. indet

  Plate 9 (page 26, samples PR-SB28 + PR-SB30):
    PR-SB28 (late early Albian-middle Cenomanian):
      Figs. 1, 2  Archaeodictyomitra gracilis (SQUINABOL) gr.
      Fig. 3      Archaeodictyomitra sp.
      Fig. 4      Stichomitra (?) sp.
      Fig. 5      Squinabollum aff. fossile (SQUINABOL)
      Fig. 6      Neosciadiocapsidae PESSAGNO gen. et sp. indet.
    PR-SB30 (Barremian):
      Fig. 7      Pseudodictyomitra leptoconica (FOREMAN)
      Fig. 8      Pseudodictyomitra sp.
      Fig. 9      Archaeodictyomitra sp.
      Fig. 10     Tuguriella sp.
      Figs. 11, 15 Hiscocapsa (?) spp.
      Fig. 12     Hiscocapsa aff. asseni (TAN)
      Fig. 13     Hiscocapsa uterculus (PARONA)
      Fig. 14     Zhamoidellum cf. testatum JUD
      Fig. 16     Syringocapsa (?) limatum FOREMAN
      Fig. 17     Pseudocrucella (?) elisabethae (RU¨ ST)
      Fig. 18     Praeconocaryomma sp.

Author citations are stripped (the gold convention in bandini2011.jsonl
suppresses them — only the species epithet is recorded; this matches
the existing pl01-06 annotations).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rlpe.evaluation import GoldPanel, write_gold  # noqa: E402

PAPER_ID = "4f1bf415485765b8"

# Author citations stripped — bandini2011 gold convention (matches pl01-06).
# Each entry is (figure_id, panel_id, species) parsed from the PDF caption text.
NEW_ENTRIES: list[tuple[str, str, str]] = []

# ---------------------------------------------------------------------------
# Plate 7 (page 24) — 18 panels, sample PR-SB26
# ---------------------------------------------------------------------------
PL07_FIG = "od_plate_4f1bf415485765b8_p024_pl07"
PL07: list[tuple[str, str]] = [
    ("1", "Archaeodictyomitra spp."),
    ("2", "Archaeodictyomitra spp."),
    ("3", "Thanarla sp."),
    ("4", "Parvicingula sp."),
    ("5", "Pseudoeucyrtis corpulentus"),
    ("6", "Obesacapsula sp."),
    ("7", "Pseudoeucyrtis hanni"),
    ("8", "Obeliscoites cf. vinassai"),
    ("9", "Nassellaria indet"),
    ("10", "Napora sp."),
    ("11", "Nassellaria indet"),
    ("12", "Acaeniotyle cf. umbilicata"),
    ("13", "Emiluvia sp."),
    ("14", "Acanthocircus sp."),
    ("15", "Crucella sp."),
    ("16", "Pantanelliidae indet"),
    ("17", "Crucella sp."),
    ("18", "Praeconocaryomma sp."),
]
for panel, species in PL07:
    NEW_ENTRIES.append((PL07_FIG, panel, species))

# ---------------------------------------------------------------------------
# Plate 8 (page 25) — 22 panels, sample PR-SB27
# ---------------------------------------------------------------------------
PL08_FIG = "od_plate_4f1bf415485765b8_p025_pl08"
PL08: list[tuple[str, str]] = [
    ("1", "Archaeodictyomitra cf. immenhauseri"),
    ("2", "Archaeodictyomitra cf. immenhauseri"),
    ("3", "Archaeodictyomitra cf. gracilis"),
    ("4", "Archaeodictyomitra montisserei"),
    ("5", "Archaeodictyomitra montisserei"),
    ("6", "Archaeodictyomitra sp."),
    ("7", "Archaeodictyomitra sp."),
    ("8", "Stichomitra sp."),  # "(?)" uncertainty stripped (gold convention)
    ("9", "Stichomitra sp."),
    ("10", "Thanarla aff. veneta"),
    ("11", "Thanarla brouweri"),
    ("12", "Thanarla sp."),
    ("13", "Archaeospongoprunum sp."),
    ("14", "Archaeospongoprunum sp."),
    ("15", "Archaeospongoprunum sp."),
    ("16", "Stylosphaera sp."),
    ("17", "Stylosphaera sp."),
    ("18", "Halesium cf. palmatum"),
    ("19", "Pantanellium sp."),
    ("20", "Pantanellium sp."),
    ("21", "Quadrigastrum lapideum"),
    ("22", "Spumellaria indet"),
]
for panel, species in PL08:
    NEW_ENTRIES.append((PL08_FIG, panel, species))

# ---------------------------------------------------------------------------
# Plate 9 (page 26) — 18 panels, samples PR-SB28 + PR-SB30
# ---------------------------------------------------------------------------
PL09_FIG = "od_plate_4f1bf415485765b8_p026_pl09"
PL09: list[tuple[str, str]] = [
    ("1", "Archaeodictyomitra gracilis"),
    ("2", "Archaeodictyomitra gracilis"),
    ("3", "Archaeodictyomitra sp."),
    ("4", "Stichomitra sp."),
    ("5", "Squinabollum aff. fossile"),
    ("6", "Neosciadiocapsidae indet"),
    ("7", "Pseudodictyomitra leptoconica"),
    ("8", "Pseudodictyomitra sp."),
    ("9", "Archaeodictyomitra sp."),
    ("10", "Tuguriella sp."),
    ("11", "Hiscocapsa sp."),
    ("12", "Hiscocapsa aff. asseni"),
    ("13", "Hiscocapsa uterculus"),
    ("14", "Zhamoidellum cf. testatum"),
    ("15", "Hiscocapsa sp."),
    ("16", "Syringocapsa limatum"),
    ("17", "Pseudocrucella elisabethae"),
    ("18", "Praeconocaryomma sp."),
]
for panel, species in PL09:
    NEW_ENTRIES.append((PL09_FIG, panel, species))


def main() -> int:
    out_path = Path("work/bandini_pl070809_verified_gold.jsonl")
    panels = [
        GoldPanel(paper_id=PAPER_ID, figure_id=fig, panel_id=panel, species=sp)
        for fig, panel, sp in NEW_ENTRIES
    ]
    write_gold(panels, out_path)
    print(f"Wrote {len(panels)} verified gold entries to {out_path}")
    # Per-plate counts
    from collections import Counter
    c = Counter(fig for fig, _, _ in NEW_ENTRIES)
    for fig, n in c.most_common():
        print(f"  {fig}: {n} panels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())