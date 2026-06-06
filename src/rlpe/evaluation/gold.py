"""Ground-truth loader for RLPE evaluation.

A gold file is a JSONL with one record per panel::

    {"paper_id": "hollis2006", "figure_id": "plate_1", "panel_id": "1", "species": "Genus species"}
    {"paper_id": "hollis2006", "figure_id": "plate_1", "panel_id": "2", "species": "Genus species"}

The ``panel_id`` here is the *label printed in the figure* (1, 2, 3, A, B, 12a).
The species is the canonical Latin binomial or genus+sp abbreviation. Empty
species means "no species assigned in the gold" (some panels are unlabelled
or are scale bars).

The gold files are stored in ``data/gold/<paper>.jsonl`` and are also
published in the same schemas/ directory for downstream consumers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


GOLD_SCHEMA_VERSION = "1.0.0"


@dataclass(slots=True)
class GoldPanel:
    paper_id: str
    figure_id: str
    panel_id: str | None
    species: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "paper_id": self.paper_id,
            "figure_id": self.figure_id,
            "panel_id": self.panel_id,
            "species": self.species,
        }


def load_gold(path: Path) -> list[GoldPanel]:
    out: list[GoldPanel] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(GoldPanel(
                paper_id=str(d["paper_id"]),
                figure_id=str(d["figure_id"]),
                panel_id=d.get("panel_id"),
                species=d.get("species"),
            ))
    return out


def write_gold(panels: Iterable[GoldPanel], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for p in panels:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def match_panel(gold: GoldPanel, pred_paper_id: str, pred_panel_id: str | None) -> bool:
    """Decide whether a predicted panel corresponds to a gold panel.

    Match rule: same paper_id and the predicted label is contained in the
    gold label (so "12a" matches gold "12", "12b" matches gold "12b", etc.).
    Empty/missing panel_id on either side is treated as a non-match
    (an unlabelled panel is not a hit).
    """
    if gold.paper_id != pred_paper_id:
        return False
    if not gold.panel_id or not pred_panel_id:
        return False
    g = gold.panel_id.strip()
    p = pred_panel_id.strip()
    if not g or not p:
        return False
    return g == p or p.startswith(g) or g.startswith(p)
