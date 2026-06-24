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
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

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
            out.append(
                GoldPanel(
                    paper_id=str(d["paper_id"]),
                    figure_id=str(d["figure_id"]),
                    panel_id=d.get("panel_id"),
                    species=d.get("species"),
                )
            )
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


def _extension_is_alpha(shorter: str, longer: str) -> bool:
    """Return True if ``longer`` is ``shorter`` extended by alphabetic content.

    A "sub-label" relationship: gold "5" and pred "5a" should match (5a is
    5 with letter suffix a). But gold "5" and pred "10" must NOT match —
    "10" is "1" extended by a digit, which means a different panel.

    Examples:
      "5"   + "5a"  → True  (alphabetic suffix)
      "5"   + "5bc" → True  (alphabetic suffix)
      "5"   + "10"  → False (numeric suffix — different panel)
      "A"   + "Aa"  → True  (alphabetic suffix)
      "A"   + "A1"  → False (numeric suffix — different panel)
      "12"  + "12a" → True
      "12a" + "12b" → False (not a prefix relationship at all)
    """
    if not longer.startswith(shorter):
        return False
    suffix = longer[len(shorter) :]
    if not suffix:
        return False
    return suffix.isalpha()


def match_panel(gold: GoldPanel, pred_paper_id: str, pred_panel_id: str | None) -> bool:
    """Decide whether a predicted panel corresponds to a gold panel.

    Match rules:
      - paper_ids must match exactly
      - empty/missing panel_id on either side is a non-match
      - exact string match always matches
      - prefix match is allowed ONLY when the longer label extends the
        shorter with ALPHABETIC content (e.g. gold "5" + pred "5a",
        gold "12" + pred "12a"). Pure-numeric extensions like "5"/"10"
        are distinct panels and must not match. Numeric-prefix-then-letter
        ("A" + "A1") is also rejected because "1" is numeric suffix.
    """
    if gold.paper_id != pred_paper_id:
        return False
    if not gold.panel_id or not pred_panel_id:
        return False
    g = gold.panel_id.strip()
    p = pred_panel_id.strip()
    if not g or not p:
        return False
    if g == p:
        return True
    if len(g) < len(p):
        return _extension_is_alpha(g, p)
    return _extension_is_alpha(p, g)
