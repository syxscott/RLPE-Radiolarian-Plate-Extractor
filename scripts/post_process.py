"""Post-processing for M3 panel extraction output.

Four utilities:
  - parse_open_nomenclature: split 'Genus cf. species' into (sp, qualifier)
  - dedup_panels: remove exact-duplicate (fig, panel, species) rows
  - filter_low_confidence: drop rows with confidence < threshold
  - normalize_panel_id: strip 'Fig. N' / 'Pl. N' / 'Plate N' prefix

All functions are pure (no LLM call, no gold reference) — they
operate only on the pred rows returned by M3.
"""

from __future__ import annotations

import re
from typing import Any

_QUALIFIER_RE = re.compile(r"\b(cf|aff|vel|similar)\b\.?\s*", re.IGNORECASE)


def parse_open_nomenclature(species: str | None) -> tuple[str | None, str | None]:
    """Split a species string into (species, qualifier).

    'Genus cf. species' → ('Genus species', 'cf.')
    'Genus aff. species' → ('Genus species', 'aff.')
    'Genus species'      → ('Genus species', None)
    """
    if species is None:
        return None, None
    species = species.strip()
    if not species:
        return None, None
    qual_match = _QUALIFIER_RE.search(species)
    if qual_match is None:
        return species, None
    qualifier = qual_match.group(1).lower() + "."
    clean = _QUALIFIER_RE.sub("", species, count=1).strip()
    return clean, qualifier


_PANEL_PREFIX_RE = re.compile(
    r"^\s*(?:Figs?|Plates|Plate|Pl|表|図版)\.?\s*",
    re.IGNORECASE,
)


def normalize_panel_id(label: str | None) -> str:
    """Strip 'Fig. N' / 'Pl. N' / 'Plate N' prefix; collapse whitespace."""
    if not label:
        return ""
    cleaned = _PANEL_PREFIX_RE.sub("", label)
    return re.sub(r"\s+", " ", cleaned).strip()


def dedup_panels(panels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop exact duplicates by (paper_id, figure_id, panel_id, species).

    When duplicates exist, keep the one with highest confidence.
    On ties (same key, same confidence), the first encountered row is kept.
    """
    best_by_key: dict[tuple, dict[str, Any]] = {}
    for p in panels:
        key = (p.get("paper_id"), p.get("figure_id"), p.get("panel_id"), p.get("species"))
        if key not in best_by_key or float(p.get("confidence", 0) or 0) > float(
            best_by_key[key].get("confidence", 0) or 0
        ):
            best_by_key[key] = p
    return list(best_by_key.values())


def filter_low_confidence(
    panels: list[dict[str, Any]], threshold: float = 0.7
) -> list[dict[str, Any]]:
    """Drop rows whose confidence is below threshold."""
    return [p for p in panels if float(p.get("confidence", 0) or 0) >= threshold]
