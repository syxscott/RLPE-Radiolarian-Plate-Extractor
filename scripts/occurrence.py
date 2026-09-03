"""Group identical species across multiple figures in a paper.

Two preds are in the same occurrence group iff:
  - same paper_id, AND
  - same normalized species (cf./aff. split, lowered, etc.)

The group id is deterministic: same input → same output.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

# Strip cf./aff. qualifiers so occurrences with the same binomial
# collapse to one group regardless of how the parser captured the
# open-nomenclature marker.
_CF_AFF_RE = re.compile(r"\b(?:cf|aff)\.\s*", re.IGNORECASE)


def _normalize_species_for_occurrence(species: str | None) -> str:
    if not species:
        return ""
    from rlpe.evaluation.metrics import _norm_species
    norm = _norm_species(species) or ""
    # Drop cf./aff. tokens (with their trailing period) so the
    # remaining tokens collapse to the bare binomial form.
    norm = _CF_AFF_RE.sub(" ", norm)
    norm = " ".join(norm.split())
    return norm


def occurrence_group_id(paper_id: str | None, species: str | None) -> str:
    pid = (paper_id or "").strip()
    sp = _normalize_species_for_occurrence(species)
    raw = f"{pid}|{sp}".encode()
    return "occ_" + hashlib.sha1(raw).hexdigest()[:6]


def add_occurrence_groups(preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for p in preds:
        q = dict(p)
        q['occurrence_group_id'] = occurrence_group_id(
            p.get('paper_id', ''), p.get('species'),
        )
        out.append(q)
    return out