"""Cross-figure reference parser.

Resolves phrases like:
  - "see Fig. 3"
  - "as in Pl. 2, fig. 1"
  - "from Figure 1 (A, B)"
  - "compared with Fig. 2C-E"

in caption text, and returns structured records so downstream consumers can
follow the link from a panel's species to the figure where it was first
described.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(slots=True)
class CrossRef:
    target_figure: str           # canonical "Fig. 3" / "Pl. 2" / "Figure 1"
    target_figure_num: str       # just the number/letter "3" / "2A"
    span: tuple[int, int]        # (start, end) in source text
    context: str                 # ±60 chars around the match
    species_hint: str | None = None  # species name found near the reference, if any

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_figure": self.target_figure,
            "target_figure_num": self.target_figure_num,
            "span": list(self.span),
            "context": self.context,
            "species_hint": self.species_hint,
        }


# "Fig." / "Figure" / "Pl." / "Plate"  +  number  +  optional panel range
_PATTERN = re.compile(
    r"\b(?:Fig|Figure|Pl|Plate)\s*\.?\s*"
    r"(\d+[A-Za-z]?)"           # figure number
    r"(?:\s*\(([A-Z\d,\-\s]+)\))?"  # optional panel range inside parens
    r"(?:\s*[A-Z](?:\s*[-–—]\s*[A-Z])?)?"  # optional trailing panel letter
    r"(?=$|\s|[.,;:\)])",
    re.IGNORECASE,
)

# A species name is two words starting with a capital letter; the first word
# is the genus, the second is the species epithet (or sp./cf./aff.)
_SPECIES_NEAR = re.compile(
    r"\b([A-Z][a-zA-Z\-]{2,}\s+(?:sp\.|spp\.|cf\.|aff\.|[a-z][a-zA-Z\-]{1,}))"
)


def parse_cross_refs(
    caption_text: str,
    current_fig_id: str = "",
) -> list[CrossRef]:
    """Find and return all cross-figure references in ``caption_text``.

    ``current_fig_id`` (e.g. "fig_2") is used to skip self-references
    ("Fig. 2" inside Fig. 2's own caption).
    """
    if not caption_text:
        return []
    # Self-reference patterns to skip: extract number from current_fig_id
    self_num = ""
    if current_fig_id:
        m = re.search(r"(\d+)", current_fig_id)
        if m:
            self_num = m.group(1)
    out: list[CrossRef] = []
    for m in _PATTERN.finditer(caption_text):
        target_num = m.group(1)
        if self_num and target_num == self_num:
            continue
        # Build canonical "Fig. N" or "Pl. N" form
        kind_raw = m.group(0).split()[0].rstrip(".").lower()
        if kind_raw.startswith("plat") or kind_raw == "pl":
            kind = "Pl"
        else:
            kind = "Fig"
        target = f"{kind}. {target_num}"
        # Extract species hint from the right side of the reference (where
        # the species name typically appears in "Fig. 2C-E shows Cromyomma sp.")
        right = caption_text[m.end():m.end() + 60]
        species_hint = None
        sp_match = _SPECIES_NEAR.search(right)
        if sp_match:
            species_hint = sp_match.group(1).strip()
        # Also try left side if right is empty
        if not species_hint:
            left = caption_text[max(0, m.start() - 80):m.start()]
            sp_match = _SPECIES_NEAR.search(left)
            if sp_match:
                species_hint = sp_match.group(1).strip()
        ctx_start = max(0, m.start() - 60)
        ctx_end = min(len(caption_text), m.end() + 60)
        out.append(
            CrossRef(
                target_figure=target,
                target_figure_num=target_num,
                span=(m.start(), m.end()),
                context=caption_text[ctx_start:ctx_end].strip(),
                species_hint=species_hint,
            )
        )
    return out
