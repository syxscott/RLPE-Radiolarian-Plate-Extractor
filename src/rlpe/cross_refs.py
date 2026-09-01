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
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CrossRef:
    target_figure: str  # canonical "Fig. 3" / "Pl. 2" / "Figure 1"
    target_figure_num: str  # just the number/letter "3" / "2A"
    span: tuple[int, int]  # (start, end) in source text
    context: str  # ±60 chars around the match
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
# Audit 2026-09-01 CR-16: previous regex used IGNORECASE flag together
# with a [A-Z] character class for the panel range. IGNORECASE made
# the [A-Z] match lowercase characters too, so a token like
# "figure 2c" was matched as figure=2 + panel="c" (lowercase)
# whereas the canonical panel letter is uppercase. Force
# ``re.ASCII`` so the class actually matches the uppercase letter
# that downstream code (panel_id validation, biozone letters)
# expects. The IGNORECASE flag remains for the keyword class
# (Fig / Figure / Pl / Plate).
_PATTERN = re.compile(
    r"\b(?:Fig|Figure|Pl|Plate)\s*\.?\s*"
    r"(\d+[A-Za-z]?)"  # figure number
    r"(?:\s*\(([A-Z\d,\-\s]+)\))?"  # optional panel range inside parens
    r"(?:\s*[A-Z](?:\s*[-–—]\s*[A-Z])?)?"  # optional trailing panel letter (uppercase only)
    r"(?=$|\s|[.,;:\)])",
    re.IGNORECASE | re.ASCII,
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
    # Self-reference patterns to skip. Anchor on a trailing
    # ``_pl\d+`` / ``_fig\d+`` / ``_plate\d+`` suffix (the convention
    # used by the production figure_id builder in
    # ``opendataloader_extractor.py``). The previous implementation
    # used a bare ``re.search(r"(\d+)")`` which grabbed the first
    # digit run — for production ids like
    # ``od_plate_bandini2011_pl03`` that yielded ``"2011"`` (the year)
    # instead of ``"03"`` (the plate number), silently failing to
    # suppress self-references.
    #
    # Audit 2026-08-16 fix: we extract both the kind ("Pl" / "Fig")
    # and the numeric suffix, then only suppress a ref when BOTH
    # match. A plate caption that mentions "Fig. 3" should NOT be
    # suppressed by a current_fig_id of ``..._pl03`` (different kind)
    # and vice versa.
    self_kind = ""
    self_num = ""
    if current_fig_id:
        m = re.search(r"_pl(\d+)$", current_fig_id)
        if m:
            self_kind = "Pl"
            self_num = m.group(1).lstrip("0") or "0"
        else:
            m2 = re.search(r"_(?:fig|figure|plate)(\d+)$", current_fig_id)
            if m2:
                self_kind = "Fig"
                self_num = m2.group(1).lstrip("0") or "0"
            else:
                # Fallback for short synthetic ids like "fig_2":
                # detect kind from leading alpha, parse trailing num.
                m3 = re.match(r"^([A-Za-z]+)_?(\d+)$", current_fig_id)
                if m3:
                    alpha = m3.group(1).lower()
                    if alpha.startswith("pl"):
                        self_kind = "Pl"
                    elif alpha.startswith("fig") or alpha.startswith("figure"):
                        self_kind = "Fig"
                    self_num = m3.group(2).lstrip("0") or "0"
    out: list[CrossRef] = []
    for m in _PATTERN.finditer(caption_text):
        target_num = m.group(1).lstrip("0") or "0"
        # Build canonical "Fig. N" or "Pl. N" form
        kind_raw = m.group(0).split()[0].rstrip(".").lower()
        if kind_raw.startswith("plat") or kind_raw == "pl":
            kind = "Pl"
        else:
            kind = "Fig"
        # Suppress self-reference only when kind AND number match.
        # Audit 2026-08-16: previously ``self_num`` alone could
        # match a different-kind ref (e.g. pl03 vs "Fig. 3" caption).
        if self_kind and kind == self_kind and target_num == self_num:
            continue
        target = f"{kind}. {target_num}"
        # Extract species hint from the right side of the reference (where
        # the species name typically appears in "Fig. 2C-E shows Cromyomma sp.")
        right = caption_text[m.end() : m.end() + 60]
        species_hint = None
        sp_match = _SPECIES_NEAR.search(right)
        if sp_match:
            species_hint = sp_match.group(1).strip()
        # Also try left side if right is empty
        if not species_hint:
            left = caption_text[max(0, m.start() - 80) : m.start()]
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
