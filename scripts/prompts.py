"""M3 prompt library — 4 templates selected by paper type.

These prompts describe general rules only (no specific taxa, no
gold references). They instruct the LLM to:
- Extract every specimen panel from the figure caption + image
- Output a strict JSON array of {label, species, confidence}
- Distinguish cf./aff./n.sp. qualifiers
- Skip non-radiolarian specimens

Prompts are intentionally generic so the eval set doesn't leak into
the prompt design.
"""
from __future__ import annotations

import re
from typing import Tuple

_BASE_OUTPUT_FORMAT = (
    "Return strict JSON array of objects with fields "
    "{label, species, confidence, panel_id}. Example: "
    '[{"label": "1", "species": "Genus species", "confidence": 0.9}, ...].'
)

_RANGE_CHART_MARKERS = ("distribution", "range chart", "biozone", "stratigraphic range")
_SEM_PLATE_MARKERS = ("scanning electron", "plate", "marker =", "bar =")
_MAP_MARKERS = ("location", "map", "schematic map", "geographic")


def _build_prompt(goal: str, special: str) -> str:
    return (
        f"You are an expert radiolarian paleontologist. {goal}\n\n"
        f"{special}\n\n"
        f"{_BASE_OUTPUT_FORMAT}\n\n"
        "Preserve taxonomic qualifiers (cf., aff., n. sp., comb. nov.).\n"
        "If a panel is NOT a radiolarian, set species=null and label=panel_id.\n"
    )


RANGE_CHART_PROMPT = _build_prompt(
    goal="Given a range chart caption and image, extract every radiolarian "
         "species and the stratigraphic range it appears in.",
    special="Output one row per (species, range) pair visible in the chart. "
            "label = species name; panel_id = the stratigraphic zone it appears in.",
)

SEM_PLATE_PROMPT = _build_prompt(
    goal="Given a plate caption and image, extract every specimen panel "
         "and identify the radiolarian species shown.",
    special="Output one row per numbered figure (Fig. 1, Fig. 2, etc.) "
            "visible in the plate. label = the figure number; panel_id = same.",
)

MAP_PROMPT = _build_prompt(
    goal="Given a map caption and image, extract any radiolarian-bearing "
         "localities mentioned.",
    special="Output one row per locality if the map shows radiolarian sites. "
            "label = the locality id (e.g. 'Loc. 5'); panel_id = same.",
)

GENERIC_PROMPT = _build_prompt(
    goal="Given a figure caption and image, extract every radiolarian "
         "specimen or locality shown.",
    special="Output one row per visible item. label = whatever the caption uses "
            "to identify the item; panel_id = same.",
)


_PREDICATE_PATTERNS = (
    (RANGE_CHART_PROMPT, _RANGE_CHART_MARKERS),
    (SEM_PLATE_PROMPT, _SEM_PLATE_MARKERS),
    (MAP_PROMPT, _MAP_MARKERS),
)


def select_prompt(caption: str) -> str:
    """Pick the most appropriate prompt template by caption keywords.

    Falls back to GENERIC_PROMPT if no markers match.
    """
    cap_lower = caption.lower()
    for prompt, markers in _PREDICATE_PATTERNS:
        for marker in markers:
            if marker in cap_lower:
                return prompt
    return GENERIC_PROMPT


def build_user_prompt(caption: str) -> str:
    """Wrap the caption into the user message sent to M3."""
    return f"Caption:\n{caption[:3000]}\n\nExtract every panel and species as JSON."