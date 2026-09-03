"""M3 prompt library — 4 templates selected by paper type.

These prompts describe general rules only (no specific taxa, no
gold references). They instruct the LLM to:
- Extract every specimen panel from the figure caption + image
- Output a strict JSON array of {label, species, confidence}
- Distinguish cf./aff./n.sp. qualifiers
- Skip non-radiolarian specimens

Prompts are intentionally generic so the eval set doesn't leak into
the prompt design. Markers are word-boundary regexes to avoid
false-positives like "plate of food" routing to the SEM template.
"""
from __future__ import annotations

import re

_BASE_OUTPUT_FORMAT = (
    "Return strict JSON array of objects with fields "
    "{label, species, confidence, panel_id}. Example: "
    '[{"label": "1", "species": "Genus species", "confidence": 0.9}, ...].'
)

# Markers are word-boundary regexes (not plain strings) to avoid
# false positives like "plate of food" matching the SEM template.
_RANGE_CHART_MARKERS = (
    re.compile(r"\b(?:distribution|range\s*chart|biozone|stratigraphic\s*range)\b", re.IGNORECASE),
)
_SEM_PLATE_MARKERS = (
    # "scanning electron" is the strongest signal (very specific).
    re.compile(r"\bscanning\s+electron\b", re.IGNORECASE),
    # "Plate 5" / "Plate V" / "Plate III" — word boundary + plate-number suffix.
    re.compile(r"\bplate\s+[ivxlcdm0-9]+\b", re.IGNORECASE),
    # "Marker = X" / "Bar = Y" are anchor strings (very specific).
    re.compile(r"\bmarker\s*=", re.IGNORECASE),
    re.compile(r"\bbar\s*=", re.IGNORECASE),
)
_MAP_MARKERS = (
    re.compile(r"\b(?:location|geographic)\b", re.IGNORECASE),
    re.compile(r"\b(?:schematic\s*)?map\b", re.IGNORECASE),
)


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

TEXT_MODE_PROMPT = _build_prompt(
    goal="Given a radiolarian paper's full text (no plate figures available), "
         "extract every radiolarian species mentioned in the text along with its location.",
    special="Output one row per species, with 'location' describing the page or section. "
            "label = the species name; panel_id = the page or section identifier. "
            "If the paper is not about Radiolaria, set species=null.",
)


def select_text_mode_prompt(caption: str) -> str:
    """Always returns TEXT_MODE_PROMPT (caller has already decided to use text mode)."""
    return TEXT_MODE_PROMPT


_PREDICATE_PATTERNS = (
    (RANGE_CHART_PROMPT, _RANGE_CHART_MARKERS),
    (SEM_PLATE_PROMPT, _SEM_PLATE_MARKERS),
    (MAP_PROMPT, _MAP_MARKERS),
)


def select_prompt(caption: str | None) -> str:
    """Pick the most appropriate prompt template by caption keywords.

    Falls back to GENERIC_PROMPT if no markers match, or if the
    input is None/empty (e.g. missing GROBID caption).
    """
    if caption is None or not caption.strip():
        return GENERIC_PROMPT
    for prompt, markers in _PREDICATE_PATTERNS:
        for pattern in markers:
            if pattern.search(caption):
                return prompt
    return GENERIC_PROMPT


def build_user_prompt(caption: str) -> str:
    """Wrap the caption into the user message sent to M3."""
    return f"Caption:\n{caption[:3000]}\n\nExtract every panel and species as JSON."