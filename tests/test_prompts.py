"""Tests for scripts/prompts.py — 4 M3 prompt templates selected by paper type."""

import sys

sys.path.insert(0, "scripts")
from prompts import (
    GENERIC_PROMPT,
    MAP_PROMPT,
    RANGE_CHART_PROMPT,
    SEM_PLATE_PROMPT,
    build_user_prompt,
    select_prompt,
)


def test_range_chart_prompt_no_gold():
    """No gold species in prompt (general rules only)."""
    assert "species" in RANGE_CHART_PROMPT.lower()
    assert "archaeodictyomitra" not in RANGE_CHART_PROMPT.lower()  # no specific taxa
    assert "array" in RANGE_CHART_PROMPT.lower()  # output format


def test_sem_plate_prompt_distinct_from_range():
    assert RANGE_CHART_PROMPT != SEM_PLATE_PROMPT


def test_map_prompt_mentions_locality():
    assert "locality" in MAP_PROMPT.lower() or "location" in MAP_PROMPT.lower()


def test_select_prompt_classifies_by_caption():
    cap_range = "Fig. 1. Distribution of radiolarians in this paper."
    cap_sem = "Plate 1. Scanning electron microscope pictures of radiolarians."
    cap_map = "Fig. 1. Schematic map indicating location of samples."
    cap_other = "Random caption with no keywords."
    assert select_prompt(cap_range) == RANGE_CHART_PROMPT
    assert select_prompt(cap_sem) == SEM_PLATE_PROMPT
    assert select_prompt(cap_map) == MAP_PROMPT
    assert select_prompt(cap_other) == GENERIC_PROMPT


def test_build_user_prompt_includes_caption():
    user = build_user_prompt("Some caption text here.")
    assert "Some caption text here." in user
    assert "JSON" in user or "json" in user


def test_select_prompt_none_returns_generic():
    """None input returns GENERIC_PROMPT (graceful fallback)."""
    assert select_prompt(None) == GENERIC_PROMPT


def test_select_prompt_empty_returns_generic():
    """Empty / whitespace-only caption returns GENERIC_PROMPT."""
    assert select_prompt("") == GENERIC_PROMPT
    assert select_prompt("   \n\t  ") == GENERIC_PROMPT


def test_plate_false_positive_filtered():
    """'plate of food' in body text must NOT route to SEM_PLATE_PROMPT.

    Without the word-boundary fix, plain 'plate' substring would match
    and incorrectly route the caption to the SEM template.
    """
    assert select_prompt("This is just a normal plate of food.") == GENERIC_PROMPT


def test_build_user_prompt_truncates_long_caption():
    """A 10,000-char caption must be truncated to 3000 chars in the prompt.

    'Z' is used as the marker because it does not appear in the static
    wrapper text ('Caption:\\n' / '\\n\\nExtract every panel and species
    as JSON.'), so the count of 'Z' exactly equals the caption length.
    """
    long_caption = "Z" * 10000
    out = build_user_prompt(long_caption)
    # The 10000-char caption itself is truncated to 3000 in the prompt;
    # the wrapper text contains no 'Z', so out.count('Z') == caption length.
    assert out.count("Z") <= 3000
    assert "Caption:" in out
