"""Tests for scripts/prompts.py — 4 M3 prompt templates selected by paper type."""

import sys
sys.path.insert(0, 'scripts')
from prompts import (
    RANGE_CHART_PROMPT,
    SEM_PLATE_PROMPT,
    MAP_PROMPT,
    GENERIC_PROMPT,
    select_prompt,
    build_user_prompt,
)


def test_range_chart_prompt_no_gold():
    """No gold species in prompt (general rules only)."""
    assert 'species' in RANGE_CHART_PROMPT.lower()
    assert 'archaeodictyomitra' not in RANGE_CHART_PROMPT.lower()  # no specific taxa
    assert 'array' in RANGE_CHART_PROMPT.lower()  # output format


def test_sem_plate_prompt_distinct_from_range():
    assert RANGE_CHART_PROMPT != SEM_PLATE_PROMPT


def test_map_prompt_mentions_locality():
    assert 'locality' in MAP_PROMPT.lower() or 'location' in MAP_PROMPT.lower()


def test_select_prompt_classifies_by_caption():
    cap_range = 'Fig. 1. Distribution of radiolarians in this paper.'
    cap_sem = 'Plate 1. Scanning electron microscope pictures of radiolarians.'
    cap_map = 'Fig. 1. Schematic map indicating location of samples.'
    cap_other = 'Random caption with no keywords.'
    assert select_prompt(cap_range) == RANGE_CHART_PROMPT
    assert select_prompt(cap_sem) == SEM_PLATE_PROMPT
    assert select_prompt(cap_map) == MAP_PROMPT
    assert select_prompt(cap_other) == GENERIC_PROMPT


def test_build_user_prompt_includes_caption():
    user = build_user_prompt('Some caption text here.')
    assert 'Some caption text here.' in user
    assert 'JSON' in user or 'json' in user