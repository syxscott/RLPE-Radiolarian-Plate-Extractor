"""Tests for TEXT_MODE_PROMPT + select_text_mode_prompt in scripts/prompts.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prompts import (
    GENERIC_PROMPT,
    MAP_PROMPT,
    RANGE_CHART_PROMPT,
    SEM_PLATE_PROMPT,
    TEXT_MODE_PROMPT,
    select_text_mode_prompt,
)


def test_text_mode_prompt_exists():
    assert isinstance(TEXT_MODE_PROMPT, str)
    assert len(TEXT_MODE_PROMPT) > 50


def test_text_mode_prompt_no_gold_taxa():
    for forbidden in ["Archaeodictyomitra", "Williriedellum", "Hiscocapsa", "praeparvicingula"]:
        assert forbidden.lower() not in TEXT_MODE_PROMPT.lower()


def test_text_mode_prompt_has_output_format():
    assert "JSON" in TEXT_MODE_PROMPT
    assert "array" in TEXT_MODE_PROMPT.lower()


def test_text_mode_prompt_requests_location():
    lower = TEXT_MODE_PROMPT.lower()
    assert any(kw in lower for kw in ["page", "location", "context"])


def test_text_mode_prompt_distinct_from_plate_prompts():
    assert TEXT_MODE_PROMPT != SEM_PLATE_PROMPT
    assert TEXT_MODE_PROMPT != RANGE_CHART_PROMPT
    assert TEXT_MODE_PROMPT != MAP_PROMPT
    assert TEXT_MODE_PROMPT != GENERIC_PROMPT


def test_select_text_mode_prompt_returns_text_mode_for_any_caption():
    assert select_text_mode_prompt("any caption here") is TEXT_MODE_PROMPT
