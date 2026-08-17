"""Regression tests for live llama.cpp + Qwen3.8-27B integration.

Audit 2026-08-17: a live probe against
``/home/user/ollama-models/Qwen3.8-27B-Q4_K_M.gguf`` running on
``http://127.0.0.1:8080`` exposed two real bugs in
``LlamaCppGemmaBackend._chat_completion``:

BUG-A: ``_build_text_prompt`` prepended a Gemma-era chat-template
prefix ``assistant\n<think>\n`` that Qwen3's chat template echoed back
verbatim. The real JSON answer came AFTER the closing ``</think>``,
but ``parse_json_from_text`` loaded the FIRST JSON object — a
placeholder / template residue like ``{"species": "species name"}`` —
instead of the real answer.

BUG-B: when Qwen3's chat template auto-injects a ``<think>...</think>``
segment, the response blob contains a placeholder JSON inside the
thinking block + the real JSON after it. The same "first JSON wins"
problem reappeared.

Fix: ``LlamaCppGemmaBackend._clean_response_text`` strips
``<think>`` / ``<answer>`` / ``` fences and returns the LAST balanced
JSON object (the real answer comes after the draft in Qwen3 output
streams). ``_build_text_prompt`` no longer prepends the Gemma template.

These tests guard the helper so a future refactor doesn't reintroduce
either bug.
"""
from __future__ import annotations

import json

import pytest

from rlpe.llm_backends import (
    LlamaCppGemmaBackend,
    _last_balanced_json_object,
)


# ---------------------------------------------------------------------------
# _last_balanced_json_object — pure helper
# ---------------------------------------------------------------------------


def test_last_balanced_json_returns_only_object():
    text = 'noise {"a": 1} more noise {"b": 2}'
    assert _last_balanced_json_object(text) == '{"b": 2}'


def test_last_balanced_json_handles_braces_in_strings():
    """Braces inside JSON string values must not throw the counter off."""
    text = 'placeholder {"a": "}"}  real {"species": "Ceratartia"}'
    assert _last_balanced_json_object(text) == '{"species": "Ceratartia"}'


def test_last_balanced_json_handles_escaped_quotes():
    text = r'{"a": "say \"hi\"", "b": 2} {"c": 3}'
    assert _last_balanced_json_object(text) == '{"c": 3}'


def test_last_balanced_json_returns_none_for_empty():
    assert _last_balanced_json_object("") is None
    assert _last_balanced_json_object("no json here") is None


def test_last_balanced_json_ignores_python_set_literals():
    """``{}`` with no colon is NOT a JSON object (e.g. ``{}``, ``{1,2}``)."""
    text = "{} not json {\"a\": 1}"
    assert _last_balanced_json_object(text) == '{"a": 1}'


def test_last_balanced_json_nested_objects():
    text = 'wrap {"outer": {"inner": 1}, "tail": 2} end'
    out = _last_balanced_json_object(text)
    assert out is not None
    parsed = json.loads(out)
    assert parsed["outer"]["inner"] == 1
    assert parsed["tail"] == 2


# ---------------------------------------------------------------------------
# LlamaCppGemmaBackend._clean_response_text — wrapper
# ---------------------------------------------------------------------------


def test_clean_strips_think_block_and_picks_real_answer():
    """BUG-B live capture from /completion on Qwen3.8-27B."""
    response = (
        '\n\n{\n  "species": "species name"\n}\n\n'
        '{\n  "species": "Ceratartia"\n}'
    )
    cleaned = LlamaCppGemmaBackend._clean_response_text(response)
    parsed = json.loads(cleaned)
    assert parsed["species"] == "Ceratartia"


def test_clean_strips_explicit_think_tags():
    response = (
        '<think>The user asked for a species. The plate shows...</think>'
        '{"species": "Tricolocapsa", "confidence": 0.7}'
    )
    cleaned = LlamaCppGemmaBackend._clean_response_text(response)
    parsed = json.loads(cleaned)
    assert parsed["species"] == "Tricolocapsa"


def test_clean_strips_answer_tags():
    response = (
        '<answer>{"species": "Pessagno", "confidence": 0.9}</answer>'
    )
    cleaned = LlamaCppGemmaBackend._clean_response_text(response)
    parsed = json.loads(cleaned)
    assert parsed["species"] == "Pessagno"


def test_clean_strips_code_fences():
    response = '```json\n{"species": "Nassellaria", "confidence": 0.5}\n```'
    cleaned = LlamaCppGemmaBackend._clean_response_text(response)
    parsed = json.loads(cleaned)
    assert parsed["species"] == "Nassellaria"


def test_clean_falls_back_to_raw_text_when_no_json():
    """If the response has no JSON at all (e.g. plain prose), return the
    cleaned prose so callers see the model's reasoning instead of an
    empty string."""
    response = "<think>...</think>I cannot identify this species."
    cleaned = LlamaCppGemmaBackend._clean_response_text(response)
    assert "cannot identify" in cleaned


def test_clean_handles_empty_string():
    assert LlamaCppGemmaBackend._clean_response_text("") == ""


def test_clean_preserves_only_json_when_no_think():
    """A clean JSON-only response must pass through untouched."""
    response = '{"species": "Spongodiscus", "confidence": 0.8}'
    cleaned = LlamaCppGemmaBackend._clean_response_text(response)
    assert json.loads(cleaned) == {"species": "Spongodiscus", "confidence": 0.8}


def test_clean_case_insensitive_think_tag():
    response = '<THINK>thoughts</THINK>{"species": "Archaeodictyomitra"}'
    cleaned = LlamaCppGemmaBackend._clean_response_text(response)
    parsed = json.loads(cleaned)
    assert parsed["species"] == "Archaeodictyomitra"


# ---------------------------------------------------------------------------
# _build_text_prompt — must NOT prepend Gemma chat-template prefix
# ---------------------------------------------------------------------------


def test_build_text_prompt_does_not_prepend_assistant_think():
    """BUG-A: prepending ``assistant\\n<think>\\n`` made Qwen3 copy the
    template back and emit its real answer AFTER the closing </think>.
    Live probe demonstrated parse_json_from_text then loaded the
    FIRST JSON (a placeholder), not the real answer.
    """
    be = LlamaCppGemmaBackend(host="http://127.0.0.1:1")  # never used
    prompt = be._build_text_prompt(
        system_prompt="You are a radiolarian assistant.",
        user_prompt="What species?",
    )
    assert "<think>" not in prompt, (
        "Prompt must not include <think> — Qwen3 chat template copies "
        "it back verbatim, polluting the response."
    )
    # Check for the Gemma-era chat-template prefix specifically
    # (``assistant\n`` followed by anything), not the word "assistant"
    # which can legitimately appear in a system prompt sentence.
    assert "assistant\n" not in prompt, (
        "Prompt must not include a chat-template assistant tag "
        "(``assistant\\n...``) — /completion has no chat template, the "
        "role tag confuses the model into thinking it's supposed to "
        "start emitting an assistant turn."
    )


def test_build_text_prompt_ends_with_json_instruction():
    """The trailing "Please output strict JSON only" cue is essential —
    without it the model sometimes returns prose."""
    be = LlamaCppGemmaBackend(host="http://127.0.0.1:1")
    prompt = be._build_text_prompt("sys", "user")
    assert "JSON" in prompt
    assert prompt.endswith("Please output strict JSON only.")


# ---------------------------------------------------------------------------
# End-to-end: parse_json_from_text integration with _clean_response_text
# ---------------------------------------------------------------------------


def test_clean_then_parse_yields_real_species_not_placeholder():
    """End-to-end: take the exact Qwen3.8-27B live response blob and
    confirm the cleaned-then-parsed species is the real one, not the
    template placeholder."""
    from rlpe.llm_backends import parse_json_from_text

    # Exact response from live probe (2026-08-17):
    response = (
        '\n\n{\n  "species": "species name"\n}\n\n'
        '{\n  "species": "Ceratartia"\n}'
    )
    cleaned = LlamaCppGemmaBackend._clean_response_text(response)
    parsed = parse_json_from_text(cleaned)
    assert parsed["species"] == "Ceratartia"
    assert parsed.get("confidence") is None or parsed.get("confidence") == 0.0