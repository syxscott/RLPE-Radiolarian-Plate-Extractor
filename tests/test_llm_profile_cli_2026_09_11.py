"""F18 (2026-09-11): multi-provider presets — CLI --llm-profile + smoke.

Locks the ``_apply_llm_profile_to_extra`` semantics (an explicit flag
wins; present-but-None fields are filled from the preset — the original
implementation used ``dict.setdefault`` which silently no-oped because
the keys always exist with None values) and exercises the early
validation path through the real parser.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlpe.cli import _apply_llm_profile_to_extra, build_parser


@dataclass
class _Profile:
    name: str
    base_url: str
    api_key: str
    model: str


class TestApplyLlmProfile:
    def test_fills_present_but_none_fields(self):
        """The exact regression: setdefault no-oped because the keys
        already existed with None values."""
        extra = {"llm_api_key": None, "llm_base_url": None, "llm_model": None}
        _apply_llm_profile_to_extra(
            extra,
            _Profile("DeepSeek", "https://ds/anthropic", "sk-ds", "deepseek-chat"),
        )
        assert extra["llm_api_key"] == "sk-ds"
        assert extra["llm_base_url"] == "https://ds/anthropic"
        assert extra["llm_model"] == "deepseek-chat"

    def test_explicit_flag_wins_over_profile(self):
        extra = {"llm_api_key": "sk-explicit", "llm_base_url": None, "llm_model": None}
        _apply_llm_profile_to_extra(
            extra,
            _Profile("DeepSeek", "https://ds/anthropic", "sk-ds", "deepseek-chat"),
        )
        assert extra["llm_api_key"] == "sk-explicit"
        assert extra["llm_model"] == "deepseek-chat"

    def test_missing_keys_are_added(self):
        extra: dict = {}
        _apply_llm_profile_to_extra(extra, _Profile("M", "https://m", "sk-m", "model-m"))
        assert extra == {
            "llm_base_url": "https://m",
            "llm_api_key": "sk-m",
            "llm_model": "model-m",
        }

    def test_profile_fields_may_be_empty(self):
        extra = {"llm_api_key": None, "llm_base_url": None, "llm_model": None}
        _apply_llm_profile_to_extra(extra, _Profile("M", "https://m", "", ""))
        # Empty preset fields resolve to None so the normal chain
        # (next preset / env) still applies.
        assert extra["llm_api_key"] is None
        assert extra["llm_model"] is None


class TestProfileFlagParsing:
    def test_flag_exists_on_parser(self):
        args = build_parser().parse_args(["--llm-profile", "MiniMax"])
        assert args.llm_profile == "MiniMax"

    def test_flag_default_none(self):
        args = build_parser().parse_args([])
        assert args.llm_profile is None
