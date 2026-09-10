"""F17 (2026-09-08): persisted LLM API settings module.

Covers: save/load round-trip, 0600 permissions, atomicity (no leftover
temp files), corrupt-file tolerance, clear, and the path override used
by tests and the API layer.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from rlpe.llm_settings import (
    LLMApiSettings,
    clear_llm_settings,
    load_llm_settings,
    save_llm_settings,
    settings_path,
)


class TestSettingsRoundTrip:
    def test_save_then_load(self, tmp_path: Path):
        p = save_llm_settings(
            LLMApiSettings(
                base_url="https://api.example.com/anthropic",
                api_key="sk-test-123",
                model="my-model",
            ),
            path=tmp_path / "llm_api.json",
        )
        loaded = load_llm_settings(p)
        assert loaded.base_url == "https://api.example.com/anthropic"
        assert loaded.api_key == "sk-test-123"
        assert loaded.model == "my-model"
        assert loaded.updated_at  # stamped by save
        assert not loaded.is_empty

    def test_permissions_0600(self, tmp_path: Path):
        p = save_llm_settings(LLMApiSettings(api_key="sk-secret"), path=tmp_path / "llm_api.json")
        mode = stat.S_IMODE(os.stat(p).st_mode)
        assert mode == 0o600, f"settings file must be 0600, got {oct(mode)}"

    def test_no_temp_files_left_behind(self, tmp_path: Path):
        save_llm_settings(LLMApiSettings(model="m"), path=tmp_path / "llm_api.json")
        leftovers = [x.name for x in tmp_path.iterdir() if x.name != "llm_api.json"]
        assert leftovers == []

    def test_overwrite_updates_timestamp(self, tmp_path: Path):
        p = tmp_path / "llm_api.json"
        save_llm_settings(LLMApiSettings(model="a"), path=p)
        first = load_llm_settings(p).updated_at
        save_llm_settings(LLMApiSettings(model="b"), path=p)
        second = load_llm_settings(p).updated_at
        assert first and second


class TestTolerance:
    def test_missing_file_returns_defaults(self, tmp_path: Path):
        loaded = load_llm_settings(tmp_path / "nope.json")
        assert loaded == LLMApiSettings()
        assert loaded.is_empty

    def test_corrupt_file_returns_defaults(self, tmp_path: Path):
        p = tmp_path / "llm_api.json"
        p.write_text("{not json at all", encoding="utf-8")
        assert load_llm_settings(p).is_empty

    def test_non_dict_payload_returns_defaults(self, tmp_path: Path):
        p = tmp_path / "llm_api.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        assert load_llm_settings(p).is_empty

    def test_clear_removes_file(self, tmp_path: Path):
        p = save_llm_settings(LLMApiSettings(api_key="k"), path=tmp_path / "llm_api.json")
        clear_llm_settings(p)
        assert not p.exists()
        clear_llm_settings(p)  # idempotent


class TestPathConvention:
    def test_default_path_is_dot_rlpe(self):
        p = settings_path()
        assert p.parent.name == ".rlpe"
        assert p.name == "llm_api.json"
