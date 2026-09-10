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


# ===========================================================================
# F18: multi-provider presets (v2 storage)
# ===========================================================================
import json  # noqa: E402

from rlpe.llm_settings import (  # noqa: E402
    ProviderConfig,
    current_provider_id,
    delete_provider,
    get_provider,
    list_providers,
    set_current_provider,
    upsert_provider,
)


class TestMultiProviderPresets:
    def test_upsert_creates_and_auto_activates_first(self, tmp_path):
        p = tmp_path / "llm_api.json"
        a = upsert_provider(
            ProviderConfig(
                name="MiniMax", base_url="https://mm/anthropic", api_key="sk-mm", model="MiniMax-M3"
            ),
            path=p,
        )
        assert a.id and a.updated_at
        assert current_provider_id(p) == a.id
        assert [x.name for x in list_providers(p)] == ["MiniMax"]

    def test_upsert_second_provider_does_not_steal_current(self, tmp_path):
        p = tmp_path / "llm_api.json"
        a = upsert_provider(ProviderConfig(name="A", base_url="https://a", model="m1"), path=p)
        b = upsert_provider(ProviderConfig(name="B", base_url="https://b", model="m2"), path=p)
        assert current_provider_id(p) == a.id
        assert {x.name for x in list_providers(p)} == {"A", "B"}
        assert a.id != b.id

    def test_explicit_activate_switches_current(self, tmp_path):
        p = tmp_path / "llm_api.json"
        upsert_provider(ProviderConfig(name="A", base_url="https://a", api_key="sk-a"), path=p)
        b = upsert_provider(ProviderConfig(name="B", base_url="https://b", api_key="sk-b"), path=p)
        assert set_current_provider(b.id, path=p) is True
        assert current_provider_id(p) == b.id
        flat = load_llm_settings(p)
        assert flat.api_key == "sk-b" and flat.profile_name == "B"

    def test_set_current_unknown_id_returns_false(self, tmp_path):
        p = tmp_path / "llm_api.json"
        upsert_provider(ProviderConfig(name="A", base_url="https://a"), path=p)
        assert set_current_provider("nope", path=p) is False

    def test_update_in_place_keeps_id(self, tmp_path):
        p = tmp_path / "llm_api.json"
        a = upsert_provider(ProviderConfig(name="A", base_url="https://a", model="m1"), path=p)
        upsert_provider(
            ProviderConfig(name="A", base_url="https://a2", model="m2", id=a.id), path=p
        )
        providers = list_providers(p)
        assert len(providers) == 1
        assert providers[0].base_url == "https://a2" and providers[0].id == a.id

    def test_delete_current_promotes_first_remaining(self, tmp_path):
        p = tmp_path / "llm_api.json"
        a = upsert_provider(ProviderConfig(name="A", base_url="https://a"), path=p)
        b = upsert_provider(ProviderConfig(name="B", base_url="https://b"), path=p)
        set_current_provider(b.id, path=p)
        assert delete_provider(b.id, path=p) is True
        assert current_provider_id(p) == a.id
        assert [x.name for x in list_providers(p)] == ["A"]

    def test_delete_last_leaves_no_current(self, tmp_path):
        p = tmp_path / "llm_api.json"
        a = upsert_provider(ProviderConfig(name="A", base_url="https://a"), path=p)
        assert delete_provider(a.id, path=p) is True
        assert list_providers(p) == []
        assert current_provider_id(p) is None
        assert load_llm_settings(p).is_empty

    def test_delete_unknown_returns_false(self, tmp_path):
        p = tmp_path / "llm_api.json"
        upsert_provider(ProviderConfig(name="A", base_url="https://a"), path=p)
        assert delete_provider("nope", path=p) is False

    def test_get_provider_by_id(self, tmp_path):
        p = tmp_path / "llm_api.json"
        a = upsert_provider(ProviderConfig(name="A", base_url="https://a", api_key="sk"), path=p)
        assert get_provider(a.id, path=p).api_key == "sk"
        assert get_provider("nope", path=p) is None


class TestV1Migration:
    def test_flat_v1_file_migrates_on_load(self, tmp_path):
        p = tmp_path / "llm_api.json"
        p.write_text(
            json.dumps(
                {
                    "base_url": "https://api.minimaxi.com/anthropic",
                    "api_key": "sk-old",
                    "model": "MiniMax-M3",
                    "updated_at": "2026-09-08T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        flat = load_llm_settings(p)
        assert flat.base_url == "https://api.minimaxi.com/anthropic"
        assert flat.api_key == "sk-old"
        assert flat.model == "MiniMax-M3"
        assert flat.profile_name == "默认"
        providers = list_providers(p)
        assert len(providers) == 1
        assert current_provider_id(p) == providers[0].id

    def test_v2_file_loads_unchanged(self, tmp_path):
        p = tmp_path / "llm_api.json"
        upsert_provider(ProviderConfig(name="A", base_url="https://a", api_key="k"), path=p)
        raw = json.loads(p.read_text(encoding="utf-8"))
        assert raw["version"] == 2
        assert load_llm_settings(p).api_key == "k"

    def test_stale_current_falls_back_to_first(self, tmp_path):
        p = tmp_path / "llm_api.json"
        upsert_provider(ProviderConfig(name="A", base_url="https://a"), path=p)
        raw = json.loads(p.read_text(encoding="utf-8"))
        raw["current"] = "deleted-id"
        p.write_text(json.dumps(raw), encoding="utf-8")
        assert current_provider_id(p) == list_providers(p)[0].id

    def test_corrupt_v2_file_is_empty(self, tmp_path):
        p = tmp_path / "llm_api.json"
        p.write_text("{broken", encoding="utf-8")
        assert list_providers(p) == []
        assert load_llm_settings(p).is_empty


class TestSaveCompatSurface:
    def test_save_updates_active_provider_keeping_name(self, tmp_path):
        p = tmp_path / "llm_api.json"
        a = upsert_provider(
            ProviderConfig(name="DeepSeek", base_url="https://ds", api_key="sk-old", model="m1"),
            path=p,
        )
        save_llm_settings(
            LLMApiSettings(base_url="https://ds", api_key="sk-new", model="m2"),
            path=p,
        )
        stored = get_provider(a.id, path=p)
        assert stored.name == "DeepSeek", "compat save must keep the preset name"
        assert stored.api_key == "sk-new" and stored.model == "m2"
