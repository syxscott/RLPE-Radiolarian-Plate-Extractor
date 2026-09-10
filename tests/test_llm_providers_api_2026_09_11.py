"""F18 (2026-09-11): /system/llm-providers multi-preset endpoints.

Covers CRUD + activate semantics through the Web API: masked keys only,
update-without-key keeps the stored secret, creating the first preset
auto-activates it, deleting the active preset promotes the first
remaining one, and unknown ids 404. The settings file is redirected
into tmp_path via BOTH bindings (module-level and the app's import-time
`llm_settings_path` alias) so tests stay hermetic in any import order.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def api(monkeypatch, tmp_path: Path):
    TestClient = pytest.importorskip("fastapi.testclient").TestClient
    monkeypatch.delenv("RLPE_API_KEY", raising=False)
    import rlpe.llm_settings as ls

    target = tmp_path / "llm_api.json"
    monkeypatch.setattr(ls, "settings_path", lambda home=None: target)
    from rlpe.api import app as api_app

    monkeypatch.setattr(api_app, "llm_settings_path", lambda: target, raising=False)
    return TestClient(api_app.app)


def _create(
    client,
    name="MiniMax",
    base_url="https://mm/anthropic",
    api_key="sk-mm-123456",
    model="MiniMax-M3",
    **extra,
):
    body = {"name": name, "base_url": base_url, "model": model, **extra}
    if api_key is not None:
        body["api_key"] = api_key
    return client.post("/system/llm-providers", json=body)


class TestProviderCrud:
    def test_create_first_auto_activates(self, api):
        r = _create(api)
        assert r.status_code == 200
        entry = r.json()
        assert entry["name"] == "MiniMax" and entry["current"] is True
        assert entry["api_key_set"] is True
        assert "sk-mm-123456" not in (entry["api_key_preview"] or "")
        listing = api.get("/system/llm-providers").json()
        assert listing["current"] == entry["id"]
        assert len(listing["providers"]) == 1

    def test_raw_key_never_echoed(self, api):
        entry = _create(api).json()
        payload = api.get("/system/llm-providers").text
        assert "sk-mm-123456" not in payload
        assert entry["api_key_preview"]  # masked preview present

    def test_create_second_does_not_change_current(self, api):
        first = _create(api).json()
        _create(
            api,
            name="DeepSeek",
            base_url="https://ds/anthropic",
            api_key="sk-ds",
            model="deepseek-chat",
        )
        listing = api.get("/system/llm-providers").json()
        assert listing["current"] == first["id"]
        assert [p["name"] for p in listing["providers"]] == ["MiniMax", "DeepSeek"]

    def test_update_without_key_keeps_secret(self, api):
        entry = _create(api).json()
        r = api.post(
            "/system/llm-providers",
            json={
                "id": entry["id"],
                "name": "MiniMax",
                "base_url": "https://mm2/anthropic",
                "model": "MiniMax-M3",
            },
        )
        assert r.status_code == 200
        updated = r.json()
        assert updated["base_url"] == "https://mm2/anthropic"
        assert updated["api_key_set"] is True, "absent api_key must keep the stored key"
        # And the stored key still resolves.
        from rlpe.llm_backends import resolve_llm_api_key

        assert resolve_llm_api_key({}) == "sk-mm-123456"

    def test_update_empty_key_clears_secret(self, api):
        entry = _create(api).json()
        r = api.post(
            "/system/llm-providers",
            json={
                "id": entry["id"],
                "name": "MiniMax",
                "base_url": "https://mm/anthropic",
                "model": "m",
                "api_key": "",
            },
        )
        assert r.json()["api_key_set"] is False

    def test_create_key_without_base_url_rejected(self, api):
        r = api.post(
            "/system/llm-providers",
            json={
                "name": "X",
                "base_url": "",
                "api_key": "sk-orphan",
                "model": "m",
            },
        )
        assert r.status_code == 400

    def test_update_unknown_id_404(self, api):
        r = api.post(
            "/system/llm-providers",
            json={
                "id": "nope",
                "name": "X",
                "base_url": "https://x",
                "model": "m",
            },
        )
        assert r.status_code == 404


class TestActivateAndDelete:
    def test_activate_switches(self, api):
        _create(api)
        second = _create(
            api,
            name="DeepSeek",
            base_url="https://ds/anthropic",
            api_key="sk-ds",
            model="deepseek-chat",
        ).json()
        r = api.post(f"/system/llm-providers/{second['id']}/activate")
        assert r.status_code == 200
        listing = api.get("/system/llm-providers").json()
        assert listing["current"] == second["id"]
        assert listing["providers"][1]["current"] is True
        # The resolve chain now serves the switched preset.
        from rlpe.llm_backends import resolve_llm_api_key, resolve_llm_model

        assert resolve_llm_api_key({}) == "sk-ds"
        assert resolve_llm_model({}) == "deepseek-chat"

    def test_activate_unknown_404(self, api):
        assert api.post("/system/llm-providers/nope/activate").status_code == 404

    def test_delete_active_promotes_first_remaining(self, api):
        first = _create(api).json()
        second = _create(
            api,
            name="DeepSeek",
            base_url="https://ds/anthropic",
            api_key="sk-ds",
            model="deepseek-chat",
        ).json()
        api.post(f"/system/llm-providers/{second['id']}/activate")
        r = api.delete(f"/system/llm-providers/{second['id']}")
        assert r.status_code == 200
        listing = api.get("/system/llm-providers").json()
        assert listing["current"] == first["id"]
        assert [p["name"] for p in listing["providers"]] == ["MiniMax"]

    def test_delete_last_leaves_empty(self, api):
        entry = _create(api).json()
        api.delete(f"/system/llm-providers/{entry['id']}")
        listing = api.get("/system/llm-providers").json()
        assert listing["providers"] == [] and listing["current"] is None

    def test_delete_unknown_404(self, api):
        assert api.delete("/system/llm-providers/nope").status_code == 404


class TestLlmStatusActiveProfile:
    def test_status_reports_active_profile_name(self, api):
        _create(api, name="DeepSeek")
        data = api.get("/system/llm-status").json()
        assert data["active_profile"] == "DeepSeek"


class TestBudgetCeiling:
    def test_131072_accepted_and_higher_rejected(self, api, monkeypatch):
        """F18: the thinking-budget ceiling moved 32000 → 131072."""
        from rlpe.api.app import JobOptions

        assert JobOptions(llm_thinking_budget_tokens=131_072).llm_thinking_budget_tokens == 131_072
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            JobOptions(llm_thinking_budget_tokens=131_073)
