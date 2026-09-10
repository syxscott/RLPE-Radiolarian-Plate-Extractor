"""F17 (2026-09-08): /system/llm-config endpoints.

Covers the persisted-API-settings round trip through the Web API:
GET returns masked data only (never the raw key), POST merges fields
(absent api_key = keep the saved key; empty string = clear), and every
response echoes the settings path. The real settings file is
neutralised via the ``llm_settings_path`` indirection so tests never
touch the developer's actual ~/.rlpe/llm_api.json.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rlpe.llm_settings import LLMApiSettings


@pytest.fixture()
def api(monkeypatch, tmp_path: Path):
    TestClient = pytest.importorskip("fastapi.testclient").TestClient
    # Deterministic auth + redirect the settings file into tmp_path.
    monkeypatch.delenv("RLPE_API_KEY", raising=False)
    monkeypatch.setenv("RLPE_SETTINGS_HOME", str(tmp_path))
    import rlpe.llm_settings as ls

    monkeypatch.setattr(ls, "settings_path", lambda home=None: tmp_path / "llm_api.json")
    from rlpe.api import app as api_app

    return TestClient(api_app.app)


class TestLlmConfigRoundTrip:
    def test_get_empty_config(self, api):
        r = api.get("/system/llm-config")
        assert r.status_code == 200
        data = r.json()
        assert data["base_url"] == ""
        assert data["model"] == ""
        assert data["api_key_set"] is False
        assert data["api_key_preview"] is None
        assert data["settings_path"].endswith("llm_api.json")

    def test_post_saves_and_get_masks_key(self, api, tmp_path: Path):
        r = api.post(
            "/system/llm-config",
            json={
                "base_url": "https://api.example.com/anthropic",
                "api_key": "sk-secret-value-123456",
                "model": "example-model",
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["base_url"] == "https://api.example.com/anthropic"
        assert data["model"] == "example-model"
        assert data["api_key_set"] is True
        assert "sk-secret-value-123456" not in (data["api_key_preview"] or "")
        # Persisted file really contains the key (0600 file, user-owned).
        raw = (tmp_path / "llm_api.json").read_text()
        assert "sk-secret-value-123456" in raw

    def test_post_without_key_keeps_saved_key(self, api):
        api.post(
            "/system/llm-config",
            json={
                "base_url": "https://a.example.com/anthropic",
                "api_key": "sk-first",
                "model": "m1",
            },
        )
        # Edit only the model — the saved key must survive.
        r = api.post("/system/llm-config", json={"model": "m2"})
        assert r.status_code == 200
        assert r.json()["model"] == "m2"
        # And the backend resolver still sees the saved key.
        from rlpe.llm_backends import resolve_llm_api_key

        assert resolve_llm_api_key({}) == "sk-first"

    def test_post_empty_key_clears_it(self, api):
        api.post(
            "/system/llm-config",
            json={"base_url": "https://a.example.com/anthropic", "api_key": "sk-first"},
        )
        r = api.post("/system/llm-config", json={"api_key": ""})
        assert r.status_code == 200
        assert r.json()["api_key_set"] is False

    def test_post_key_without_base_url_rejected(self, api):
        r = api.post("/system/llm-config", json={"api_key": "sk-orphan"})
        assert r.status_code == 400

    def test_post_unknown_field_rejected(self, api, tmp_path: Path):
        # extra="forbid" surfaces typos; the app's validation handling
        # converts RequestValidationError, so use a non-raising client.
        TestClient = pytest.importorskip("fastapi.testclient").TestClient
        from rlpe.api import app as api_app

        client = TestClient(api_app.app, raise_server_exceptions=False)
        r = client.post("/system/llm-config", json={"api_kye": "typo"})
        # The app's global handler surfaces RequestValidationError as a
        # non-2xx JSON error; the contract under test is "a typo'd field
        # is never silently accepted".
        assert r.status_code >= 400, r.status_code
        assert "RequestValidationError" in r.text or r.status_code in (400, 422)


class TestLlmStatusReflectsSavedConfig:
    def test_saved_config_beats_env_precedence(self, api, monkeypatch):
        """Saved settings win over env for endpoint/model display, and
        key_source reports the saved file."""
        from rlpe.llm_settings import save_llm_settings

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-key-123456")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://env.example.com/anthropic")
        monkeypatch.setenv("ANTHROPIC_MODEL", "env-model")
        save_llm_settings(
            LLMApiSettings(
                base_url="https://saved.example.com/anthropic",
                api_key="sk-saved-key-123456",
                model="saved-model",
            )
        )
        r = api.get("/system/llm-status")
        assert r.status_code == 200
        data = r.json()
        assert data["key_configured"] is True
        assert data["key_source"] == "saved:~/.rlpe/llm_api.json"
        assert data["active_endpoint"] == "https://saved.example.com/anthropic"
        assert data["active_model"] == "saved-model"

    def test_env_used_when_nothing_saved(self, api, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-key-123456")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://env.example.com/anthropic")
        monkeypatch.setenv("ANTHROPIC_MODEL", "env-model")
        r = api.get("/system/llm-status")
        data = r.json()
        assert data["key_configured"] is True
        assert data["key_source"] == "env:ANTHROPIC_API_KEY"
        assert data["active_endpoint"] == "https://env.example.com/anthropic"
        assert data["active_model"] == "env-model"


class TestTestLlmUsesSavedConfig:
    def test_test_llm_reports_missing_config(self, api, monkeypatch):
        """No key/base_url/model anywhere → a clear MissingConfig error
        (F17 removed the vendor defaults that used to mask this)."""
        for var in (
            "ANTHROPIC_API_KEY",
            "MiniMax_API_KEY",
            "MINIMAX_API_KEY",
            "ANTHROPIC_BASE_URL",
            "MiniMax_BASE_URL",
            "ANTHROPIC_MODEL",
            "MiniMax_MODEL",
        ):
            monkeypatch.delenv(var, raising=False)
        r = api.post("/system/test-llm", json={})
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is False
        assert data["error_type"] == "MissingKey"
