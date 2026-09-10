"""F18 review hardening (2026-09-11): ApiTab migration + threaded test.

Covers the two issues found in the post-implementation review:
1. One-time migration of legacy QSettings credentials
   (``MiniMax_api_key`` / ``llm_api_key`` / ``m3_model`` / ``llm_model``)
   into the presets file — an upgrade from pre-F18 must never strand the
   user's key.
2. The connection test must run its HTTP call on a worker thread, never
   the GUI thread, and report back through the queued ``_test_finished``
   signal.

Runs widgets offscreen like ``test_gui_path_label_splitter_2026_09_05``
(never drives ``QEventLoop.exec()``). The presets file and QSettings are
redirected into ``tmp_path`` so the developer's real configuration is
never touched.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from rlpe.llm_settings import (
    ProviderConfig,
    current_provider_id,
    list_providers,
    load_llm_settings,
)


@pytest.fixture()
def qt_app():
    pytest.importorskip("PySide6.QtWidgets")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture()
def isolated_stores(monkeypatch, tmp_path: Path):
    """Redirect BOTH stores (presets file + QSettings) into tmp_path."""
    target = tmp_path / "llm_api.json"
    import rlpe.llm_settings as ls

    monkeypatch.setattr(ls, "settings_path", lambda home=None: target)

    from PySide6.QtCore import QSettings

    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "qs"))
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    from rlpe.gui.constants import APP_AUTHOR, APP_NAME

    qs = QSettings(APP_AUTHOR, APP_NAME)
    for key in ("MiniMax_api_key", "llm_api_key", "m3_model", "llm_model", "llm_base_url"):
        qs.remove(key)
    qs.sync()
    yield target, qs


def _make_tab(qt_app, monkeypatch, tmp_path: Path):
    from rlpe.gui.api_tab import ApiTab

    return ApiTab({})


class TestLegacyQSettingsMigration:
    def test_legacy_credentials_become_a_preset(
        self, qt_app, isolated_stores, monkeypatch, tmp_path
    ):
        target, qs = isolated_stores
        qs.setValue("MiniMax_api_key", "sk-legacy-survivor")
        qs.setValue("m3_model", "MiniMax-M3")
        qs.sync()

        from rlpe.gui.api_tab import ApiTab

        ApiTab({})
        providers = list_providers(target)
        assert len(providers) == 1
        assert providers[0].api_key == "sk-legacy-survivor"
        assert providers[0].model == "MiniMax-M3"
        assert current_provider_id(target) == providers[0].id

    def test_migration_is_one_time_only(self, qt_app, isolated_stores, monkeypatch, tmp_path):
        target, qs = isolated_stores
        qs.setValue("MiniMax_api_key", "sk-once")
        qs.sync()

        from rlpe.gui.api_tab import ApiTab

        ApiTab({})
        ApiTab({})  # second construction must not duplicate
        assert len(list_providers(target)) == 1

    def test_no_legacy_values_no_preset(self, qt_app, isolated_stores, monkeypatch, tmp_path):
        from rlpe.gui.api_tab import ApiTab

        ApiTab({})
        assert list_providers(isolated_stores[0]) == []

    def test_existing_presets_block_migration(self, qt_app, isolated_stores, monkeypatch, tmp_path):
        target, qs = isolated_stores
        from rlpe.llm_settings import upsert_provider

        upsert_provider(
            ProviderConfig(name="Existing", base_url="https://x", model="m"),
            path=target,
        )
        qs.setValue("MiniMax_api_key", "sk-legacy")
        qs.sync()

        from rlpe.gui.api_tab import ApiTab

        ApiTab({})
        providers = list_providers(target)
        assert len(providers) == 1
        assert providers[0].name == "Existing"


class TestThreadedConnectionTest:
    def test_http_call_runs_off_the_gui_thread(
        self, qt_app, isolated_stores, monkeypatch, tmp_path
    ):
        """The infer_text call must execute on a worker thread; the
        result reaches _on_test_finished through the queued signal."""
        from PySide6.QtWidgets import QMessageBox

        import rlpe.llm_backends as lb
        from rlpe.gui.api_tab import ApiTab

        target, _qs = isolated_stores
        from rlpe.llm_settings import upsert_provider

        upsert_provider(
            ProviderConfig(name="P", base_url="https://p", api_key="sk", model="m"),
            path=target,
        )
        tab = _make_tab(qt_app, monkeypatch, tmp_path)

        main_thread = threading.main_thread()
        seen_threads: list[threading.Thread] = []
        shown: list[str] = []

        def _record(kind):
            def _show(*a, **k):
                shown.append(kind)
                return QMessageBox.StandardButton.Ok

            return _show

        monkeypatch.setattr(QMessageBox, "information", _record("info"))
        monkeypatch.setattr(QMessageBox, "warning", _record("warn"))
        monkeypatch.setattr("rlpe.llm_backends.resolve_llm_api_key", lambda extra=None: "sk")
        monkeypatch.setattr(
            "rlpe.llm_backends.resolve_llm_base_url", lambda extra=None: "http://localhost:0"
        )
        monkeypatch.setattr("rlpe.llm_backends.resolve_llm_model", lambda extra=None: "m")

        class SlowBackend:
            def __init__(self, **_):
                pass

            def infer_text(self, **_):
                seen_threads.append(threading.current_thread())
                time.sleep(0.05)
                return {
                    "fallback_used": False,
                    "model_version": "m",
                    "usage": {"input_tokens": 1},
                }

        monkeypatch.setattr(lb, "AnthropicCompatBackend", SlowBackend)

        tab._on_test_connection()
        # The worker runs on a daemon thread; pump the event loop until
        # the queued signal delivers the outcome to the GUI thread.
        deadline = time.time() + 5.0
        while time.time() < deadline and not shown:
            qt_app.processEvents()
            time.sleep(0.02)

        assert seen_threads, "backend call never ran"
        assert seen_threads[0] is not main_thread, (
            "infer_text ran on the GUI thread — the window would freeze"
        )
        assert shown == ["info"], shown
        assert tab._test_btn.isEnabled()

    def test_button_disabled_while_running(self, qt_app, isolated_stores, monkeypatch, tmp_path):
        from rlpe.gui.api_tab import ApiTab

        target, _qs = isolated_stores
        from rlpe.llm_settings import upsert_provider

        upsert_provider(
            ProviderConfig(name="P", base_url="https://p", api_key="sk", model="m"),
            path=target,
        )
        tab = _make_tab(qt_app, monkeypatch, tmp_path)
        monkeypatch.setattr("rlpe.llm_backends.AnthropicCompatBackend", lambda **_: object())

        # Stub the resolve chain so _on_test_connection builds a backend
        # without network, then verify the button disables synchronously.
        monkeypatch.setattr("rlpe.llm_backends.resolve_llm_api_key", lambda extra=None: "sk")
        monkeypatch.setattr(
            "rlpe.llm_backends.resolve_llm_base_url", lambda extra=None: "http://localhost:0"
        )
        monkeypatch.setattr("rlpe.llm_backends.resolve_llm_model", lambda extra=None: "m")
        # The worker would call infer_text on `object()` — stub the worker
        # to keep this test focused on the button state transitions.
        monkeypatch.setattr(tab, "_test_connection_worker", lambda backend, model: None)
        tab._on_test_connection()
        assert tab._test_btn.isEnabled() is False
        qt_app.processEvents()
