"""API 配置 tab (F18) — multi-provider LLM API presets.

The cc-switch model inside the desktop GUI: save several named
Anthropic-compatible provider presets (MiniMax / DeepSeek / Kimi /
...) and switch the active one at any time. Presets live in the shared
``~/.rlpe/llm_api.json`` (see :mod:`rlpe.llm_settings`) so the Web UI
and the CLI resolve the same active configuration.

The raw API key is never displayed after it is saved — the list shows
only that a key exists; an empty key field on save means "keep the
stored key".
"""

from __future__ import annotations

import threading
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from .i18n import _tr
from .i18n_widgets import tr_button, tr_checkbox, tr_groupbox, tr_label
from .styles import SPACE_L, SPACE_M, SPACE_S
from .utils import get_gui_logger


class ApiTab(QWidget):
    """Provider preset management — the GUI half of the F18 feature."""

    # Emitted after the presets file changed (other tabs may refresh).
    providers_changed = Signal()
    # Emitted by the connection-test worker thread; queued into the GUI
    # thread so the QMessageBox calls stay on the main thread.
    _test_finished = Signal(dict, str)

    def __init__(self, settings: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._editing_id: str | None = None  # None = form creates a new preset
        self._log = get_gui_logger()
        self._build_ui()
        self._migrate_legacy_qsettings()
        self._reload()

    def _migrate_legacy_qsettings(self) -> None:
        """One-time migration: pre-F18 GUI versions stored the provider
        key/model in QSettings ("MiniMax_api_key" / "llm_api_key" /
        "m3_model" / "llm_model"). If the presets file has no presets
        yet, promote those values into an initial preset so an upgrade
        never strands the user's key (F18 review fix)."""
        from ..llm_settings import ProviderConfig, list_providers, upsert_provider

        try:
            if list_providers():
                return
            key = str(
                self._qsettings_legacy_value("MiniMax_api_key")
                or self._qsettings_legacy_value("llm_api_key")
                or ""
            )
            model = str(
                self._qsettings_legacy_value("m3_model")
                or self._qsettings_legacy_value("llm_model")
                or ""
            )
            base_url = str(self._qsettings_legacy_value("llm_base_url") or "")
            if not (key or model or base_url):
                return
            upsert_provider(
                ProviderConfig(
                    name=self._tr_default_name(),
                    base_url=base_url,
                    api_key=key,
                    model=model,
                )
            )
            self._log.info("api_tab: migrated legacy QSettings API config into presets file")
        except Exception:
            self._log.debug("legacy API-config migration failed", exc_info=True)

    @staticmethod
    def _qsettings_legacy_value(key: str) -> Any:
        from PySide6.QtCore import QSettings

        from .constants import APP_AUTHOR, APP_NAME

        return QSettings(APP_AUTHOR, APP_NAME).value(key, "") or ""

    @staticmethod
    def _tr_default_name() -> str:
        return _tr("apitab.default_name")

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACE_L, SPACE_L, SPACE_L, SPACE_L)
        outer.setSpacing(SPACE_M)

        hint = QLabel(_tr("apitab.hint"))
        hint.setWordWrap(True)
        outer.addWidget(hint)

        # ---- preset list ----
        list_group = tr_groupbox("apitab.list_group")
        list_layout = QVBoxLayout(list_group)
        list_layout.setSpacing(SPACE_S)

        self._provider_list = QListWidget()
        self._provider_list.setMinimumHeight(140)
        self._provider_list.itemDoubleClicked.connect(lambda _item: self._load_selected_into_form())
        self._provider_list.itemSelectionChanged.connect(self._update_button_states)
        list_layout.addWidget(self._provider_list)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(SPACE_S)
        self._activate_btn = tr_button("apitab.activate")
        self._activate_btn.clicked.connect(self._on_activate)
        self._edit_btn = tr_button("apitab.edit")
        self._edit_btn.clicked.connect(self._load_selected_into_form)
        self._delete_btn = tr_button("apitab.delete")
        self._delete_btn.clicked.connect(self._on_delete)
        self._test_btn = tr_button("apitab.test")
        self._test_btn.clicked.connect(self._on_test_connection)
        self._test_finished.connect(self._on_test_finished)
        for btn in (self._activate_btn, self._edit_btn, self._delete_btn, self._test_btn):
            btn_row.addWidget(btn)
        btn_row.addStretch(1)
        list_layout.addLayout(btn_row)
        outer.addWidget(list_group)
        self._update_button_states()

        # ---- editor form ----
        form_group = tr_groupbox("apitab.form_group")
        form_layout = QFormLayout(form_group)
        form_layout.setHorizontalSpacing(SPACE_L)
        form_layout.setVerticalSpacing(SPACE_S)

        self._form_title = QLabel(_tr("apitab.form.new"))
        form_layout.addRow(self._form_title)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText(_tr("apitab.name.hint"))
        form_layout.addRow(tr_label("apitab.name"), self._name_edit)

        self._base_url_edit = QLineEdit()
        self._base_url_edit.setPlaceholderText(_tr("apitab.base_url.hint"))
        form_layout.addRow(tr_label("apitab.base_url"), self._base_url_edit)

        self._key_edit = QLineEdit()
        self._key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._key_edit.setPlaceholderText(_tr("apitab.key.hint"))
        self._show_key = tr_checkbox("apitab.key.show")
        self._show_key.toggled.connect(
            lambda checked: self._key_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        form_layout.addRow(tr_label("apitab.key"), self._key_edit)
        form_layout.addRow(QLabel(""), self._show_key)

        self._model_edit = QLineEdit()
        self._model_edit.setPlaceholderText(_tr("apitab.model.hint"))
        form_layout.addRow(tr_label("apitab.model"), self._model_edit)

        save_row = QHBoxLayout()
        save_row.setSpacing(SPACE_S)
        self._save_btn = tr_button("apitab.save")
        self._save_btn.clicked.connect(self._on_save)
        self._new_btn = tr_button("apitab.new")
        self._new_btn.clicked.connect(self._reset_form)
        save_row.addWidget(self._save_btn)
        save_row.addWidget(self._new_btn)
        save_row.addStretch(1)
        form_layout.addRow(save_row)

        outer.addWidget(form_group)
        outer.addStretch(1)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def _update_button_states(self) -> None:
        """Row-scoped actions need a selected preset; the connection test
        only needs a configured provider."""
        has_selection = self._selected_provider_id() is not None
        for btn in (self._activate_btn, self._edit_btn, self._delete_btn):
            btn.setEnabled(has_selection)

    def _selected_provider_id(self) -> str | None:
        item = self._provider_list.currentItem()
        return item.data(0x0100) if item is not None else None  # Qt.UserRole

    def _reload(self) -> None:
        """Refresh the list from the shared presets file."""
        from ..llm_settings import current_provider_id, list_providers, load_llm_settings

        self._provider_list.clear()
        current = current_provider_id()
        for p in list_providers():
            marker = f"  {_tr('apitab.active_marker')}" if p.id == current else ""
            key_state = _tr("apitab.key.saved") if p.api_key else _tr("apitab.key.missing")
            item = QListWidgetItem(
                f"{p.name}  ·  {p.model or '?'}  ·  {p.base_url or '?'}  ·  {key_state}{marker}"
            )
            item.setData(0x0100, p.id)  # Qt.UserRole
            self._provider_list.addItem(item)
        # Summary line for the currently active preset.
        flat = load_llm_settings()
        summary = (
            f"{_tr('apitab.current')}: {flat.profile_name or '—'}"
            f"  ·  {flat.model or '—'}  ·  {flat.base_url or '—'}"
        )
        self._form_title.setToolTip(summary)
        self._update_button_states()
        # Mirror the active preset into the shared settings cache so the
        # Run tab's summary (if it displays one) stays current. The run
        # itself resolves via the file — see pipeline_worker.
        self._settings["llm_profile_name"] = flat.profile_name
        self._settings["llm_base_url"] = flat.base_url
        self._settings["llm_model"] = flat.model
        self._settings["llm_api_key"] = flat.api_key

    def _reset_form(self) -> None:
        self._editing_id = None
        self._form_title.setText(_tr("apitab.form.new"))
        self._name_edit.clear()
        self._base_url_edit.clear()
        self._key_edit.clear()
        self._key_edit.setPlaceholderText(_tr("apitab.key.hint"))
        self._model_edit.clear()

    def _load_selected_into_form(self) -> None:
        from ..llm_settings import get_provider

        pid = self._selected_provider_id()
        if pid is None:
            return
        p = get_provider(pid)
        if p is None:
            return
        self._editing_id = p.id
        self._form_title.setText(f"{_tr('apitab.form.edit')}: {p.name}")
        self._name_edit.setText(p.name)
        self._base_url_edit.setText(p.base_url)
        self._key_edit.clear()
        self._key_edit.setPlaceholderText(
            f"{_tr('apitab.key.saved_hint')}" if p.api_key else _tr("apitab.key.hint")
        )
        self._model_edit.setText(p.model)

    def _on_save(self) -> None:
        from ..llm_settings import ProviderConfig, upsert_provider

        base_url = self._base_url_edit.text().strip()
        if not base_url:
            QMessageBox.warning(self, _tr("apitab.save"), _tr("apitab.error.base_url"))
            return
        # F19 review: catch a scheme-less address at save time instead of
        # letting the run fail later with the SSRF-guard error.
        from urllib.parse import urlparse

        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            QMessageBox.warning(self, _tr("apitab.save"), _tr("apitab.error.scheme"))
            return
        key_val = self._key_edit.text().strip()
        stored_key = key_val
        if self._editing_id and not key_val:
            # Empty field on edit = keep the stored key.
            from ..llm_settings import get_provider

            existing = get_provider(self._editing_id)
            stored_key = existing.api_key if existing else ""
        upsert_provider(
            ProviderConfig(
                name=self._name_edit.text().strip() or _tr("apitab.default_name"),
                base_url=base_url,
                api_key=stored_key,
                model=self._model_edit.text().strip(),
                id=self._editing_id or "",
            ),
            activate=self._editing_id is None and self._is_first_preset(),
        )
        self._reset_form()
        self._reload()
        self.providers_changed.emit()

    @staticmethod
    def _is_first_preset() -> bool:
        from ..llm_settings import list_providers

        return not list_providers()

    def _on_delete(self) -> None:
        from ..llm_settings import delete_provider

        pid = self._selected_provider_id()
        if pid is None:
            return
        answer = QMessageBox.question(
            self,
            _tr("apitab.delete"),
            _tr("apitab.delete.confirm"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        delete_provider(pid)
        if self._editing_id == pid:
            self._reset_form()
        self._reload()
        self.providers_changed.emit()

    def _on_activate(self) -> None:
        from ..llm_settings import set_current_provider

        pid = self._selected_provider_id()
        if pid is None:
            return
        set_current_provider(pid)
        self._reload()
        self.providers_changed.emit()

    def _on_test_connection(self) -> None:
        """Ping the ACTIVE preset on a worker thread.

        The HTTP call must NOT run on the GUI thread — a slow or hung
        endpoint would freeze the whole window for up to the backend
        timeout (F18 review fix). Config resolution happens here (fast,
        local); the request runs in a daemon thread and reports back
        through a queued Qt signal so the dialogs stay on the main
        thread.
        """
        from ..llm_backends import (
            AnthropicCompatBackend,
            resolve_llm_api_key,
            resolve_llm_base_url,
            resolve_llm_model,
        )

        api_key = resolve_llm_api_key()
        if not api_key:
            QMessageBox.warning(self, _tr("apitab.test"), _tr("apitab.test.no_key"))
            return
        base_url = resolve_llm_base_url()
        model = resolve_llm_model()
        if not base_url or not model:
            QMessageBox.warning(self, _tr("apitab.test"), _tr("apitab.test.no_config"))
            return
        try:
            backend = AnthropicCompatBackend(
                api_key=api_key,
                base_url=base_url,
                model=model,
                max_output_tokens=64,
                thinking_budget_tokens=0,
                enable_thinking=False,
                timeout_sec=15,
                max_retries=1,
                max_concurrent=1,
            )
        except Exception as exc:
            QMessageBox.warning(self, _tr("apitab.test"), str(exc))
            return

        self._test_btn.setEnabled(False)
        self._test_btn.setText(_tr("apitab.test.running"))
        threading.Thread(
            target=self._test_connection_worker,
            args=(backend, model),
            daemon=True,
        ).start()

    def _test_connection_worker(self, backend: Any, fallback_model: str) -> None:
        """Blocking ping; runs on a daemon thread. The result is
        marshalled back to the GUI thread through a queued signal."""
        try:
            result = backend.infer_text(
                system_prompt="You are a connection test. Reply with exactly: OK",
                user_prompt="ping",
            )
        except Exception as exc:
            result = {
                "fallback_used": True,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        self._test_finished.emit(result, fallback_model)

    def _on_test_finished(self, result: dict[str, Any], fallback_model: str) -> None:
        """Back on the GUI thread: render the ping outcome."""
        self._test_btn.setEnabled(True)
        self._test_btn.setText(_tr("apitab.test"))
        if result.get("fallback_used") and result.get("error_type", "").lower() not in {
            "jsonparseerror",
            "valueerror",
        }:
            QMessageBox.warning(self, _tr("apitab.test"), str(result.get("error") or "API error"))
        else:
            usage = result.get("usage") or {}
            QMessageBox.information(
                self,
                _tr("apitab.test"),
                _tr("apitab.test.ok").format(
                    model=result.get("model_version") or fallback_model,
                    tokens=usage.get("input_tokens"),
                ),
            )
