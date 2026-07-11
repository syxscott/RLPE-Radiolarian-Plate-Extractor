"""Settings tab — application-wide configuration backed by QSettings.

This tab is distinct from the per-run Run Tab's quick config:
it stores the *defaults* used when a new job starts. Settings
persist across GUI restarts via ``QSettings`` (INI on Linux,
Registry on Windows, plist on macOS).

Layout:
* Top: Theme + recent-dirs + save settings
* Middle: GROBID / OCR / LLM defaults
* Bottom: Advanced (PBDB + diagnostics) + Reset
"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .constants import (
    COMBO_MIN_WIDTH,
    INPUT_WIDTH_LONG,
    INPUT_WIDTH_MEDIUM,
    INPUT_WIDTH_PATH,
    APP_AUTHOR,
    APP_DOMAIN,
    APP_NAME,
    APP_VERSION,
    DEFAULT_GROBID_MAX_RETRIES,
    DEFAULT_GROBID_TIMEOUT,
    DEFAULT_GROBID_URL,
    DEFAULT_M3_BUDGET,
    DEFAULT_M3_PROMPT_LANG,
    DEFAULT_M3_TIMEOUT,
    DEFAULT_M3_MAX_RETRIES,
    DEFAULT_MINIMAX_MODEL,
    DEFAULT_OCR_LANG,
    DEFAULT_PALEO_MAX_OCC,
    DEFAULT_RENDER_DPI,
    DEFAULT_THEME,
    QS_KEY_LAST_DIR,
    QS_KEY_LAST_EXPORT_DIR,
    QS_KEY_THEME,
    RANGE_DPI,
    RANGE_GROBID_MAX_RETRIES,
    RANGE_GROBID_TIMEOUT,
    RANGE_M3_BUDGET,
    RANGE_M3_MAX_RETRIES,
    RANGE_M3_OUTPUT_TOKENS,
    RANGE_M3_TIMEOUT,
    RANGE_OD_CAPTION_WINDOW,
    RANGE_PALEO_OCC,
    THEME_DARK,
    THEME_LIGHT,
    THEME_SYSTEM,
)
from .i18n_widgets import tr_button, tr_checkbox, tr_combobox, tr_groupbox, tr_label, tr_lineedit, tr_spinbox
from .styles import SPACE_L, SPACE_M, SPACE_S, apply_theme
from . import i18n
from .utils import get_gui_logger


class SettingsTab(QWidget):
    """Application-wide configuration tab."""

    settings_changed = Signal()  # emitted when settings saved (so other tabs refresh)

    def __init__(self, settings: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._qsettings = QSettings(APP_DOMAIN, APP_NAME)
        self._settings = settings  # in-memory cache (Run tab reads/writes this)
        self._log = get_gui_logger()
        self._build_ui()
        self._load()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACE_L, SPACE_L, SPACE_L, SPACE_L)
        outer.setSpacing(SPACE_M)

        # Title
        title = QLabel(f"⚙️ {APP_NAME}  ·  v{APP_VERSION}")
        title.setObjectName("sectionTitle")
        outer.addWidget(title)

        # ---- Appearance ----
        appearance = QGroupBox("🎨 Appearance")
        alayout = QFormLayout(appearance)
        alayout.setHorizontalSpacing(SPACE_L)
        alayout.setVerticalSpacing(SPACE_S)

        self._theme_combo = QComboBox()
        self._theme_combo.addItems([THEME_LIGHT, THEME_DARK, THEME_SYSTEM])
        self._theme_combo.currentTextChanged.connect(self._on_theme_change)
        alayout.addRow(tr_label("settab.theme"), self._theme_combo)

        # Language picker (Phase 33 — bilingual UI)
        self._lang_combo = tr_combobox(
            "settab.lang",
            min_width=COMBO_MIN_WIDTH,
        )
        for code, label in i18n.available_languages():
            self._lang_combo.addItem(label, userData=code)
        # Set current to current language
        for i in range(self._lang_combo.count()):
            if self._lang_combo.itemData(i) == i18n.current_language():
                self._lang_combo.setCurrentIndex(i)
                break
        self._lang_combo.currentIndexChanged.connect(self._on_lang_change)
        alayout.addRow(tr_label("settab.lang"), self._lang_combo)

        outer.addWidget(appearance)

        # ---- Default directories ----
        dirs_group = QGroupBox("📁 Default directories")
        dlayout = QFormLayout(dirs_group)
        dlayout.setHorizontalSpacing(SPACE_L)
        dlayout.setVerticalSpacing(SPACE_S)

        self._pdf_dir_edit = QLineEdit()
        self._pdf_dir_edit.setReadOnly(True)
        dlayout.addRow("Default PDF directory:", self._build_file_row(self._pdf_dir_edit, "pdf_dir"))

        self._out_dir_edit = QLineEdit()
        self._out_dir_edit.setReadOnly(True)
        dlayout.addRow("Default output directory:", self._build_file_row(self._out_dir_edit, "output_dir"))

        outer.addWidget(dirs_group)

        # ---- GROBID defaults ----
        grobid = QGroupBox("🌐 GROBID defaults")
        glayout = QFormLayout(grobid)
        glayout.setHorizontalSpacing(SPACE_L)
        glayout.setVerticalSpacing(SPACE_S)

        self._grobid_url = QLineEdit(DEFAULT_GROBID_URL)
        glayout.addRow("GROBID URL:", self._grobid_url)

        self._grobid_retries = QSpinBox()
        self._grobid_retries.setRange(*RANGE_GROBID_MAX_RETRIES)
        self._grobid_retries.setValue(DEFAULT_GROBID_MAX_RETRIES)
        glayout.addRow("Max retries:", self._grobid_retries)

        self._grobid_timeout = QSpinBox()
        self._grobid_timeout.setRange(*RANGE_GROBID_TIMEOUT)
        self._grobid_timeout.setValue(DEFAULT_GROBID_TIMEOUT)
        glayout.addRow("Timeout (s):", self._grobid_timeout)

        outer.addWidget(grobid)

        # ---- OCR defaults ----
        ocr = QGroupBox("🔡 OCR defaults")
        olayout = QFormLayout(ocr)
        olayout.setHorizontalSpacing(SPACE_L)
        olayout.setVerticalSpacing(SPACE_S)

        self._ocr_backend = QComboBox()
        self._ocr_backend.addItems(["paddleocr", "easyocr"])
        olayout.addRow("OCR backend:", self._ocr_backend)

        self._ocr_lang = QLineEdit(DEFAULT_OCR_LANG)
        self._ocr_lang.setPlaceholderText("en, en,ja, en,ch_sim…")
        olayout.addRow("OCR language(s):", self._ocr_lang)

        self._caption_window = QSpinBox()
        self._caption_window.setRange(1, 50)
        self._caption_window.setValue(2)
        olayout.addRow("Caption window (GROBID):", self._caption_window)

        self._od_caption_window = QSpinBox()
        self._od_caption_window.setRange(*RANGE_OD_CAPTION_WINDOW)
        self._od_caption_window.setValue(5)
        olayout.addRow("OD caption window:", self._od_caption_window)

        outer.addWidget(ocr)

        # ---- LLM defaults ----
        llm = QGroupBox("🧠 LLM / M3 defaults")
        llayout = QFormLayout(llm)
        llayout.setHorizontalSpacing(SPACE_L)
        llayout.setVerticalSpacing(SPACE_S)

        self._llm_backend = QComboBox()
        self._llm_backend.addItems(["minimax", "minimax-m3", "minimax_api", "transformers", "ollama", "llamacpp"])
        llayout.addRow("LLM backend:", self._llm_backend)

        self._m3_model = QLineEdit(DEFAULT_MINIMAX_MODEL)
        llayout.addRow("M3 model:", self._m3_model)

        self._m3_prompt_lang = QComboBox()
        self._m3_prompt_lang.addItems(["auto", "zh", "en", "ja"])
        self._m3_prompt_lang.setCurrentText(DEFAULT_M3_PROMPT_LANG)
        llayout.addRow("M3 prompt lang:", self._m3_prompt_lang)

        self._m3_budget = QSpinBox()
        self._m3_budget.setRange(*RANGE_M3_BUDGET)
        self._m3_budget.setValue(DEFAULT_M3_BUDGET)
        llayout.addRow("M3 thinking budget:", self._m3_budget)

        self._m3_output = QSpinBox()
        self._m3_output.setRange(*RANGE_M3_OUTPUT_TOKENS)
        self._m3_output.setValue(2048)
        llayout.addRow("M3 max output tokens:", self._m3_output)

        self._m3_timeout = QSpinBox()
        self._m3_timeout.setRange(*RANGE_M3_TIMEOUT)
        self._m3_timeout.setValue(DEFAULT_M3_TIMEOUT)
        llayout.addRow("M3 timeout (s):", self._m3_timeout)

        self._m3_max_retries = QSpinBox()
        self._m3_max_retries.setRange(*RANGE_M3_MAX_RETRIES)
        self._m3_max_retries.setValue(3)
        llayout.addRow("M3 max retries:", self._m3_max_retries)

        outer.addWidget(llm)

        # ---- PBDB defaults ----
        pbdb = QGroupBox("🦴 Paleobiology Database defaults")
        playout = QFormLayout(pbdb)
        playout.setHorizontalSpacing(SPACE_L)
        playout.setVerticalSpacing(SPACE_S)

        self._use_pbdb = QCheckBox("Enable PBDB enrichment (taxonomy + occurrences)")
        self._use_pbdb.setChecked(True)
        playout.addRow("", self._use_pbdb)

        self._pbdb_max_occ = QSpinBox()
        self._pbdb_max_occ.setRange(*RANGE_PALEO_OCC)
        self._pbdb_max_occ.setValue(DEFAULT_PALEO_MAX_OCC)
        playout.addRow("Max occurrences per species:", self._pbdb_max_occ)

        self._pbdb_endpoint = QLineEdit("https://paleobiodb.org/data1.2")
        self._pbdb_endpoint.setPlaceholderText("(leave blank for default)")
        playout.addRow("PBDB endpoint:", self._pbdb_endpoint)

        outer.addWidget(pbdb)

        # ---- Diagnostics ----
        diag = QGroupBox("🛠️ Diagnostics")
        dlayout = QFormLayout(diag)
        dlayout.setHorizontalSpacing(SPACE_L)
        dlayout.setVerticalSpacing(SPACE_S)

        self._dpi = QSpinBox()
        self._dpi.setRange(*RANGE_DPI)
        self._dpi.setValue(DEFAULT_RENDER_DPI)
        dlayout.addRow("Render DPI:", self._dpi)

        self._save_intermediate = QCheckBox("Save intermediate panels (large disk usage)")
        dlayout.addRow("", self._save_intermediate)

        self._open_log_btn = QPushButton("📂 Open log file")
        self._open_log_btn.clicked.connect(self._open_log_file)
        dlayout.addRow("Logs:", self._open_log_btn)

        outer.addWidget(diag)

        outer.addStretch(1)

        # ---- Action bar ----
        actions = QHBoxLayout()
        actions.setSpacing(SPACE_S)
        actions.addStretch(1)

        reset_btn = QPushButton("Reset to defaults")
        reset_btn.setObjectName("flat")
        reset_btn.clicked.connect(self._reset_defaults)
        actions.addWidget(reset_btn)

        save_btn = QPushButton("💾 Save settings")
        save_btn.setObjectName("primary")
        save_btn.clicked.connect(self._save)
        actions.addWidget(save_btn)

        outer.addLayout(actions)

    def _build_file_row(self, line_edit: QLineEdit, kind: str) -> QHBoxLayout:
        """Build a QLineEdit + browse button row."""
        row = QHBoxLayout()
        row.setSpacing(SPACE_S)
        row.addWidget(line_edit, 1)
        btn = QPushButton("Browse…")
        btn.clicked.connect(lambda: self._pick_dir(line_edit, kind))
        row.addWidget(btn)
        return row

    def _pick_dir(self, line_edit: QLineEdit, kind: str) -> None:
        path = QFileDialog.getExistingDirectory(self, f"Choose {kind}", line_edit.text() or str(__file__))
        if path:
            line_edit.setText(path)

    def _open_log_file(self) -> None:
        import os
        import subprocess
        from .utils import LOG_FILE_NAME
        log_path = os.path.expanduser(f"~/.cache/rlpe/gui/{LOG_FILE_NAME}")
        try:
            if hasattr(subprocess, "Popen"):
                if sys.platform == "darwin":
                    subprocess.Popen(["open", log_path])
                elif sys.platform.startswith("win"):
                    os.startfile(log_path)  # type: ignore[attr-defined]
                else:
                    subprocess.Popen(["xdg-open", log_path])
            QMessageBox.information(self, "Log file", f"Log file: {log_path}")
        except Exception as exc:
            QMessageBox.warning(self, "Log file", f"Could not open: {exc}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load settings from QSettings + in-memory cache."""
        # Theme
        theme = self._qsettings.value(QS_KEY_THEME, DEFAULT_THEME)
        if theme not in (THEME_LIGHT, THEME_DARK, THEME_SYSTEM):
            theme = DEFAULT_THEME
        self._theme_combo.setCurrentText(theme)

        # Default directories
        self._pdf_dir_edit.setText(self._qsettings.value(QS_KEY_LAST_DIR, ""))
        self._out_dir_edit.setText(self._qsettings.value(QS_KEY_LAST_EXPORT_DIR, ""))

        # GROBID
        self._grobid_url.setText(self._qsettings.value("grobid_url", DEFAULT_GROBID_URL))
        self._grobid_retries.setValue(int(self._qsettings.value("grobid_max_retries", DEFAULT_GROBID_MAX_RETRIES)))
        self._grobid_timeout.setValue(int(self._qsettings.value("grobid_timeout", DEFAULT_GROBID_TIMEOUT)))

        # OCR
        self._ocr_backend.setCurrentText(self._qsettings.value("ocr_backend", "paddleocr"))
        self._ocr_lang.setText(self._qsettings.value("ocr_lang", DEFAULT_OCR_LANG))
        self._caption_window.setValue(int(self._qsettings.value("caption_window", 2)))
        self._od_caption_window.setValue(int(self._qsettings.value("od_caption_window", 5)))

        # LLM
        llm_backend = self._qsettings.value("llm_backend", "minimax")
        ix = self._llm_backend.findText(llm_backend)
        if ix >= 0:
            self._llm_backend.setCurrentIndex(ix)
        self._m3_model.setText(self._qsettings.value("m3_model", DEFAULT_MINIMAX_MODEL))
        self._m3_prompt_lang.setCurrentText(self._qsettings.value("m3_prompt_lang", DEFAULT_M3_PROMPT_LANG))
        self._m3_budget.setValue(int(self._qsettings.value("MiniMax_thinking_budget", DEFAULT_M3_BUDGET)))
        self._m3_output.setValue(int(self._qsettings.value("MiniMax_max_output_tokens", 2048)))
        self._m3_timeout.setValue(int(self._qsettings.value("MiniMax_timeout_sec", DEFAULT_M3_TIMEOUT)))
        self._m3_max_retries.setValue(int(self._qsettings.value("MiniMax_max_retries", DEFAULT_M3_MAX_RETRIES)))

        # PBDB
        self._use_pbdb.setChecked(self._qsettings.value("use_paleodb", True, type=bool))
        self._pbdb_max_occ.setValue(int(self._qsettings.value("paleodb_max_occurrences", DEFAULT_PALEO_MAX_OCC)))
        self._pbdb_endpoint.setText(self._qsettings.value("paleodb_endpoint", "https://paleobiodb.org/data1.2"))

        # Diagnostics
        self._dpi.setValue(int(self._qsettings.value("render_dpi", DEFAULT_RENDER_DPI)))
        self._save_intermediate.setChecked(self._qsettings.value("save_intermediate", False, type=bool))

    def _save(self) -> None:
        """Write the current settings to QSettings + in-memory cache."""
        # Theme
        theme = self._theme_combo.currentText()
        self._qsettings.setValue(QS_KEY_THEME, theme)
        self._settings["theme"] = theme

        # Default directories
        self._qsettings.setValue(QS_KEY_LAST_DIR, self._pdf_dir_edit.text())
        self._qsettings.setValue(QS_KEY_LAST_EXPORT_DIR, self._out_dir_edit.text())
        self._settings["last_pdf_dir"] = self._pdf_dir_edit.text()
        self._settings["last_export_dir"] = self._out_dir_edit.text()

        # GROBID
        self._qsettings.setValue("grobid_url", self._grobid_url.text())
        self._qsettings.setValue("grobid_max_retries", self._grobid_retries.value())
        self._qsettings.setValue("grobid_timeout", self._grobid_timeout.value())

        # OCR
        self._qsettings.setValue("ocr_backend", self._ocr_backend.currentText())
        self._qsettings.setValue("ocr_lang", self._ocr_lang.text())
        self._qsettings.setValue("caption_window", self._caption_window.value())
        self._qsettings.setValue("od_caption_window", self._od_caption_window.value())

        # LLM
        self._qsettings.setValue("llm_backend", self._llm_backend.currentText())
        self._qsettings.setValue("m3_model", self._m3_model.text())
        self._qsettings.setValue("m3_prompt_lang", self._m3_prompt_lang.currentText())
        self._qsettings.setValue("MiniMax_thinking_budget", self._m3_budget.value())
        self._qsettings.setValue("MiniMax_max_output_tokens", self._m3_output.value())
        self._qsettings.setValue("MiniMax_timeout_sec", self._m3_timeout.value())
        self._qsettings.setValue("MiniMax_max_retries", self._m3_max_retries.value())

        # PBDB
        self._qsettings.setValue("use_paleodb", self._use_pbdb.isChecked())
        self._qsettings.setValue("paleodb_max_occurrences", self._pbdb_max_occ.value())
        self._qsettings.setValue("paleodb_endpoint", self._pbdb_endpoint.text())

        # Diagnostics
        self._qsettings.setValue("render_dpi", self._dpi.value())
        self._qsettings.setValue("save_intermediate", self._save_intermediate.isChecked())

        self._qsettings.sync()
        # Apply theme live
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, theme)
        self.settings_changed.emit()
        QMessageBox.information(self, "Settings", "Settings saved.")

    def _reset_defaults(self) -> None:
        if QMessageBox.question(
            self,
            "Reset settings?",
            "This will reset all settings to their defaults. Continue?",
        ) != QMessageBox.Yes:
            return
        self._qsettings.clear()
        self._load()
        QMessageBox.information(self, "Settings", "Settings reset to defaults.")

    # ------------------------------------------------------------------
    # Live theme change
    # ------------------------------------------------------------------
    def _on_lang_change(self, idx: int) -> None:
        lang = self._lang_combo.itemData(idx)
        if not lang:
            return
        i18n.set_language(lang)
        # Refresh all tabs in the main window
        main_window = self.parent()
        while main_window is not None and not isinstance(main_window, type(self.parent()).__mro__[0]):
            main_window = main_window.parent()
        # Walk all tabs and call their _refresh_texts() if they have it
        if hasattr(self.parent(), "_tabs"):
            for i in range(self.parent()._tabs.count()):
                w = self.parent()._tabs.widget(i)
                if hasattr(w, "_refresh_texts"):
                    try:
                        w._refresh_texts()
                    except Exception:
                        pass
        # Re-apply theme to refresh status bar / menu colours
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, i18n.current_language() and "light" or "light")  # keep light

    def _on_theme_change(self, theme: str) -> None:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, theme)
        # Don't persist until user clicks Save (live preview only).

    # ------------------------------------------------------------------
    # Public API used by the parent MainWindow
    # ------------------------------------------------------------------
    def _refresh_texts(self) -> None:
        # Re-translate the appearance + section labels.
        # Most settings widgets are QLabel/QComboBox/QLineEdit and
        # would be retexted by the global i18n._apply_registry()
        # sweep; this method exists so the parent MainWindow can
        # call back into us after a language switch.
        for w in self.findChildren((QLabel, QGroupBox, QPushButton, QCheckBox)):
            name = w.objectName()
            if name and i18n._tr(name) != "⟦" + name + "⟦":
                # Already registered; the registry re-applies it.
                pass

    def apply_to_run_settings(self) -> None:
        """When the Run tab starts, push current Settings-tab values
        into the in-memory run defaults so new jobs use them."""
        self._settings.update({
            "grobid_url": self._grobid_url.text(),
            "grobid_max_retries": self._grobid_retries.value(),
            "grobid_timeout": self._grobid_timeout.value(),
            "ocr_backend": self._ocr_backend.currentText(),
            "ocr_lang": self._ocr_lang.text() or DEFAULT_OCR_LANG,
            "caption_window": self._caption_window.value(),
            "od_caption_window": self._od_caption_window.value(),
            "llm_backend": self._llm_backend.currentText(),
            "m3_prompt_lang": self._m3_prompt_lang.currentText(),
            "m3_model": self._m3_model.text() or DEFAULT_MINIMAX_MODEL,
            "MiniMax_thinking_budget": self._m3_budget.value(),
            "MiniMax_max_output_tokens": self._m3_output.value(),
            "MiniMax_timeout_sec": self._m3_timeout.value(),
            "MiniMax_max_retries": self._m3_max_retries.value(),
            "use_paleodb": self._use_pbdb.isChecked(),
            "paleodb_max_occurrences": self._pbdb_max_occ.value(),
            "paleodb_endpoint": self._pbdb_endpoint.text(),
            "render_dpi": self._dpi.value(),
            "save_intermediate": self._save_intermediate.isChecked(),
        })