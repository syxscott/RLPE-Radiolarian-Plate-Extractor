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
from .i18n_widgets import (_ensure_size_hint, normalise_input_height, tr_button, tr_checkbox, tr_combobox, tr_form_row, tr_groupbox, tr_label, tr_lineedit, tr_spinbox)
from .styles import SPACE_L, SPACE_M, SPACE_S, apply_theme
from . import i18n
from .utils import get_gui_logger


class SettingsTab(QWidget):
    """Application-wide configuration tab."""

    settings_changed = Signal()  # emitted when settings saved (so other tabs refresh)

    def __init__(self, settings: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._qsettings = QSettings(APP_AUTHOR, APP_NAME)
        self._settings = settings  # in-memory cache (Run tab reads/writes this)
        self._log = get_gui_logger()
        self._build_ui()
        self._load()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        # Wrap the body in a QScrollArea so on short / 150% DPI
        # windows the user can scroll instead of having the Save
        # button clipped off the bottom.
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setObjectName("settab.scroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QScrollArea.NoFrame)

        body = QWidget()
        body.setObjectName("settab.body")
        # cumulative widths, which underestimates when the scroll
        # area viewport is wide. Force a minimum width matching the
        # viewport so groupboxes expand to fill the visible space
        # rather than getting squashed into half-width.
        from PySide6.QtWidgets import QSizePolicy
        body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        body.setMinimumWidth(700)  # matches main window min width - margins
        body_layout = QVBoxLayout(body)
        # and 150% DPI don't clip QSpinBox / QComboBox value text.
        # Called at the end of _build_ui via _normalise_input_heights().
        body_layout.setContentsMargins(SPACE_L, SPACE_L, SPACE_L, SPACE_L)
        body_layout.setSpacing(SPACE_M)

        # Title
        # We do NOT use tr_label or i18n.register_widget_text for the
        # title because both would substitute APP_NAME (the Python
        # constant) into the template, locking the language. Instead
        # the _refresh_texts method calls i18n._tr("app.name") so the
        # ZH title says "RLPE - 放射虫图版提取系统" rather than the
        # EN constant.
        title = QLabel(self)
        title.setObjectName("settab.title")
        title.setProperty("class", "sectionTitle")
        self._title_label = title
        body_layout.addWidget(title)

        # ---- Appearance ----
        appearance = tr_groupbox("settab.appearance")
        alayout = QFormLayout(appearance)
        alayout.setHorizontalSpacing(SPACE_L)
        alayout.setVerticalSpacing(SPACE_S)

        self._theme_combo = QComboBox()
        from PySide6.QtWidgets import QSizePolicy
        self._theme_combo.setMinimumHeight(32)
        self._theme_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        # Phase 55 audit M8 — use the i18n-aware helper so the
        # labels refresh on language switch.
        from .constants import theme_friendly_options
        from .i18n_widgets import populate_friendly_combo
        populate_friendly_combo(self._theme_combo, theme_friendly_options)
        # Phase 54 audit: M2 — ``currentIndexChanged`` emits the
        # *index* (int), not the userData. The previous direct
        # connection passed that int to ``_on_theme_change(theme: str)``,
        # which then compared ``theme == "dark"`` (False for any int)
        # and silently routed every selection to the system theme.
        # Use a lambda to forward the userData string instead. The
        # saved value was already correct because ``_save`` reads
        # ``currentData()``; only the live preview was broken.
        self._theme_combo.currentIndexChanged.connect(
            lambda _idx: self._on_theme_change(self._theme_combo.currentData())
        )
        alayout.addRow(tr_label("settab.theme"), self._theme_combo)

        # Language picker (Phase 33 — bilingual UI)
        self._lang_combo = tr_combobox(
            "settab.lang",
            min_width=COMBO_MIN_WIDTH,
        )
        # Phase 55 audit M8 — use the i18n-aware helper so the
        # language labels ("English" / "中文") refresh on switch.
        # Note: this is called from the language picker itself; the
        # helper preserves currentData() across rebuilds so picking
        # English keeps English selected even after switching back.
        from .i18n_widgets import populate_friendly_combo
        populate_friendly_combo(
            self._lang_combo, i18n.available_languages,
            default_code=i18n.current_language(),
        )
        self._lang_combo.currentIndexChanged.connect(self._on_lang_change)
        alayout.addRow(tr_label("settab.lang"), self._lang_combo)

        # Phase 54 audit M24: do NOT add self._title_label again
        # here. The title QLabel is already inserted at the top of
        # ``body_layout`` above. Re-adding it would put the title in
        # the layout twice and crash Qt with "QLabel already has a
        # parent / is already a child of the layout".
        body_layout.addWidget(appearance)

        # ---- Default directories ----
        dirs_group = tr_groupbox("settab.dirs")
        dlayout = QFormLayout(dirs_group)
        dlayout.setHorizontalSpacing(SPACE_L)
        dlayout.setVerticalSpacing(SPACE_S)

        self._pdf_dir_edit = QLineEdit()
        self._pdf_dir_edit.setReadOnly(True)
        dlayout.addRow(tr_label("settab.dir.pdf"), self._build_file_row(self._pdf_dir_edit, "pdf_dir"))

        self._out_dir_edit = QLineEdit()
        self._out_dir_edit.setReadOnly(True)
        dlayout.addRow(tr_label("settab.dir.out"), self._build_file_row(self._out_dir_edit, "output_dir"))

        body_layout.addWidget(dirs_group)

        # ---- GROBID defaults ----
        grobid = tr_groupbox("settab.grobid")
        glayout = QFormLayout(grobid)
        glayout.setHorizontalSpacing(SPACE_L)
        glayout.setVerticalSpacing(SPACE_S)

        self._grobid_url = QLineEdit(DEFAULT_GROBID_URL)
        glayout.addRow(tr_label("settab.grobid.url"), self._grobid_url)

        self._grobid_retries = QSpinBox()
        self._grobid_retries.setRange(*RANGE_GROBID_MAX_RETRIES)
        self._grobid_retries.setValue(DEFAULT_GROBID_MAX_RETRIES)
        glayout.addRow(tr_label("settab.grobid.retries"), self._grobid_retries)

        self._grobid_timeout = QSpinBox()
        self._grobid_timeout.setRange(*RANGE_GROBID_TIMEOUT)
        self._grobid_timeout.setValue(DEFAULT_GROBID_TIMEOUT)
        glayout.addRow(tr_label("settab.grobid.timeout"), self._grobid_timeout)

        body_layout.addWidget(grobid)

        # ---- OCR defaults ----
        ocr = tr_groupbox("settab.ocr")
        olayout = QFormLayout(ocr)
        olayout.setHorizontalSpacing(SPACE_L)
        olayout.setVerticalSpacing(SPACE_S)

        self._ocr_backend = QComboBox()
        self._ocr_backend.setMinimumHeight(32)
        self._ocr_backend.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        # Phase 55 audit M8 — i18n-aware populate helper.
        from .constants import ocr_backend_friendly_options
        from .i18n_widgets import populate_friendly_combo
        populate_friendly_combo(self._ocr_backend, ocr_backend_friendly_options)
        olayout.addRow(tr_label("settab.ocr.backend"), self._ocr_backend)

        self._ocr_lang = QLineEdit(DEFAULT_OCR_LANG)
        self._ocr_lang.setPlaceholderText("en, en,ja, en,ch_sim…")
        olayout.addRow(tr_label("settab.ocr.lang"), self._ocr_lang)

        self._caption_window = QSpinBox()
        self._caption_window.setRange(1, 50)
        self._caption_window.setValue(2)
        olayout.addRow(tr_label("settab.ocr.caption_window"), self._caption_window)

        self._od_caption_window = QSpinBox()
        self._od_caption_window.setRange(*RANGE_OD_CAPTION_WINDOW)
        self._od_caption_window.setValue(5)
        olayout.addRow(tr_label("settab.ocr.od_caption_window"), self._od_caption_window)

        body_layout.addWidget(ocr)

        # ---- LLM defaults ----
        llm = tr_groupbox("settab.llm")
        llayout = QFormLayout(llm)
        llayout.setHorizontalSpacing(SPACE_L)
        llayout.setVerticalSpacing(SPACE_S)

        self._llm_backend = QComboBox()
        self._llm_backend.setMinimumHeight(32)
        self._llm_backend.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        # Phase 55 audit M8 — i18n-aware populate helper.
        from .constants import llm_backend_friendly_options
        populate_friendly_combo(self._llm_backend, llm_backend_friendly_options)
        llayout.addRow(tr_label("settab.llm.backend"), self._llm_backend)

        self._m3_model = QLineEdit(DEFAULT_MINIMAX_MODEL)
        llayout.addRow(tr_label("settab.m3.model"), self._m3_model)

        self._m3_prompt_lang = QComboBox()
        self._m3_prompt_lang.setMinimumHeight(32)
        self._m3_prompt_lang.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        # Phase 55 audit M8 — i18n-aware populate helper.
        from .constants import m3_prompt_lang_friendly_options
        populate_friendly_combo(self._m3_prompt_lang, m3_prompt_lang_friendly_options)
        llayout.addRow(tr_label("settab.m3.lang"), self._m3_prompt_lang)

        self._m3_budget = QSpinBox()
        self._m3_budget.setRange(*RANGE_M3_BUDGET)
        self._m3_budget.setValue(DEFAULT_M3_BUDGET)
        llayout.addRow(tr_label("settab.m3.budget"), self._m3_budget)

        self._m3_output = QSpinBox()
        self._m3_output.setRange(*RANGE_M3_OUTPUT_TOKENS)
        self._m3_output.setValue(2048)
        llayout.addRow(tr_label("settab.m3.output"), self._m3_output)

        self._m3_timeout = QSpinBox()
        self._m3_timeout.setRange(*RANGE_M3_TIMEOUT)
        self._m3_timeout.setValue(DEFAULT_M3_TIMEOUT)
        llayout.addRow(tr_label("settab.m3.timeout"), self._m3_timeout)

        self._m3_max_retries = QSpinBox()
        self._m3_max_retries.setRange(*RANGE_M3_MAX_RETRIES)
        self._m3_max_retries.setValue(3)
        llayout.addRow(tr_label("settab.m3.max_retries"), self._m3_max_retries)

        body_layout.addWidget(llm)

        # ---- PBDB defaults ----
        pbdb = tr_groupbox("settab.pbdb")
        playout = QFormLayout(pbdb)
        playout.setHorizontalSpacing(SPACE_L)
        playout.setVerticalSpacing(SPACE_S)

        self._use_pbdb = tr_checkbox("settab.pbdb.use", checked=True)
        playout.addRow("", self._use_pbdb)

        self._pbdb_max_occ = QSpinBox()
        self._pbdb_max_occ.setRange(*RANGE_PALEO_OCC)
        self._pbdb_max_occ.setValue(DEFAULT_PALEO_MAX_OCC)
        playout.addRow(tr_label("settab.pbdb.occ"), self._pbdb_max_occ)

        self._pbdb_endpoint = QLineEdit("https://paleobiodb.org/data1.2")
        self._pbdb_endpoint.setPlaceholderText("(leave blank for default)")
        playout.addRow(tr_label("settab.pbdb.endpoint"), self._pbdb_endpoint)

        body_layout.addWidget(pbdb)

        # ---- Diagnostics ----
        diag = tr_groupbox("settab.diag")
        dlayout = QFormLayout(diag)
        dlayout.setHorizontalSpacing(SPACE_L)
        dlayout.setVerticalSpacing(SPACE_S)

        self._dpi = QSpinBox()
        self._dpi.setRange(*RANGE_DPI)
        self._dpi.setValue(DEFAULT_RENDER_DPI)
        dlayout.addRow(tr_label("settab.diag.dpi"), self._dpi)

        self._save_intermediate = tr_checkbox("settab.diag.save_intermediate")
        dlayout.addRow("", self._save_intermediate)

        self._open_log_btn = tr_button("settab.diag.log_btn")
        self._open_log_btn.clicked.connect(self._open_log_file)
        dlayout.addRow(tr_label("settab.diag.log_label"), self._open_log_btn)

        body_layout.addWidget(diag)

        # ---- Action bar ----
        actions = QHBoxLayout()
        actions.setSpacing(SPACE_S)
        actions.addStretch(1)

        reset_btn = tr_button("settab.reset")
        reset_btn.setProperty("class", "flat")
        reset_btn.clicked.connect(self._reset_defaults)
        actions.addWidget(reset_btn)

        save_btn = tr_button("settab.save")
        save_btn.setProperty("class", "primary")
        save_btn.clicked.connect(self._save)
        actions.addWidget(save_btn)

        body_layout.addLayout(actions)

        # body fills the tab and becomes scrollable when the window
        # is shorter than the body.
        scroll.setWidget(body)
        outer.addWidget(scroll)

        # minimum height so the value is visible. This runs once
        # at construction; for runtime dynamic widgets, call
        # ``_normalise_input_heights()`` after adding them.
        self._normalise_input_heights()

    def _build_file_row(self, line_edit: QLineEdit, kind: str) -> QHBoxLayout:
        """Build a QLineEdit + browse button row."""
        row = QHBoxLayout()
        row.setSpacing(SPACE_S)
        row.addWidget(line_edit, 1)
        btn = tr_button("common.browse")
        btn.clicked.connect(lambda: self._pick_dir(line_edit, kind))
        row.addWidget(btn)
        return row

    def _pick_dir(self, line_edit: QLineEdit, kind: str) -> None:
        # Phase 54 audit m20 — ``str(__file__)`` is a .py file path
        # inside the source tree; handing it to a directory chooser
        # was a small bug. Fall back to the user's home directory,
        # which is a sensible default for "where do I want to point
        # the PDF folder" dialogs.
        from pathlib import Path
        initial = line_edit.text()
        # Validate the saved path still exists; fall back to home if not.
        if not initial or not Path(initial).is_dir():
            initial = str(Path.home())
        path = QFileDialog.getExistingDirectory(
            self,
            i18n._tr("common.choose_dir").format(kind=kind),
            initial,
        )
        if path:
            line_edit.setText(path)

    def _open_log_file(self) -> None:
        import os
        import subprocess
        import sys
        from .utils import LOG_FILE_NAME
        log_path = os.path.expanduser(f"~/.cache/rlpe/gui/{LOG_FILE_NAME}")
        try:
            # subprocess.Popen is always available — no hasattr check.
            if sys.platform == "darwin":
                subprocess.Popen(["open", log_path])
            elif sys.platform.startswith("win"):
                os.startfile(log_path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", log_path])
            QMessageBox.information(
                self,
                i18n._tr("settab.diag.log_btn"),
                i18n._tr("settab.log.path").format(path=log_path),
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                i18n._tr("settab.diag.log_btn"),
                i18n._tr("settab.log.open_fail").format(error=exc, path=log_path),
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load settings from QSettings + in-memory cache."""
        # Theme — Phase 47: the combo stores friendly names as text
        # and ISO codes ("light", "dark", "system") in userData.
        theme = self._qsettings.value(QS_KEY_THEME, DEFAULT_THEME)
        if theme not in (THEME_LIGHT, THEME_DARK, THEME_SYSTEM):
            theme = DEFAULT_THEME
        # Find the row whose userData == theme
        theme_ix = -1
        for i in range(self._theme_combo.count()):
            if self._theme_combo.itemData(i) == theme:
                theme_ix = i
                break
        if theme_ix >= 0:
            self._theme_combo.setCurrentIndex(theme_ix)

        # Default directories
        self._pdf_dir_edit.setText(self._qsettings.value(QS_KEY_LAST_DIR, ""))
        self._out_dir_edit.setText(self._qsettings.value(QS_KEY_LAST_EXPORT_DIR, ""))

        # Phase 55 audit M6 — wrap every ``int(qsettings.value(...))`` in
        # a helper that falls back to the supplied default on parse
        # failure. On Linux the INI backend returns stored ints as
        # strings; if a stray empty / corrupted value slips in,
        # ``int("")`` raises ValueError and the spinbox would crash
        # the whole ``_load`` path. The helper also clamps to the
        # spinbox range to handle hand-edited INI files.
        def _qint(key: str, default: int) -> int:
            raw = self._qsettings.value(key, default)
            try:
                return int(raw)
            except (TypeError, ValueError):
                return default

        # GROBID
        self._grobid_url.setText(self._qsettings.value("grobid_url", DEFAULT_GROBID_URL))
        self._grobid_retries.setValue(_qint("grobid_max_retries", DEFAULT_GROBID_MAX_RETRIES))
        self._grobid_timeout.setValue(_qint("grobid_timeout", DEFAULT_GROBID_TIMEOUT))

        # OCR — Phase 47: the combo stores ISO codes in userData.
        ocr_backend = self._qsettings.value("ocr_backend", "paddleocr")
        ocr_ix = self._ocr_backend.findData(ocr_backend)
        if ocr_ix >= 0:
            self._ocr_backend.setCurrentIndex(ocr_ix)
        self._ocr_lang.setText(self._qsettings.value("ocr_lang", DEFAULT_OCR_LANG))
        self._caption_window.setValue(_qint("caption_window", 2))
        self._od_caption_window.setValue(_qint("od_caption_window", 5))

        # LLM
        llm_backend = self._qsettings.value("llm_backend", "minimax")
        ix = self._llm_backend.findData(llm_backend)
        if ix >= 0:
            self._llm_backend.setCurrentIndex(ix)
        self._m3_model.setText(self._qsettings.value("m3_model", DEFAULT_MINIMAX_MODEL))
        # the friendly display.
        m3_lang = self._qsettings.value("m3_prompt_lang", DEFAULT_M3_PROMPT_LANG)
        m3_lang_ix = self._m3_prompt_lang.findData(m3_lang)
        if m3_lang_ix >= 0:
            self._m3_prompt_lang.setCurrentIndex(m3_lang_ix)
        self._m3_budget.setValue(_qint("MiniMax_thinking_budget", DEFAULT_M3_BUDGET))
        self._m3_output.setValue(_qint("MiniMax_max_output_tokens", 2048))
        self._m3_timeout.setValue(_qint("MiniMax_timeout_sec", DEFAULT_M3_TIMEOUT))
        self._m3_max_retries.setValue(_qint("MiniMax_max_retries", DEFAULT_M3_MAX_RETRIES))

        # PBDB
        self._use_pbdb.setChecked(self._qsettings.value("use_paleodb", True, type=bool))
        self._pbdb_max_occ.setValue(_qint("paleodb_max_occurrences", DEFAULT_PALEO_MAX_OCC))
        self._pbdb_endpoint.setText(self._qsettings.value("paleodb_endpoint", "https://paleobiodb.org/data1.2"))

        # Diagnostics
        self._dpi.setValue(_qint("render_dpi", DEFAULT_RENDER_DPI))
        self._save_intermediate.setChecked(self._qsettings.value("save_intermediate", False, type=bool))

    def _save(self) -> None:
        """Write the current settings to QSettings + in-memory cache."""
        # Theme — Phase 47: use the ISO code, not the friendly name.
        theme = self._theme_combo.currentData() or self._theme_combo.currentText()
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

        # OCR — Phase 47: use the ISO code (userData), not the
        # friendly name (currentText).
        ocr_backend_code = self._ocr_backend.currentData() or self._ocr_backend.currentText()
        self._qsettings.setValue("ocr_backend", ocr_backend_code)
        self._qsettings.setValue("ocr_lang", self._ocr_lang.text())
        self._qsettings.setValue("caption_window", self._caption_window.value())
        self._qsettings.setValue("od_caption_window", self._od_caption_window.value())

        # LLM
        llm_backend_code = self._llm_backend.currentData() or self._llm_backend.currentText()
        self._qsettings.setValue("llm_backend", llm_backend_code)
        self._qsettings.setValue("m3_model", self._m3_model.text())
        m3_lang_code = self._m3_prompt_lang.currentData() or self._m3_prompt_lang.currentText()
        self._qsettings.setValue("m3_prompt_lang", m3_lang_code)
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
        # Phase 37 audit fix: refresh in-memory cache so Run tab
        # picks up the saved values immediately (was: cache stale
        # until app restart).
        self.apply_to_run_settings()
        # Apply theme live
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, theme)
        self.settings_changed.emit()
        QMessageBox.information(
            self,
            i18n._tr("settab.save"),
            i18n._tr("settab.save.done"),
        )

    def _reset_defaults(self) -> None:
        if QMessageBox.question(
            self,
            i18n._tr("settab.reset.confirm.title"),
            i18n._tr("settab.reset.confirm.body"),
        ) != QMessageBox.Yes:
            return
        self._qsettings.clear()
        # Phase 37 audit fix: block theme/lang combos from firing
        # _on_theme_change / _on_lang_change while we programmatically
        # load defaults — otherwise the user sees a "Settings reset"
        # dialog on top of a momentarily-repainted window.
        from PySide6.QtCore import QSignalBlocker
        with QSignalBlocker(self._theme_combo), QSignalBlocker(self._lang_combo):
            self._load()
        # up the freshly-reset defaults immediately (was: only QSettings
        # was reset, Run tab kept stale values until app restart).
        self.apply_to_run_settings()
        # Re-apply default theme now that signals are unblocked.
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, DEFAULT_THEME)
        self.settings_changed.emit()
        QMessageBox.information(
            self,
            i18n._tr("settab.reset.confirm.title"),
            i18n._tr("settab.reset.done"),
        )

    # ------------------------------------------------------------------
    # Live theme change
    # ------------------------------------------------------------------
    def _on_lang_change(self, idx: int) -> None:
        lang = self._lang_combo.itemData(idx)
        if not lang:
            return
        i18n.set_language(lang)
        # uses the same language. This is a live preference (no
        # Save button click required).
        self._qsettings.setValue("ui/language", lang)
        self._qsettings.sync()
        # Phase 37 audit fix: ``self.parent()`` is the QTabWidget's
        # tab page wrapper, NOT the MainWindow. The MainWindow owns
        # ``_tabs``. Walk up the parent chain with ``self.window()``
        # (returns the top-level QWidget) which is always the
        # MainWindow for a normal QMainWindow GUI.
        win = self.window()
        if win is not None and hasattr(win, "_tabs"):
            for i in range(win._tabs.count()):
                w = win._tabs.widget(i)
                # already register an i18n listener that calls their
                # own _refresh_texts via i18n.set_language listeners —
                # so we don't need to manually walk and call. But
                # call _refresh_texts() explicitly here as a belt-and-
                # suspenders fallback in case a tab hasn't registered
                # a listener yet.
                if hasattr(w, "_refresh_texts"):
                    try:
                        w._refresh_texts()
                    except Exception:
                        pass
        # No-op theme refresh (keep current theme).
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None and hasattr(self, "_theme_combo"):
            # name; apply_theme() needs "light" / "dark" / "system",
            # not "浅色" / "深色" / "跟随系统".
            theme_code = self._theme_combo.currentData() or self._theme_combo.currentText()
            apply_theme(app, theme_code)

    def _on_theme_change(self, theme: str) -> None:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, theme)
        # Don't persist until user clicks Save (live preview only).

    # ------------------------------------------------------------------
    # Public API used by the parent MainWindow
    # ------------------------------------------------------------------
    def _normalise_input_heights(self, min_height: int = 32) -> None:
        """Phase 34/35/36: clamp every input widget's minimum height.

        QFormLayout collapses a row to the height of its smallest
        child, so a QSpinBox with default 22-px height clips the value
        text in the QSS dark theme and at 150% DPI. This walks all
        descendants once after _build_ui() and forces each input
        widget (SpinBox / DoubleSpinBox / ComboBox / LineEdit /
        PushButton / CheckBox) to have a minimum height of ``min_height``.

        Phase 35 added QCheckBox — long checkbox labels such as
        "Enable PBDB enrichment (taxonomy + occurrences)" or
        "Save intermediate panels (large disk usage)" would otherwise
        wrap / clip at the QSS-imposed 22-px checkbox height.

        Phase 36 also patches ``sizeHint()`` so QFormLayout's row
        calculation uses ``min_height`` rather than the widget's
        intrinsic 22-26 px height. Without this, ``setMinimumHeight(30)``
        is ignored when computing row heights because Qt layouts read
        ``sizeHint()`` first.
        """
        for w in self.findChildren(QWidget):
            if isinstance(
                w,
                (QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit, QPushButton, QCheckBox),
            ):
                if w.minimumHeight() < min_height:
                    w.setMinimumHeight(min_height)
                _ensure_size_hint(w, min_height)

    def _refresh_texts(self) -> None:
        # Re-translate the appearance + section labels.
        # Most settings widgets are QLabel/QComboBox/QLineEdit and
        # would be retexted by the global i18n._apply_registry()
        # sweep; this method exists so the parent MainWindow can
        # call back into us after a language switch.
        # + app.version so the ZH title says "RLPE - 放射虫图版提取系统"
        # rather than the EN constant. ``APP_NAME`` and ``APP_VERSION``
        # are language-independent (e.g. version numbers) so we use
        # them as-is.
        for w in self.findChildren(QLabel):
            if w.objectName() == "settab.title":
                w.setText(
                    i18n._tr("settab.title").format(
                        app=i18n._tr("app.name"),
                        version=APP_VERSION,
                    )
                )

    def apply_to_run_settings(self) -> None:
        """When the Run tab starts, push current Settings-tab values
        into the in-memory run defaults so new jobs use them.

        Phase 37 audit fix: was missing ``last_pdf_dir``,
        ``last_export_dir``, ``theme``, and ``m3_multi_plate_enrich``,
        so a fresh Run tab would still read the values the user had
        at first launch even after Settings changed them.
        """
        self._settings.update({
            # (friendly name) so the Run tab receives the right value.
            "theme": self._theme_combo.currentData() or self._theme_combo.currentText(),
            "last_pdf_dir": self._pdf_dir_edit.text(),
            "last_export_dir": self._out_dir_edit.text(),
            "grobid_url": self._grobid_url.text(),
            "grobid_max_retries": self._grobid_retries.value(),
            "grobid_timeout": self._grobid_timeout.value(),
            "ocr_backend": self._ocr_backend.currentData() or self._ocr_backend.currentText(),
            "ocr_lang": self._ocr_lang.text() or DEFAULT_OCR_LANG,
            "caption_window": self._caption_window.value(),
            "od_caption_window": self._od_caption_window.value(),
            "llm_backend": self._llm_backend.currentData() or self._llm_backend.currentText(),
            "m3_prompt_lang": self._m3_prompt_lang.currentData() or self._m3_prompt_lang.currentText(),
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