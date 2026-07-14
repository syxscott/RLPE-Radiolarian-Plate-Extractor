"""Run tab — pick a PDF, configure pipeline, start a job."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .constants import (
    BUTTON_MIN_HEIGHT,
    BUTTON_PRIMARY_HEIGHT,
    COMBO_MIN_WIDTH,
    DEFAULT_GROBID_MAX_RETRIES,
    DEFAULT_GROBID_TIMEOUT,
    DEFAULT_GROBID_URL,
    DEFAULT_LLM_BACKEND,
    DEFAULT_M3_BUDGET,
    DEFAULT_M3_PROMPT_LANG,
    DEFAULT_M3_TIMEOUT,
    DEFAULT_M3_MAX_RETRIES,
    DEFAULT_M3_OUTPUT_TOKENS,
    DEFAULT_MINIMAX_MODEL,
    DEFAULT_OCR_BACKEND,
    DEFAULT_OCR_LANG,
    DEFAULT_PALEO_MAX_OCC,
    DEFAULT_RENDER_DPI,
    INPUT_WIDTH_LONG,
    INPUT_WIDTH_OCR_LANG,
    INPUT_WIDTH_PATH,
    INPUT_WIDTH_SHORT,
    QS_KEY_LAST_DIR,
    RANGE_DPI,
    RANGE_GROBID_MAX_RETRIES,
    RANGE_GROBID_TIMEOUT,
    RANGE_M3_BUDGET,
    RANGE_M3_OUTPUT_TOKENS,
    RANGE_M3_TIMEOUT,
    RANGE_M3_MAX_RETRIES,
    RANGE_OD_CAPTION_WINDOW,
    RANGE_PALEO_OCC,
)
from .i18n_widgets import (
    tr_button, tr_checkbox, tr_combobox, tr_doublespinbox,
    tr_form_row, tr_groupbox, tr_label, tr_lineedit, tr_spinbox,
)
from . import i18n
from .styles import SPACE_L, SPACE_M, SPACE_S, SPACE_XL
from .pipeline_worker import PipelineWorker
from .utils import (
    file_size_human,
    get_gui_logger,
    short_path,
)


class RunTab(QWidget):
    """First tab — configure + start a single-paper pipeline run."""

    # Emitted to the parent (MainWindow) to broadcast job state.
    job_started = Signal(str, str)        # job_id, pdf_path
    job_progress = Signal(str, int, int, str)  # job_id, cur, total, msg
    job_finished = Signal(str, list)     # job_id, results
    job_failed = Signal(str, str)        # job_id, error

    def __init__(self, settings: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._log = get_gui_logger()
        self._worker: PipelineWorker | None = None
        self._current_job_id: str | None = None
        self._build_ui()
        self._connect_signals()
        # Phase 54 audit m9: register an i18n listener so the
        # progress format / status strings re-render on language
        # switch. Previously the format was only refreshed when
        # ``_refresh_formats`` was called manually, which the parent
        # MainWindow only did on certain paths.
        i18n.add_listener(lambda _lang: self._refresh_formats())

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACE_L, SPACE_L, SPACE_L, SPACE_L)
        outer.setSpacing(SPACE_M)

        # ---- File picker row ----
        file_group = tr_groupbox("runtab.input_group")
        file_layout = QHBoxLayout(file_group)
        file_layout.setSpacing(SPACE_S)

        self._path_edit = tr_lineedit(
            "runtab.input.placeholder",
            min_width=INPUT_WIDTH_PATH * 2,  # paths are long
        )
        self._path_edit.setReadOnly(True)
        file_layout.addWidget(self._path_edit, 1)

        browse_btn = tr_button("runtab.browse", object_name="runtab.browse")
        browse_btn.setMinimumHeight(BUTTON_MIN_HEIGHT)
        browse_btn.clicked.connect(self._on_browse)
        file_layout.addWidget(browse_btn)

        clear_btn = tr_button("runtab.clear", object_name="runtab.clear")
        clear_btn.setProperty("class", "flat")  # QSS object-name selector
        clear_btn.setMinimumHeight(BUTTON_MIN_HEIGHT)
        clear_btn.clicked.connect(self._on_clear)
        file_layout.addWidget(clear_btn)

        outer.addWidget(file_group)

        # ---- Output dir row ----
        out_group = tr_groupbox("runtab.out_group")
        out_layout = QHBoxLayout(out_group)
        out_layout.setSpacing(SPACE_S)
        self._out_edit = tr_lineedit(
            "runtab.out.placeholder",
            min_width=INPUT_WIDTH_PATH * 2,
        )
        self._out_edit.setReadOnly(True)
        out_layout.addWidget(self._out_edit, 1)
        out_btn = tr_button("runtab.out.choose", object_name="runtab.out.choose")
        out_btn.setMinimumHeight(BUTTON_MIN_HEIGHT)
        out_btn.clicked.connect(self._on_pick_outdir)
        out_layout.addWidget(out_btn)
        open_btn = tr_button("runtab.out.open", object_name="runtab.out.open")
        open_btn.setMinimumHeight(BUTTON_MIN_HEIGHT)
        open_btn.clicked.connect(self._on_open_outdir)
        out_layout.addWidget(open_btn)
        outer.addWidget(out_group)

        # ---- Config form (split into Basic / Advanced sections) ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        cfg_widget = QWidget()
        cfg_layout = QVBoxLayout(cfg_widget)
        cfg_layout.setContentsMargins(0, 0, 0, 0)
        cfg_layout.setSpacing(SPACE_M)

        # Basic config grid
        basic = tr_groupbox("runtab.basic_group")
        basic_layout = QGridLayout(basic)
        basic_layout.setHorizontalSpacing(SPACE_L)
        basic_layout.setVerticalSpacing(SPACE_S)
        row = 0

        # OCR backend — Phase 48: friendly names (PaddleOCR (推荐) /
        # EasyOCR (多语言)) instead of raw "paddleocr"/"easyocr". The
        # backend still receives the raw ISO code via currentData()
        # (or currentText() fallback).
        from .constants import ocr_backend_friendly_options
        from PySide6.QtWidgets import QSizePolicy
        basic_layout.addWidget(tr_label("runtab.label.ocr_backend"), row, 0)
        self._ocr_combo = QComboBox()
        self._ocr_combo.setObjectName("runtab.ocr_backend")
        self._ocr_combo.setMinimumHeight(32)
        self._ocr_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        # Phase 55 audit M8 — populate via the i18n-aware helper so
        # the displayed friendly names refresh on language switch.
        from .i18n_widgets import populate_friendly_combo
        populate_friendly_combo(
            self._ocr_combo, ocr_backend_friendly_options,
            default_code=DEFAULT_OCR_BACKEND,
        )
        basic_layout.addWidget(self._ocr_combo, row, 1)

        basic_layout.addWidget(tr_label("runtab.label.ocr_lang"), row, 2)
        # Phase 46: a QComboBox with friendly language names
        # ("English" / "中文 (简体)" / "日本語") instead of the
        # unfriendly ISO codes ("en" / "ch_sim" / "ja"). The ISO
        # codes are still stored in the userData so the OCR backend
        # still receives the right value. setEditable(True) lets
        # power users still type a custom comma-separated list
        # (e.g. "en,ja,ch_sim") for advanced use cases.
        from .constants import ocr_lang_friendly_options
        self._ocr_lang_edit = QComboBox()
        self._ocr_lang_edit.setObjectName("runtab.ocr_lang")
        self._ocr_lang_edit.setEditable(True)
        # Phase 36: ensure input row height is 32 (matches tr_form_row).
        # Phase 46 + 47: tr_combobox factory does this for combos
        # created via the factory, but we're constructing the QComboBox
        # directly to use friendly names. Set min_height manually.
        from PySide6.QtWidgets import QSizePolicy
        self._ocr_lang_edit.setMinimumHeight(32)
        self._ocr_lang_edit.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        # Also fix the internal QLineEdit (the editable text field
        # inside the combo) — Phase 35 tests walk all QLineEdits
        # and expect min_height >= 30.
        self._ocr_lang_edit.lineEdit().setMinimumHeight(32)
        # Phase 55 audit M8 — use the i18n-aware helper so the
        # friendly names ("English" / "中文") update on language switch.
        from .i18n_widgets import populate_friendly_combo
        populate_friendly_combo(
            self._ocr_lang_edit, ocr_lang_friendly_options,
            default_code=DEFAULT_OCR_LANG,
        )
        self._ocr_lang_edit.setMinimumWidth(INPUT_WIDTH_OCR_LANG)
        # Phase 55 audit F-4 — the tooltip used to be hardcoded
        # English. Translate it via the i18n registry so the tooltip
        # follows the language switch.
        from . import i18n as _i18n_tooltip
        self._ocr_lang_edit.setObjectName("runtab.ocr_lang")
        _i18n_tooltip.register_widget_text(
            "runtab.ocr_lang", "toolTip", "runtab.ocr_lang.tooltip",
        )
        self._ocr_lang_edit.setToolTip(
            _i18n_tooltip._tr("runtab.ocr_lang.tooltip")
        )
        basic_layout.addWidget(self._ocr_lang_edit, row, 3)
        row += 1

        # GROBID
        basic_layout.addWidget(tr_label("runtab.label.grobid_url"), row, 0)
        self._grobid_edit = tr_lineedit(
            "runtab.label.grobid_url",
            min_width=INPUT_WIDTH_LONG,
            text=DEFAULT_GROBID_URL,
        )
        basic_layout.addWidget(self._grobid_edit, row, 1, 1, 3)
        row += 1

        basic_layout.addWidget(tr_label("runtab.label.grobid_retries"), row, 0)
        self._grobid_retries = tr_spinbox(
            "runtab.label.grobid_retries",
            min_width=INPUT_WIDTH_SHORT,
            min_val=RANGE_GROBID_MAX_RETRIES[0],
            max_val=RANGE_GROBID_MAX_RETRIES[1],
            value=DEFAULT_GROBID_MAX_RETRIES,
        )
        basic_layout.addWidget(self._grobid_retries, row, 1)
        basic_layout.addWidget(tr_label("runtab.label.grobid_timeout"), row, 2)
        self._grobid_timeout = tr_spinbox(
            "runtab.label.grobid_timeout",
            min_width=INPUT_WIDTH_SHORT,
            min_val=RANGE_GROBID_TIMEOUT[0],
            max_val=RANGE_GROBID_TIMEOUT[1],
            value=DEFAULT_GROBID_TIMEOUT,
        )
        basic_layout.addWidget(self._grobid_timeout, row, 3)
        row += 1

        # Caption windows
        basic_layout.addWidget(tr_label("runtab.label.caption_window"), row, 0)
        self._caption_window = tr_spinbox(
            "runtab.label.caption_window",
            min_width=INPUT_WIDTH_SHORT,
            min_val=1, max_val=50, value=2,
        )
        self._caption_window.setToolTip("GROBID caption→page lookup window (Phase 28)")
        basic_layout.addWidget(self._caption_window, row, 1)
        basic_layout.addWidget(tr_label("runtab.label.od_caption_window"), row, 2)
        self._od_caption_window = tr_spinbox(
            "runtab.label.od_caption_window",
            min_width=INPUT_WIDTH_SHORT,
            min_val=RANGE_OD_CAPTION_WINDOW[0],
            max_val=RANGE_OD_CAPTION_WINDOW[1],
            value=5,
        )
        self._od_caption_window.setToolTip("OpenDataLoader caption↔image cross-page window (Phase 28)")
        basic_layout.addWidget(self._od_caption_window, row, 3)
        row += 1

        # Workers
        basic_layout.addWidget(tr_label("runtab.label.workers"), row, 0)
        self._workers = tr_spinbox(
            "runtab.label.workers",
            min_width=INPUT_WIDTH_SHORT,
            min_val=1, max_val=32, value=1,
        )
        basic_layout.addWidget(self._workers, row, 1)
        basic_layout.addWidget(tr_label("runtab.label.panel_score"), row, 2)
        self._panel_score = tr_doublespinbox(
            min_width=INPUT_WIDTH_SHORT,
            min_val=0.0, max_val=1.0, value=0.80, step=0.05,
        )
        basic_layout.addWidget(self._panel_score, row, 3)
        row += 1

        # Use GPU
        basic_layout.addWidget(tr_label("runtab.label.use_gpu"), row, 0)
        self._gpu_check = tr_checkbox(
            "runtab.gpu_check",
            checked=True,
        )
        basic_layout.addWidget(self._gpu_check, row, 1, 1, 3)

        cfg_layout.addWidget(basic)

        # Advanced config
        adv = tr_groupbox("runtab.adv_group")
        adv_layout = QFormLayout(adv)
        adv_layout.setHorizontalSpacing(SPACE_L)
        adv_layout.setVerticalSpacing(SPACE_S)

        self._llm_combo = QComboBox()
        # Phase 47: friendly backend names instead of raw tokens.
        from .constants import llm_backend_friendly_options
        from PySide6.QtWidgets import QSizePolicy
        self._llm_combo.setMinimumHeight(32)
        self._llm_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        # Phase 55 audit M8 — populate via the i18n-aware helper.
        from .i18n_widgets import populate_friendly_combo
        populate_friendly_combo(
            self._llm_combo, llm_backend_friendly_options,
            default_code=DEFAULT_LLM_BACKEND,
        )
        lbl, w = tr_form_row("runtab.label.llm_backend", self._llm_combo)
        adv_layout.addRow(lbl, w)

        self._m3_lang = QComboBox()
        # Phase 47: friendly M3 prompt language names.
        from .constants import m3_prompt_lang_friendly_options
        from PySide6.QtWidgets import QSizePolicy
        self._m3_lang.setMinimumHeight(32)
        self._m3_lang.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        # Phase 55 audit M8 — populate via the i18n-aware helper.
        populate_friendly_combo(
            self._m3_lang, m3_prompt_lang_friendly_options,
            default_code=DEFAULT_M3_PROMPT_LANG,
        )
        lbl, w = tr_form_row("runtab.label.m3_lang", self._m3_lang)
        adv_layout.addRow(lbl, w)

        self._m3_model_edit = QLineEdit(DEFAULT_MINIMAX_MODEL)
        lbl, w = tr_form_row("runtab.label.m3_model", self._m3_model_edit)
        adv_layout.addRow(lbl, w)

        self._m3_budget = QSpinBox()
        self._m3_budget.setRange(*RANGE_M3_BUDGET)
        self._m3_budget.setValue(1024)
        lbl, w = tr_form_row("runtab.label.m3_budget", self._m3_budget)
        adv_layout.addRow(lbl, w)

        self._m3_output = QSpinBox()
        self._m3_output.setRange(*RANGE_M3_OUTPUT_TOKENS)
        self._m3_output.setValue(2048)
        lbl, w = tr_form_row("runtab.label.m3_output", self._m3_output)
        adv_layout.addRow(lbl, w)

        self._m3_timeout = QSpinBox()
        self._m3_timeout.setRange(*RANGE_M3_TIMEOUT)
        self._m3_timeout.setValue(DEFAULT_M3_TIMEOUT)
        lbl, w = tr_form_row("runtab.label.m3_timeout", self._m3_timeout)
        adv_layout.addRow(lbl, w)

        self._m3_max_retries = QSpinBox()
        self._m3_max_retries.setRange(*RANGE_M3_MAX_RETRIES)
        self._m3_max_retries.setValue(3)
        lbl, w = tr_form_row("runtab.label.m3_max_retries", self._m3_max_retries)
        adv_layout.addRow(lbl, w)

        self._paleodb_check = tr_checkbox("runtab.use_pbdb", checked=True)
        adv_layout.addRow("", self._paleodb_check)

        self._paleodb_occ = QSpinBox()
        self._paleodb_occ.setRange(*RANGE_PALEO_OCC)
        self._paleodb_occ.setValue(DEFAULT_PALEO_MAX_OCC)
        lbl, w = tr_form_row("runtab.label.paleodb_occ", self._paleodb_occ)
        adv_layout.addRow(lbl, w)

        self._geo_vision = tr_checkbox("runtab.geo_vision", checked=True)
        adv_layout.addRow("", self._geo_vision)

        self._m3_stage3 = tr_checkbox("runtab.m3_stage3", checked=True)
        adv_layout.addRow("", self._m3_stage3)

        self._m3_multi_plate = tr_checkbox("runtab.m3_multi_plate", checked=True)
        adv_layout.addRow("", self._m3_multi_plate)

        self._od_fallback = tr_checkbox("runtab.od_fallback", checked=True)
        adv_layout.addRow("", self._od_fallback)

        self._save_intermediate = tr_checkbox("runtab.save_intermediate", checked=False)
        adv_layout.addRow("", self._save_intermediate)

        self._dpi = QSpinBox()
        self._dpi.setRange(*RANGE_DPI)
        self._dpi.setValue(DEFAULT_RENDER_DPI)
        lbl, w = tr_form_row("runtab.label.dpi", self._dpi)
        adv_layout.addRow(lbl, w)

        cfg_layout.addWidget(adv)
        cfg_layout.addStretch(1)
        scroll.setWidget(cfg_widget)
        outer.addWidget(scroll, 1)

        # ---- Action bar ----
        action_bar = QFrame()
        action_bar.setObjectName("actionBar")
        action_layout = QHBoxLayout(action_bar)
        action_layout.setContentsMargins(0, SPACE_M, 0, 0)
        action_layout.setSpacing(SPACE_S)

        self._start_btn = tr_button("runtab.start", object_name="runtab.start")
        self._start_btn.setProperty("class", "primary")
        self._start_btn.setMinimumHeight(BUTTON_PRIMARY_HEIGHT)
        self._start_btn.clicked.connect(self._on_start)
        action_layout.addWidget(self._start_btn)

        self._cancel_btn = tr_button("runtab.cancel", object_name="runtab.cancel")
        self._cancel_btn.setProperty("class", "danger")
        self._cancel_btn.setMinimumHeight(BUTTON_MIN_HEIGHT)
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        action_layout.addWidget(self._cancel_btn)

        action_layout.addStretch(1)

        self._status_label = tr_label("runtab.status.idle", object_name="runtab.status_label")
        self._status_label.setProperty("class", "status")
        action_layout.addWidget(self._status_label)

        outer.addWidget(action_bar)

        # ---- Progress bar ----
        self._progress = QProgressBar()
        self._progress.setValue(0)
        self._progress.setFormat(i18n._tr("runtab.progress.idle"))
        self._progress.setMinimumHeight(20)
        outer.addWidget(self._progress)

        # Phase 55 audit B3 — split the live status line from the
        # static empty-state hint. The live status is rewritten on
        # every progress tick and on the worker's `status_changed`
        # signal; registering it in the i18n registry (via tr_label)
        # would cause the next ``set_language`` call to overwrite
        # the live text with the "no PDF selected" hint translation
        # mid-job, even though the pipeline is happily running.
        #
        # Instead we keep TWO labels stacked:
        #   * ``_hint_label``  — static empty-state hint (translatable)
        #   * ``_progress_msg`` — live status (NOT in i18n registry)
        self._hint_label = tr_label("runtab.prompt.no_pdf.body")
        self._hint_label.setWordWrap(True)
        outer.addWidget(self._hint_label)

        self._progress_msg = QLabel("")
        self._progress_msg.setWordWrap(True)
        self._progress_msg.setObjectName("runtab_progress_msg")
        outer.addWidget(self._progress_msg)
        # The live status starts hidden — once the pipeline emits its
        # first status_changed we show it and hide the hint.
        self._progress_msg.hide()

    def _connect_signals(self) -> None:
        """Wire file picker path → enable start button."""
        self._path_edit.textChanged.connect(self._on_path_changed)

    # ------------------------------------------------------------------
    # Public API used by the parent MainWindow
    # ------------------------------------------------------------------

    def collect_settings(self) -> dict[str, Any]:
        """Return the current settings as a flat dict (worker-ready)."""
        # Phase 46: ocr_lang is now a QComboBox with friendly names
        # but ISO codes in userData. Prefer currentData() (the ISO
        # code the user selected). Fall back to currentText() if the
        # user typed a custom comma-separated list (e.g. "en,ja").
        ocr_lang_data = self._ocr_lang_edit.currentData()
        if ocr_lang_data:
            ocr_lang = ocr_lang_data
        else:
            ocr_lang = self._ocr_lang_edit.currentText().strip() or "en"
        return {
            "ocr_backend": self._ocr_combo.currentData() or self._ocr_combo.currentText(),
            "ocr_lang": ocr_lang,
            "grobid_url": self._grobid_edit.text().strip(),
            "grobid_max_retries": self._grobid_retries.value(),
            "grobid_timeout": self._grobid_timeout.value(),
            "caption_window": self._caption_window.value(),
            "od_caption_window": self._od_caption_window.value(),
            "num_workers": self._workers.value(),
            "min_panel_score": self._panel_score.value(),
            "use_gpu": self._gpu_check.isChecked(),
            "llm_backend": self._llm_combo.currentData() or self._llm_combo.currentText(),
            "m3_prompt_lang": self._m3_lang.currentData() or self._m3_lang.currentText(),
            "m3_model": self._m3_model_edit.text().strip() or DEFAULT_MINIMAX_MODEL,
            "MiniMax_thinking_budget": self._m3_budget.value(),
            "MiniMax_max_output_tokens": self._m3_output.value(),
            "MiniMax_timeout_sec": self._m3_timeout.value(),
            "MiniMax_max_retries": self._m3_max_retries.value(),
            "use_paleodb": self._paleodb_check.isChecked(),
            "paleodb_max_occurrences": self._paleodb_occ.value(),
            "use_geo_vision": self._geo_vision.isChecked(),
            "use_m3_stage3": self._m3_stage3.isChecked(),
            "m3_multi_plate_enrich": self._m3_multi_plate.isChecked(),
            "disable_od_fallback": not self._od_fallback.isChecked(),
            "save_intermediate": self._save_intermediate.isChecked(),
            "render_dpi": self._dpi.value(),
        }

    def apply_settings(self, settings: dict[str, Any]) -> None:
        """Restore settings from the QSettings tab on startup."""
        s = dict(settings)
        if "ocr_backend" in s:
            # Phase 48: ocr_backend combo has friendly names as text
            # and ISO codes ("paddleocr" / "easyocr") in userData.
            # Use findData first, fall back to findText for legacy
            # configs that stored the friendly name.
            backend = str(s["ocr_backend"])
            ix = self._ocr_combo.findData(backend)
            if ix < 0:
                ix = self._ocr_combo.findText(backend)
            if ix >= 0:
                self._ocr_combo.setCurrentIndex(ix)
        if "ocr_lang" in s:
            # Phase 46: try to find the ISO code in the combo's
            # userData first (e.g. "en", "ch_sim"). If not found
            # (e.g. legacy config with comma-separated "en,ja"),
            # setText so the user sees their custom string.
            lang_str = str(s["ocr_lang"])
            ix = self._ocr_lang_edit.findData(lang_str)
            if ix >= 0:
                self._ocr_lang_edit.setCurrentIndex(ix)
            else:
                self._ocr_lang_edit.setCurrentText(lang_str)
        if "grobid_url" in s:
            self._grobid_edit.setText(str(s["grobid_url"]))
        if "grobid_max_retries" in s:
            self._grobid_retries.setValue(int(s["grobid_max_retries"]))
        if "grobid_timeout" in s:
            self._grobid_timeout.setValue(int(s["grobid_timeout"]))
        if "caption_window" in s:
            self._caption_window.setValue(int(s["caption_window"]))
        if "od_caption_window" in s:
            self._od_caption_window.setValue(int(s["od_caption_window"]))
        if "num_workers" in s:
            self._workers.setValue(int(s["num_workers"]))
        if "min_panel_score" in s:
            self._panel_score.setValue(float(s["min_panel_score"]))
        if "use_gpu" in s:
            self._gpu_check.setChecked(bool(s["use_gpu"]))
        if "llm_backend" in s:
            # Phase 53: _llm_combo stores ISO codes in userData and
            # friendly names as itemText (Phase 47 pattern). Use
            # findData (the ISO code) instead of findText, otherwise
            # settings saved as e.g. "minimax" don't match the
            # friendly label "MiniMax-M3 (推荐)".
            backend = str(s["llm_backend"])
            ix = self._llm_combo.findData(backend)
            if ix < 0:
                ix = self._llm_combo.findText(backend)
            if ix >= 0:
                self._llm_combo.setCurrentIndex(ix)
        if "m3_prompt_lang" in s:
            # Phase 53: same fix as llm_backend — M3 prompt lang combo
            # uses friendly names in itemText + ISO codes in userData.
            lang = str(s["m3_prompt_lang"])
            ix = self._m3_lang.findData(lang)
            if ix < 0:
                ix = self._m3_lang.findText(lang)
            if ix >= 0:
                self._m3_lang.setCurrentIndex(ix)
        if "m3_model" in s:
            self._m3_model_edit.setText(str(s["m3_model"]))
        for k, sb in (
            ("MiniMax_thinking_budget", self._m3_budget),
            ("MiniMax_max_output_tokens", self._m3_output),
            ("MiniMax_timeout_sec", self._m3_timeout),
            ("MiniMax_max_retries", self._m3_max_retries),
            ("paleodb_max_occurrences", self._paleodb_occ),
            ("render_dpi", self._dpi),
        ):
            if k in s:
                sb.setValue(int(s[k]))
        for k, cb in (
            ("use_paleodb", self._paleodb_check),
            ("use_geo_vision", self._geo_vision),
            ("use_m3_stage3", self._m3_stage3),
            ("m3_multi_plate_enrich", self._m3_multi_plate),
            ("save_intermediate", self._save_intermediate),
        ):
            if k in s:
                cb.setChecked(bool(s[k]))
        # OD fallback checkbox is inverse-named (it's "allow")
        if "disable_od_fallback" in s:
            self._od_fallback.setChecked(not bool(s["disable_od_fallback"]))

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _refresh_formats(self) -> None:
        """Refresh status / progress / idle / start strings after
        a language switch. Called by the parent MainWindow."""
        self._progress.setFormat(i18n._tr("runtab.progress.idle"))
        # Toggle button enabled state so the trigger text updates
        has = bool(self._path_edit.text().strip())
        if self._worker is None:
            self._start_btn.setEnabled(has)

    def _on_path_changed(self, text: str) -> None:
        has = bool(text and Path(text).exists())
        self._start_btn.setEnabled(has and self._worker is None)

    def _on_browse(self) -> None:
        settings = self._settings
        last_dir = settings.get("last_pdf_dir", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(
            self,
            i18n._tr("runtab.browse.title"),
            last_dir,
            "PDF files (*.pdf);;All files (*)",
        )
        if not path:
            return
        self._set_pdf_path(Path(path))
        settings["last_pdf_dir"] = str(Path(path).parent)

    def _on_clear(self) -> None:
        self._path_edit.clear()
        self._out_edit.clear()

    def _on_pick_outdir(self) -> None:
        settings = self._settings
        last_dir = settings.get("last_export_dir", str(Path.home()))
        path = QFileDialog.getExistingDirectory(
            self,
            i18n._tr("runtab.out.choose.title"),
            last_dir,
        )
        if not path:
            return
        self._out_edit.setText(path)
        settings["last_export_dir"] = path

    def _on_open_outdir(self) -> None:
        path = self._out_edit.text().strip()
        if not path:
            QMessageBox.information(
                self,
                i18n._tr("runtab.out.no_outdir.title"),
                i18n._tr("runtab.out.no_outdir.body"),
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _set_pdf_path(self, path: Path) -> None:
        self._path_edit.setText(str(path))
        # If no out dir yet, default to <path_parent>/<stem>_rlpe_out
        if not self._out_edit.text().strip():
            stem = path.stem
            default_out = path.parent / f"{stem}_rlpe_out"
            self._out_edit.setText(str(default_out))

    def _on_start(self) -> None:
        # Phase 41 audit fix (BLOCKER): guard against double-click on
        # Start. Without this guard, two PipelineWorker threads could
        # be created, the second overwriting self._worker and leaking
        # the first. QThread can only be started() once; re-starting
        # raises RuntimeError which we surface to the user.
        if self._worker is not None and self._worker.isRunning():
            self._log.warning(
                "Start clicked while another job is running; ignored."
            )
            return
        pdf = self._path_edit.text().strip()
        out_dir = self._out_edit.text().strip()
        if not pdf or not Path(pdf).exists():
            QMessageBox.warning(
                self,
                i18n._tr("runtab.prompt.no_pdf.title"),
                i18n._tr("runtab.prompt.no_pdf.body"),
            )
            return
        if not out_dir:
            QMessageBox.warning(
                self,
                i18n._tr("runtab.prompt.no_outdir.title"),
                i18n._tr("runtab.prompt.no_outdir.body"),
            )
            return
        out_path = Path(out_dir)
        try:
            out_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(
                self,
                i18n._tr("runtab.prompt.no_outdir.title"),
                i18n._tr("runtab.prompt.no_outdir.body")
                + f"\n\n{type(exc).__name__}: {exc}",
            )
            return
        work_dir = out_path / "work"
        work_dir.mkdir(parents=True, exist_ok=True)
        self._settings["last_export_dir"] = str(out_path)
        self._settings["last_pdf_dir"] = str(Path(pdf).parent)

        settings = self.collect_settings()
        self._worker = PipelineWorker(settings, Path(pdf), work_dir, parent=self)
        self._current_job_id = self._make_job_id(pdf)
        # Wire signals
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._log_to_statusbar)
        self._worker.status_changed.connect(self._on_status)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_thread_done)
        # Toggle buttons
        self._start_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress.setRange(0, 0)
        self._progress.setFormat(i18n._tr("runtab.progress.starting"))
        self._show_live_progress(i18n._tr("runtab.progress.init"))
        self._status_label.setText(i18n._tr("runtab.status.starting"))
        self._status_label.setProperty("status", "running")
        self._log.info("Starting job %s on %s", self._current_job_id, pdf)
        self.job_started.emit(self._current_job_id, pdf)
        self._worker.start()

    def _on_cancel(self) -> None:
        if self._worker is None:
            return
        self._log.info("Cancellation requested for %s", self._current_job_id)
        # Phase 42: the worker's request_cancel() sets the cooperative
        # cancel event AND calls requestInterruption. The pipeline
        # polls cancel_event between PDFs and short-circuits the
        # run; previously this only stopped progress forwarding.
        self._worker.request_cancel()
        self._cancel_btn.setEnabled(False)
        self._status_label.setText(i18n._tr("runtab.status.cancelling"))

    @staticmethod
    def _make_job_id(pdf_path: str | Path) -> str:
        stem = Path(pdf_path).stem
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)[:40]
        return f"{safe}-{_dt.datetime.now().strftime('%H%M%S')}"

    # ------------------------------------------------------------------
    # Worker signal handlers
    # ------------------------------------------------------------------

    def _on_progress(self, current: int, total: int, message: str) -> None:
        if total > 0:
            if self._progress.maximum() != total:
                self._progress.setRange(0, total)
            self._progress.setValue(current)
            # Phase 54 audit m9: QProgressBar.setFormat treats '%' as
            # a format specifier. Progress messages can contain '%'
            # (e.g. "matched 95% of panels") which would raise
            # ValueError from the underlying QString formatter. Escape
            # any literal '%' before passing the message in.
            safe_message = (message or "").replace("%", "%%")
            self._progress.setFormat(f"{{value}} / {{maxValue}}  ·  {{message}}".replace("{value}", "%v").replace("{maxValue}", "%m").replace("{message}", safe_message))
        else:
            self._progress.setRange(0, 0)
        self._show_live_progress(message or "Working…")
        if self._current_job_id:
            self.job_progress.emit(self._current_job_id, current, total, message)

    def _log_to_statusbar(self, line: str) -> None:
        # Append to the status label (truncate to last 200 chars)
        existing = self._status_label.text()
        new = (existing + "  »  " + line)[-200:]
        self._status_label.setText(new)

    def _show_live_progress(self, message: str) -> None:
        """Write a live status message to ``_progress_msg`` and hide
        the static empty-state hint.

        Phase 55 audit B3 — live status is intentionally NOT in the
        i18n registry. Only the static hint at the bottom of the tab
        is translated. As soon as the pipeline emits its first
        progress/status event we hide the hint and show the live
        label so the user sees what the pipeline is doing right
        now. Once the job is done we keep showing the live label
        (so the user can see "completed in 38s" etc.); the hint
        re-appears when ``_reset_to_idle`` is called below.
        """
        self._hint_label.hide()
        self._progress_msg.setText(message or "")
        self._progress_msg.show()

    def _reset_to_idle(self) -> None:
        """Restore the empty-state hint and hide the live label.

        Called from ``_on_thread_done`` (the worker is gone) and
        from ``_clear_form`` (the user picked a new PDF).
        """
        self._progress_msg.clear()
        self._progress_msg.hide()
        self._hint_label.show()

    def _on_status(self, status: str) -> None:
        self._status_label.setText(status)
        # Status colour via property is QSS-driven — see styles.py.

    def _on_finished(self, results: list) -> None:
        self._log.info("Job %s finished with %d rows", self._current_job_id, len(results))
        if self._current_job_id:
            self.job_finished.emit(self._current_job_id, results)

    def _on_failed(self, error: str) -> None:
        self._log.error("Job %s failed: %s", self._current_job_id, error)
        QMessageBox.critical(
            self,
            i18n._tr("runtab.prompt.error.title"),
            error,
        )
        if self._current_job_id:
            self.job_failed.emit(self._current_job_id, error)

    def _on_thread_done(self) -> None:
        # Phase 43: QThread cleanup. The previous code just set
        # ``self._worker = None`` which left the worker object alive
        # until Python's GC ran. If the user closed the window (or
        # hit Ctrl-C) before Python GC, the QThread's C++ object
        # was still running and the RuntimeError "QThread: Destroyed
        # while thread is still running" was raised at process exit.
        # Fixed: call quit() (asks the thread's event loop to stop)
        # and wait() (block until the thread actually exits) before
        # dropping the reference.
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.quit()
            if not worker.wait(2000):  # 2s timeout
                # Thread didn't exit cleanly; ask it to terminate.
                worker.terminate()
                worker.wait(500)
        self._worker = None
        self._current_job_id = None
        self._start_btn.setEnabled(bool(self._path_edit.text().strip()))
        self._cancel_btn.setEnabled(False)
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._progress.setFormat(i18n._tr("runtab.status.done"))
        # Phase 55 audit B3 — keep the final status visible (don't
        # reset to the empty-state hint) so the user can read the
        # completion message.
        self._progress_msg.setText(i18n._tr("runtab.progress.done"))