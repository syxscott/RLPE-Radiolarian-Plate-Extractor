"""Run tab — pick a PDF, configure pipeline, start a job."""

from __future__ import annotations

import datetime as _dt
import functools
from pathlib import Path
from typing import Any

from PySide6.QtCore import QRegularExpression, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QRegularExpressionValidator
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import i18n
from .constants import (
    BUTTON_MIN_HEIGHT,
    BUTTON_PRIMARY_HEIGHT,
    DEFAULT_GROBID_MAX_RETRIES,
    DEFAULT_GROBID_TIMEOUT,
    DEFAULT_GROBID_URL,
    DEFAULT_LLM_BACKEND,
    DEFAULT_M3_PROMPT_LANG,
    DEFAULT_M3_TIMEOUT,
    DEFAULT_MINIMAX_MODEL,
    DEFAULT_OCR_BACKEND,
    DEFAULT_OCR_LANG,
    DEFAULT_PALEO_MAX_OCC,
    DEFAULT_RENDER_DPI,
    INPUT_WIDTH_LONG,
    INPUT_WIDTH_OCR_LANG,
    INPUT_WIDTH_PATH,
    INPUT_WIDTH_SHORT,
    RANGE_DPI,
    RANGE_GROBID_MAX_RETRIES,
    RANGE_GROBID_TIMEOUT,
    RANGE_M3_BUDGET,
    RANGE_M3_MAX_RETRIES,
    RANGE_M3_OUTPUT_TOKENS,
    RANGE_M3_TIMEOUT,
    RANGE_OD_CAPTION_WINDOW,
    RANGE_PALEO_OCC,
)
from .i18n_widgets import (
    tr_button,
    tr_checkbox,
    tr_doublespinbox,
    tr_form_row,
    tr_groupbox,
    tr_label,
    tr_lineedit,
    tr_spinbox,
)
from .pipeline_worker import PipelineWorker
from .styles import SPACE_L, SPACE_M, SPACE_S
from .utils import (
    get_gui_logger,
)

# URL regex used for GROBID URL validation (same pattern as settings_tab).
_URL_RX = r"^https?://[^\s/$.?#].[^\s]*$"


class RunTab(QWidget):
    """First tab — configure + start a single-paper pipeline run."""

    # Emitted to the parent (MainWindow) to broadcast job state.
    # Phase F-2 (M-10/M-16): job_started now carries output_dir (str)
    # so MainWindow can record the correct path on JobRecord without
    # re-deriving it from pdf_path / stem. The output_dir is
    # <work_dir>/output where work_dir = <user_selected>/work.
    job_started = Signal(str, str, str)  # job_id, pdf_path, output_dir
    job_progress = Signal(str, int, int, str)  # job_id, cur, total, msg
    job_finished = Signal(str, list)  # job_id, results
    job_failed = Signal(str, str)  # job_id, error

    def __init__(self, settings: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._log = get_gui_logger()
        self._worker: PipelineWorker | None = None
        self._current_job_id: str | None = None
        # M-21: saved widget references for run-time input locking
        self._browse_pdf_btn: QWidget | None = None
        self._clear_pdf_btn: QWidget | None = None
        self._output_browse_btn: QWidget | None = None
        self._open_output_btn: QWidget | None = None
        self._build_ui()
        self._connect_signals()
        # Register an i18n listener so progress format strings re-render
        # on language switch. Use a bound method so closeEvent can remove
        # the listener by identity without accumulating stale references.
        self._i18n_listener = self._on_language_changed
        i18n.add_listener(self._i18n_listener)

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
        self._browse_pdf_btn = browse_btn

        clear_btn = tr_button("runtab.clear", object_name="runtab.clear")
        clear_btn.setProperty("class", "flat")  # QSS object-name selector
        clear_btn.setMinimumHeight(BUTTON_MIN_HEIGHT)
        clear_btn.clicked.connect(self._on_clear)
        file_layout.addWidget(clear_btn)
        self._clear_pdf_btn = clear_btn

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
        self._output_browse_btn = out_btn
        open_btn = tr_button("runtab.out.open", object_name="runtab.out.open")
        open_btn.setMinimumHeight(BUTTON_MIN_HEIGHT)
        open_btn.clicked.connect(self._on_open_outdir)
        out_layout.addWidget(open_btn)
        self._open_output_btn = open_btn
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

        # OCR backend — friendly names shown in the UI, raw code stored in userData.
        from PySide6.QtWidgets import QSizePolicy

        from .constants import ocr_backend_friendly_options

        basic_layout.addWidget(tr_label("runtab.label.ocr_backend"), row, 0)
        self._ocr_combo = QComboBox()
        self._ocr_combo.setObjectName("runtab.ocr_backend")
        self._ocr_combo.setMinimumHeight(32)
        self._ocr_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        from .i18n_widgets import populate_friendly_combo

        populate_friendly_combo(
            self._ocr_combo,
            ocr_backend_friendly_options,
            default_code=DEFAULT_OCR_BACKEND,
        )
        basic_layout.addWidget(self._ocr_combo, row, 1)

        basic_layout.addWidget(tr_label("runtab.label.ocr_lang"), row, 2)
        # Friendly language names in the combo; ISO codes stored in userData.
        # setEditable allows custom comma-separated lists (e.g. "en,ja").
        from .constants import ocr_lang_friendly_options

        self._ocr_lang_edit = QComboBox()
        self._ocr_lang_edit.setObjectName("runtab.ocr_lang")
        self._ocr_lang_edit.setEditable(True)
        from PySide6.QtWidgets import QSizePolicy

        self._ocr_lang_edit.setMinimumHeight(32)
        self._ocr_lang_edit.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._ocr_lang_edit.lineEdit().setMinimumHeight(32)
        from .i18n_widgets import populate_friendly_combo

        populate_friendly_combo(
            self._ocr_lang_edit,
            ocr_lang_friendly_options,
            default_code=DEFAULT_OCR_LANG,
        )
        self._ocr_lang_edit.setMinimumWidth(INPUT_WIDTH_OCR_LANG)
        # Translate tooltip via the i18n registry.
        from . import i18n as _i18n_tooltip

        self._ocr_lang_edit.setObjectName("runtab.ocr_lang")
        _i18n_tooltip.register_widget_text(
            "runtab.ocr_lang",
            "toolTip",
            "runtab.ocr_lang.tooltip",
        )
        self._ocr_lang_edit.setToolTip(_i18n_tooltip._tr("runtab.ocr_lang.tooltip"))
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
        self._grobid_edit.setValidator(
            QRegularExpressionValidator(QRegularExpression(_URL_RX), self._grobid_edit)
        )
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
            min_val=1,
            max_val=50,
            value=2,
        )
        self._caption_window.setToolTip("GROBID caption→page lookup window")
        basic_layout.addWidget(self._caption_window, row, 1)
        basic_layout.addWidget(tr_label("runtab.label.od_caption_window"), row, 2)
        self._od_caption_window = tr_spinbox(
            "runtab.label.od_caption_window",
            min_width=INPUT_WIDTH_SHORT,
            min_val=RANGE_OD_CAPTION_WINDOW[0],
            max_val=RANGE_OD_CAPTION_WINDOW[1],
            value=5,
        )
        self._od_caption_window.setToolTip("OpenDataLoader caption↔image cross-page window")
        basic_layout.addWidget(self._od_caption_window, row, 3)
        row += 1

        # Workers
        basic_layout.addWidget(tr_label("runtab.label.workers"), row, 0)
        self._workers = tr_spinbox(
            "runtab.label.workers",
            min_width=INPUT_WIDTH_SHORT,
            min_val=1,
            max_val=32,
            value=1,
        )
        basic_layout.addWidget(self._workers, row, 1)
        basic_layout.addWidget(tr_label("runtab.label.panel_score"), row, 2)
        self._panel_score = tr_doublespinbox(
            min_width=INPUT_WIDTH_SHORT,
            min_val=0.0,
            max_val=1.0,
            value=0.80,
            step=0.05,
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
        # Friendly backend names — stored as userData, displayed as itemText.
        from PySide6.QtWidgets import QSizePolicy

        from .constants import llm_backend_friendly_options

        self._llm_combo.setMinimumHeight(32)
        self._llm_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        from .i18n_widgets import populate_friendly_combo

        populate_friendly_combo(
            self._llm_combo,
            llm_backend_friendly_options,
            default_code=DEFAULT_LLM_BACKEND,
        )
        lbl, w = tr_form_row("runtab.label.llm_backend", self._llm_combo)
        adv_layout.addRow(lbl, w)

        self._m3_lang = QComboBox()
        from PySide6.QtWidgets import QSizePolicy

        from .constants import m3_prompt_lang_friendly_options

        self._m3_lang.setMinimumHeight(32)
        self._m3_lang.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        populate_friendly_combo(
            self._m3_lang,
            m3_prompt_lang_friendly_options,
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
            # Audit 2026-07-26 M5: forward YOLO config from the shared
            # Settings-tab dict. YOLO controls live in SettingsTab, not
            # here, so read from self._settings (populated by
            # SettingsTab.apply_to_run_settings + MainWindow.
            # _load_settings_cache). Without this the worker received
            # use_yolo_figures=False regardless of GUI state.
            "use_yolo_figures": bool(self._settings.get("use_yolo_figures", False)),
            "yolo_model_path": str(self._settings.get("yolo_model_path", "")),
            "yolo_conf_threshold": float(self._settings.get("yolo_conf_threshold", 0.25)),
            "yolo_iou_threshold": float(self._settings.get("yolo_iou_threshold", 0.45)),
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
            # NOTE: _od_fallback checkbox is "use OD fallback" (positive),
            # but the pipeline config key is "disable_od_fallback" (negative).
            # Invert here; matched by apply_settings below.
            "disable_od_fallback": not self._od_fallback.isChecked(),
            "save_intermediate": self._save_intermediate.isChecked(),
            "render_dpi": self._dpi.value(),
        }

    def apply_settings(self, settings: dict[str, Any]) -> None:
        """Restore settings from the QSettings tab on startup."""
        s = dict(settings)
        if "ocr_backend" in s:
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

    def _on_language_changed(self, _lang: str) -> None:
        """Rebuild format strings on language switch (i18n listener)."""
        # Phase 56 audit: defer refresh to next event loop iteration so
        # it doesn't race with worker signal handlers on the main thread.
        from PySide6.QtCore import QTimer as _QTimer

        _QTimer.singleShot(0, self._refresh_formats)

    def _remove_i18n_listener(self) -> None:
        """Remove our i18n listener when the widget is destroyed."""
        listener = getattr(self, "_i18n_listener", None)
        if listener is not None:
            try:
                i18n.remove_listener(listener)
            except Exception:
                pass

    def closeEvent(self, event) -> None:  # noqa: N802
        """Phase 56 audit: stop worker and remove i18n listener on tab close."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_cancel()
            self._worker.quit()
            # Do NOT wait() synchronously — it blocks the GUI thread for up
            # to 2 s.  _on_thread_done (connected to finished) handles the
            # synchronous wait when the thread actually exits.
        self._remove_i18n_listener()
        super().closeEvent(event)

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

    # M-19: PDF path validation helper
    @staticmethod
    def _validate_pdf_path(p: str) -> str | None:
        """Return None if valid, an error message string if invalid."""
        if not p:
            return i18n._tr("runtab.error.invalid_pdf")
        path = Path(p)
        if not path.is_file():
            return i18n._tr("runtab.error.invalid_pdf")
        if path.suffix.lower() != ".pdf":
            return i18n._tr("runtab.error.invalid_pdf")
        return None

    # M-21: lock/unlock input widgets during pipeline execution
    def _set_inputs_locked(self, locked: bool) -> None:
        """Disable (or re-enable) all input widgets while pipeline runs."""
        state = not locked
        for w in (
            self._browse_pdf_btn,
            self._clear_pdf_btn,
            self._output_browse_btn,
            self._open_output_btn,
            self._path_edit,
            self._out_edit,
            self._grobid_edit,
        ):
            if w is not None:
                w.setEnabled(state)
        # Also lock spinboxes/combos that are not already disabled
        for w in (self._caption_window, self._od_caption_window):
            try:
                w.setEnabled(state)
            except AttributeError:
                pass  # not all callers have these spinboxes
        if locked:
            self.setStyleSheet("RunTab QWidget:disabled { background-color: #f0f0f0; }")
        else:
            self.setStyleSheet("")

    def _on_start(self) -> None:
        # Start. Without this guard, two PipelineWorker threads could
        # be created, the second overwriting self._worker and leaking
        # the first. QThread can only be started() once; re-starting
        # raises RuntimeError which we surface to the user.
        if self._worker is not None and self._worker.isRunning():
            self._log.warning("Start clicked while another job is running; ignored.")
            return
        pdf = self._path_edit.text().strip()
        out_dir = self._out_edit.text().strip()
        # M-19: validate PDF path (must be a real .pdf file)
        pdf_error = self._validate_pdf_path(pdf)
        if pdf_error:
            QMessageBox.critical(self, i18n._tr("runtab.prompt.no_pdf.title"), pdf_error)
            self._log.warning("Invalid PDF path: %s", pdf)
            return
        if not out_dir:
            QMessageBox.warning(
                self,
                i18n._tr("runtab.prompt.no_outdir.title"),
                i18n._tr("runtab.prompt.no_outdir.body"),
            )
            return
        # M-19: validate GROBID URL format
        if not self._grobid_edit.hasAcceptableInput():
            QMessageBox.critical(
                self,
                i18n._tr("runtab.prompt.no_pdf.title"),
                i18n._tr("runtab.error.invalid_grobid_url"),
            )
            self._log.warning("Invalid GROBID URL: %s", self._grobid_edit.text())
            return
        out_path = Path(out_dir)
        work_dir = out_path / "work"
        # M-17: both mkdir calls in the same try/except so OSError from
        # either the output dir or the work sub-dir is caught uniformly.
        try:
            out_path.mkdir(parents=True, exist_ok=True)
            work_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._log.error("Cannot create output dir %s: %s", work_dir, exc, exc_info=True)
            QMessageBox.critical(
                self,
                i18n._tr("runtab.prompt.no_pdf.title"),
                f"{type(exc).__name__}: {exc}",
            )
            self._start_btn.setEnabled(bool(self._path_edit.text().strip()))
            self._status_label.setText(i18n._tr("runtab.status.idle"))
            return
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
        # audit 2026-07-31: bind the worker instance into the slot.
        # The plain ``self._on_thread_done`` was connected WITHOUT the
        # worker reference; in batch mode MainWindow starts the next
        # worker synchronously from ``_on_job_finished``, so by the
        # time THIS worker's ``finished`` fires, ``self._worker``
        # already points at the next job — and the old handler
        # quit()+terminate()d the *new* worker (batch job 2 died
        # ~2s in). The partial carries the owning worker so the
        # handler can only ever touch its own thread.
        self._worker.finished.connect(functools.partial(self._on_thread_done, self._worker))
        # Toggle buttons + lock inputs
        self._start_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._set_inputs_locked(True)
        self._progress.setRange(0, 0)
        self._progress.setFormat(i18n._tr("runtab.progress.starting"))
        self._show_live_progress(i18n._tr("runtab.progress.init"))
        self._status_label.setText(i18n._tr("runtab.status.starting"))
        self._status_label.setProperty("status", "running")
        self._log.info("Starting job %s on %s", self._current_job_id, pdf)
        # Phase F-2 (M-10/M-16): emit the actual output_dir so MainWindow
        # can record it on JobRecord. output_dir = <out_path>/output;
        # worker.work_dir = <out_path>/work.
        output_dir = str(out_path / "output")
        self.job_started.emit(self._current_job_id, pdf, output_dir)
        self._worker.start()

    def _on_cancel(self) -> None:
        if self._worker is None:
            return
        self._log.info("Cancellation requested for %s", self._current_job_id)
        # cancel event AND calls requestInterruption. The pipeline
        # polls cancel_event between PDFs and short-circuits the
        # run; previously this only stopped progress forwarding.
        self._worker.request_cancel()
        self._cancel_btn.setEnabled(False)
        self._status_label.setText(i18n._tr("runtab.status.cancelling"))
        self._status_label.setProperty("status", "cancelling")
        self._status_label.style().unpolish(self._status_label)
        self._status_label.style().polish(self._status_label)

    def shutdown(self) -> None:
        """M-27: Clean shutdown of the pipeline worker.

        Called from MainWindow.closeEvent. Requests worker interruption,
        waits up to 30s for it to exit, and clears state. Unlike the
        internal _on_thread_done path this does NOT restore buttons or
        unlock inputs — that is the caller's responsibility after this
        returns.
        """
        if self._worker is None:
            return
        if self._worker.isRunning():
            self._worker.requestInterruption()
            try:
                if not self._worker.wait(30000):
                    self._log.critical(
                        "PipelineWorker did not exit within 30s after "
                        "requestInterruption in shutdown(); thread will be "
                        "reclaimed on process exit."
                    )
            except RuntimeError:
                # wait() raises RuntimeError if the thread has already
                # finished. Safe to ignore.
                pass


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
            # a format specifier. Progress messages can contain '%'
            # (e.g. "matched 95% of panels") which would raise
            # ValueError from the underlying QString formatter. Escape
            # any literal '%' before passing the message in.
            # audit 2026-07-31: QProgressBar.setFormat does NOT
            # interpret % in the message text (only the %v/%m tokens);
            # escaping to %% made the user see "matched 95%% of
            # panels". Pass the message through unchanged.
            safe_message = message or ""
            self._progress.setFormat(
                "{value} / {maxValue}  ·  {message}".replace("{value}", "%v")
                .replace("{maxValue}", "%m")
                .replace("{message}", safe_message)
            )
        else:
            self._progress.setRange(0, 0)
        self._show_live_progress(message or i18n._tr("runtab.progress.working"))
        if self._current_job_id:
            self.job_progress.emit(self._current_job_id, current, total, message)

    def _log_to_statusbar(self, line: str) -> None:
        # Append to the status label (truncate to last 200 chars)
        existing = self._status_label.text()
        new = (existing + "  »  " + line)[-200:]
        self._status_label.setText(new)

    def _show_live_progress(self, message: str) -> None:
        """Write a live status message to the progress label.

        The static hint label is hidden while a job is running and
        re-shown by ``_reset_to_idle`` when the job completes.
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
        # M-22: mark terminal outcome so _on_thread_done knows this succeeded
        self._pending_outcome = "success"
        self._log.info("Job %s finished with %d rows", self._current_job_id, len(results))
        # Phase 56 audit: clear live progress message so it doesn't linger
        self._reset_to_idle()
        if self._current_job_id:
            self.job_finished.emit(self._current_job_id, results)

    def _on_failed(self, error: str) -> None:
        # M-22: mark terminal outcome so _on_thread_done knows this failed/cancelled
        cancelled = "cancelled" in (error or "").lower() or "取消" in (error or "")
        self._pending_outcome = "cancelled" if cancelled else "failed"
        self._log.error("Job %s failed: %s", self._current_job_id, error)
        # audit 2026-07-31: a user cancellation is not a pipeline
        # failure — no red error dialog, status shows "cancelled".
        if cancelled:
            self._status_label.setProperty("status", "cancelled")
            self._status_label.style().unpolish(self._status_label)
            self._status_label.style().polish(self._status_label)
            self._status_label.setText(i18n._tr("runtab.status.cancelled"))
        else:
            # Phase 56 audit: set status QSS property to "failed" before
            # dialog so the label colour reflects the failed state.
            self._status_label.setProperty("status", "failed")
            self._status_label.style().unpolish(self._status_label)
            self._status_label.style().polish(self._status_label)

    def _on_thread_done(self, worker: PipelineWorker | None = None) -> None:
        # audit 2026-07-31: the handler now receives the worker that
        # actually finished (bound via functools.partial at connect
        # time). It must NEVER touch ``self._worker`` — in batch mode
        # MainWindow starts the next job synchronously from
        # ``_on_job_finished``, so by the time this slot runs
        # ``self._worker`` points at the NEXT job. Acting on
        # ``self._worker`` quit()+terminate()d the next job (batch
        # job 2 died ~2s in) and disconnected its signals.
        if worker is None:
            return
        # M-22: read the terminal outcome set by _on_finished / _on_failed.
        # If neither ran (shouldn't happen), treat as cancelled.
        outcome = getattr(self, "_pending_outcome", "cancelled")
        # Phase 56 audit: disconnect signals before cleanup so late
        # signals from a dying thread don't reach stale slots.
        try:
            worker.progress.disconnect(self._on_progress)
            worker.log_line.disconnect(self._log_to_statusbar)
            worker.status_changed.disconnect(self._on_status)
            worker.finished_ok.disconnect(self._on_finished)
            worker.failed.disconnect(self._on_failed)
            worker.finished.disconnect()
        except (TypeError, RuntimeError):
            pass  # already disconnected or worker partially deleted
        if worker.isRunning():
            # audit 2026-08-01 D20: replace ``worker.terminate()`` with
            # ``requestInterruption()`` + a 30s bounded wait. ``terminate()``
            # forcibly kills the QThread mid-Python execution, which can
            # orphan subprocesses (OpenDataLoader JVM, in-flight LLM HTTP
            # requests) and leave partial temp dirs like
            # ``od_output/<paper_id>/`` behind. ``requestInterruption()``
            # sets the interrupt flag the pipeline polls between stages;
            # the 30s wait gives it time to flush and clean up. If the
            # worker is still running after 30s, log a warning rather than
            # forcibly killing it — the OS will reclaim the thread on
            # process exit, and forcibly killing a worker that owns a
            # live JVM is worse than letting the parent process exit normally.
            worker.requestInterruption()
            try:
                if not worker.wait(30000):  # 30s timeout
                    self._log.warning(
                        "PipelineWorker did not exit within 30s after "
                        "requestInterruption; leaving thread alive (process "
                        "exit will reclaim it). OpenDataLoader JVM or LLM "
                        "HTTP request may still be in flight."
                    )
            except RuntimeError:
                # M-27: wait() raises RuntimeError if the thread has already
                # finished or is being finished by another means. Ignore it.
                pass
        # Only reset the shared state when this worker is still the
        # active one (single-job mode). In batch mode the next worker
        # is already running; clearing its bookkeeping would corrupt
        # the batch state machine.
        if self._worker is worker:
            self._worker = None
            self._current_job_id = None
            self._start_btn.setEnabled(bool(self._path_edit.text().strip()))
            self._cancel_btn.setEnabled(False)
            self._set_inputs_locked(False)
            # M-22: set UI state based on terminal outcome
            if outcome == "success":
                self._progress.setRange(0, 1)
                self._progress.setValue(1)
                self._status_label.setProperty("status", "idle")
                self._status_label.style().unpolish(self._status_label)
                self._status_label.style().polish(self._status_label)
                self._status_label.setText(i18n._tr("runtab.status.done"))
                self._progress_msg.setText(i18n._tr("runtab.progress.done"))
            elif outcome == "cancelled":
                self._progress.setRange(0, 1)
                self._progress.setValue(0)
                self._status_label.setProperty("status", "cancelled")
                self._status_label.style().unpolish(self._status_label)
                self._status_label.style().polish(self._status_label)
                self._status_label.setText(i18n._tr("runtab.status.cancelled"))
                self._progress_msg.clear()
            else:  # "failed"
                self._status_label.setProperty("status", "failed")
                self._status_label.style().unpolish(self._status_label)
                self._status_label.style().polish(self._status_label)
                self._progress_msg.clear()
