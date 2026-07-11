"""Run tab — pick a PDF, configure pipeline, start a job."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QCheckBox,
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

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACE_L, SPACE_L, SPACE_L, SPACE_L)
        outer.setSpacing(SPACE_M)

        # ---- File picker row ----
        file_group = QGroupBox("📄 Input PDF")
        file_layout = QHBoxLayout(file_group)
        file_layout.setSpacing(SPACE_S)

        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Pick a radiolarian paper PDF to extract plates from…")
        self._path_edit.setReadOnly(True)
        file_layout.addWidget(self._path_edit, 1)

        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse)
        file_layout.addWidget(browse_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("flat")
        clear_btn.clicked.connect(self._on_clear)
        file_layout.addWidget(clear_btn)

        outer.addWidget(file_group)

        # ---- Output dir row ----
        out_group = QGroupBox("💾 Output directory")
        out_layout = QHBoxLayout(out_group)
        out_layout.setSpacing(SPACE_S)
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("Where to write manifests / figures / xlsx…")
        self._out_edit.setReadOnly(True)
        out_layout.addWidget(self._out_edit, 1)
        out_btn = QPushButton("Choose…")
        out_btn.clicked.connect(self._on_pick_outdir)
        out_layout.addWidget(out_btn)
        open_btn = QToolButton()
        open_btn.setText("Open")
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
        basic = QGroupBox("⚙️ Basic configuration")
        basic_layout = QGridLayout(basic)
        basic_layout.setHorizontalSpacing(SPACE_L)
        basic_layout.setVerticalSpacing(SPACE_S)
        row = 0

        # OCR backend
        basic_layout.addWidget(QLabel("OCR backend:"), row, 0)
        from PySide6.QtWidgets import QComboBox
        self._ocr_combo = QComboBox()
        self._ocr_combo.addItems(["paddleocr", "easyocr"])
        self._ocr_combo.setCurrentText(DEFAULT_OCR_BACKEND)
        basic_layout.addWidget(self._ocr_combo, row, 1)

        basic_layout.addWidget(QLabel("OCR language(s):"), row, 2)
        self._ocr_lang_edit = QLineEdit(DEFAULT_OCR_LANG)
        self._ocr_lang_edit.setPlaceholderText("e.g. en, ja, ch_sim")
        self._ocr_lang_edit.setToolTip("Comma-separated language list (EasyOCR accepts multi-lang)")
        basic_layout.addWidget(self._ocr_lang_edit, row, 3)
        row += 1

        # GROBID
        basic_layout.addWidget(QLabel("GROBID URL:"), row, 0)
        self._grobid_edit = QLineEdit(DEFAULT_GROBID_URL)
        basic_layout.addWidget(self._grobid_edit, row, 1, 1, 3)
        row += 1

        basic_layout.addWidget(QLabel("GROBID retries:"), row, 0)
        self._grobid_retries = QSpinBox()
        self._grobid_retries.setRange(*RANGE_GROBID_MAX_RETRIES)
        self._grobid_retries.setValue(DEFAULT_GROBID_MAX_RETRIES)
        basic_layout.addWidget(self._grobid_retries, row, 1)
        basic_layout.addWidget(QLabel("GROBID timeout (s):"), row, 2)
        self._grobid_timeout = QSpinBox()
        self._grobid_timeout.setRange(*RANGE_GROBID_TIMEOUT)
        self._grobid_timeout.setValue(DEFAULT_GROBID_TIMEOUT)
        basic_layout.addWidget(self._grobid_timeout, row, 3)
        row += 1

        # Caption windows
        basic_layout.addWidget(QLabel("Caption window (GROBID):"), row, 0)
        self._caption_window = QSpinBox()
        self._caption_window.setRange(1, 50)
        self._caption_window.setValue(2)
        self._caption_window.setToolTip("GROBID caption→page lookup window (Phase 28)")
        basic_layout.addWidget(self._caption_window, row, 1)
        basic_layout.addWidget(QLabel("OD caption window:"), row, 2)
        self._od_caption_window = QSpinBox()
        self._od_caption_window.setRange(*RANGE_OD_CAPTION_WINDOW)
        self._od_caption_window.setValue(5)
        self._od_caption_window.setToolTip("OpenDataLoader caption↔image cross-page window (Phase 28)")
        basic_layout.addWidget(self._od_caption_window, row, 3)
        row += 1

        # Workers
        basic_layout.addWidget(QLabel("Workers:"), row, 0)
        self._workers = QSpinBox()
        self._workers.setRange(1, 32)
        self._workers.setValue(1)
        basic_layout.addWidget(self._workers, row, 1)
        basic_layout.addWidget(QLabel("Panel score:"), row, 2)
        self._panel_score = QDoubleSpinBox()
        self._panel_score.setRange(0.0, 1.0)
        self._panel_score.setSingleStep(0.05)
        self._panel_score.setValue(0.80)
        basic_layout.addWidget(self._panel_score, row, 3)
        row += 1

        # Use GPU
        basic_layout.addWidget(QLabel("Use GPU:"), row, 0)
        self._gpu_check = QCheckBox("Auto-detect CUDA at startup")
        self._gpu_check.setChecked(True)
        basic_layout.addWidget(self._gpu_check, row, 1, 1, 3)

        cfg_layout.addWidget(basic)

        # Advanced config
        adv = QGroupBox("🔬 Advanced (LLM / M3 / PBDB)")
        adv_layout = QFormLayout(adv)
        adv_layout.setHorizontalSpacing(SPACE_L)
        adv_layout.setVerticalSpacing(SPACE_S)

        self._llm_combo = QComboBox()
        self._llm_combo.addItems(["minimax", "minimax-m3", "minimax_api", "transformers", "ollama", "llamacpp"])
        self._llm_combo.setCurrentText(DEFAULT_LLM_BACKEND)
        adv_layout.addRow("LLM backend:", self._llm_combo)

        self._m3_lang = QComboBox()
        self._m3_lang.addItems(["auto", "zh", "en", "ja"])
        self._m3_lang.setCurrentText(DEFAULT_M3_PROMPT_LANG)
        adv_layout.addRow("M3 prompt lang:", self._m3_lang)

        self._m3_model_edit = QLineEdit(DEFAULT_MINIMAX_MODEL)
        adv_layout.addRow("M3 model:", self._m3_model_edit)

        self._m3_budget = QSpinBox()
        self._m3_budget.setRange(*RANGE_M3_BUDGET)
        self._m3_budget.setValue(1024)
        adv_layout.addRow("M3 thinking budget:", self._m3_budget)

        self._m3_output = QSpinBox()
        self._m3_output.setRange(*RANGE_M3_OUTPUT_TOKENS)
        self._m3_output.setValue(2048)
        adv_layout.addRow("M3 max output tokens:", self._m3_output)

        self._m3_timeout = QSpinBox()
        self._m3_timeout.setRange(*RANGE_M3_TIMEOUT)
        self._m3_timeout.setValue(DEFAULT_M3_TIMEOUT)
        adv_layout.addRow("M3 timeout (s):", self._m3_timeout)

        self._m3_max_retries = QSpinBox()
        self._m3_max_retries.setRange(*RANGE_M3_MAX_RETRIES)
        self._m3_max_retries.setValue(3)
        adv_layout.addRow("M3 max retries:", self._m3_max_retries)

        self._paleodb_check = QCheckBox("Use Paleobiology Database for taxonomy + occurrence enrichment")
        self._paleodb_check.setChecked(True)
        adv_layout.addRow("", self._paleodb_check)

        self._paleodb_occ = QSpinBox()
        self._paleodb_occ.setRange(*RANGE_PALEO_OCC)
        self._paleodb_occ.setValue(DEFAULT_PALEO_MAX_OCC)
        adv_layout.addRow("PBDB max occurrences:", self._paleodb_occ)

        self._geo_vision = QCheckBox("Multi-modal geology vision (Round 6)")
        self._geo_vision.setChecked(True)
        adv_layout.addRow("", self._geo_vision)

        self._m3_stage3 = QCheckBox("M3 stage 3 (panel refinement)")
        self._m3_stage3.setChecked(True)
        adv_layout.addRow("", self._m3_stage3)

        self._m3_multi_plate = QCheckBox("M3 multi-plate enrichment (Round 7)")
        self._m3_multi_plate.setChecked(True)
        adv_layout.addRow("", self._m3_multi_plate)

        self._od_fallback = QCheckBox("Allow OpenDataLoader fallback when GROBID fails (Phase 29)")
        self._od_fallback.setChecked(True)
        adv_layout.addRow("", self._od_fallback)

        self._save_intermediate = QCheckBox("Save intermediate panels (large disk usage)")
        self._save_intermediate.setChecked(False)
        adv_layout.addRow("", self._save_intermediate)

        self._dpi = QSpinBox()
        self._dpi.setRange(*RANGE_DPI)
        self._dpi.setValue(DEFAULT_RENDER_DPI)
        adv_layout.addRow("Render DPI:", self._dpi)

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

        self._start_btn = QPushButton("▶  Start extraction")
        self._start_btn.setObjectName("primary")
        self._start_btn.setMinimumHeight(34)
        self._start_btn.clicked.connect(self._on_start)
        action_layout.addWidget(self._start_btn)

        self._cancel_btn = QPushButton("⏹  Cancel")
        self._cancel_btn.setObjectName("danger")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        action_layout.addWidget(self._cancel_btn)

        action_layout.addStretch(1)

        self._status_label = QLabel("Idle")
        self._status_label.setProperty("class", "status")
        action_layout.addWidget(self._status_label)

        outer.addWidget(action_bar)

        # ---- Progress bar ----
        self._progress = QProgressBar()
        self._progress.setValue(0)
        self._progress.setFormat("Idle  (%v / %m)")
        self._progress.setMinimumHeight(20)
        outer.addWidget(self._progress)

        self._progress_msg = QLabel("Pick a PDF above to begin.")
        self._progress_msg.setWordWrap(True)
        outer.addWidget(self._progress_msg)

    def _connect_signals(self) -> None:
        """Wire file picker path → enable start button."""
        self._path_edit.textChanged.connect(self._on_path_changed)

    # ------------------------------------------------------------------
    # Public API used by the parent MainWindow
    # ------------------------------------------------------------------

    def collect_settings(self) -> dict[str, Any]:
        """Return the current settings as a flat dict (worker-ready)."""
        return {
            "ocr_backend": self._ocr_combo.currentText(),
            "ocr_lang": self._ocr_lang_edit.text().strip() or "en",
            "grobid_url": self._grobid_edit.text().strip(),
            "grobid_max_retries": self._grobid_retries.value(),
            "grobid_timeout": self._grobid_timeout.value(),
            "caption_window": self._caption_window.value(),
            "od_caption_window": self._od_caption_window.value(),
            "num_workers": self._workers.value(),
            "min_panel_score": self._panel_score.value(),
            "use_gpu": self._gpu_check.isChecked(),
            "llm_backend": self._llm_combo.currentText(),
            "m3_prompt_lang": self._m3_lang.currentText(),
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
            ix = self._ocr_combo.findText(str(s["ocr_backend"]))
            if ix >= 0:
                self._ocr_combo.setCurrentIndex(ix)
        if "ocr_lang" in s:
            self._ocr_lang_edit.setText(str(s["ocr_lang"]))
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
            ix = self._llm_combo.findText(str(s["llm_backend"]))
            if ix >= 0:
                self._llm_combo.setCurrentIndex(ix)
        if "m3_prompt_lang" in s:
            ix = self._m3_lang.findText(str(s["m3_prompt_lang"]))
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

    def _on_path_changed(self, text: str) -> None:
        has = bool(text and Path(text).exists())
        self._start_btn.setEnabled(has and self._worker is None)

    def _on_browse(self) -> None:
        settings = self._settings
        last_dir = settings.get("last_pdf_dir", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose a radiolarian paper PDF",
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
            "Choose an output directory",
            last_dir,
        )
        if not path:
            return
        self._out_edit.setText(path)
        settings["last_export_dir"] = path

    def _on_open_outdir(self) -> None:
        path = self._out_edit.text().strip()
        if not path:
            QMessageBox.information(self, "Output", "No output directory set yet.")
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
        pdf = self._path_edit.text().strip()
        out_dir = self._out_edit.text().strip()
        if not pdf or not Path(pdf).exists():
            QMessageBox.warning(self, "Missing PDF", "Please choose a valid PDF file first.")
            return
        if not out_dir:
            QMessageBox.warning(self, "Missing output dir", "Please choose an output directory.")
            return
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
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
        self._progress.setFormat("Starting…")
        self._progress_msg.setText("Pipeline initialising…")
        self._status_label.setText("Starting")
        self._status_label.setProperty("status", "running")
        self._log.info("Starting job %s on %s", self._current_job_id, pdf)
        self.job_started.emit(self._current_job_id, pdf)
        self._worker.start()

    def _on_cancel(self) -> None:
        if self._worker is None:
            return
        self._log.info("Cancellation requested for %s", self._current_job_id)
        self._worker.requestInterruption()
        self._cancel_btn.setEnabled(False)
        self._status_label.setText("Cancelling…")

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
            self._progress.setFormat(f"{{value}} / {{maxValue}}  ·  {{message}}".replace("{value}", "%v").replace("{maxValue}", "%m").replace("{message}", message))
        else:
            self._progress.setRange(0, 0)
        self._progress_msg.setText(message or "Working…")
        if self._current_job_id:
            self.job_progress.emit(self._current_job_id, current, total, message)

    def _log_to_statusbar(self, line: str) -> None:
        # Append to the status label (truncate to last 200 chars)
        existing = self._status_label.text()
        new = (existing + "  »  " + line)[-200:]
        self._status_label.setText(new)

    def _on_status(self, status: str) -> None:
        self._status_label.setText(status)
        # Status colour via property is QSS-driven — see styles.py.

    def _on_finished(self, results: list) -> None:
        self._log.info("Job %s finished with %d rows", self._current_job_id, len(results))
        if self._current_job_id:
            self.job_finished.emit(self._current_job_id, results)

    def _on_failed(self, error: str) -> None:
        self._log.error("Job %s failed: %s", self._current_job_id, error)
        QMessageBox.critical(self, "Pipeline error", error)
        if self._current_job_id:
            self.job_failed.emit(self._current_job_id, error)

    def _on_thread_done(self) -> None:
        self._worker = None
        self._current_job_id = None
        self._start_btn.setEnabled(bool(self._path_edit.text().strip()))
        self._cancel_btn.setEnabled(False)
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._progress.setFormat("Done")
        self._progress_msg.setText("Pipeline finished. See Results tab.")