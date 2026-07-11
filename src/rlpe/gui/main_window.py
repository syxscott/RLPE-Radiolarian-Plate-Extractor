"""Main window — QMainWindow hosting the four tabs + menu / toolbar / statusbar.

Tabs:
  0: Run       — pick PDF + start extraction
  1: Jobs      — job queue + history
  2: Results   — row-by-row browser with image preview
  3: Settings  — app-wide configuration (QSettings-backed)

The window also has a menu bar, a toolbar with quick actions, and
a status bar with permanent + temporary widgets. We use a
``QStackedLayout`` *within* the QTabWidget so cross-tab references
work — for example, opening a job from the Jobs tab switches to
the Results tab and loads the rows.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QIcon,
    QKeySequence,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QWidget,
)

from .batch_dialog import BatchDialog
from . import i18n
from .i18n_widgets import tr_label
from .styles import apply_theme
from .utils import (
    fmt_count,
    fmt_duration,
    get_gui_logger,
    is_linux,
    is_macos,
    is_windows,
    short_path,
    single_instance_lock,
)


def _make_app_icon() -> QIcon:
    """Build a simple programmatic app icon (no PNG asset needed).

    We draw a microscope-on-circle SVG to avoid shipping binary
    assets. The icon is rendered into a QPixmap at the requested
    size so it works on all platforms including the system tray.
    """
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QPainter, QPixmap
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    # Background circle
    p.setBrush(QColor("#1f77b4"))
    p.setPen(QColor("#14507a"))
    p.drawEllipse(QRectF(2, 2, 60, 60))
    # Inner accent
    p.setPen(QColor("#ffffff"))
    f = p.font()
    f.setPointSize(28)
    f.setBold(True)
    p.setFont(f)
    p.drawText(QRectF(0, 0, 64, 64), Qt.AlignCenter, "R")
    p.end()
    return QIcon(pm)


from .constants import (
    APP_AUTHOR,
    APP_DOMAIN,
    APP_NAME,
    APP_VERSION,
    MAIN_WINDOW_DEFAULT_SIZE,
    MAIN_WINDOW_MIN_SIZE,
    QS_KEY_GEOMETRY,
    QS_KEY_LAST_DIR,
    QS_KEY_LAST_EXPORT_DIR,
    QS_KEY_STATE,
    QS_KEY_THEME,
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    TAB_JOBS,
    TAB_RESULTS,
    TAB_RUN,
    TAB_SETTINGS,
    THEME_LIGHT,
)
from .styles import SPACE_M, SPACE_S, apply_theme
from .jobs_tab import JobRecord, JobsTab
from .results_tab import ResultsTab
from .run_tab import RunTab
from .settings_tab import SettingsTab
from .styles import apply_theme
from .utils import (
    fmt_count,
    fmt_duration,
    get_gui_logger,
    is_linux,
    is_macos,
    is_windows,
    short_path,
    single_instance_lock,
)


# ============================================================
# Main window
# ============================================================
class MainWindow(QMainWindow):
    """Top-level window for the RLPE GUI."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._log = get_gui_logger()
        self._qsettings = QSettings(APP_DOMAIN, APP_NAME)
        self._settings = self._load_settings_cache()
        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        self._build_statusbar()
        self._wire_signals()
        self._restore_window_state()

    # ------------------------------------------------------------------
    # Settings cache
    # ------------------------------------------------------------------
    def _load_settings_cache(self) -> dict[str, Any]:
        """In-memory mirror of the QSettings keys the Run tab reads.

        The Run tab reads ``self._settings[key]`` for last-used
        directories + defaults. We populate from QSettings so the
        Run tab doesn't have to read QSettings directly.
        """
        return {
            "last_pdf_dir": str(self._qsettings.value(QS_KEY_LAST_DIR, str(__file__))),
            "last_export_dir": str(self._qsettings.value(QS_KEY_LAST_EXPORT_DIR, str(Path.home()))),
            "grobid_url": self._qsettings.value("grobid_url", "http://localhost:8070"),
            "grobid_max_retries": int(self._qsettings.value("grobid_max_retries", 3)),
            "grobid_timeout": int(self._qsettings.value("grobid_timeout", 300)),
            "ocr_backend": self._qsettings.value("ocr_backend", "paddleocr"),
            "ocr_lang": self._qsettings.value("ocr_lang", "en"),
            "caption_window": int(self._qsettings.value("caption_window", 2)),
            "od_caption_window": int(self._qsettings.value("od_caption_window", 5)),
            "llm_backend": self._qsettings.value("llm_backend", "minimax"),
            "m3_prompt_lang": self._qsettings.value("m3_prompt_lang", "auto"),
            "m3_model": self._qsettings.value("m3_model", "MiniMax-M3"),
            "MiniMax_thinking_budget": int(self._qsettings.value("MiniMax_thinking_budget", 1024)),
            "MiniMax_max_output_tokens": int(self._qsettings.value("MiniMax_max_output_tokens", 2048)),
            "MiniMax_timeout_sec": int(self._qsettings.value("MiniMax_timeout_sec", 60)),
            "MiniMax_max_retries": int(self._qsettings.value("MiniMax_max_retries", 3)),
            "use_paleodb": bool(self._qsettings.value("use_paleodb", True)),
            "paleodb_max_occurrences": int(self._qsettings.value("paleodb_max_occurrences", 25)),
            "paleodb_endpoint": self._qsettings.value("paleodb_endpoint", "https://paleobiodb.org/data1.2"),
            "render_dpi": int(self._qsettings.value("render_dpi", 200)),
            "save_intermediate": bool(self._qsettings.value("save_intermediate", False)),
        }

    # ------------------------------------------------------------------
    # UI assembly
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.setWindowTitle(f"{APP_NAME}  v{APP_VERSION}")
        self.setWindowIcon(_make_app_icon())
        self.resize(*MAIN_WINDOW_DEFAULT_SIZE)
        self.setMinimumSize(*MAIN_WINDOW_MIN_SIZE)

        # Central tab widget
        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.North)
        self._tabs.setDocumentMode(True)

        # Tab 0: Run
        self._run_tab = RunTab(self._settings)
        self._tabs.addTab(self._run_tab, "▶  Run")

        # Tab 1: Jobs
        self._jobs_tab = JobsTab()
        self._tabs.addTab(self._jobs_tab, "📋  Jobs")

        # Tab 2: Results
        self._results_tab = ResultsTab()
        self._tabs.addTab(self._results_tab, "📊  Results")

        # Tab 3: Settings
        self._settings_tab = SettingsTab(self._settings)
        self._tabs.addTab(self._settings_tab, "⚙️  Settings")

        # Pre-populate the Run tab's quick config from persisted defaults
        self._run_tab.apply_settings(self._settings)

        self.setCentralWidget(self._tabs)

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_act = QAction("📂  &Open PDF…", self)
        open_act.setShortcut(QKeySequence.Open)
        open_act.triggered.connect(self._on_open_pdf)
        file_menu.addAction(open_act)

        batch_act = QAction("📚  &Batch…", self)
        batch_act.setShortcut(QKeySequence("Ctrl+B"))
        batch_act.triggered.connect(self._on_open_batch)
        file_menu.addAction(batch_act)

        file_menu.addSeparator()

        out_act = QAction("📁  Open output &directory…", self)
        out_act.triggered.connect(self._on_open_outdir)
        file_menu.addAction(out_act)

        file_menu.addSeparator()

        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # View menu
        view_menu = menubar.addMenu("&View")

        for label, tab_idx in (
            ("&Run tab",      TAB_RUN),
            ("&Jobs tab",     TAB_JOBS),
            ("&Results tab",  TAB_RESULTS),
            ("&Settings tab", TAB_SETTINGS),
        ):
            act = QAction(label, self)
            act.setShortcut(QKeySequence(f"Ctrl+{tab_idx + 1}"))
            act.triggered.connect(lambda _=False, i=tab_idx: self._tabs.setCurrentIndex(i))
            view_menu.addAction(act)

        view_menu.addSeparator()

        # Theme switcher
        theme_menu = view_menu.addMenu("🎨  &Theme")
        for theme in (THEME_LIGHT, "dark", "system"):
            act = QAction(theme.title() if theme != "system" else "System default", self)
            act.triggered.connect(lambda _=False, t=theme: self._apply_theme(t))
            theme_menu.addAction(act)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")

        # Open log file
        log_act = QAction("📂  Open &log file", self)
        log_act.triggered.connect(self._open_log_file)
        tools_menu.addAction(log_act)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_act = QAction("&About RLPE", self)
        about_act.triggered.connect(self._on_about)
        help_menu.addAction(about_act)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main toolbar")
        toolbar.setObjectName("mainToolBar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(toolbar)

        # Quick actions
        open_act = QAction("📂  Open PDF", self)
        open_act.triggered.connect(self._on_open_pdf)
        toolbar.addAction(open_act)

        batch_act = QAction("📚  Batch…", self)
        batch_act.triggered.connect(self._on_open_batch)
        toolbar.addAction(batch_act)

        toolbar.addSeparator()

        for label, tab_idx, shortcut in (
            ("▶  Run",      TAB_RUN,      "Ctrl+1"),
            ("📋  Jobs",    TAB_JOBS,     "Ctrl+2"),
            ("📊  Results", TAB_RESULTS,  "Ctrl+3"),
            ("⚙️  Settings", TAB_SETTINGS, "Ctrl+4"),
        ):
            act = QAction(label, self)
            act.setShortcut(QKeySequence(shortcut))
            act.triggered.connect(lambda _=False, i=tab_idx: self._tabs.setCurrentIndex(i))
            toolbar.addAction(act)

        toolbar.addSeparator()
        about_act = QAction("ℹ️  About", self)
        about_act.triggered.connect(self._on_about)
        toolbar.addAction(about_act)

    def _build_statusbar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)

        # Permanent status indicator
        self._status_perm = QLabel("Ready")
        sb.addPermanentWidget(self._status_perm, 1)

        # Mini progress bar
        self._mini_progress = QProgressBar()
        self._mini_progress.setMaximumWidth(200)
        self._mini_progress.setRange(0, 1)
        self._mini_progress.setValue(0)
        self._mini_progress.setVisible(False)
        sb.addPermanentWidget(self._mini_progress)

        # Settings version
        ver = QLabel(f"v{APP_VERSION}")
        ver.setObjectName("metricLabel")
        sb.addPermanentWidget(ver)

    def _wire_signals(self) -> None:
        # Run tab → Jobs tab + Results tab + status bar
        self._run_tab.job_started.connect(self._on_job_started)
        self._run_tab.job_progress.connect(self._on_job_progress)
        self._run_tab.job_finished.connect(self._on_job_finished)
        self._run_tab.job_failed.connect(self._on_job_failed)

        # Jobs tab → Results tab (open in results, retry, etc.)
        self._jobs_tab.open_results_requested.connect(self._open_results)
        self._jobs_tab.retry_requested.connect(self._on_retry)

        # Settings tab → live apply to Run tab
        self._settings_tab.settings_changed.connect(self._on_settings_changed)

    # ------------------------------------------------------------------
    # Window state persistence
    # ------------------------------------------------------------------
    def _restore_window_state(self) -> None:
        geometry = self._qsettings.value(QS_KEY_GEOMETRY)
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = self._qsettings.value(QS_KEY_STATE)
        if state is not None:
            self.restoreState(state)
        # Apply theme
        theme = self._qsettings.value(QS_KEY_THEME, THEME_LIGHT)
        self._apply_theme(theme)

    def _save_window_state(self) -> None:
        self._qsettings.setValue(QS_KEY_GEOMETRY, self.saveGeometry())
        self._qsettings.setValue(QS_KEY_STATE, self.saveState())
        self._qsettings.sync()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._save_window_state()
        # If a job is running, confirm.
        # (The worker is a QThread; it will keep running but Qt
        # will mark it for deletion once we close. The user is
        # warned so they can cancel the job first.)
        # NOTE: We don't currently track "any thread running"; the
        # worker checks itself. For the MVP we just let the OS
        # clean up. Future: prompt the user to confirm.
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Menu / toolbar slots
    # ------------------------------------------------------------------
    def _on_open_pdf(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open PDF",
            self._settings.get("last_pdf_dir", str(Path.home())),
            "PDF files (*.pdf)",
        )
        if not path:
            return
        self._run_tab._set_pdf_path(Path(path))
        self._settings["last_pdf_dir"] = str(Path(path).parent)
        self._tabs.setCurrentIndex(TAB_RUN)

    def _on_open_batch(self) -> None:
        dlg = BatchDialog(self._settings, parent=self)
        dlg.batch_started.connect(self._on_batch_started)
        dlg.exec()

    def _on_open_outdir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open output directory", self._settings.get("last_export_dir", str(Path.home())))
        if path:
            self._settings["last_export_dir"] = path
            from PySide6.QtGui import QDesktopServices
            from PySide6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "About RLPE",
            f"<h3>{APP_NAME}</h3>"
            f"<p><b>Version:</b> {APP_VERSION}<br>"
            f"<b>Author:</b> {APP_AUTHOR}</p>"
            f"<p>Native Qt6 desktop GUI for radiolarian plate extraction. "
            f"Powered by PySide6, the RLPE pipeline, FastAPI, and PBDB.</p>"
            f"<p>Phase 32 (MVP) — full feature parity with the Web UI plus "
            f"native image preview with bbox overlay.</p>"
        )

    def _apply_theme(self, theme: str) -> None:
        apply_theme(QApplication.instance(), theme)
        self._qsettings.setValue(QS_KEY_THEME, theme)
        self._log.info("Theme applied: %s", theme)

    def _open_log_file(self) -> None:
        import os
        import subprocess
        from .utils import LOG_FILE_NAME
        log_path = os.path.expanduser(f"~/.cache/rlpe/gui/{LOG_FILE_NAME}")
        try:
            if is_macos():
                subprocess.Popen(["open", log_path])
            elif is_windows():
                os.startfile(log_path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", log_path])
        except Exception as exc:
            QMessageBox.warning(self, "Log file", f"Could not open: {exc}\n\nPath: {log_path}")

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------
    def _on_job_started(self, job_id: str, pdf_path: str) -> None:
        job = JobRecord(
            job_id=job_id,
            pdf_path=pdf_path,
            output_dir=str(Path(pdf_path).parent / f"{Path(pdf_path).stem}_rlpe_out"),
            status=STATUS_RUNNING,
            started_at=time.time(),
            settings=self._run_tab.collect_settings(),
        )
        self._jobs_tab.add_or_update_job(job)
        self._status_perm.setText(f"Job {job_id} running…")
        self._mini_progress.setRange(0, 0)  # indeterminate
        self._mini_progress.setVisible(True)
        # Auto-switch to Jobs tab
        self._tabs.setCurrentIndex(TAB_JOBS)

    def _on_job_progress(self, job_id: str, current: int, total: int, msg: str) -> None:
        self._jobs_tab.update_progress(job_id, current, total, msg)
        if total > 0:
            self._mini_progress.setRange(0, total)
            self._mini_progress.setValue(current)
        self._status_perm.setText(f"{msg}  ({current}/{total})" if total > 0 else msg)

    def _on_job_finished(self, job_id: str, rows: list[dict[str, Any]]) -> None:
        # Update jobs tab
        self._jobs_tab.mark_done(job_id, rows)
        # Update results tab
        job_dir = self._settings.get("last_export_dir", "")
        self._results_tab.load_job(job_id, rows, job_dir)
        # Status
        self._status_perm.setText(f"Job {job_id} done — {len(rows):,} rows")
        self._mini_progress.setRange(0, 1)
        self._mini_progress.setValue(1)
        QTimer.singleShot(2000, lambda: self._mini_progress.setVisible(False))
        # Auto-switch to results tab
        self._tabs.setCurrentIndex(TAB_RESULTS)

    def _on_job_failed(self, job_id: str, error: str) -> None:
        self._jobs_tab.mark_failed(job_id, error)
        self._status_perm.setText(f"Job {job_id} failed")
        self._mini_progress.setVisible(False)
        QMessageBox.critical(self, "Pipeline error", f"Job {job_id} failed:\n\n{error[:1000]}")

    # ------------------------------------------------------------------
    # Jobs tab → open in results / retry
    # ------------------------------------------------------------------
    def _open_results(self, job_id: str) -> None:
        # Find the job in the jobs tab
        from PySide6.QtCore import Qt as _Qt
        jobs = getattr(self._jobs_tab, "_jobs", {})
        if job_id not in jobs:
            return
        job = jobs[job_id]
        self._results_tab.load_job(job.job_id, job.rows, job.output_dir)
        self._tabs.setCurrentIndex(TAB_RESULTS)

    def _on_retry(self, job_id: str, settings: dict) -> None:
        # Re-run the job with the same PDF + (possibly updated) settings
        from PySide6.QtCore import Qt as _Qt
        jobs = getattr(self._jobs_tab, "_jobs", {})
        if job_id not in jobs:
            return
        job = jobs[job_id]
        path = Path(job.pdf_path)
        if not path.exists():
            QMessageBox.warning(self, "Retry", f"Original file no longer exists:\n{path}")
            return
        # Re-push the PDF + settings into Run tab and start
        self._run_tab._set_pdf_path(path)
        # Apply settings
        self._run_tab.apply_settings(settings)
        self._tabs.setCurrentIndex(TAB_RUN)

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------
    def _on_batch_started(self, pdfs: list[Path], batch_settings: dict) -> None:
        # Build a list of job_id placeholders
        from .jobs_tab import JobRecord
        for i, p in enumerate(pdfs):
            jid = f"batch-{i:02d}-{p.stem}"
            job = JobRecord(
                job_id=jid,
                pdf_path=str(p),
                output_dir=str(Path(batch_settings["last_export_dir"]) / f"{p.stem}_rlpe_out"),
                status=STATUS_QUEUED,
                started_at=time.time(),
                settings=batch_settings,
            )
            self._jobs_tab.add_or_update_job(job)
        # Hand off to Run tab; the Run tab worker is single-job
        # so we currently run serially via a small orchestrator:
        self._batch_run(pdfs, batch_settings)

    def _batch_run(self, pdfs: list[Path], batch_settings: dict) -> None:
        """Serial-batch runner: for each PDF, start a PipelineWorker and
        wait for ``finished_ok`` / ``failed`` before starting the next.
        """
        if not pdfs:
            return
        self._batch_pdfs = list(pdfs)
        self._batch_settings = batch_settings
        self._batch_index = 0
        self._start_next_batch_job()

    def _start_next_batch_job(self) -> None:
        if self._batch_index >= len(self._batch_pdfs):
            self._status_perm.setText("Batch complete.")
            return
        pdf = self._batch_pdfs[self._batch_index]
        self._batch_index += 1
        # Use the existing Run tab mechanism to start a job
        self._run_tab._set_pdf_path(pdf)
        self._run_tab.apply_settings(self._batch_settings)
        # Force start
        # The Run tab has a private _on_start(); we emulate it
        # by setting the path, settings, and calling _on_start.
        # For simplicity we directly call:
        self._run_tab._on_start()

    # Patch into the run tab to advance the batch on each completion
    def _refresh_texts(self) -> None:
        """Re-apply all menu / tab labels after a language switch."""
        for i in range(self._tabs.count()):
            key = ("tab.run", "tab.jobs", "tab.results", "tab.settings")[i]
            self._tabs.setTabText(i, i18n._tr(key))
        self.setWindowTitle(i18n._tr("app.title") + f"  v{i18n._tr('app.title')}".replace("RLPE - Radiolarian Plate Extractor", "").strip())
        # Re-render the tab bar widget tree (objects registered via
        # setObjectName are retexted by i18n._apply_registry)
        # (already done by set_language in i18n.py)

    def _on_settings_changed(self) -> None:
        # Re-apply settings to Run tab
        self._settings_tab.apply_to_run_settings()
        self._run_tab.apply_settings(self._settings)