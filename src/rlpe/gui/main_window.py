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
        self._qsettings = QSettings(APP_AUTHOR, APP_NAME)
        self._settings = self._load_settings_cache()
        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        self._build_statusbar()
        self._wire_signals()
        self._restore_window_state()
        # Phase 49: scan disk for previously completed jobs (mirrors
        # what the Web API does at startup) and populate the JobsTab.
        # Without this, restarting the GUI loses visibility into jobs
        # completed via the CLI / Web UI / a prior GUI session.
        self._load_recent_jobs()

    def _load_recent_jobs(self) -> None:
        """Phase 49: scan service_work/ and work/ for completed jobs.

        Logs a status-bar message summarising how many were loaded.

        Phase 51: if at least one job was loaded, auto-select the most
        recently finished one in the Results tab and switch to it. This
        matches user expectations — when they open the GUI to "see the
        data I extracted", the Run tab (which is the default empty tab)
        is the wrong place to start. They want to see results.
        """
        try:
            n = self._jobs_tab.load_recent_jobs_from_disk()
        except Exception as exc:  # never let startup scan block the GUI
            self._log.warning("Phase 49 startup scan failed: %s", exc)
            return
        if not n:
            return
        self._log.info("Loaded %d recent job(s) from disk", n)
        # Show a transient status bar message in the user's language.
        self.statusBar().showMessage(
            i18n._tr("main.recent_loaded").format(n=n), 5000
        )
        # Phase 51: auto-open the most recently finished job in
        # the Results tab. Sort by finished_at (descending); fall
        # back to job_id if finished_at is missing.
        try:
            jobs = self._jobs_tab._jobs
            if not jobs:
                return
            latest = max(
                jobs.values(),
                key=lambda j: (j.finished_at, j.job_id),
            )
            self._results_tab.load_job(
                latest.job_id, latest.rows, latest.output_dir,
            )
            self._tabs.setCurrentIndex(TAB_RESULTS)
        except Exception as exc:
            # Never let auto-load block GUI startup.
            self._log.warning("Phase 51 auto-open most recent failed: %s", exc)

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
        # Phase 36: load persisted language preference (default zh_CN).
        # The i18n module defaults to zh_CN at import time; the user's
        # last choice overrides it here.
        saved_lang = self._qsettings.value("ui/language", "zh_CN")
        if saved_lang in ("en", "zh_CN"):
            i18n.set_language(saved_lang)

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

        # Phase 40: register _refresh_texts as an i18n listener so
        # tab labels (▶ Run / 📋 Jobs / 📊 Results / ⚙️ Settings)
        # and the window title translate on language switch. Without
        # this the tabs are stuck in the language they were first
        # created in. Also call _refresh_texts() once at the end of
        # _build_ui so the FIRST PAINT shows the labels in the
        # current language (zh_CN by default), not the bare English
        # strings we passed to addTab().
        i18n.add_listener(lambda _lang: self._refresh_texts())
        self._refresh_texts()

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        # Phase 39: use tr_menu / tr_action so all menu labels
        # translate on language switch. The QAction objects are
        # registered in the i18n._MENU_ACTIONS list (i18n_widgets)
        # because they aren't QWidgets, so the registry's
        # allWidgets() sweep can't find them.
        from .i18n_widgets import tr_action, tr_menu

        # File menu
        file_menu = tr_menu("menu.file", parent=menubar)
        menubar.addAction(file_menu.menuAction())

        open_act = tr_action("menu.file.open", parent=self)
        open_act.setShortcut(QKeySequence.Open)
        open_act.triggered.connect(self._on_open_pdf)
        file_menu.addAction(open_act)

        batch_act = tr_action("menu.file.batch", parent=self)
        batch_act.setShortcut(QKeySequence("Ctrl+B"))
        batch_act.triggered.connect(self._on_open_batch)
        file_menu.addAction(batch_act)

        file_menu.addSeparator()

        out_act = tr_action("menu.file.outdir", parent=self)
        out_act.triggered.connect(self._on_open_outdir)
        file_menu.addAction(out_act)

        file_menu.addSeparator()

        quit_act = tr_action("menu.file.quit", parent=self)
        quit_act.setShortcut(QKeySequence.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # View menu
        view_menu = tr_menu("menu.view", parent=menubar)
        menubar.addAction(view_menu.menuAction())

        for key, tab_idx in (
            ("menu.view.run",      TAB_RUN),
            ("menu.view.jobs",     TAB_JOBS),
            ("menu.view.results",  TAB_RESULTS),
            ("menu.view.settings", TAB_SETTINGS),
        ):
            act = tr_action(key, parent=self)
            act.setShortcut(QKeySequence(f"Ctrl+{tab_idx + 1}"))
            act.triggered.connect(lambda _=False, i=tab_idx: self._tabs.setCurrentIndex(i))
            view_menu.addAction(act)

        view_menu.addSeparator()

        # Theme switcher
        theme_menu = tr_menu("menu.theme", parent=view_menu)
        view_menu.addAction(theme_menu.menuAction())
        for theme_key, theme in (
            ("menu.theme.light",  THEME_LIGHT),
            ("menu.theme.dark",   "dark"),
            ("menu.theme.system", "system"),
        ):
            act = tr_action(theme_key, parent=self)
            act.triggered.connect(lambda _=False, t=theme: self._apply_theme(t))
            theme_menu.addAction(act)

        # Tools menu
        tools_menu = tr_menu("menu.tools", parent=menubar)
        menubar.addAction(tools_menu.menuAction())

        # Open log file
        log_act = tr_action("menu.tools.log", parent=self)
        log_act.triggered.connect(self._open_log_file)
        tools_menu.addAction(log_act)

        # Help menu
        help_menu = tr_menu("menu.help", parent=menubar)
        menubar.addAction(help_menu.menuAction())

        about_act = tr_action("menu.help.about", parent=self)
        about_act.triggered.connect(self._on_about)
        help_menu.addAction(about_act)

    def _build_toolbar(self) -> None:
        # Phase 39: use tr_action so toolbar labels translate on
        # language switch. The QToolBar widget itself is a QWidget
        # so its title can be retexted via the i18n registry.
        from .i18n_widgets import tr_action
        toolbar = QToolBar(i18n._tr("toolbar.title"))
        toolbar.setObjectName("mainToolBar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(toolbar)
        i18n.register_widget_text("mainToolBar", "windowTitle", "toolbar.title")

        # Quick actions
        open_act = tr_action("toolbar.open", parent=self)
        open_act.triggered.connect(self._on_open_pdf)
        toolbar.addAction(open_act)

        batch_act = tr_action("toolbar.batch", parent=self)
        batch_act.triggered.connect(self._on_open_batch)
        toolbar.addAction(batch_act)

        toolbar.addSeparator()

        for key, tab_idx, shortcut in (
            ("toolbar.run",      TAB_RUN,      "Ctrl+1"),
            ("toolbar.jobs",     TAB_JOBS,     "Ctrl+2"),
            ("toolbar.results",  TAB_RESULTS,  "Ctrl+3"),
            ("toolbar.settings", TAB_SETTINGS, "Ctrl+4"),
        ):
            act = tr_action(key, parent=self)
            act.setShortcut(QKeySequence(shortcut))
            act.triggered.connect(lambda _=False, i=tab_idx: self._tabs.setCurrentIndex(i))
            toolbar.addAction(act)

        toolbar.addSeparator()
        about_act = tr_action("toolbar.about", parent=self)
        about_act.triggered.connect(self._on_about)
        toolbar.addAction(about_act)

    def _build_statusbar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)

        # Phase 39: use a plain QLabel + i18n registry so the status
        # text translates on language switch. We track which i18n key
        # the status bar is currently displaying in ``_status_key``
        # so a language switch can re-render the right key.
        self._status_perm = QLabel()
        self._status_perm.setObjectName("statusBar.main")
        self._status_key = "main.idle"
        self._status_kwargs: dict = {}
        self._status_perm.setText(i18n._tr(self._status_key))
        i18n.register_widget_text("statusBar.main", "text", "main.idle")
        # Register a listener so the status bar re-renders on language
        # switch using the current key.
        i18n.add_listener(self._refresh_status_text)
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

    def _refresh_status_text(self, _lang: str) -> None:
        """Phase 39: re-render the status bar with the current i18n key
        when the language switches."""
        try:
            self._status_perm.setText(
                i18n._tr(self._status_key).format(**self._status_kwargs)
            )
        except Exception:
            pass

    def _set_status(self, key: str, **kwargs) -> None:
        """Phase 39: set the status bar text via an i18n key + kwargs.
        Replaces direct setText calls so the language switch can
        re-render via _refresh_status_text."""
        self._status_key = key
        self._status_kwargs = kwargs
        try:
            self._status_perm.setText(i18n._tr(key).format(**kwargs))
        except Exception:
            # Fallback: set without format if template has unexpected keys
            self._status_perm.setText(i18n._tr(key))

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
        # Phase 44: robust shutdown order. The previous code
        # just saved window state and called super().closeEvent(),
        # which could leave the PipelineWorker QThread running
        # with the parent widget destroyed → RuntimeError.
        self._save_window_state()
        # 1) Stop the worker if it's running. The Run tab's
        #    _on_thread_done handles quit() + wait() but it only
        #    fires on the worker's `finished` signal, which we
        #    may not see if we're closing mid-job. So we wait
        #    synchronously here.
        worker = getattr(self._run_tab, "_worker", None)
        if worker is not None and worker.isRunning():
            try:
                worker.request_cancel()
            except RuntimeError:
                pass
            # Wait up to 5s for the thread to honour the cancel.
            if not worker.wait(5000):
                worker.terminate()
                worker.wait(500)
        # 2) Flush QSettings so any unsaved state survives the
        #    process exit (especially important on Windows where
        #    QSettings is backed by the registry and changes
        #    are only flushed on app exit, not on every setValue).
        from PySide6.QtCore import QSettings
        QSettings(APP_AUTHOR, APP_NAME).sync()
        # 3) Accept the event (window closes).
        event.accept()
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
        # Phase 39: use i18n keys for the About dialog so it
        # translates on language switch. The body uses {fmt} placeholders
        # for the version / author / app name.
        QMessageBox.about(
            self,
            i18n._tr("app.about.title"),
            i18n._tr("app.about.body").format(
                version=APP_VERSION,
                author=APP_AUTHOR,
            ),
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
        self._set_status("main.running", id=job_id)
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
        self._set_status("main.done", id=job_id, rows=f"{len(rows):,}")
        self._mini_progress.setRange(0, 1)
        self._mini_progress.setValue(1)
        QTimer.singleShot(2000, lambda: self._mini_progress.setVisible(False))
        # Auto-switch to results tab
        self._tabs.setCurrentIndex(TAB_RESULTS)

    def _on_job_failed(self, job_id: str, error: str) -> None:
        self._jobs_tab.mark_failed(job_id, error)
        self._set_status("main.failed", id=job_id)
        self._mini_progress.setVisible(False)
        # Phase 42: if a batch is running and the operator asked
        # for "stop on first error", halt the batch here. The
        # _batch_pdfs list is reset so the next _start_next_batch_job
        # call returns immediately.
        if (
            getattr(self, "_batch_pdfs", None)
            and self._batch_settings.get("_stop_on_error", False)
        ):
            remaining = len(self._batch_pdfs) - self._batch_index
            self._batch_pdfs = []  # halt
            self._set_status("main.batch_stopped_on_error", failed=job_id, remaining=remaining)
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
            self._set_status("main.batch_complete")
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
        """Re-apply all menu / tab labels after a language switch.

        Phase 40: also refresh the window title — the old code
        concatenated ``i18n._tr("app.title")`` twice (once directly,
        once in an f-string ``v{i18n._tr('app.title')}`` whose
        ``.replace("RLPE - Radiolarian Plate Extractor", "")`` was
        meant to strip the EN title to leave just "v1.1.0", but in
        zh_CN the title was different so the replace did nothing and
        the window title became e.g.
        ``"RLPE - 放射虫图版提取系统  vRLPE - 放射虫图版提取系统"``.
        Fixed: use APP_VERSION constant directly (it's a number, not
        translated).
        """
        for i in range(self._tabs.count()):
            key = ("tab.run", "tab.jobs", "tab.results", "tab.settings")[i]
            self._tabs.setTabText(i, i18n._tr(key))
        self.setWindowTitle(f"{i18n._tr('app.title')}  v{APP_VERSION}")
        # Re-render the tab bar widget tree (objects registered via
        # setObjectName are retexted by i18n._apply_registry)
        # (already done by set_language in i18n.py)

    def _on_settings_changed(self) -> None:
        # Re-apply settings to Run tab
        self._settings_tab.apply_to_run_settings()
        self._run_tab.apply_settings(self._settings)