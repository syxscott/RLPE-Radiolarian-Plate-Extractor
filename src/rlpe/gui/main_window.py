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
from .styles import apply_theme
from .utils import (
    get_gui_logger,
    is_linux,
    is_macos,
    is_windows,
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
from .styles import SPACE_M, SPACE_S
from .jobs_tab import JobRecord, JobsTab
from .results_tab import ResultsTab
from .run_tab import RunTab
from .settings_tab import SettingsTab


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
        # Phase 56 audit: build the full UI (was stuck after _qbool return)
        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        self._build_statusbar()
        self._wire_signals()
        self._restore_window_state()
        # Phase 49: load completed jobs from disk so the Jobs tab
        # shows historical results even after a GUI restart.
        self._load_recent_jobs()

    @staticmethod
    def _qbool(qsettings: QSettings, key: str, default: bool) -> bool:
        """Phase 55 audit: correctly parse QSettings-stored booleans.

        QSettings stores bools as strings ("true"/"false"). Wrapping in
        bool() is wrong because bool("false") == True (non-empty string).
        The correct approach is to use type=bool OR compare lowercase.
        """
        val = qsettings.value(key, default, type=bool)
        if isinstance(val, bool):
            return val
        return str(val).lower() in ("true", "1", "yes")

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
        # the Results tab. Sort by finished_at (descending); fall
        # back to job_id if finished_at is missing.
        try:
            jobs = self._jobs_tab._jobs
            if not jobs:
                return
            latest = max(
                jobs.values(),
                key=lambda j: (j.finished_at or -1, j.job_id),
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
            # Phase 54 audit m6: default was str(__file__) which
            # pointed at main_window.py inside the source tree. On
            # first run this caused the file dialog to open inside
            # src/rlpe/gui, which is rarely where a user's PDFs
            # actually live. Fall back to the user's home directory.
            "last_pdf_dir": str(self._qsettings.value(QS_KEY_LAST_DIR, str(Path.home()))),
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
            "use_paleodb": self._qbool(self._qsettings, "use_paleodb", True),
            "paleodb_max_occurrences": int(self._qsettings.value("paleodb_max_occurrences", 25)),
            "paleodb_endpoint": self._qsettings.value("paleodb_endpoint", "https://paleobiodb.org/data1.2"),
            "render_dpi": int(self._qsettings.value("render_dpi", 200)),
            "save_intermediate": self._qbool(self._qsettings, "save_intermediate", False),
        }

    # ------------------------------------------------------------------
    # UI assembly
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
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

        # Register _refresh_texts as an i18n listener so tab labels and
        # the window title translate on language switch. Using a bound
        # method (not a lambda) lets closeEvent remove the listener
        # by identity without accumulating stale references.
        self._i18n_listener = self._on_language_changed
        i18n.add_listener(self._i18n_listener)
        self._refresh_texts()

    def _build_menu(self) -> None:
        menubar = self.menuBar()

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

        # text translates on language switch. We track which i18n key
        # the status bar is currently displaying in ``_status_key``
        # so a language switch can re-render the right key.
        self._status_perm = QLabel()
        self._status_perm.setObjectName("statusBar.main")
        self._status_key = "main.idle"
        self._status_kwargs: dict = {}
        self._status_perm.setText(i18n._tr(self._status_key))
        i18n.register_widget_text("statusBar.main", "text", "main.idle")
        # Phase 55 audit B4: store as bound method attribute so
        # closeEvent can remove the same listener by identity.
        self._status_i18n_listener = self._refresh_status_text
        i18n.add_listener(self._status_i18n_listener)
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
        """Re-render the status bar with the current i18n key."""
        try:
            self._status_perm.setText(
                i18n._tr(self._status_key).format(**self._status_kwargs)
            )
        except Exception:
            pass

    def _on_language_changed(self, _lang: str) -> None:
        """Rebuild menu/tab labels on language switch (i18n listener)."""
        self._refresh_texts()

    def _set_status(self, key: str, **kwargs) -> None:
        """Set the status bar text via an i18n key + kwargs."""
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
        self._remove_i18n_listeners()
        self._save_window_state()
        self._stop_pipeline_worker()
        self._flush_settings()
        event.accept()
        super().closeEvent(event)

    def _remove_i18n_listeners(self) -> None:
        """Drop all i18n listeners registered on construction.

        Phase 55 audit B4: without this, every MainWindow construction
        accumulated one dead listener that fired on language switch after
        the widget was destroyed (RuntimeError: wrapped C/C++ object
        has been deleted).
        """
        listener = getattr(self, "_i18n_listener", None)
        if listener is not None:
            i18n.remove_listener(listener)
        status_listener = getattr(self, "_status_i18n_listener", None)
        if status_listener is not None:
            i18n.remove_listener(status_listener)
        jobs_remove = getattr(self._jobs_tab, "_remove_i18n_listener", None)
        if jobs_remove:
            jobs_remove()
        results_remove = getattr(self._results_tab, "_remove_i18n_listener", None)
        if results_remove:
            results_remove()
        run_remove = getattr(self._run_tab, "_remove_i18n_listener", None)
        if run_remove:
            run_remove()

    def _stop_pipeline_worker(self) -> None:
        """Stop the PipelineWorker QThread synchronously on shutdown.

        Phase 44: the worker's ``finished`` signal only fires when the
        thread exits normally; closing mid-job would leave it running
        with the parent widget destroyed → RuntimeError.  We request
        cancel, wait up to 5 s, then terminate if needed.
        Phase 55 audit B2: capture the job id before requesting cancel
        so the Jobs tab row can be stamped STATUS_CANCELLED and the
        disk-scan zombie-job problem (Phase 49) is avoided.
        """
        worker = getattr(self._run_tab, "_worker", None)
        if worker is None or not worker.isRunning():
            return
        current_job_id = getattr(self._run_tab, "_current_job_id", None)
        try:
            worker.request_cancel()
        except RuntimeError:
            pass
        if not worker.wait(5000):
            worker.terminate()
            worker.wait(500)
        if current_job_id:
            try:
                self._jobs_tab.mark_cancelled(current_job_id)
            except Exception:  # pragma: no cover — defensive
                pass

    def _flush_settings(self) -> None:
        """Flush QSettings to persistent storage on shutdown.

        Phase 56 audit fix: write back in-memory _settings changes to QSettings
        before syncing. Without this, last_pdf_dir and last_export_dir changes
        made via the Run tab are lost on restart.

        On Windows QSettings is registry-backed; changes are only
        flushed on process exit, not on every setValue.
        """
        for key, value in self._settings.items():
            self._qsettings.setValue(key, value)
        self._qsettings.sync()

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
        from PySide6.QtCore import QFileInfo
        log_path = Path(os.path.expanduser(f"~/.cache/rlpe/gui/{LOG_FILE_NAME}"))
        # Phase 55 audit — on a fresh install the log file does not
        # exist yet, so the previous ``xdg-open``/``open`` call would
        # raise FileNotFoundError and pop a confusing yellow warning.
        # Show a friendly info dialog instead — the file will appear
        # after the first pipeline run.
        if not log_path.exists():
            QMessageBox.information(
                self,
                i18n._tr("settab.log.title"),
                i18n._tr("settab.log.not_yet").format(path=str(log_path)),
            )
            return
        try:
            if is_macos():
                subprocess.Popen(["open", str(log_path)])
            elif is_windows():
                os.startfile(str(log_path))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(log_path)])
        except Exception as exc:
            QMessageBox.warning(
                self,
                i18n._tr("settab.log.title"),
                i18n._tr("settab.log.open_fail").format(
                    error=f"{type(exc).__name__}: {exc}", path=str(log_path),
                ),
            )

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
        # Phase 54 audit: M8 — prefer the per-job output_dir recorded on
        # the JobRecord (if any) over the global ``last_export_dir``
        # setting. The previous code always used the global setting, so
        # a batch job with per-job output dirs opened the wrong
        # folder's exports. Fall back to the global setting for legacy
        # single-PDF jobs that never recorded their own output_dir.
        job_record = self._jobs_tab._jobs.get(job_id) if hasattr(self, "_jobs_tab") else None
        job_dir = (
            getattr(job_record, "output_dir", None)
            or self._settings.get("last_export_dir", "")
        )
        self._results_tab.load_job(job_id, rows, job_dir)
        # Status
        self._set_status("main.done", id=job_id, rows=f"{len(rows):,}")
        self._mini_progress.setRange(0, 1)
        self._mini_progress.setValue(1)
        QTimer.singleShot(2000, lambda: self._mini_progress.setVisible(False))
        # Auto-switch to results tab
        self._tabs.setCurrentIndex(TAB_RESULTS)
        # Phase 54 audit: M3 — advance the serial batch queue. The
        # previous version declared a batch helper at line 702-719 that
        # started the *first* job and bumped the index, but nothing
        # invoked ``_start_next_batch_job`` on completion. The dangling
        # comment at line 719 ("Patch into the run tab to advance the
        # batch on each completion") was never implemented, so a
        # 10-PDF batch would actually only run the first one. We now
        # re-enter the helper here whenever a batch is in flight and
        # there are still PDFs to process.
        if (
            getattr(self, "_batch_pdfs", None)
            and self._batch_index < len(self._batch_pdfs)
        ):
            self._start_next_batch_job()

    def _on_job_failed(self, job_id: str, error: str) -> None:
        self._jobs_tab.mark_failed(job_id, error)
        self._set_status("main.failed", id=job_id)
        self._mini_progress.setVisible(False)
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
            # Phase 55 audit M2 — honour the batch's "produce a
            # consolidated .xlsx at the end" checkbox. The previous
            # code never read ``_xlsx_at_end`` so the BatchDialog
            # checkbox was silently ignored. When the flag is set
            # and we have a valid output_dir, ask the Results tab
            # to export a workbook combining every batch job's rows.
            self._set_status("main.batch_complete")
            batch_settings = getattr(self, "_batch_settings", {}) or {}
            if batch_settings.get("_xlsx_at_end"):
                try:
                    self._export_batch_xlsx()
                except Exception as exc:  # pragma: no cover
                    self._log.warning(
                        "batch xlsx_at_end export failed: %s", exc
                    )
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

    def _export_batch_xlsx(self) -> None:
        """Phase 55 audit M2 — export a single workbook combining all
        batch job rows, opened with a Save dialog so the user picks
        the destination. Falls back to the batch output_dir when
        the user cancels.
        """
        from PySide6.QtWidgets import QFileDialog
        import datetime as _dt

        all_rows: list[dict[str, Any]] = []
        for job in getattr(self._jobs_tab, "_jobs", {}).values():
            for row in (getattr(job, "rows", None) or []):
                all_rows.append(row)
        if not all_rows:
            return
        default_dir = (
            (self._batch_settings or {}).get("last_export_dir")
            or str(Path.home())
        )
        default_name = (
            f"rlpe_batch_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            i18n._tr("restab.export.xlsx.title"),
            str(Path(default_dir) / default_name),
            "Excel files (*.xlsx)",
        )
        if not path:
            return
        # Phase 56 audit: build run_output from all batch rows and export via xlsx.
        from ..exporters.xlsx import write_xlsx
        run_output = {
            "schema_version": "1.0.0",
            "provenance": {"job_id": "batch", "source": "rlpe-gui"},
            "papers": [],
            "figures": [],
            "panels": all_rows,
            "taxa": [],
            "samples": [],
            "geology_contexts": [
                g for r in all_rows for g in ((r.get("metadata") or {}).get("geology_links") or [])
            ],
            "localities": [
                {"country": g.get("country"), "locality": g.get("locality")}
                for r in all_rows for g in ((r.get("metadata") or {}).get("geology_links") or [])
                if g.get("country") or g.get("locality")
            ],
            "paleo_coordinates": [],
            "warnings": [],
        }
        write_xlsx(run_output, str(path))

    # Patch into the run tab to advance the batch on each completion
    def _refresh_texts(self) -> None:
        """Re-apply all menu / tab labels after a language switch."""
        for i in range(self._tabs.count()):
            key = ("tab.run", "tab.jobs", "tab.results", "tab.settings")[i]
            self._tabs.setTabText(i, i18n._tr(key))
        self.setWindowTitle(f"{i18n._tr('app.title')}  v{APP_VERSION}")

    def _on_settings_changed(self) -> None:
        # Re-apply settings to Run tab
        self._settings_tab.apply_to_run_settings()
        self._run_tab.apply_settings(self._settings)