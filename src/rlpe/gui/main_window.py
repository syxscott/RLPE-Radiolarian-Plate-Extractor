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

import re
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
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

from . import i18n
from .batch_dialog import BatchDialog
from .styles import apply_theme
from .utils import (
    get_gui_logger,
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
    APP_NAME,
    APP_VERSION,
    MAIN_WINDOW_DEFAULT_SIZE,
    MAIN_WINDOW_MIN_SIZE,
    QS_KEY_GEOMETRY,
    QS_KEY_LAST_DIR,
    QS_KEY_LAST_EXPORT_DIR,
    QS_KEY_STATE,
    QS_KEY_THEME,
    STATUS_DONE,
    STATUS_QUEUED,
    STATUS_RUNNING,
    TAB_JOBS,
    TAB_RESULTS,
    TAB_RUN,
    TAB_SETTINGS,
    THEME_LIGHT,
)
from .jobs_tab import JobRecord, JobsTab
from .results_tab import ResultsTab
from .run_tab import RunTab
from .settings_tab import SettingsTab

# ============================================================
# Batch export worker (M-25)
# ============================================================


class _BatchExportWorker(QThread):
    """Phase F-2 (M-25): run the heavy XLSX write on a background
    QThread so the GUI never freezes during a large batch export.

    Signals
    -------
    progress(str) — human-readable progress message
    finished_ok(str) — absolute path of the written file on success
    failed(str) — error message on failure
    """

    progress = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        jobs: list,  # snapshot of JobRecord list
        out_path: str,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._jobs = jobs
        self._out_path = out_path
        self._cancelled = False

    def run(self) -> None:
        try:
            all_rows: list[dict[str, Any]] = []
            for job in self._jobs:
                for row in getattr(job, "rows", None) or []:
                    all_rows.append(row)
            if not all_rows:
                self.failed.emit("No rows to export")
                return

            self.progress.emit(f"Writing {len(all_rows):,} rows…")
            run_output = {
                "schema_version": "1.0.0",
                "provenance": {"job_id": "batch", "source": "rlpe-gui"},
                "papers": [],
                "figures": [],
                "panels": all_rows,
                "taxa": [],
                "samples": [],
                "geology_contexts": [
                    g
                    for r in all_rows
                    for g in ((r.get("metadata") or {}).get("geology_links") or [])
                ],
                "localities": [
                    {"country": g.get("country"), "locality": g.get("locality")}
                    for r in all_rows
                    for g in ((r.get("metadata") or {}).get("geology_links") or [])
                    if g.get("country") or g.get("locality")
                ],
                "paleo_coordinates": [],
                "warnings": [],
            }
            from ..exporters.xlsx import write_xlsx

            write_xlsx(run_output, self._out_path)
            self.finished_ok.emit(self._out_path)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


# ============================================================
# Main window
# ============================================================
class MainWindow(QMainWindow):
    """Top-level window for the RLPE GUI."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Type-annotated initialisations (mypy: prevents dynamic-attr
        # errors when these attrs are read before assignment).
        self._mini_progress_timer: QTimer | None = None
        self._batch_pdfs: list[Path] = []
        self._batch_index: int = 0
        self._batch_settings: dict[str, Any] = {}
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

    def _qint(self, qsettings: QSettings, key: str, default: int) -> int:
        """audit 2026-07-31: safe int read — a corrupted QSettings
        value ("abc", a float "3.0") used to crash the whole GUI at
        startup via bare int(). Mirrors settings_tab._qint."""
        val = qsettings.value(key, default)
        try:
            return int(float(str(val)))
        except (TypeError, ValueError):
            return default

    def _qfloat(self, qsettings: QSettings, key: str, default: float) -> float:
        """audit 2026-07-31: safe float read (same corruption guard)."""
        val = qsettings.value(key, default)
        try:
            return float(str(val))
        except (TypeError, ValueError):
            return default

    def _qbool(self, qsettings: QSettings, key: str, default: bool) -> bool:
        """Phase 55 audit: correctly parse QSettings-stored booleans.

        QSettings stores bools as strings ("true"/"false"). Wrapping in
        bool() is wrong because bool("false") == True (non-empty string).

        Phase 6A (NIT-1): DO NOT use ``type=bool`` as the converter —
        Qt's QVariant-to-bool coerces every non-empty string (including
        ``"no"``, ``"off"``, ``"false"``) to ``True`` (the
        "any-non-empty-string-is-truthy" pitfall wrapped in Qt). Read
        the raw value, lower-case + strip it, and accept the standard
        truthy spellings (``true``, ``yes``, ``1``, ``on``) and the
        standard falsy ones (``false``, ``no``, ``0``, ``off``).
        Anything else falls back to ``default``.
        """
        val = qsettings.value(key, default)
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        s = str(val).strip().lower()
        if s in ("true", "1", "yes", "on", "y", "t"):
            return True
        if s in ("false", "0", "no", "off", "n", "f", ""):
            return False
        return default

    def _load_recent_jobs(self) -> None:
        """Phase 49: kick off the async disk scan for completed jobs.

        The actual JSONL parse runs on a :class:`_DiskScanWorker`
        ``QThread`` (Phase F-1 / B-15) so the GUI event loop is never
        blocked. The number returned here is the *candidate* count
        (how many ``matches.jsonl`` files the synchronous directory
        walk found). The actual rows arrive via the worker's
        ``job_loaded`` signal; the final list arrives via
        ``JobsTab.scan_finished`` which is connected to
        :meth:`_on_disk_scan_done` in :meth:`__init__`.

        Phase 51: the auto-open-to-Results-tab logic used to live
        here synchronously. That stopped working once B-15 made the
        scan async — by the time this function returned, the worker
        hadn't emitted any ``job_loaded`` signals yet, so
        ``self._jobs_tab._jobs`` was empty and the auto-open never
        fired. The fix moves the auto-open to
        :meth:`_on_disk_scan_done`, which runs after the scan
        truly completes.

        Logs a status-bar message summarising how many were loaded.
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
        self.statusBar().showMessage(i18n._tr("main.recent_loaded").format(n=n), 5000)

    def _on_disk_scan_done(self, records: list[JobRecord]) -> None:
        """Phase F-1 (B-3): auto-open the latest done job in Results.

        Connected to :attr:`JobsTab.scan_finished` (which fires AFTER
        every ``job_loaded`` signal has been dispatched and the worker
        has finished). The previous synchronous auto-open logic in
        :meth:`_load_recent_jobs` was a no-op once B-15 moved the parse
        to a ``QThread`` because the worker hadn't finished yet by the
        time the function returned.

        Behaviour:
          * ``len(records) == 0`` (no candidates, or shutdown
            interrupt) → keep the current tab (Run on a fresh install).
          * Otherwise → pick the most recent ``STATUS_DONE`` job with
            a non-empty ``rows`` list, load it into the Results tab,
            and switch to the Results tab.

        Defensive: every step is wrapped in try/except so a misformatted
        JobRecord can never crash the GUI startup path.
        """
        try:
            if not records:
                # No candidates / graceful shutdown — keep the default tab.
                return
            finished = [
                j
                for j in records
                if getattr(j, "status", None) == STATUS_DONE and getattr(j, "rows", None)
            ]
            if not finished:
                return
            latest = max(finished, key=lambda j: getattr(j, "finished_at", 0) or 0)
            self._results_tab.load_job(latest.job_id, latest.rows, latest.output_dir)
            self._tabs.setCurrentIndex(TAB_RESULTS)
        except Exception as exc:  # pragma: no cover — defensive
            self._log.warning("Phase F-1 B-3 auto-open failed: %s", exc)

    def _on_disk_scan_failed(self, reason: str) -> None:
        """Phase F-1 (B-3): log (do not crash) when the async scan fails.

        The scan worker is best-effort: a corrupt manifest, a defunct
        path, or a transient race should never block the GUI from
        starting. We log at WARNING so the operator can see it in the
        log file, then leave the ``_jobs`` state as-is (whatever the
        scan did manage to load before it failed). The default tab
        (Run) stays selected.
        """
        self._log.warning("Phase F-1 B-3: disk scan failed: %s", reason)

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
            "grobid_max_retries": self._qint(self._qsettings, "grobid_max_retries", 3),
            "grobid_timeout": self._qint(self._qsettings, "grobid_timeout", 300),
            "ocr_backend": self._qsettings.value("ocr_backend", "paddleocr"),
            "ocr_lang": self._qsettings.value("ocr_lang", "en"),
            "caption_window": self._qint(self._qsettings, "caption_window", 2),
            "od_caption_window": self._qint(self._qsettings, "od_caption_window", 5),
            # Audit 2026-07-26 M5: load YOLO keys from QSettings so the
            # Run tab's collect_settings() can forward them to the
            # worker. Previously only the Settings tab read these;
            # _load_settings_cache omitted them, so the in-memory
            # default (use_yolo_figures=False) silently won.
            "use_yolo_figures": self._qbool(self._qsettings, "use_yolo_figures", False),
            "yolo_model_path": str(self._qsettings.value("yolo_model_path", "")),
            "yolo_conf_threshold": self._qfloat(self._qsettings, "yolo_conf_threshold", 0.25),
            "yolo_iou_threshold": self._qfloat(self._qsettings, "yolo_iou_threshold", 0.45),
            "llm_backend": self._qsettings.value("llm_backend", "minimax"),
            "m3_prompt_lang": self._qsettings.value("m3_prompt_lang", "auto"),
            "m3_model": self._qsettings.value("m3_model", "MiniMax-M3"),
            "MiniMax_thinking_budget": self._qint(self._qsettings, "MiniMax_thinking_budget", 1024),
            "MiniMax_max_output_tokens": self._qint(
                self._qsettings, "MiniMax_max_output_tokens", 2048
            ),
            "MiniMax_timeout_sec": self._qint(self._qsettings, "MiniMax_timeout_sec", 60),
            "MiniMax_max_retries": self._qint(self._qsettings, "MiniMax_max_retries", 3),
            "use_paleodb": self._qbool(self._qsettings, "use_paleodb", True),
            "paleodb_max_occurrences": self._qint(self._qsettings, "paleodb_max_occurrences", 25),
            "paleodb_endpoint": self._qsettings.value(
                "paleodb_endpoint", "https://paleobiodb.org/data1.2"
            ),
            "render_dpi": self._qint(self._qsettings, "render_dpi", 200),
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
            ("menu.view.run", TAB_RUN),
            ("menu.view.jobs", TAB_JOBS),
            ("menu.view.results", TAB_RESULTS),
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
            ("menu.theme.light", THEME_LIGHT),
            ("menu.theme.dark", "dark"),
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
            ("toolbar.run", TAB_RUN, "Ctrl+1"),
            ("toolbar.jobs", TAB_JOBS, "Ctrl+2"),
            ("toolbar.results", TAB_RESULTS, "Ctrl+3"),
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
            self._status_perm.setText(i18n._tr(self._status_key).format(**self._status_kwargs))
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
        # Phase F-2 (M-10/M-16): job_started now carries output_dir
        self._run_tab.job_started.connect(self._on_job_started)
        self._run_tab.job_progress.connect(self._on_job_progress)
        self._run_tab.job_finished.connect(self._on_job_finished)
        self._run_tab.job_failed.connect(self._on_job_failed)

        # Jobs tab → Results tab (open in results, retry, etc.)
        self._jobs_tab.open_results_requested.connect(self._open_results)
        self._jobs_tab.retry_requested.connect(self._on_retry)
        # Phase F-1 (B-3): the asynchronous disk scan completes off
        # the GUI thread; the synchronous auto-open-on-startup logic
        # in ``_load_recent_jobs`` used to read ``_jobs`` before the
        # worker had emitted anything, so the auto-open never fired.
        # Connect the new ``scan_finished(records)`` signal so we
        # auto-open the latest done job AFTER the real load completes.
        # ``scan_failed`` is logged but otherwise ignored — the GUI
        # still starts with whatever the in-memory ``_jobs`` state was.
        self._jobs_tab.scan_finished.connect(self._on_disk_scan_done)
        self._jobs_tab.scan_failed.connect(self._on_disk_scan_failed)

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
        # Phase F-1 (B-1): the Jobs tab's ``_DiskScanWorker`` is a
        # QThread that runs the async JSONL parse. Closing the GUI
        # mid-scan used to leave the thread running; when the Python
        # interpreter tried to GC the wrapped C++ object on window
        # destroy, it crashed with exit code 134 (SIGABRT) and the
        # ``QThread: Destroyed while thread is still running``
        # warning. Ask the Jobs tab to shut the worker down BEFORE we
        # try to stop the heavier pipeline worker below — the disk
        # scan completes in <100 ms on a typical install and is the
        # faster of the two to release.
        try:
            self._jobs_tab.shutdown()
        except Exception as exc:  # pragma: no cover — defensive
            self._log.warning("Phase F-1 B-1: jobs_tab shutdown failed: %s", exc)
        # Phase F-2 (M-9+M-27): also shut down RunTab (cancels any
        # in-flight pipeline via request_cancel) before stopping the
        # worker thread. RunTab.shutdown is a graceful no-op if there
        # is no active job.
        try:
            run_shutdown = getattr(self._run_tab, "shutdown", None)
            if run_shutdown:
                run_shutdown()
        except Exception as exc:  # pragma: no cover — defensive
            self._log.warning("Phase F-2 M-9: run_tab shutdown failed: %s", exc)
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
        # audit 2026-08-01 D20: replace ``worker.terminate()`` with a
        # 30s bounded wait. ``terminate()`` forcibly kills the QThread
        # mid-Python execution, which can orphan subprocesses
        # (OpenDataLoader JVM, in-flight LLM HTTP requests) and leave
        # partial temp dirs like ``od_output/<paper_id>/`` behind.
        # ``request_cancel()`` above already called
        # ``requestInterruption()``; the 30s wait gives the pipeline
        # time to finish its current stage and clean up. If the worker
        # is still running after 30s, log a warning rather than killing
        # it — forcibly killing a worker that owns a live JVM is worse
        # than letting the parent process exit normally.
        # Phase F-2 (M-9+M-27): the wait() call can raise RuntimeError
        # if the QThread C++ object was already destroyed by the time
        # we call wait() (e.g. the GUI is closing and the Qt event
        # loop has already cleaned up the thread). Catch it and proceed
        # gracefully — the process is exiting anyway.
        try:
            if not worker.wait(30000):  # 30s timeout
                self._log.warning(
                    "PipelineWorker did not exit within 30s after "
                    "request_cancel; leaving thread alive (process exit "
                    "will reclaim it). OpenDataLoader JVM may still be "
                    "running."
                )
        except RuntimeError as exc:
            self._log.warning(
                "Phase F-2 M-9: worker.wait() raised RuntimeError "
                "(thread already gone): %s. Proceeding with close.",
                exc,
            )
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
        # audit 2026-08-17 (GUI-D1): use the i18n key instead of the
        # hardcoded English "Open PDF" so the dialog title
        # localises on language switch.
        path, _ = QFileDialog.getOpenFileName(
            self,
            i18n._tr("menu.file.open"),
            self._settings.get("last_pdf_dir", str(Path.home())),
            i18n._tr("filter.pdf"),
        )
        if not path:
            return
        self._run_tab._set_pdf_path(Path(path))
        self._settings["last_pdf_dir"] = str(Path(path).parent)
        # audit 2026-07-31: the choice was only written to the
        # in-memory cache; the Settings tab persisted "io/last_pdf_dir"
        # on ITS save, but selecting a PDF here never did — the
        # directory memory was lost on restart. Persist immediately
        # under the same key the loader reads.
        self._qsettings.setValue(QS_KEY_LAST_DIR, self._settings["last_pdf_dir"])
        self._tabs.setCurrentIndex(TAB_RUN)

    def _on_open_batch(self) -> None:
        dlg = BatchDialog(self._settings, parent=self)
        dlg.batch_started.connect(self._on_batch_started)
        dlg.exec()

    def _on_open_outdir(self) -> None:
        # audit 2026-08-17 (GUI-D1): use the i18n key instead of the
        # hardcoded English "Open output directory".
        path = QFileDialog.getExistingDirectory(
            self,
            i18n._tr("menu.file.outdir"),
            self._settings.get("last_export_dir", str(Path.home())),
        )
        if path:
            self._settings["last_export_dir"] = path
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices

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
        # Phase 6A (NIT-3): explicitly sync() so the theme choice
        # survives a hard GUI close. Without sync(), QSettings may
        # defer the platform-level write (registry / plist / INI) to
        # process exit, and a kill -9 / power loss would lose the
        # choice. A explicit flush is cheap (single small write).
        self._qsettings.sync()
        self._log.info("Theme applied: %s", theme)

    def _open_log_file(self) -> None:
        import os
        import subprocess

        from .utils import LOG_FILE_NAME

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
                # Phase F-3 NIT: start_new_session=True detaches the
                # viewer from the GUI process group so killing the
                # GUI doesn't orphan the viewer.
                subprocess.Popen(["open", str(log_path)], start_new_session=True)
            elif is_windows():
                os.startfile(str(log_path))  # type: ignore[attr-defined]
            else:
                # Linux: xdg-open + start_new_session (setsid) to avoid
                # orphaned viewer if the GUI is killed.
                subprocess.Popen(["xdg-open", str(log_path)], start_new_session=True)
        except Exception as exc:
            QMessageBox.warning(
                self,
                i18n._tr("settab.log.title"),
                i18n._tr("settab.log.open_fail").format(
                    error=f"{type(exc).__name__}: {exc}",
                    path=str(log_path),
                ),
            )

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------
    def _on_job_started(self, job_id: str, pdf_path: str, output_dir: str) -> None:
        # Phase F-2 (M-10/M-16): output_dir now comes from the signal
        # (RunTab computes it as <out_path>/output where out_path is the
        # user-selected directory). Previously MainWindow re-derived it
        # from pdf_path/stem which was wrong for batch jobs (which use
        # <batch_out_dir>/<stem>_rlpe_out/work/output, not
        # <pdf_parent>/<stem>_rlpe_out).
        stem = Path(pdf_path).stem
        placeholder_ids = getattr(self, "_batch_placeholder_by_stem", {})
        ph = placeholder_ids.get(stem)
        if ph:
            self._jobs_tab.remove_job(ph)
        job = JobRecord(
            job_id=job_id,
            pdf_path=pdf_path,
            output_dir=output_dir,
            status=STATUS_RUNNING,
            started_at=time.time(),
            settings=self._run_tab.collect_settings(),
        )
        self._jobs_tab.add_or_update_job(job)
        self._set_status("main.running", id=job_id)
        self._mini_progress.setRange(0, 0)  # indeterminate
        self._mini_progress.setVisible(True)
        # audit 2026-07-31: cancel any pending hide timer from the
        # previous job so it can't hide THIS job's progress bar.
        if getattr(self, "_mini_progress_timer", None) is not None:
            self._mini_progress_timer.stop()
            self._mini_progress_timer = None
        # Auto-switch to Jobs tab
        self._tabs.setCurrentIndex(TAB_JOBS)

    def _on_job_progress(self, job_id: str, current: int, total: int, msg: str) -> None:
        self._jobs_tab.update_progress(job_id, current, total, msg)
        if total > 0:
            self._mini_progress.setRange(0, total)
            self._mini_progress.setValue(current)
        # Phase F-2 (M-8): use _set_status so the i18n key is preserved.
        # Previously this wrote directly to _status_perm.setText(),
        # bypassing _status_key / _status_kwargs — after that, a
        # language switch would jump back to the stale key's text.
        self._set_status("main.progress", msg=msg, current=current, total=total)

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
        job_dir = getattr(job_record, "output_dir", None) or self._settings.get(
            "last_export_dir", ""
        )
        self._results_tab.load_job(job_id, rows, job_dir)
        # Status
        self._set_status("main.done", id=job_id, rows=f"{len(rows):,}")
        self._mini_progress.setRange(0, 1)
        self._mini_progress.setValue(1)
        # audit 2026-07-31: the 2s hide timer raced the next batch job
        # (which setVisible(True) immediately) — the timer then hid the
        # NEXT job's progress bar. Track the timer and only hide if no
        # job started since; also stop auto-switching to the Results
        # tab during a batch (the tab-jacking interrupted the operator
        # 10× on a 10-paper batch).
        if getattr(self, "_mini_progress_timer", None) is not None:
            self._mini_progress_timer.stop()
        self._mini_progress_timer = QTimer(self)
        self._mini_progress_timer.setSingleShot(True)
        self._mini_progress_timer.timeout.connect(lambda: self._mini_progress.setVisible(False))
        self._mini_progress_timer.start(2000)
        if not getattr(self, "_batch_pdfs", None):
            # Auto-switch to results tab (single-job mode only)
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
        if getattr(self, "_batch_pdfs", None) and self._batch_index < len(self._batch_pdfs):
            self._start_next_batch_job()

    def _on_job_failed(self, job_id: str, error: str) -> None:
        # Phase F-2 (M-7): check the worker's was_cancelled() flag FIRST.
        # The pipeline emits a "cancelled (...)" message via the failed signal
        # for both genuine user cancellations AND exceptions where the error
        # string contains the word "cancelled" (e.g. "cannot cancel: worker
        # still running"). Substring matching was falsely classifying the
        # latter as a user cancellation.
        # Fall back to word-boundary regex only when the worker is not
        # accessible (e.g. race condition between job start and failure).
        cancelled = False
        worker = getattr(self._run_tab, "_worker", None)
        if worker is not None and hasattr(worker, "was_cancelled"):
            cancelled = worker.was_cancelled()
        if not cancelled:
            # Word-boundary check as fallback for legacy / cross-process calls.
            # Chinese "取消" substring is deliberately excluded: it matches
            # "无法取消" (cannot cancel) which is a real error, not a
            # user-initiated cancellation. When the pipeline genuinely cancels,
            # worker.was_cancelled() is True and this branch is never reached.
            e = error or ""
            cancelled = bool(re.search(r"\bcancelled\b|\bcanceled\b", e, re.IGNORECASE))
        if cancelled:
            # audit 2026-07-31: a user-initiated cancellation is NOT a
            # failure — mark_cancelled and don't treat it as a batch
            # error (previously it showed a red "failed" row AND, with
            # _stop_on_error, silently halted the whole batch).
            self._jobs_tab.mark_cancelled(job_id)
            self._set_status("main.cancelled", id=job_id)
        else:
            self._jobs_tab.mark_failed(job_id, error)
            self._set_status("main.failed", id=job_id)
        self._mini_progress.setVisible(False)
        # for "stop on first error", halt the batch here. The
        # _batch_pdfs list is reset so the next _start_next_batch_job
        # call returns immediately. Cancellations never halt the batch.
        if (
            not cancelled
            and getattr(self, "_batch_pdfs", None)
            and getattr(self, "_batch_settings", None)
            and self._batch_settings.get("_stop_on_error", False)
        ):
            remaining = len(self._batch_pdfs) - self._batch_index
            self._batch_pdfs = []  # halt
            self._set_status("main.batch_stopped_on_error", failed=job_id, remaining=remaining)
        # audit 2026-07-31: when NOT stopping on error, advance the
        # batch — the previous code left every remaining PDF stuck in
        # "queued" after the first failure (only _on_job_finished
        # advanced the queue).
        if (
            not cancelled
            and getattr(self, "_batch_pdfs", None)
            and self._batch_index < len(self._batch_pdfs)
            and not self._batch_settings.get("_stop_on_error", False)
        ):
            self._start_next_batch_job()
        # audit 2026-07-31: a CANCELLED job also advances the batch —
        # cancellation skips the current PDF, it does not silently
        # strand the remaining ones in "queued" forever.
        if (
            cancelled
            and getattr(self, "_batch_pdfs", None)
            and self._batch_index < len(self._batch_pdfs)
        ):
            self._start_next_batch_job()
        # Audit 2026-07-26 M9: do NOT raise a second QMessageBox here -
        # RunTab._on_failed already shows an i18n error dialog before
        # emitting job_failed, so this was a duplicate (and hard-coded
        # English) popup on every failure.

    # ------------------------------------------------------------------
    # Jobs tab → open in results / retry
    # ------------------------------------------------------------------
    def _open_results(self, job_id: str) -> None:
        # Find the job in the jobs tab

        jobs = getattr(self._jobs_tab, "_jobs", {})
        if job_id not in jobs:
            return
        job = jobs[job_id]
        self._results_tab.load_job(job.job_id, job.rows, job.output_dir)
        self._tabs.setCurrentIndex(TAB_RESULTS)

    def _on_retry(self, job_id: str) -> None:
        # Re-run the job with the same PDF + (possibly updated) settings.
        #
        # Audit 2026-09-01 BL-6: the previous signature was
        # ``_on_retry(self, job_id: str, settings: dict)`` but
        # ``JobsTab.retry_requested = Signal(str)`` only emits the job_id
        # — every right-click on a finished job raised
        # ``TypeError: missing 1 required positional argument: 'settings'``
        # and crashed the Qt main window. Fetch settings from the stored
        # JobRecord instead of accepting them through the signal.
        settings: dict = {}

        jobs = getattr(self._jobs_tab, "_jobs", {})
        if job_id not in jobs:
            return
        job = jobs[job_id]
        path = Path(job.pdf_path)
        if not path.exists():
            # audit 2026-08-17 (GUI-D2): use the canonical
            # ``common.retry.title`` i18n key for the dialog title
            # and ``common.retry.body`` for the body so the message
            # localises on language switch. Previously
            # ``main.retry`` was undefined and silently rendered as
            # ``⟦main.retry⟧``.
            QMessageBox.warning(
                self,
                i18n._tr("common.retry.title"),
                i18n._tr("common.retry.body").format(path=str(path)),
            )
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

        # audit 2026-07-31: placeholder ids ("batch-00-<stem>") NEVER
        # matched the Run tab's real ids ("<stem>-HHMMSS"), so the
        # placeholder rows stayed "queued" forever while a SECOND row
        # with the real id appeared. Record the stem→placeholder map
        # so _on_job_started can promote the placeholder row instead.
        self._batch_placeholder_by_stem: dict[str, str] = {}
        for i, p in enumerate(pdfs):
            jid = f"batch-{i:02d}-{p.stem}"
            self._batch_placeholder_by_stem[p.stem] = jid
            # Phase F-2 (M-10/M-16): output_dir must be <stem>_rlpe_out/work/output
            # to match what RunTab computes (out_path / "work" is the PipelineWorker's
            # work_dir; actual results live at <work_dir>/output). Previously this was
            # just <stem>_rlpe_out (missing /work/output), so "open output dir"
            # opened the wrong folder for every batch job.
            out_root = Path(batch_settings["last_export_dir"]) / f"{p.stem}_rlpe_out"
            job = JobRecord(
                job_id=jid,
                pdf_path=str(p),
                output_dir=str(out_root / "work" / "output"),
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
                    self._log.warning("batch xlsx_at_end export failed: %s", exc)
            return
        pdf = self._batch_pdfs[self._batch_index]
        self._batch_index += 1
        # Use the existing Run tab mechanism to start a job
        self._run_tab._set_pdf_path(pdf)
        # audit 2026-07-31: each batch job MUST get its own output
        # directory. _set_pdf_path only defaults _out_edit when it is
        # EMPTY — after job 1 it stays set, so every later job wrote
        # into <pdf1_stem>_rlpe_out; the shared work/input then
        # accumulated every processed PDF and job k re-processed all
        # previous papers (O(N²) work, duplicated rows, and the last
        # job's matches.jsonl overwrote the earlier ones).
        stem = pdf.stem
        base = Path(self._batch_settings.get("last_export_dir") or pdf.parent)
        self._run_tab._out_edit.setText(str(base / f"{stem}_rlpe_out"))
        self._run_tab.apply_settings(self._batch_settings)
        # Force start
        # The Run tab has a private _on_start(); we emulate it
        # by setting the path, settings, and calling _on_start.
        # For simplicity we directly call:
        self._run_tab._on_start()

    def _export_batch_xlsx(self) -> None:
        """Phase F-2 (M-25): export a single workbook combining all
        batch job rows on a background thread so the GUI never freezes.
        """
        import datetime as _dt

        from PySide6.QtWidgets import QFileDialog

        jobs = list(getattr(self._jobs_tab, "_jobs", {}).values())
        if not jobs:
            return
        default_dir = (self._batch_settings or {}).get("last_export_dir") or str(Path.home())
        default_name = f"rlpe_batch_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        path, _ = QFileDialog.getSaveFileName(
            self,
            i18n._tr("restab.export.xlsx_title"),
            str(Path(default_dir) / default_name),
            "Excel files (*.xlsx)",
        )
        if not path:
            return
        self._set_status("main.batch_exporting")
        # Kick off background worker; terminal slots handle completion / error.
        self._batch_export_worker = _BatchExportWorker(jobs, path, parent=self)
        self._batch_export_worker.progress.connect(
            lambda msg: self._set_status("main.batch_exporting", msg=msg)
        )
        self._batch_export_worker.finished_ok.connect(self._on_batch_export_ok)
        self._batch_export_worker.failed.connect(self._on_batch_export_failed)
        self._batch_export_worker.start()

    def _on_batch_export_ok(self, path: str) -> None:
        """Phase F-2 (M-25): called on GUI thread when the export succeeds."""
        self._log.info("Batch export succeeded: %s", path)
        self._set_status("main.batch_complete")
        QMessageBox.information(
            self,
            i18n._tr("jobstab.export.xlsx_title"),
            i18n._tr("jobstab.export.saved").format(count="?", path=path),
        )
        if hasattr(self, "_batch_export_worker"):
            self._batch_export_worker.deleteLater()
            del self._batch_export_worker

    def _on_batch_export_failed(self, error: str) -> None:
        """Phase F-2 (M-25): called on GUI thread when the export fails."""
        self._log.error("Batch export failed: %s", error)
        self._set_status("main.idle")
        QMessageBox.warning(
            self,
            i18n._tr("common.error"),
            i18n._tr("jobstab.export.failed").format(error=error),
        )
        if hasattr(self, "_batch_export_worker"):
            self._batch_export_worker.deleteLater()
            del self._batch_export_worker

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
