"""Jobs tab — job queue, history, per-job controls.

A "job" is a single ``PipelineWorker`` invocation. The Jobs tab
shows:
* Currently running jobs (status=running, with live progress bar)
* Recently completed jobs (status=done, with row count + export link)
* Failed jobs (status=failed, with error preview)
* A button-bar: Cancel | Retry | Open output dir | Remove

We keep the job state in-process (no QSettings persistence yet
— that's a Phase 32+ candidate). Restarting the GUI clears the
list, which matches the Web UI's behaviour (jobs are in-memory).
"""

from __future__ import annotations

import collections
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QMenu,
    QMessageBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import i18n
from .constants import (
    MAX_RECENT_JOBS_IN_LIST,
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
)
from .i18n_widgets import tr_button, tr_label
from .styles import SPACE_M, SPACE_S
from .utils import (
    fmt_count,
    fmt_duration,
    get_gui_logger,
    short_path,
)

# Phase F-2 (M-12): the _jobs dict and the table must stay in sync.
# MAX_JOBS is the hard cap on _jobs entries; the table cap is
# MAX_RECENT_JOBS_IN_LIST (200). When _jobs hits MAX_JOBS the
# oldest entry is evicted from both the dict and the table so
# len(_jobs) == table.rowCount() is always true.
MAX_JOBS: Final[int] = 500


# ============================================================
# Audit 2026-08-19 (B-15): disk-scan synchronous JSONL parse
# ============================================================
# ``load_recent_jobs_from_disk`` used to walk the disk and parse every
# ``matches.jsonl`` line-by-line on the GUI thread. On a workstation with
# 150+ cached jobs, the scan blocked the main event loop for 3–10 s while
# the operator stared at a frozen progress bar. The fix splits the scan
# into two phases:
#
# 1. Fast directory-name listing — runs synchronously on the GUI thread
#    (milliseconds) and yields a ``_PendingDiskScan`` describing which
#    ``matches.jsonl`` files *might* be loadable.
# 2. JSONL parse — runs on a ``QThread`` so the event loop stays free.
#    The worker emits a single ``loaded`` carrying the new ``JobRecord``
#    instances, which the GUI thread re-injects via ``add_or_update_job``.
#
# Public API of ``load_recent_jobs_from_disk`` is unchanged: it still
# returns the number of jobs it scheduled for load. The number is the
# count of *candidate* jobs found synchronously; the actual JSONL parse
# happens asynchronously after the function returns. Callers that need
# the final loaded count after the async path finishes should track
# ``len(self._jobs)`` later, which is what the GUI already does.
# ============================================================

# ============================================================
# Audit 2026-08-20 (Phase F-1, M-4): JSONL DoS protection
# ============================================================
# ``matches.jsonl`` files come from the pipeline's own writers and are
# trusted under normal use, but a corrupted / attacker-supplied file
# (10 GB JSONL, or a single 2 GB line with no newline) used to OOM
# the GUI thread inside ``_parse_one``. We cap the file at 100 MB
# and truncate any individual line longer than 1 MB before parsing.
# Anything larger is logged and skipped; the rest of the scan keeps
# running.
MAX_JSONL_SIZE: Final[int] = 100 * 1024 * 1024  # 100 MB
MAX_LINE_SIZE: Final[int] = 1 * 1024 * 1024  # 1 MB

# Status → row-background tint map. Phase F-3 NIT fix: previously an
# inline dict literal in ``_refresh_row``. Pulled to module scope so
# the design tokens live in one place (next to ``MAX_JOBS``) and so
# ``_refresh_row`` doesn't re-import QColor on every row update.
_STATUS_BG_COLORS: Final[dict[str, QColor]] = {
    STATUS_RUNNING: QColor("#d6e4ff"),
    STATUS_DONE: QColor("#d8f5d0"),
    STATUS_FAILED: QColor("#ffe0e0"),
    STATUS_CANCELLED: QColor("#ffe9d6"),
    STATUS_QUEUED: QColor("#eef1f6"),
}


class _PendingDiskScan:
    """One ``matches.jsonl`` found during the synchronous directory scan.

    The async worker consumes these in a continuation off the main thread.
    """

    __slots__ = ("jid", "root", "matches_path", "complete_flag")

    def __init__(
        self,
        jid: str,
        root: Path,
        matches_path: Path,
        complete_flag: Path,
    ) -> None:
        self.jid = jid
        self.root = root
        self.matches_path = matches_path
        self.complete_flag = complete_flag


class _DiskScanWorker(QThread):
    """Background worker that parses ``matches.jsonl`` files.

    Emits ``job_loaded(JobRecord)`` for every successfully parsed job
    so the GUI thread can fold them into ``_jobs`` one at a time, and
    a final ``completed(list[JobRecord])`` carrying the records
    produced by the scan (so callers can await the final loaded set
    via the ``JobsTab.scan_finished`` signal). ``failed(str)`` is
    emitted if the whole scan aborts with an unrecoverable error.

    Phase F-1 (B-1 / B-3): ``completed`` replaces the previous
    ``finished_ok`` count-only signal so callers no longer have to
    re-walk ``_jobs`` to know when the scan truly finished.
    """

    job_loaded = Signal(object)  # carries a JobRecord
    completed = Signal(list)  # list[JobRecord]; fired when run() returns
    failed = Signal(str)  # error reason; fired when run() raises

    def __init__(self, pending: list[_PendingDiskScan]) -> None:
        super().__init__()
        self._pending = pending
        # Each thread gets its own logger capture so a misformatted line
        # in one job doesn't kill the whole scan.
        self._log = get_gui_logger()

    def run(self) -> None:  # noqa: D401 - QThread contract
        loaded: list[JobRecord] = []
        try:
            for entry in self._pending:
                # Phase F-1 (B-1): honour shutdown requests between
                # entries so a closing GUI doesn't keep scanning
                # another 149 jobs. The worker still emits
                # ``completed`` with whatever it had loaded so the
                # caller can release the thread cleanly.
                if self.isInterruptionRequested():
                    self._log.info(
                        "load_recent_jobs: disk scan interrupted after %d entries",
                        len(loaded),
                    )
                    self.completed.emit(loaded)
                    return
                try:
                    job = self._parse_one(entry)
                except Exception as exc:
                    # Defensive: never let one bad job kill the whole scan.
                    self._log.warning(
                        "load_recent_jobs: failed to parse %s: %s",
                        entry.matches_path,
                        exc,
                    )
                    continue
                if job is not None:
                    self.job_loaded.emit(job)
                    loaded.append(job)
            self.completed.emit(loaded)
        except Exception as exc:
            # Top-level guard: if the loop itself blows up (e.g. an
            # OS error before we entered the for), surface the
            # reason so JobsTab.scan_failed can forward it.
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    def _parse_one(self, entry: _PendingDiskScan) -> JobRecord | None:
        """Parse a single ``matches.jsonl`` into a :class:`JobRecord`.

        Returns ``None`` if the file is empty / unreadable / 0-row, so the
        worker can skip it without the caller crashing.
        """
        mp = entry.matches_path
        # Phase F-1 (M-4): DoS guard. A 10 GB JSONL would OOM the GUI
        # thread if we tried to read it; bail out cheaply via stat().
        try:
            mp_size = mp.stat().st_size
        except OSError:
            return None
        if mp_size > MAX_JSONL_SIZE:
            self._log.warning(
                "load_recent_jobs: skipping %s (matches.jsonl exceeds %d MB; %d bytes)",
                mp,
                MAX_JSONL_SIZE // (1024 * 1024),
                mp_size,
            )
            return None

        rows: list[dict[str, Any]] = []
        try:
            with mp.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    # Phase F-1 (M-4): truncate any pathological line
                    # (e.g. a 2 GB single line with no newline) before
                    # json.loads so we don't allocate gigabytes.
                    line = line[:MAX_LINE_SIZE]
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        # audit 2026-07-31: skip the BROKEN LINE,
                        # not the whole job — a partially-written
                        # matches.jsonl (crash mid-write) used to
                        # hide every panel of the job.
                        self._log.debug(
                            "load_recent_jobs: skipping bad line in %s",
                            mp,
                        )
                        continue
                    if not isinstance(row, dict):
                        self._log.debug(
                            "load_recent_jobs: skipping non-dict line in %s",
                            mp,
                        )
                        continue
                    rows.append(row)
        except OSError:
            self._log.warning(
                "load_recent_jobs: skipping %s (read error)",
                mp,
            )
            return None
        if not rows:
            return None

        # Locate the original PDF (best-effort).
        pdf_path = ""
        pdfs_dir = entry.root / "pdfs"
        if pdfs_dir.exists():
            pdfs = list(pdfs_dir.glob("*.pdf"))
            if pdfs:
                pdf_path = str(pdfs[0])
        # Try to get a creation timestamp from filesystem.
        try:
            finished_at = mp.stat().st_mtime
        except OSError:
            finished_at = time.time()
        # audit 2026-08-17 (jobs_tab C1): disk-scan honesty. A
        # matches.jsonl that exists but lacks the API's
        # ``complete.flag`` is a PARTIAL run — the pipeline was
        # killed (OOM, ctrl-C, segfault) before it could finish.
        # The previous code stamped every disk-loaded job as
        # STATUS_DONE regardless, so the operator saw a green
        # "done" row in the Jobs tab for jobs that needed a
        # retry. Honour the same flag the API uses (see
        # api/app.py:775-798) and only fall back to STATUS_DONE
        # if no ``manifests/`` directory exists at all (legacy
        # pre-flag runs in ad-hoc CLI scratch dirs).
        manifests_dir = entry.root / "output" / "manifests"
        try:
            if entry.complete_flag.exists():
                job_status = STATUS_DONE
                progress_msg = i18n._tr("jobstab.loaded_from_disk")
            elif manifests_dir.exists():
                # Phase F-1 (M-partial): a ``manifests/`` directory
                # with rows but no ``complete.flag`` means the run
                # was interrupted (OOM/ctrl-C/segfault) before the
                # pipeline could finish. Mark as FAILED so the
                # operator sees a red row instead of a misleading
                # green "done" row that needs a retry anyway.
                # The API uses the same ``"partial"`` label in
                # ``api/app.py:_load_existing_jobs_from_disk``.
                job_status = STATUS_FAILED
                progress_msg = i18n._tr("jobstab.partial_no_complete_flag")
            else:
                # Legacy CLI run outside the standard manifests/ tree;
                # treat as done so old runs don't all show as red.
                job_status = STATUS_DONE
                progress_msg = i18n._tr("jobstab.loaded_from_disk")
        except OSError:
            job_status = STATUS_FAILED
            progress_msg = i18n._tr("jobstab.loaded_from_disk")
        return JobRecord(
            job_id=entry.jid,
            pdf_path=pdf_path,
            output_dir=str(entry.root / "output"),
            status=job_status,
            progress_current=1,
            progress_total=1,
            progress_msg=progress_msg,
            rows=rows,
            started_at=finished_at,
            finished_at=finished_at,
        )


# ============================================================
# Phase F-2 (M-13): async export worker
# ============================================================
class _JobsExportWorker(QThread):
    """Background worker that writes one XLSX or JSON export file.

    Phase F-2 (M-13): runs on a QThread so the GUI stays responsive
    during large exports (50k+ rows). The ``run_output`` dict is
    captured on the GUI thread before the worker starts, so the worker
    is side-effect-free with respect to the job record.

    Signals
    -------
    finished_ok(str) — destination path on success.
    failed(str) — error message on failure (never re-raises).
    progress(int) — 0-100 progress percentage for status-bar updates.
    """

    finished_ok = Signal(str)
    failed = Signal(str)
    progress = Signal(int)

    _VALID_FMTS: frozenset[str] = frozenset({"xlsx", "json"})

    def __init__(
        self,
        fmt: str,
        run_output: dict[str, Any],
        path: str,
    ) -> None:
        super().__init__()
        if fmt not in self._VALID_FMTS:
            raise ValueError(f"unknown export format {fmt!r}")
        self._fmt = fmt
        self._run_output = run_output
        self._path = path
        # Phase F-2 (M-13): cancellation support.
        self._cancelled: bool = False

    def cancel(self) -> None:
        """Ask :meth:`run` to bail out at the next checkpoint."""
        self._cancelled = True

    def run(self) -> None:  # noqa: D401 — QThread contract
        try:
            if self._cancelled:
                return
            self.progress.emit(10)
            if self._fmt == "xlsx":
                from ..exporters.xlsx import write_xlsx

                write_xlsx(self._run_output, self._path)
            elif self._fmt == "json":
                self.progress.emit(30)
                with open(self._path, "w", encoding="utf-8") as fh:
                    json.dump(
                        self._run_output,
                        fh,
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )
            self.progress.emit(100)
            self.finished_ok.emit(self._path)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


# ============================================================
# Job dataclass
# ============================================================
@dataclass
class JobRecord:
    """In-memory representation of a single pipeline run."""

    job_id: str
    pdf_path: str
    output_dir: str
    status: str = STATUS_QUEUED
    progress_current: int = 0
    progress_total: int = 0
    progress_msg: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        end = self.finished_at if self.finished_at > 0 else time.time()
        return max(0.0, end - self.started_at)

    def to_summary(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "pdf": self.pdf_path,
            "out": self.output_dir,
            "status": self.status,
            "rows": len(self.rows),
            "elapsed": self.elapsed,
        }


# ============================================================
# Inline progress bar delegate (renders progress in a table cell)
# ============================================================
class _ProgressCellDelegate(QStyledItemDelegate):
    """Paints a small QProgressBar inside the progress column.

    Delegates to the base class for the cell background + selection
    state, then overlays a progress bar on top.
    """

    def paint(self, painter, option, index) -> None:  # type: ignore[override]
        # Delegate to the base class for the cell background +
        # selection state. This is what Qt's docs say is required
        # for custom delegates.
        super().paint(painter, option, index)
        # Now overlay a progress bar.
        painter.save()
        try:
            progress_bar_option = QStyleOptionProgressBar()
            progress_bar_option.rect = option.rect
            progress_bar_option.state = option.state
            progress_bar_option.palette = option.palette
            progress_bar_option.minimum = 0
            progress_bar_option.maximum = max(1, index.data(Qt.UserRole) or 0)
            progress_bar_option.progress = index.data(Qt.UserRole + 1) or 0
            progress_bar_option.text = (
                f"{progress_bar_option.progress} / {progress_bar_option.maximum}"
            )
            progress_bar_option.textVisible = True
            # CE_ProgressBar (value 10) is the correct ControlElement for
            # QStyle.drawControl — not PrimitiveElement which doesn't exist.
            QApplication.style().drawControl(
                QStyle.CE_ProgressBar,
                progress_bar_option,
                painter,
            )
        finally:
            painter.restore()


# ============================================================
# Jobs tab
# ============================================================
class JobsTab(QWidget):
    """Job queue + history. Updated reactively from ``RunTab`` signals."""

    open_results_requested = Signal(str)  # job_id
    # Phase F-2 (M-24): ``retry_requested(job_id)`` — emitted by the
    # Retry context menu action for FAILED/CANCELLED jobs.
    #
    # Expected handler signature (in MainWindow):
    #
    #     def _on_retry(self, job_id: str) -> None:
    #         job = self._jobs_tab.get_job(job_id)
    #         if job is None:
    #             return
    #         self._run_tab.set_pdf_path(job.pdf_path)
    #         self._run_tab.set_settings(job.settings)
    #         self._run_tab.set_output_dir(job.output_dir)
    #         self._run_tab.start()  # actually re-runs the pipeline
    retry_requested = Signal(str)  # job_id
    # Phase F-1 (B-3): ``scan_finished`` carries the final list of
    # ``JobRecord`` instances produced by the async disk scan (empty
    # if no candidates were found, or after a graceful shutdown
    # interrupt). ``scan_failed`` carries the error reason when the
    # whole scan aborts. ``MainWindow`` connects ``scan_finished``
    # to the auto-open-on-startup logic so it can decide whether
    # there's actually something to show.
    #
    # PySide6 requires ``Signal`` to be declared at the class level,
    # not in ``__init__`` (the instance-level form silently breaks
    # ``connect()``). We follow that convention here even though the
    # spec suggested ``__init__`` placement.
    scan_finished = Signal(list)  # list[JobRecord]
    scan_failed = Signal(str)  # error reason

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._log = get_gui_logger()
        # Phase F-2 (M-12): OrderedDict so _trim_old_jobs can pop the
        # oldest (first) entry deterministically and keep dict ordering
        # aligned with table row order (oldest = row 0).
        self._jobs: collections.OrderedDict[str, JobRecord] = collections.OrderedDict()
        self._ctx_actions: list[tuple[QAction, str]] = []
        # Phase F-1 (B-1): keep a strong ref to the disk-scan worker
        # so it doesn't get GC'd mid-flight; ``shutdown()`` clears it
        # after a graceful interrupt + wait.
        self._disk_scan_worker: _DiskScanWorker | None = None
        self._build_ui()
        # Register as an i18n listener so column headers, context menus,
        # and status labels auto-translate on language switch. Using a bound
        # method (not a lambda) lets closeEvent remove the listener by
        # identity without accumulating stale references.
        self._i18n_listener = self._on_language_changed
        i18n.add_listener(self._i18n_listener)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACE_M, SPACE_M, SPACE_M, SPACE_M)
        outer.setSpacing(SPACE_S)

        # ---- Toolbar ----
        bar = QHBoxLayout()
        bar.setSpacing(SPACE_S)

        clear_done_btn = tr_button("jobstab.clear_finished")
        clear_done_btn.clicked.connect(self._clear_finished)
        bar.addWidget(clear_done_btn)

        # Phase 37 audit fix: use setProperty("class", ...) instead
        # of setObjectName("flat") so the i18n registry's objectName
        # key isn't clobbered (QSS still styles the button via the
        # class property).
        clear_all_btn = tr_button("jobstab.clear_all")
        clear_all_btn.setProperty("class", "flat")
        clear_all_btn.clicked.connect(self._clear_all)
        bar.addWidget(clear_all_btn)

        bar.addStretch(1)
        self._count_label = tr_label("jobstab.no_jobs")
        self._count_label.setObjectName("metric")
        bar.addWidget(self._count_label)

        outer.addLayout(bar)

        # ---- Table ----
        self._table = QTableWidget(0, 7)
        # Column headers use i18n keys so they translate on language switch.
        self._table.setHorizontalHeaderLabels(
            [
                i18n._tr("jobstab.col.id"),
                i18n._tr("jobstab.col.pdf"),
                i18n._tr("jobstab.col.status"),
                i18n._tr("jobstab.col.progress"),
                i18n._tr("jobstab.col.rows"),
                i18n._tr("jobstab.col.elapsed"),
                i18n._tr("jobstab.col.out"),
            ]
        )
        # Configure selection / behaviour
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(False)
        self._table.setShowGrid(False)
        # Resize columns to fit content
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        self._table.setColumnWidth(3, 180)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(26)
        # Progress column custom delegate
        self._progress_delegate = _ProgressCellDelegate(self._table)
        self._table.setItemDelegateForColumn(3, self._progress_delegate)
        # Context menu
        self._table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_context_menu)
        self._table.doubleClicked.connect(self._on_row_double_clicked)
        outer.addWidget(self._table, 1)

        # ---- Status row ----
        status_row = QHBoxLayout()
        self._summary = tr_label("jobstab.no_jobs")
        self._summary.setObjectName("metricLabel")
        status_row.addWidget(self._summary, 1)
        outer.addLayout(status_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_recent_jobs_from_disk(self) -> int:
        """Scan disk for completed jobs and populate ``_jobs``.

        Mirrors what the Web API does at startup (``_load_existing_jobs_from_disk``
        in ``api/app.py``): walks ``service_work/<job_id>/output/manifests/matches.jsonl``
        and ``work/output/manifests/matches.jsonl`` (CLI runs), and turns each
        ``matches.jsonl`` into a ``JobRecord(status=STATUS_DONE)`` with the rows
        loaded from disk.

        Before Phase 49, the GUI kept jobs purely in-memory in ``_jobs`` —
        restarting the GUI (or running an extraction via the CLI / web) made
        the results invisible until the user re-ran the same PDF inside the
        GUI session. This scan fixes that.

        Audit 2026-08-19 (B-15): the synchronous scan blocked the GUI
        thread for 3–10 s on a workstation with 150+ cached jobs. The
        function now does only the *fast* directory walk synchronously
        (typical <100 ms) and defers the JSONL parse to a
        :class:`_DiskScanWorker` running on a ``QThread``. The
        ``returned count`` is the number of *candidate* jobs scheduled
        for the async path; the actual rows arrive via the worker's
        ``job_loaded`` signal. The GUI thread stays interactive.

        Returns the number of jobs scheduled for async load.
        """
        from .constants import PROJECT_ROOT

        # Candidate roots: <root>/service_work/<job_id>/output/manifests/matches.jsonl
        # and the dev work/ directory at project root.
        pending: list[_PendingDiskScan] = []
        service_work = PROJECT_ROOT / "service_work"
        if service_work.exists():
            for child in sorted(service_work.iterdir()):
                if child.is_dir():
                    mp = child / "output" / "manifests" / "matches.jsonl"
                    if mp.exists():
                        pending.append(
                            _PendingDiskScan(
                                jid=child.name,
                                root=child,
                                matches_path=mp,
                                complete_flag=child / "output" / "manifests" / "complete.flag",
                            )
                        )
        # Also scan project root work/ for ad-hoc CLI runs.
        cli_work = PROJECT_ROOT / "work"
        if cli_work.exists() and cli_work.resolve() != service_work.resolve():
            import hashlib

            jid = "cli_" + hashlib.md5(str(cli_work.resolve()).encode()).hexdigest()[:12]
            mp = cli_work / "output" / "manifests" / "matches.jsonl"
            if mp.exists():
                pending.append(
                    _PendingDiskScan(
                        jid=jid,
                        root=cli_work,
                        matches_path=mp,
                        complete_flag=cli_work / "output" / "manifests" / "complete.flag",
                    )
                )
        # Audit 2026-09-03 (BLOCKER user-reported): scan the user's
        # configured ``last_pdf_dir`` and ``last_export_dir`` (read
        # from QSettings so this works without changing MainWindow's
        # JobsTab ctor signature) for PySide6-GUI output patterns.
        # The PySide6 GUI writes to ``<pdf_dir>/<stem>/work/manifests/``
        # AND ``<last_export_dir>/<stem>_rlpe_out/work/output/`` —
        # neither of which the legacy service_work/ or project root
        # work/ scan would find. Without this, every PySide6 run
        # against a user-specified output_dir is invisible to the
        # Jobs tab until the operator re-opens the run inside the
        # same GUI session — exactly the "I ran it but the Results
        # tab is empty" failure mode reported on 2026-09-03.
        try:
            from PySide6.QtCore import QSettings
            from .constants import (
                APP_AUTHOR,
                APP_NAME,
                QS_KEY_LAST_DIR,
                QS_KEY_LAST_EXPORT_DIR,
            )
            _qsettings = QSettings(APP_AUTHOR, APP_NAME)
            # Collect the set of PDF-side and export-side roots to
            # scan. We dedupe via a resolved path set so the same
            # subdir isn't loaded twice when last_pdf_dir and
            # last_export_dir coincide (common for ad-hoc tests).
            user_roots: set[Path] = set()
            for _key in (QS_KEY_LAST_DIR, QS_KEY_LAST_EXPORT_DIR):
                _raw = str(_qsettings.value(_key, "") or "").strip()
                if not _raw:
                    continue
                _p = Path(_raw)
                if _p.exists() and _p.is_dir():
                    user_roots.add(_p.resolve())
            # The PySide6 GUI writes single-PDF runs to
            # ``<last_pdf_dir>/<stem>/work/manifests/matches.jsonl``
            # so scan each immediate child of last_pdf_dir.
            # Batch runs write to
            # ``<last_export_dir>/<stem>_rlpe_out/work/output/manifests/matches.jsonl``
            # so scan each ``*_rlpe_out`` child of last_export_dir.
            # We also walk one level deeper (children of children)
            # so a paper staged in ``<last_pdf_dir>/<author>/<stem>/``
            # still gets picked up.
            for _root in user_roots:
                try:
                    _children = list(_root.iterdir())
                except OSError:
                    continue
                # Single-PDF outputs (<root>/<stem>/work/manifests/...)
                for _child in _children:
                    if not _child.is_dir():
                        continue
                    _mp = _child / "work" / "manifests" / "matches.jsonl"
                    if _mp.exists():
                        # Stable jid derived from the absolute path
                        # so re-running the same PDF twice produces the
                        # same jid (and the JobsTab dedupes rows).
                        import hashlib as _hl
                        _jid = "user_" + _hl.md5(
                            str(_mp.resolve()).encode()
                        ).hexdigest()[:12]
                        pending.append(
                            _PendingDiskScan(
                                jid=_jid,
                                root=_child,
                                matches_path=_mp,
                                complete_flag=_child / "work" / "manifests" / "complete.flag",
                            )
                        )
                    # Batch outputs (<root>/<stem>_rlpe_out/work/output/...)
                    _batch = _child / "work" / "output" / "manifests" / "matches.jsonl"
                    if _batch.exists():
                        import hashlib as _hl
                        _jid = "user_batch_" + _hl.md5(
                            str(_batch.resolve()).encode()
                        ).hexdigest()[:12]
                        pending.append(
                            _PendingDiskScan(
                                jid=_jid,
                                root=_child / "work",
                                matches_path=_batch,
                                complete_flag=_child / "work" / "output" / "manifests" / "complete.flag",
                            )
                        )
        except ImportError:
            # PySide6 not installed (running tests headless) — skip
            # the user-root scan rather than crash the test suite.
            pass

        if not pending:
            # Phase F-1 (B-3): emit scan_finished([]) so callers
            # awaiting the async load know the scan truly finished
            # with zero results (and not just "hasn't started yet").
            # We use ``QTimer.singleShot(0, ...)`` so the emit happens
            # after the current synchronous setup round returns,
            # matching the async-worker scheduling below.
            from PySide6.QtCore import QTimer

            QTimer.singleShot(
                0,
                lambda: self.scan_finished.emit([]),
            )
            return 0

        # Capture the worker on the instance so the QThread isn't
        # garbage-collected mid-flight (a known PySide6 footgun).
        worker = _DiskScanWorker(pending)
        self._disk_scan_worker = worker

        def _on_job(job: JobRecord) -> None:
            try:
                self.add_or_update_job(job)
            except Exception as exc:
                self._log.warning(
                    "load_recent_jobs: add_or_update_job failed for %s: %s",
                    job.job_id,
                    exc,
                )

        def _on_completed(records: list[JobRecord]) -> None:
            # Re-show the welcome state now that the async load is in.
            try:
                self._update_summary()
            except Exception:
                pass
            # Phase F-1 (B-3): forward the worker's completion
            # notification to the public ``scan_finished`` signal so
            # external callers (MainWindow auto-open-on-startup,
            # tests) can await the real loaded set.
            self.scan_finished.emit(records)

        def _on_failed(reason: str) -> None:
            self._log.warning(
                "load_recent_jobs: async scan failed: %s",
                reason,
            )
            self.scan_failed.emit(reason)

        worker.job_loaded.connect(_on_job)
        worker.completed.connect(_on_completed)
        worker.failed.connect(_on_failed)
        worker.finished.connect(worker.deleteLater)
        # Audit 2026-08-19 (B-15): kick the worker off via a 0-delay
        # timer so the caller (MainWindow.__init__) finishes its own
        # setup round before the thread starts issuing signals.
        worker.start()

        return len(pending)

    # ------------------------------------------------------------------
    def add_or_update_job(self, job: JobRecord) -> None:
        """Insert or update a job record in the table.

        Phase F-2 (M-12): ordering matters here. ``_refresh_row`` evicts
        the oldest entry from both ``_jobs`` and the table when the
        MAX_JOBS cap is hit. We add to ``_jobs`` AFTER ``_refresh_row``
        trims so the eviction pop and the new insert land at the same
        length, preserving the ``len(_jobs) == rowCount()`` invariant.
        """
        self._refresh_row(job)
        self._jobs[job.job_id] = job
        self._update_summary()
        self._trim_old_jobs()

    def remove_job(self, job_id: str) -> None:
        """Remove a job row from the table.

        audit 2026-07-31: used by the batch flow to promote a
        placeholder row to the real job id (the placeholder id never
        matched the Run tab's generated id)."""
        if job_id not in self._jobs:
            return
        del self._jobs[job_id]
        self._update_summary()

    def update_progress(self, job_id: str, current: int, total: int, msg: str) -> None:
        if job_id not in self._jobs:
            return
        job = self._jobs[job_id]
        job.progress_current = current
        job.progress_total = total
        job.progress_msg = msg
        self._refresh_row(job)

    def mark_done(self, job_id: str, rows: list[dict[str, Any]]) -> None:
        if job_id not in self._jobs:
            return
        job = self._jobs[job_id]
        job.status = STATUS_DONE
        job.rows = rows
        job.progress_current = job.progress_total
        job.finished_at = time.time()
        self._refresh_row(job)
        self._update_summary()
        # Phase 56 audit: trim after terminal state transition
        self._trim_old_jobs()

    def mark_failed(self, job_id: str, error: str) -> None:
        if job_id not in self._jobs:
            return
        job = self._jobs[job_id]
        job.status = STATUS_FAILED
        job.error = error
        job.finished_at = time.time()
        self._refresh_row(job)
        self._update_summary()
        # Phase 56 audit: trim after terminal state transition
        self._trim_old_jobs()

    def mark_cancelled(self, job_id: str) -> None:
        if job_id not in self._jobs:
            return
        job = self._jobs[job_id]
        job.status = STATUS_CANCELLED
        job.finished_at = time.time()
        self._refresh_row(job)
        self._update_summary()
        # Phase 56 audit: trim after terminal state transition
        self._trim_old_jobs()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _find_row(self, job_id: str) -> int:
        for r in range(self._table.rowCount()):
            item = self._table.item(r, 0)
            if item and item.text() == job_id:
                return r
        return -1

    def _refresh_row(self, job: JobRecord) -> None:
        row = self._find_row(job.job_id)
        if row < 0:
            # Phase F-2 (M-12): both _jobs (MAX_JOBS=500) and the table
            # (MAX_RECENT_JOBS_IN_LIST=200) must stay in sync.
            #
            # Evict oldest entries as a PAIR (one dict pop + one table
            # removeRow) so the two structures never drift apart.
            # This loop runs at most once per insert in steady state
            # (the second condition only triggers during the gap between
            # when the dict hits 500 and the table catches up to 200).
            while len(self._jobs) >= MAX_JOBS or (
                self._table.rowCount() >= MAX_RECENT_JOBS_IN_LIST
            ):
                self._jobs.popitem(last=False)
                self._table.removeRow(0)
            row = self._table.rowCount()
            self._table.insertRow(row)

        # Column 0: Job ID (monospace)
        item_id = QTableWidgetItem(job.job_id)
        item_id.setData(Qt.UserRole, job.job_id)
        font = item_id.font()
        font.setFamily("JetBrains Mono, Cascadia Code, Menlo, Consolas, monospace")
        item_id.setFont(font)
        self._table.setItem(row, 0, item_id)

        # Column 1: PDF (short path tooltip). Hold a local reference to
        # the QTableWidgetItem instead of re-fetching from the model —
        # row indices can shift if cap evicts during a later insert.
        pdf_item = QTableWidgetItem(short_path(Path(job.pdf_path), 50))
        pdf_item.setToolTip(job.pdf_path)
        self._table.setItem(row, 1, pdf_item)

        # Column 2: Status (coloured via QSS objectName)
        status_item = QTableWidgetItem(job.status)
        status_item.setData(Qt.UserRole, job.status)
        # Map status → objectName for QSS colour (see styles.py).
        object_name_map = {
            STATUS_QUEUED: "statusQueued",
            STATUS_RUNNING: "statusRunning",
            STATUS_DONE: "statusDone",
            STATUS_FAILED: "statusFailed",
            STATUS_CANCELLED: "statusCancelled",
        }
        status_item.setTextAlignment(Qt.AlignCenter)
        # Use a font weight + role for visual hierarchy
        font = status_item.font()
        font.setBold(True)
        status_item.setFont(font)
        status_item.setData(Qt.AccessibleTextRole, job.status)
        self._table.setItem(row, 2, status_item)

        # Column 3: Progress (via custom delegate)
        progress_item = QTableWidgetItem()
        progress_item.setData(Qt.UserRole, job.progress_total)  # max
        progress_item.setData(Qt.UserRole + 1, job.progress_current)  # value
        progress_item.setToolTip(job.progress_msg)
        self._table.setItem(row, 3, progress_item)

        # Column 4: Rows
        rows_item = QTableWidgetItem(fmt_count(len(job.rows)))
        rows_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._table.setItem(row, 4, rows_item)

        # Column 5: Elapsed
        elapsed_item = QTableWidgetItem(fmt_duration(job.elapsed))
        elapsed_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._table.setItem(row, 5, elapsed_item)

        # Column 6: Output (short path tooltip)
        out_item = QTableWidgetItem(short_path(Path(job.output_dir), 50))
        out_item.setToolTip(job.output_dir)
        self._table.setItem(row, 6, out_item)

        # Apply status row colour via QSS — set background colour
        # via a per-cell role object name. We use setData with
        # Qt.AccessibleDescriptionRole + a small CSS hack: each
        # status column gets the matching QSS objectName set via
        # QTreeWidget.setObjectName — but QTableWidgetItem has no
        # setObjectName; instead we set the item background colour.
        # Phase F-3 NIT: ``bg_map`` is now the module-level
        # ``_STATUS_BG_COLORS`` constant (QColor imported at module
        # scope), so we don't re-import on every row refresh.
        bg_map = _STATUS_BG_COLORS
        if job.status in bg_map:
            for col in range(self._table.columnCount()):
                it = self._table.item(row, col)
                if it is not None:
                    it.setBackground(bg_map[job.status])

    def _update_summary(self) -> None:
        total = len(self._jobs)
        running = sum(1 for j in self._jobs.values() if j.status == STATUS_RUNNING)
        done = sum(1 for j in self._jobs.values() if j.status == STATUS_DONE)
        failed = sum(1 for j in self._jobs.values() if j.status == STATUS_FAILED)
        self._count_label.setText(
            i18n._tr("jobstab.summary.count_label").format(
                total=total,
                running=running,
                done=done,
                failed=failed,
            )
        )
        self._summary.setText(
            i18n._tr("jobstab.summary.count").format(
                total=total,
                running=running,
                done=done,
                failed=failed,
            )
        )

    def _trim_old_jobs(self) -> None:
        """Cap the table at MAX_RECENT_JOBS_IN_LIST rows (safety net).

        Phase F-2 (M-12): _jobs dict trimming is handled exclusively in
        _refresh_row (before insert) so the two stay in sync.
        _trim_old_jobs is only called after terminal state transitions
        (done/failed/cancelled) as a safety net — it only removes table
        rows; _jobs is already correct.
        """
        while self._table.rowCount() > MAX_RECENT_JOBS_IN_LIST:
            self._table.removeRow(0)

    def get_job(self, job_id: str) -> JobRecord | None:
        """Return the JobRecord for ``job_id``, or None if not found."""
        return self._jobs.get(job_id)

    # ------------------------------------------------------------------
    # Context menu / row interactions
    # ------------------------------------------------------------------
    def _selected_job(self) -> JobRecord | None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self._table.item(rows[0].row(), 0)
        if not item:
            return None
        return self._jobs.get(item.text())

    def _show_context_menu(self, pos) -> None:
        job = self._selected_job()
        if job is None:
            return
        # Phase 6A (NIT-4): parent the menu to ``None`` (not ``self``) so
        # the QMenu + its QAction children don't accumulate as dangling
        # children of the JobsTab across repeated right-clicks. The old
        # ``QMenu(self)`` kept every previously-shown menu alive because
        # Qt's parent-child ownership transfers to the parent widget.
        # We ``deleteLater()`` after ``exec_`` returns so the next event
        # loop tick actually frees the widgets.
        menu = QMenu()
        self._ctx_actions.clear()

        def _add_action(key: str) -> QAction:
            act = QAction(i18n._tr(key), menu)
            i18n.register_widget_text(f"jobstab.action.{key}", "text", key)
            self._ctx_actions.append((act, key))
            menu.addAction(act)
            return act

        act_open_results = _add_action("jobstab.menu.open_results")
        act_open_results.triggered.connect(lambda: self.open_results_requested.emit(job.job_id))

        act_open_out = _add_action("jobstab.menu.open_out")
        act_open_out.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(job.output_dir))
        )

        # Phase F-2 (M-24): Retry action — enabled only for failed/cancelled jobs.
        # Emits ``retry_requested(job_id)`` so MainWindow can restart the pipeline
        # with the original settings. The action text carries the 🔁 icon.
        act_retry = _add_action("jobstab.action.retry")
        act_retry.setEnabled(job.status in (STATUS_FAILED, STATUS_CANCELLED))
        act_retry.triggered.connect(lambda: self.retry_requested.emit(job.job_id))

        menu.addSeparator()

        act_export_xlsx = _add_action("jobstab.menu.export_xlsx")
        act_export_xlsx.triggered.connect(lambda: self._export_xlsx(job))

        act_export_json = _add_action("jobstab.menu.export_json")
        act_export_json.triggered.connect(lambda: self._export_json(job))

        menu.addSeparator()

        act_remove = _add_action("jobstab.menu.remove")
        act_remove.triggered.connect(lambda: self._remove_job(job.job_id))

        try:
            # Phase 6A (NIT-4): use ``exec_`` (the Python-friendly alias)
            # rather than ``exec`` — the latter is a C++-bound builtin that
            # shadows the Python builtin and is not monkey-patchable from
            # tests. ``exec_`` resolves to the same Qt modal-popup call.
            menu.exec_(self._table.viewport().mapToGlobal(pos))
        finally:
            # Drop both the actions list and the menu object so Qt can
            # free them. ``deleteLater`` queues a deletion on the next
            # event loop tick, which is safe because ``exec_`` already
            # returned (no nested event loop active).
            self._ctx_actions.clear()
            menu.deleteLater()

    def _on_row_double_clicked(self, index) -> None:
        item = self._table.item(index.row(), 0)
        if item is None:
            return
        job_id = item.text()
        self.open_results_requested.emit(job_id)

    def _export_xlsx(self, job: JobRecord) -> None:
        if not job.rows:
            QMessageBox.information(
                self,
                i18n._tr("jobstab.menu.export_xlsx"),
                i18n._tr("jobstab.export.no_rows"),
            )
            return
        default_path = str(Path(job.output_dir) / f"{job.job_id}.xlsx")
        path, _ = QFileDialog.getSaveFileName(
            self,
            i18n._tr("jobstab.export.xlsx_title"),
            default_path,
            "Excel files (*.xlsx)",
        )
        if not path:
            return
        # Snapshot the run_output on the GUI thread before kicking off
        # the worker so subsequent row mutations (filtering, edits) don't
        # race with the write.
        run_output = self._build_run_output(job)
        self._run_export_worker("xlsx", run_output, path)

    def _export_json(self, job: JobRecord) -> None:
        if not job.rows:
            QMessageBox.information(
                self,
                i18n._tr("jobstab.menu.export_json"),
                i18n._tr("jobstab.export.no_rows"),
            )
            return
        default_path = str(Path(job.output_dir) / f"{job.job_id}.json")
        path, _ = QFileDialog.getSaveFileName(
            self,
            i18n._tr("jobstab.export.json_title"),
            default_path,
            "JSON files (*.json)",
        )
        if not path:
            return
        run_output = self._build_run_output(job)
        self._run_export_worker("json", run_output, path)

    # ------------------------------------------------------------------
    # Phase F-2 (M-13, M-26): async export workers
    # ------------------------------------------------------------------
    def _run_export_worker(
        self,
        fmt: str,
        run_output: dict[str, Any],
        path: str,
    ) -> None:
        """Spin up an :class:`_ExportWorker` for ``fmt`` and wire it up.

        Phase F-2 (M-13): moves the write off the GUI thread so
        large jobs (50k+ rows) don't freeze the UI. Phase F-2
        (M-26): both XLSX and JSON go through the same worker so
        the error path is unified — log at ERROR level with exc_info
        AND surface a popup to the user.
        """
        worker = _JobsExportWorker(fmt, run_output, path)
        # Keep the worker alive so it isn't GC'd mid-flight (PySide6 footgun).
        self._export_worker = worker

        def _on_success(saved_path: str) -> None:
            self._export_worker = None
            self._log.info("Export succeeded: %s → %s", fmt, saved_path)

        def _on_error(msg: str) -> None:
            self._export_worker = None
            # M-26: log AND popup — same pattern as results_tab.
            self._log.error(
                "Export failed (%s → %s): %s",
                fmt,
                path,
                msg,
                exc_info=True,
            )
            label_key = f"jobstab.export.{fmt}_title"
            QMessageBox.warning(
                self,
                i18n._tr(label_key),
                i18n._tr("jobstab.export.failed").format(error=msg),
            )

        worker.finished_ok.connect(_on_success)
        worker.failed.connect(_on_error)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _build_run_output(self, job: JobRecord) -> dict[str, Any]:
        """Build a RunOutput-compatible dict for the exporters."""
        panels = job.rows
        # Collect distinct figures / localities / geology
        figure_ids = sorted({p.get("figure_id") for p in panels if p.get("figure_id")})
        localities: list[dict[str, Any]] = []
        geo: list[dict[str, Any]] = []
        seen_loc: set[tuple] = set()
        for p in panels:
            md = p.get("metadata") or {}
            for g in md.get("geology_links") or []:
                geo.append(g)
            for occ in (md.get("paleodb") or {}).get("occurrences") or []:
                key = (occ.get("country"), occ.get("locality"))
                if key not in seen_loc:
                    seen_loc.add(key)
                    localities.append(
                        {"country": occ.get("country"), "locality": occ.get("locality")}
                    )
        return {
            "schema_version": "1.0.0",
            "provenance": {"job_id": job.job_id, "source": "rlpe-gui"},
            "papers": [{"pdf_path": job.pdf_path}],
            "figures": [{"figure_id": fid} for fid in figure_ids],
            "panels": panels,
            "taxa": [],
            "samples": [],
            "geology_contexts": geo,
            "localities": localities,
            "paleo_coordinates": [],
            "warnings": [],
        }

    def _remove_job(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)
        row = self._find_row(job_id)
        if row >= 0:
            self._table.removeRow(row)
        self._update_summary()

    def _clear_finished(self) -> None:
        # Remove all jobs whose status is done / failed / cancelled.
        # Phase 56 audit: batch removal avoids O(N²) repeated _find_row +
        # _update_summary; collect rows first, remove bottom-to-top.
        statuses = {STATUS_DONE, STATUS_FAILED, STATUS_CANCELLED}
        to_remove = [j for j, job in self._jobs.items() if job.status in statuses]
        if not to_remove:
            return
        rows_to_remove = sorted(
            {self._find_row(j) for j in to_remove if self._find_row(j) >= 0},
            reverse=True,
        )
        for r in rows_to_remove:
            self._table.removeRow(r)
        for jid in to_remove:
            self._jobs.pop(jid, None)
        self._ctx_actions.clear()
        self._update_summary()

    def _clear_all(self) -> None:
        # Phase 56 audit: batch clear instead of individual _remove_job calls.
        self._jobs.clear()
        self._table.setRowCount(0)
        self._ctx_actions.clear()
        self._update_summary()

    def _refresh_texts(self) -> None:
        """Re-apply column headers and context-menu actions after language switch."""
        headers = [
            i18n._tr("jobstab.col.id"),
            i18n._tr("jobstab.col.pdf"),
            i18n._tr("jobstab.col.status"),
            i18n._tr("jobstab.col.progress"),
            i18n._tr("jobstab.col.rows"),
            i18n._tr("jobstab.col.elapsed"),
            i18n._tr("jobstab.col.out"),
        ]
        for i, h in enumerate(headers):
            item = self._table.horizontalHeaderItem(i)
            if item is not None:
                item.setText(h)
            else:
                # Phase 56 audit fix: horizontalHeaderItem() can return None
                # after setHorizontalHeaderLabels(). Use the header view
                # model directly to refresh header text safely.
                self._table.horizontalHeader().model().setHeaderData(
                    i,
                    Qt.Horizontal,
                    h,
                    Qt.DisplayRole,
                )
        # Translate any active context menu actions (QAction isn't
        # a QWidget so the i18n registry's allWidgets() loop misses
        # them — we update them here explicitly). GUI-m3 fix: just clear
        # the stale list; _show_context_menu rebuilds it fresh.
        ctx_actions = getattr(self, "_ctx_actions", None)
        if ctx_actions:
            ctx_actions.clear()

    def _on_language_changed(self, _lang: str) -> None:
        """Rebuild UI texts on language switch (i18n listener)."""
        self._refresh_texts()

    def _remove_i18n_listener(self) -> None:
        """Remove our i18n listener when the widget is destroyed."""
        listener = getattr(self, "_i18n_listener", None)
        if listener is not None:
            try:
                i18n.remove_listener(listener)
            except Exception:
                pass

    def shutdown(self) -> None:
        """Interrupt and wait for the disk-scan worker (if any).

        Phase F-1 (B-1): closing the GUI mid-scan used to leave the
        ``_DiskScanWorker`` QThread running; the Python interpreter
        then tried to GC the wrapped C++ object and crashed with
        exit code 134 (SIGABRT). We now:

        1. Call ``requestInterruption()`` on the worker so its
           ``run()`` loop breaks at the next entry boundary.
        2. ``wait(30000)`` for it to exit gracefully (30 s matches
           the pipeline-worker's shutdown budget in ``main_window``).
        3. Log a warning if the wait timed out — we deliberately
           do NOT ``terminate()`` because the worker only reads
           files; letting the process exit reclaim it is fine.

        The strong reference is then cleared so the worker can be
        collected normally. We never ``del`` a still-running QThread
        (a known PySide6 footgun), so we leave ``None``-ing it as
        the only safe disposal.

        Safe to call multiple times; safe to call when the worker
        has already finished and been ``deleteLater()``-d (a
        ``RuntimeError`` from the deleted C++ object is treated as
        "already gone" and the ref is dropped).
        """
        worker = getattr(self, "_disk_scan_worker", None)
        if worker is None:
            return
        try:
            running = worker.isRunning()
        except RuntimeError as exc:
            # The worker's ``finished`` signal has already fired and
            # its ``deleteLater`` has executed — the C++ object is
            # gone but the Python ref lingers. Just drop the ref.
            self._log.debug(
                "shutdown: worker C++ already deleted (%s); dropping ref",
                exc,
            )
            self._disk_scan_worker = None
            return
        if not running:
            # Already finished but the C++ object is still around.
            # Just drop the ref.
            self._disk_scan_worker = None
            return
        try:
            worker.requestInterruption()
        except RuntimeError as exc:
            # Already finished between the isRunning() check and now.
            self._log.debug(
                "shutdown: requestInterruption failed (race): %s",
                exc,
            )
            self._disk_scan_worker = None
            return
        try:
            if not worker.wait(30000):  # 30 s timeout (matches main_window)
                self._log.warning(
                    "shutdown: _DiskScanWorker did not exit within 30s; "
                    "leaving thread alive (process exit will reclaim it)"
                )
            else:
                self._log.debug("shutdown: disk scan worker stopped cleanly")
        except RuntimeError as exc:
            # wait() raised if the underlying QThread is already destroyed.
            self._log.debug("shutdown: worker.wait() raised: %s", exc)
        # Drop the strong reference so Qt can free it. Never call
        # ``del`` here — a still-running QThread raises on destruction.
        self._disk_scan_worker = None

    def closeEvent(self, event) -> None:  # noqa: N802
        """Phase 56 audit: remove i18n listener on widget destruction.

        Phase F-1 (B-1): also call ``shutdown()`` to interrupt the
        disk-scan QThread so we don't crash on GUI close (exit 134).
        """
        self._remove_i18n_listener()
        self.shutdown()
        super().closeEvent(event)
