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

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressBar,
    QStyle,
    QPushButton,
    QStyledItemDelegate,
    QStyleOptionProgressBar,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QApplication,
)

from .constants import (
    MAX_RECENT_JOBS_IN_LIST,
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
)
from .styles import SPACE_M, SPACE_S
from . import i18n
from .i18n_widgets import tr_button, tr_label
from .utils import (
    file_size_human,
    fmt_count,
    fmt_duration,
    get_gui_logger,
    short_path,
)


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

    Phase 43 audit fix: the previous implementation skipped calling
    ``QStyledItemDelegate.paint`` for the cell background, leaving
    the painter in a state where Qt's BackingStore still had an
    active painter when endPaint() was called, producing
    "QBackingStore::endPaint() called with active painter" warnings
    on every refresh. Fixed: call the base class first to draw the
    cell, then draw the progress bar on top.
    """

    def paint(self, painter, option, index) -> None:  # type: ignore[override]
        # Delegate to the base class for the cell background +
        # selection state. This is what Qt's docs say is required
        # for custom delegates.
        super().paint(painter, option, index)
        # Now overlay a progress bar. Phase 45: wrap in painter.save()
        # / painter.restore() so the drawControl's pen / brush changes
        # don't leak into the next delegate's paint call.
        painter.save()
        try:
            progress_bar_option = QStyleOptionProgressBar()
            progress_bar_option.rect = option.rect
            progress_bar_option.state = option.state
            progress_bar_option.palette = option.palette
            progress_bar_option.minimum = 0
            progress_bar_option.maximum = max(1, index.data(Qt.UserRole) or 0)
            progress_bar_option.progress = index.data(Qt.UserRole + 1) or 0
            progress_bar_option.text = f"{progress_bar_option.progress} / {progress_bar_option.maximum}"
            progress_bar_option.textVisible = True
            # Phase 54 audit: B1 — was using
            # ``QStyledItemDelegate.PrimitiveElement.ProgressBar``, which
            # does not exist (``QStyledItemDelegate`` has no
            # ``PrimitiveElement`` attribute). The error fired inside
            # ``paint()``, propagating an ``AttributeError`` out of the
            # Qt paint call and corrupting the jobs-tab render. The
            # correct element for ``QStyle.drawControl`` is the
            # ``ControlElement`` ``CE_ProgressBar`` (value 10).
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
    retry_requested = Signal(str, dict)  # job_id, settings

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._log = get_gui_logger()
        self._jobs: dict[str, JobRecord] = {}
        self._build_ui()
        # Phase 37 audit fix: register as an i18n listener so
        # column headers, context menu items, and status labels
        # auto-translate when the language switches (was: the
        # MainWindow had to walk every tab and manually call
        # _refresh_texts, which it didn't).
        # The listener signature is ``Callable[[str], None]`` so
        # we wrap with a lambda to discard the lang argument —
        # without this, ``fn(lang)`` would TypeError on _refresh_texts
        # (which takes no args) and set_language's ``except: pass``
        # would silently swallow the error.
        i18n.add_listener(lambda _lang: self._refresh_texts())

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

        # Phase 37: translated count label, default text from i18n.
        self._count_label = tr_label("jobstab.no_jobs")
        self._count_label.setObjectName("metric")
        bar.addWidget(self._count_label)

        outer.addLayout(bar)

        # ---- Table ----
        self._table = QTableWidget(0, 7)
        # Phase 37 audit fix: use i18n keys for column headers so
        # they translate on language switch. ``_refresh_texts``
        # (called via the i18n listener registered in __init__)
        # re-applies them when set_language() runs.
        self._table.setHorizontalHeaderLabels([
            i18n._tr("jobstab.col.id"),
            i18n._tr("jobstab.col.pdf"),
            i18n._tr("jobstab.col.status"),
            i18n._tr("jobstab.col.progress"),
            i18n._tr("jobstab.col.rows"),
            i18n._tr("jobstab.col.elapsed"),
            i18n._tr("jobstab.col.out"),
        ])
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
        # Phase 37: translated status label.
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

        Returns the number of jobs loaded.
        """
        from .constants import PROJECT_ROOT
        loaded = 0

        # Candidate roots: <root>/service_work/<job_id>/output/manifests/matches.jsonl
        # and the dev work/ directory at project root.
        roots: list[tuple[Path, str]] = []
        service_work = PROJECT_ROOT / "service_work"
        if service_work.exists():
            for child in sorted(service_work.iterdir()):
                if child.is_dir():
                    roots.append((child, child.name))
        # Also scan project root work/ for ad-hoc CLI runs.
        cli_work = PROJECT_ROOT / "work"
        if cli_work.exists() and cli_work.resolve() != service_work.resolve():
            import hashlib
            jid = "cli_" + hashlib.md5(
                str(cli_work.resolve()).encode()
            ).hexdigest()[:12]
            roots.append((cli_work, jid))

        for root, jid in roots:
            matches_path = root / "output" / "manifests" / "matches.jsonl"
            if not matches_path.exists():
                continue
            try:
                rows: list[dict[str, Any]] = []
                with matches_path.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))
            except (OSError, json.JSONDecodeError):
                # Corrupt or partial file — skip silently. Logged at
                # the GUI logger for the curious.
                self._log.warning(
                    "load_recent_jobs: skipping %s (read/parse error)",
                    matches_path,
                )
                continue
            if not rows:
                continue
            # Locate the original PDF (best-effort).
            pdf_path = ""
            pdfs_dir = root / "pdfs"
            if pdfs_dir.exists():
                pdfs = list(pdfs_dir.glob("*.pdf"))
                if pdfs:
                    pdf_path = str(pdfs[0])
            # Try to get a creation timestamp from filesystem.
            try:
                finished_at = matches_path.stat().st_mtime
            except OSError:
                finished_at = time.time()
            job = JobRecord(
                job_id=jid,
                pdf_path=pdf_path,
                output_dir=str(root / "output"),
                status=STATUS_DONE,
                progress_current=1,
                progress_total=1,
                progress_msg=i18n._tr("jobstab.loaded_from_disk"),
                rows=rows,
                started_at=finished_at,
                finished_at=finished_at,
            )
            self.add_or_update_job(job)
            loaded += 1
        return loaded

    # ------------------------------------------------------------------
    def add_or_update_job(self, job: JobRecord) -> None:
        """Insert or update a job record in the table."""
        self._jobs[job.job_id] = job
        self._refresh_row(job)
        self._update_summary()
        self._trim_old_jobs()

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

    def mark_failed(self, job_id: str, error: str) -> None:
        if job_id not in self._jobs:
            return
        job = self._jobs[job_id]
        job.status = STATUS_FAILED
        job.error = error
        job.finished_at = time.time()
        self._refresh_row(job)
        self._update_summary()

    def mark_cancelled(self, job_id: str) -> None:
        if job_id not in self._jobs:
            return
        job = self._jobs[job_id]
        job.status = STATUS_CANCELLED
        job.finished_at = time.time()
        self._refresh_row(job)
        self._update_summary()

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
            row = self._table.rowCount()
            self._table.insertRow(row)
            # Cap the table at MAX_RECENT_JOBS_IN_LIST rows.
            while self._table.rowCount() > MAX_RECENT_JOBS_IN_LIST:
                self._table.removeRow(0)

        # Column 0: Job ID (monospace)
        item_id = QTableWidgetItem(job.job_id)
        item_id.setData(Qt.UserRole, job.job_id)
        font = item_id.font()
        font.setFamily("JetBrains Mono, Cascadia Code, Menlo, Consolas, monospace")
        item_id.setFont(font)
        self._table.setItem(row, 0, item_id)

        # Column 1: PDF (short path tooltip)
        self._table.setItem(row, 1, QTableWidgetItem(short_path(Path(job.pdf_path), 50)))
        self._table.item(row, 1).setToolTip(job.pdf_path)

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
        progress_item.setData(Qt.UserRole, job.progress_total)   # max
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
        from PySide6.QtGui import QColor

        bg_map = {
            STATUS_RUNNING: QColor("#d6e4ff"),
            STATUS_DONE: QColor("#d8f5d0"),
            STATUS_FAILED: QColor("#ffe0e0"),
            STATUS_CANCELLED: QColor("#ffe9d6"),
            STATUS_QUEUED: QColor("#eef1f6"),
        }
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
                total=total, running=running, done=done, failed=failed,
            )
        )
        self._summary.setText(
            i18n._tr("jobstab.summary.count").format(
                total=total, running=running, done=done, failed=failed,
            )
        )

    def _trim_old_jobs(self) -> None:
        """Cap the table at MAX_RECENT_JOBS_IN_LIST rows."""
        while self._table.rowCount() > MAX_RECENT_JOBS_IN_LIST:
            # Remove the oldest row (top of table)
            self._table.removeRow(0)

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
        menu = QMenu(self)
        # Phase 37 audit fix: register i18n keys for each QAction
        # so the menu items translate on language switch. We
        # store the actions on ``self`` so _refresh_texts can
        # update their text later.
        self._ctx_actions: list[tuple[QAction, str]] = []

        def _add_action(key: str) -> QAction:
            act = QAction(i18n._tr(key), self)
            i18n.register_widget_text(f"jobstab.action.{key}", "text", key)
            self._ctx_actions.append((act, key))
            menu.addAction(act)
            return act

        act_open_results = _add_action("jobstab.menu.open_results")
        act_open_results.triggered.connect(lambda: self.open_results_requested.emit(job.job_id))

        act_open_out = _add_action("jobstab.menu.open_out")
        act_open_out.triggered.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(job.output_dir)))

        menu.addSeparator()

        act_export_xlsx = _add_action("jobstab.menu.export_xlsx")
        act_export_xlsx.triggered.connect(lambda: self._export_xlsx(job))

        act_export_json = _add_action("jobstab.menu.export_json")
        act_export_json.triggered.connect(lambda: self._export_json(job))

        menu.addSeparator()

        act_remove = _add_action("jobstab.menu.remove")
        act_remove.triggered.connect(lambda: self._remove_job(job.job_id))

        menu.exec(self._table.viewport().mapToGlobal(pos))

    def _on_row_double_clicked(self, index) -> None:
        job_id = self._table.item(index.row(), 0).text()
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
        try:
            from ..export import export_csv  # legacy alias if no xlsx

            # Try the xlsx exporter (Round 24)
            from ..cli_export import export_run_output_to_xlsx  # type: ignore
        except ImportError:
            try:
                from ..exporters.xlsx import write_xlsx
                # Build a RunOutput-compatible dict from the job
                run_output = self._build_run_output(job)
                write_xlsx(run_output, path)
                QMessageBox.information(
                    self,
                    i18n._tr("jobstab.menu.export_xlsx"),
                    i18n._tr("jobstab.export.saved").format(
                        count=len(job.rows), path=path,
                    ),
                )
                return
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    i18n._tr("jobstab.menu.export_xlsx"),
                    i18n._tr("jobstab.export.failed").format(
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                )
                return
        # Fallback: call the cli_export function if available
        try:
            from ..cli_export import export_run_output_to_xlsx
            run_output = self._build_run_output(job)
            export_run_output_to_xlsx(run_output, path)
            QMessageBox.information(
                self,
                i18n._tr("jobstab.menu.export_xlsx"),
                i18n._tr("jobstab.export.saved").format(
                    count=len(job.rows), path=path,
                ),
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                i18n._tr("jobstab.menu.export_xlsx"),
                i18n._tr("jobstab.export.failed").format(
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )

    def _export_json(self, job: JobRecord) -> None:
        if not job.rows:
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
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self._build_run_output(job), fh, indent=2, ensure_ascii=False, default=str)
            QMessageBox.information(
                self,
                i18n._tr("jobstab.menu.export_json"),
                i18n._tr("jobstab.export.saved").format(
                    count=len(job.rows), path=path,
                ),
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                i18n._tr("jobstab.menu.export_json"),
                i18n._tr("jobstab.export.failed").format(error=str(exc)),
            )

    def _build_run_output(self, job: JobRecord) -> dict[str, Any]:
        """Build a RunOutput-compatible dict for the exporters."""
        panels = job.rows
        # Collect distinct figures / localities / geology
        figure_ids = sorted({p.get("figure_id") for p in panels if p.get("figure_id")})
        localities: list[dict[str, Any]] = []
        geo: list[dict[str, Any]] = []
        seen_loc: set[tuple] = set()
        for p in panels:
            for k, v in (p.get("metadata") or {}).items():
                if k == "geology_links":
                    for g in v or []:
                        geo.append(g)
            for k, v in (p.get("metadata") or {}).items():
                if k == "paleodb":
                    for occ in (v or {}).get("occurrences", []):
                        key = (occ.get("country"), occ.get("locality"))
                        if key not in seen_loc:
                            seen_loc.add(key)
                            localities.append({"country": occ.get("country"), "locality": occ.get("locality")})
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
        # Remove all jobs whose status is done / failed / cancelled
        for status in (STATUS_DONE, STATUS_FAILED, STATUS_CANCELLED):
            to_remove = [j for j, job in self._jobs.items() if job.status == status]
            for jid in to_remove:
                self._remove_job(jid)

    def _clear_all(self) -> None:
        for jid in list(self._jobs.keys()):
            self._remove_job(jid)

    def _refresh_texts(self) -> None:
        """Re-apply column headers / buttons / context-menu actions
        after language switch. Phase 37 audit fix: also walks
        ``self._ctx_actions`` (set by _show_context_menu) so the
        right-click menu items translate too."""
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
        # Translate any active context menu actions (QAction isn't
        # a QWidget so the i18n registry's allWidgets() loop misses
        # them — we update them here explicitly).
        for action, key in getattr(self, "_ctx_actions", []):
            try:
                action.setText(i18n._tr(key))
            except RuntimeError:
                # Action may have been destroyed if the menu closed
                self._ctx_actions.remove((action, key))