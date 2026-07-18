"""Batch manager — queue and run multiple PDFs in serial.

The CLI has a ``--num-workers`` flag for parallel batch. The GUI
implements serial batch (one job at a time) because:
  * Parallel batch + Qt's signal/slot model + cv2/torch has
    measurable deadlock risk. We don't want to debug a GUI freeze
    the day before a demo.
  * Single-GPU machines (which is most radiolarian researchers)
    don't benefit much from parallel OCR anyway.
  * Serial batch is the safest UX: each job completes before
    the next starts, so per-job error handling is simple.

The batch dialog is launched from the main window's File menu.
It reuses the existing ``PipelineWorker`` (one job at a time) plus
the Jobs / Results tabs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .constants import BATCH_DIALOG_DEFAULT_SIZE
from .i18n_widgets import tr_button, tr_checkbox, tr_groupbox, tr_label
from .styles import SPACE_M, SPACE_S
from .utils import file_size_human, get_gui_logger, short_path
from . import i18n


class BatchDialog(QDialog):
    """Modal dialog for queuing multiple PDFs."""

    batch_started = Signal(list, dict)  # list[Path], settings
    batch_closed = Signal()

    def __init__(self, settings: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._log = get_gui_logger()
        self._settings = settings
        self._pdfs: list[Path] = []        # ordered list for UI display
        self._pdfs_set: set[Path] = set()  # Phase 56 audit: O(1) duplicate-check + remove
        self._build_ui()

    def _build_ui(self) -> None:
        # Phase 54 audit m5: window title was hardcoded English. Use
        # the i18n key batch.title so the title translates on
        # language switch.
        self.setWindowTitle(i18n._tr("batch.title"))
        self.resize(*BATCH_DIALOG_DEFAULT_SIZE)
        outer = QVBoxLayout(self)
        outer.setSpacing(SPACE_M)
        outer.setContentsMargins(SPACE_M, SPACE_M, SPACE_M, SPACE_M)

        # ---- File list ----
        # Phase 54 audit m5: group title was hardcoded English. Use
        # tr_groupbox("batch.files") so it translates on language
        # switch.
        files_group = tr_groupbox("batch.files")
        fg_layout = QVBoxLayout(files_group)
        fg_layout.setSpacing(SPACE_S)

        # File list + buttons
        list_row = QHBoxLayout()
        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._file_list.setAlternatingRowColors(True)
        list_row.addWidget(self._file_list, 1)

        file_btns = QVBoxLayout()
        add_btn = tr_button("batch.add")
        add_btn.clicked.connect(self._on_add)
        file_btns.addWidget(add_btn)

        add_dir_btn = tr_button("batch.add_dir")
        add_dir_btn.clicked.connect(self._on_add_dir)
        file_btns.addWidget(add_dir_btn)

        file_btns.addSpacing(SPACE_S)

        remove_btn = tr_button("batch.remove")
        remove_btn.clicked.connect(self._on_remove)
        file_btns.addWidget(remove_btn)

        clear_btn = tr_button("batch.clear")
        clear_btn.clicked.connect(self._on_clear_all)
        file_btns.addWidget(clear_btn)

        file_btns.addStretch(1)
        list_row.addLayout(file_btns)
        fg_layout.addLayout(list_row)

        # Drag & drop support
        self._file_list.setAcceptDrops(True)
        self._file_list.viewport().setAcceptDrops(True)
        self._file_list.setDragDropMode(QAbstractItemView.DropOnly)
        self._file_list.dragEnterEvent = self._drag_enter  # type: ignore[assignment]
        self._file_list.dragMoveEvent = self._drag_enter   # type: ignore[assignment]
        self._file_list.dropEvent = self._drop_files     # type: ignore[assignment]

        outer.addWidget(files_group, 1)

        # ---- Options ----
        # Phase 54 audit m5: group title was hardcoded English. Use
        # tr_groupbox("batch.options") so it translates on language
        # switch.
        opts_group = tr_groupbox("batch.options")
        og = QFormLayout(opts_group)
        og.setHorizontalSpacing(SPACE_M)
        og.setVerticalSpacing(SPACE_S)

        self._out_dir_edit = QLineEdit(self._settings.get("last_export_dir", str(Path.home() / "rlpe_batch_out")))
        og.addRow(tr_label("batch.outdir"), self._out_dir_edit)

        out_row = QHBoxLayout()
        out_row.addWidget(self._out_dir_edit, 1)
        out_btn = tr_button("common.browse")
        out_btn.clicked.connect(self._on_pick_out)
        out_row.addWidget(out_btn)
        og.addRow("", _wrap(out_row))

        self._stop_on_error = tr_checkbox("batch.stop_on_error")
        self._stop_on_error.setChecked(False)
        og.addRow("", self._stop_on_error)

        self._xlsx_at_end = tr_checkbox("batch.xlsx_at_end")
        self._xlsx_at_end.setChecked(True)
        og.addRow("", self._xlsx_at_end)

        outer.addWidget(opts_group)

        # ---- Bottom action bar ----
        bar = QHBoxLayout()
        bar.setSpacing(SPACE_S)

        self._count_label = tr_label("batch.count.zero")
        self._count_label.setObjectName("metric")
        bar.addWidget(self._count_label)

        bar.addStretch(1)

        cancel_btn = tr_button("batch.cancel")
        cancel_btn.clicked.connect(self.reject)
        bar.addWidget(cancel_btn)

        start_btn = tr_button("batch.start")
        # Phase 54 audit m5: setObjectName("primary") was clobbered by
        # tr_button which sets the object name to the i18n key. Use
        # setProperty("class", "primary") so the stylesheet can still
        # target the start button.
        start_btn.setProperty("class", "primary")
        start_btn.clicked.connect(self._on_start)
        bar.addWidget(start_btn)

        outer.addLayout(bar)

    # ------------------------------------------------------------------
    # Drag & drop
    # ------------------------------------------------------------------
    def _drag_enter(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def _drop_files(self, event) -> None:
        for url in event.mimeData().urls():
            if url.isLocalFile():
                p = Path(url.toLocalFile())
                if p.is_file() and p.suffix.lower() == ".pdf":
                    self._add_pdf(p)
                elif p.is_dir():
                    for sub in sorted(p.glob("*.pdf")):
                        self._add_pdf(sub)

    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------
    def _add_pdf(self, path: Path) -> None:
        if not path.exists():
            return
        if path in self._pdfs_set:  # Phase 56 audit: O(1) via set
            return
        self._pdfs.append(path)
        self._pdfs_set.add(path)
        item = QListWidgetItem(f"{path.name}  ·  {file_size_human(path)}")
        item.setData(Qt.UserRole, str(path))
        item.setToolTip(str(path))
        self._file_list.addItem(item)
        self._count_label.setText(f"{len(self._pdfs)} PDFs queued")

    def _on_add(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            i18n._tr("batch.add.title"),
            self._settings.get("last_pdf_dir", str(Path.home())),
            "PDF files (*.pdf)",
        )
        for p in paths:
            self._add_pdf(Path(p))

    def _on_add_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            i18n._tr("batch.add_dir.title"),
            self._settings.get("last_pdf_dir", str(Path.home())),
        )
        if not path:
            return
        for sub in sorted(Path(path).glob("*.pdf")):
            self._add_pdf(sub)

    def _on_remove(self) -> None:
        # Phase 56 audit: use set for O(1) removal instead of list.remove O(n)
        for item in self._file_list.selectedItems():
            row = self._file_list.row(item)
            path = Path(item.data(Qt.UserRole))
            if path in self._pdfs_set:
                self._pdfs_set.discard(path)
                self._pdfs.remove(path)
            self._file_list.takeItem(row)
        self._count_label.setText(i18n._tr("batch.count").format(n=len(self._pdfs)))

    def _on_clear_all(self) -> None:
        """Clear all PDFs from the list and the output directory."""
        self._file_list.clear()
        self._pdfs.clear()
        self._pdfs_set.clear()  # Phase 56 audit: clear the set too
        self._count_label.setText(i18n._tr("batch.count").format(n=0))

    def _on_pick_out(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            i18n._tr("batch.outdir.title"),
            self._out_dir_edit.text() or str(Path.home()),
        )
        if path:
            self._out_dir_edit.setText(path)

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------
    def _on_start(self) -> None:
        if not self._pdfs:
            QMessageBox.information(
                self,
                i18n._tr("batch.title"),
                i18n._tr("batch.no_pdfs"),
            )
            return
        out_dir = self._out_dir_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(
                self,
                i18n._tr("batch.no_outdir.title"),
                i18n._tr("batch.no_outdir.body"),
            )
            return
        try:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(
                self,
                i18n._tr("batch.no_outdir.title"),
                f"{i18n._tr('batch.no_outdir.body')}\n\n{type(exc).__name__}: {exc}",
            )
            return
        # Phase 55 audit m4 — verify the chosen directory is actually
        # writable BEFORE the batch starts. ``mkdir`` succeeds on a
        # read-only mount or a directory where the user lacks write
        # permission; the failure would surface only when the first
        # job tries to write a JSON, which is harder to recover from.
        import tempfile as _tf_probe
        try:
            with _tf_probe.NamedTemporaryFile(
                prefix=".rlpe_probe_", dir=out_dir, delete=True,
            ) as _probe:
                _probe.write(b"ok")
            import os as _os_probe
            if not _os_probe.access(out_dir, _os_probe.W_OK):
                raise OSError(f"Directory is not writable: {out_dir}")
        except OSError as exc:
            QMessageBox.warning(
                self,
                i18n._tr("batch.no_outdir.title"),
                i18n._tr("batch.outdir.not_writable").format(
                    path=out_dir, error=f"{type(exc).__name__}: {exc}",
                ),
            )
            return
        self._settings["last_export_dir"] = out_dir
        # Settings to pass to each PipelineWorker
        batch_settings = dict(self._settings)
        batch_settings["last_export_dir"] = out_dir
        batch_settings["_stop_on_error"] = self._stop_on_error.isChecked()
        batch_settings["_xlsx_at_end"] = self._xlsx_at_end.isChecked()
        self.batch_started.emit(list(self._pdfs), batch_settings)
        self.accept()


def _wrap(layout) -> QWidget:
    """Wrap a layout in a QWidget so it can be used as a form-row child."""
    w = QWidget()
    w.setLayout(layout)
    return w