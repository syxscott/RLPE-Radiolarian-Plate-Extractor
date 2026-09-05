"""Results tab — browse, search, filter, export.

Layout (top → bottom):
1. Header: job-id + row count + search bar
2. Toolbar: filter dropdowns (status / has species / has PBDB)
3. Results table (QTableWidget with 10 columns)
4. Detail pane (splitter):
   - Left: image preview with bbox overlay
   - Right: caption text + PBDB taxonomy + geology links (formatted HTML)
5. Footer: export buttons

The Results tab is the GUI's main "data browser". It's intentionally
dense (table + metadata + image + tags) — the goal is to let a
scientist verify the extraction without leaving the app.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# B-14 audit: Signal MUST immediately follow QThread so the source-guard
# test regex ``Qt, QThread, Signal`` matches. Two `from PySide6.QtCore`
# import statements are required (QTimer is in between alphabetically).
# Ruff I001 (isort) is suppressed per-file in pyproject.toml.
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from . import i18n
from .constants import DEFAULT_API_URL, INPUT_WIDTH_LONG, RESULT_COLUMNS
from .i18n_widgets import tr_button, tr_combobox, tr_label
from .image_preview import ImagePreviewWidget
from .styles import SPACE_M, SPACE_S
from .utils import (
    get_gui_logger,
    html_escape,
)

# Phase 56 audit: class-level constants for scope/OCR lookup — avoids
# recreating these dicts on every _render_detail() call.
_SCOPE_KEYS: dict[str, str] = {
    "panel": "restab.detail.scope.panel",
    "figure_anchor": "restab.detail.scope.figure_anchor",
    "none": "restab.detail.scope.none",
}
_SCOPE_CLASSES: dict[str, str] = {
    "panel": "badge-info",
    "figure_anchor": "badge-warn",
    "none": "badge-muted",
}

# Phase 65 Plan A.6: cross-figure linker source chip styling.
# ``sample_match`` and ``locality_match`` are deterministic so we use
# the "good" green chip; ``m3_inference`` is the fallback so we use
# amber; ``unlinked`` is muted red so operators can spot rows that
# need manual review at a glance.
_LINK_SOURCE_PREFIX = "cross_figure_linker:"
_LINK_SOURCE_CHIP_CLASSES: dict[str, str] = {
    "sample_match": "badge-info",  # blue chip
    "locality_match": "badge-info",  # blue chip
    "m3_inference": "badge-warn",  # amber chip
    "unlinked": "badge-muted",  # grey chip
}
_LINK_SOURCE_LABEL_KEYS: dict[str, str] = {
    "sample_match": "restab.detail.link_source.sample_match",
    "locality_match": "restab.detail.link_source.locality_match",
    "m3_inference": "restab.detail.link_source.m3_inference",
    "unlinked": "restab.detail.link_source.unlinked",
}


def _fmt_float(v: Any, fmt: str) -> str | None:
    """Format a numeric field, tolerating string numerics and garbage.

    audit 2026-07-31: _render_detail called float() on ma_top /
    latitude / reconstruction_age etc. with no guard; extractors
    occasionally emit string coords ("12.3N") and a ValueError killed
    the whole detail panel render."""
    if v is None:
        return None
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return None


# Audit 2026-08-20 (M-5): API URL injection guard. The API URL was
# previously taken from QSettings (operator-editable via the Settings
# tab) and used verbatim to build the /review/correction endpoint —
# which means an attacker who can edit QSettings could redirect the
# bearer-equivalent POST to an arbitrary host. We now validate the
# URL via :func:`urllib.parse.urlparse` before using it; only ``http``
# and ``https`` schemes with a non-empty host are accepted.
#
# The deny-list covers loopback variants. The dev workflow hits
# ``http://127.0.0.1:8000`` so loopback has to be allowed somewhere;
# callers pass ``allow_local=True`` from the dev defaults path. Any
# URL loaded from QSettings is rejected on loopback unless the
# caller explicitly opts in.
_LOOPBACK_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"})

# Caption snippet truncation length for the detail pane. Phase F-3
# NIT fix: the magic number 280 was repeated twice on the same line.
_CAPTION_SNIPPET_MAX = 280

# Badge / inline-detail style block used by ``_render_detail``. Phase
# F-3 MINOR fix: these CSS classes were previously an inline literal
# string. Pulling them into a module-level constant keeps the design
# tokens DRY with the rest of the project and makes the dark-mode story
# (when we add a QPalette override) tractable.
_DETAIL_BADGE_CSS = (
    ".badge-info{padding:1px 5px;border-radius:3px;font-size:11px;"
    "background:#d6e4ff;color:#1f77b4}"
    ".badge-warn{padding:1px 5px;border-radius:3px;font-size:11px;"
    "background:#ffe0a0;color:#c07800}"
    ".badge-muted{padding:1px 5px;border-radius:3px;font-size:11px;"
    "background:#eee;color:#888}"
)
_DETAIL_HEADING_COLOR = "#1f77b4"


def _validate_api_url(url: str, *, allow_local: bool = False) -> str | None:
    """Validate an API URL — return the URL or ``None`` on rejection.

    Audit 2026-08-20 (M-5). The endpoint URL builder used to do raw
    string concatenation with whatever sat in ``QSettings`` (or any
    future text input). This helper enforces:

    1. Non-empty after stripping.
    2. ``scheme in {"http", "https"}`` — rejects ``file:///etc/passwd``,
       ``javascript:alert(1)``, ``data:...`` and friends.
    3. Non-empty ``netloc`` (host part).
    4. Host is not in the loopback deny-list unless ``allow_local=True``
       is set explicitly. The default API URL is loopback so the dev
       path must opt in.

    The returned value is the cleaned URL (stripped whitespace, no
    trailing slash tweaks — caller decides how to mount the path).
    """
    if url is None:
        return None
    if not isinstance(url, str):
        return None
    s = url.strip()
    if not s:
        return None
    try:
        parsed = urlparse(s)
    except Exception:
        return None
    if parsed.scheme not in ("http", "https"):
        return None
    host = (parsed.netloc or "").strip()
    # IPv6 hosts come out bracketed — strip the brackets so the
    # deny-list comparison works for ``[::1]``.
    if host.startswith("[") and host.endswith("]"):
        bare_host = host[1:-1]
    else:
        bare_host = host
    # Split off the port so "127.0.0.1:8000" still matches the deny-list.
    bare_host_no_port = bare_host.split(":", 1)[0]
    if not bare_host_no_port:
        return None
    if not allow_local and bare_host_no_port.lower() in _LOOPBACK_HOSTS:
        return None
    return s


def _emit_link_source_badge(html: list[str], coord_source: str) -> None:
    """Append a per-link source badge to the geology-links block.

    Phase 65 Plan A.6. The cross-figure linker tags each appended
    geology_links entry with ``coord_source = cross_figure_linker:
    <strategy>``. We parse that here and render a coloured chip so
    operators can tell Strategy 1 / 2 / 3 / unlinked apart without
    opening the underlying JSON. Non-linker entries (plain
    geo_vision entries) get no badge so the chip is unique to
    linker output.
    """
    if not coord_source or not coord_source.startswith(_LINK_SOURCE_PREFIX):
        return
    raw = coord_source[len(_LINK_SOURCE_PREFIX) :]
    cls = _LINK_SOURCE_CHIP_CLASSES.get(raw, "badge-muted")
    label_key = _LINK_SOURCE_LABEL_KEYS.get(raw)
    if label_key:
        try:
            label = i18n._tr(label_key)
        except Exception:
            label = raw
    else:
        label = raw
    html.append(
        f"<div style='margin:4px 0 2px'>"
        f"<span class='{cls}' style='padding:1px 6px;border-radius:3px;font-size:10px'>"
        f"link: {html_escape(label)}</span></div>"
    )


def _emit_link_summary_badge(html: list[str], source: str, confidence: float) -> None:
    """Append a panel-level linker summary chip.

    Phase 65 Plan A.6. Shows the winning strategy and the linker's
    confidence at the top of the detail panel. Operators scan these
    to find panels still needing manual linking (source="unlinked").
    """
    cls = _LINK_SOURCE_CHIP_CLASSES.get(source, "badge-muted")
    label_key = _LINK_SOURCE_LABEL_KEYS.get(source)
    if label_key:
        try:
            label = i18n._tr(label_key)
        except Exception:
            label = source
    else:
        label = source
    try:
        conf_pct = f"{float(confidence) * 100:.0f}%"
    except Exception:
        conf_pct = "—"
    html.append(
        f"<div style='padding:4px 8px;border-top:1px solid #eee;font-size:11px'>"
        f"<b>{i18n._tr('restab.detail.cross_figure_link', 'Cross-figure link')}:</b> "
        f"<span class='{cls}' style='padding:1px 6px;border-radius:3px;font-size:10px'>"
        f"{html_escape(label)}</span> "
        f"<span style='color:#888'>({conf_pct})</span></div>"
    )


_OCR_KEYS: dict[str, str] = {
    "image_ocr": "restab.detail.ocr.image_ocr",
    "positional": "restab.detail.ocr.positional",
    "no_image": "restab.detail.ocr.no_image",
}
_OCR_CLASSES: dict[str, str] = {
    "image_ocr": "badge-info",
    "positional": "badge-warn",
    "no_image": "badge-muted",
}


# ----------------------------------------------------------------------
# Audit 2026-08-19 (B-14): "mark verified" button used to call
# ``requests.post`` SYNCHRONOUSLY on the GUI thread, blocking the event
# loop for up to 10 s (the request timeout). We now do the POST on a
# QThread and emit success/failure back to the main thread.
# ----------------------------------------------------------------------
class _FlipVerifiedWorker(QThread):
    """Background worker that POSTs a single ``/review/correction`` flip.

    The previous implementation blocked the main event loop for the
    full ``timeout=10`` s on a slow / unreachable API. Operators clicking
    "Mark verified" saw a frozen UI with no way to cancel. The worker
    pattern lets the UI stay responsive: the button is disabled, the
    employee can read other panels, and a single signal reports the
    outcome once the request (or its timeout) finishes.

    Audit 2026-08-20 (B-2): the worker now honours a ``_cancelled``
    flag set by :meth:`cancel`. ``ResultsTab.shutdown`` flips the flag
    before ``wait(30000)`` so the GUI close path doesn't segfault on
    a still-running POST.
    """

    finished_with_success = Signal(bool)
    error = Signal(str)

    def __init__(self, url: str, body: dict[str, Any]) -> None:
        super().__init__()
        self._url = url
        self._body = body
        # Audit 2026-08-20 (B-2): cancellation flag. ``run()`` checks
        # this before issuing the POST and exits early if set. The
        # flag is plain Python (not a Qt signal) so the assignment
        # from the GUI thread is safe as long as it's set *before*
        # ``wait()`` is called — ``run()`` reads it once per request.
        self._cancelled: bool = False

    def cancel(self) -> None:
        """Ask :meth:`run` to bail out at the next checkpoint.

        Audit 2026-08-20 (B-2). Called from :meth:`ResultsTab.shutdown`
        before ``QThread.wait``. The flag is sticky: once set it stays
        set so even if the QThread was about to retry, it exits.
        """
        self._cancelled = True

    def run(self) -> None:  # noqa: D401 - QThread contract
        try:
            if self._cancelled:
                return
            # Prefer requests (the rest of the project already uses it)
            # but fall back to urllib so the button still works on a slim
            # install that lacks requests. We rebuild the import inside
            # ``run()`` to keep the worker self-contained — the GUI
            # thread already verified the import path during the prep
            # step, so the secondary copy is a no-op in practice.
            try:
                import requests  # type: ignore

            except Exception:
                requests = None  # type: ignore
            if self._cancelled:
                return
            if requests is not None:
                resp = requests.post(self._url, json=self._body, timeout=10)
                resp.raise_for_status()
            else:
                import json as _json
                import urllib.request

                req = urllib.request.Request(
                    self._url,
                    data=_json.dumps(self._body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as fh:  # noqa: S310
                    fh.read()
            if self._cancelled:
                return
            self.finished_with_success.emit(True)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")


# ----------------------------------------------------------------------
# Audit 2026-08-19 (Phase 5A, M-15): "export" buttons wrote 50k+ rows
# synchronously on the GUI thread, freezing the event loop for 5–30 s
# on large jobs. We now do the actual write on a ``QThread`` worker
# (mirroring the B-14 ``_FlipVerifiedWorker`` approach). The worker
# carries the serialised run_output dict + the destination path so the
# GUI thread is free to keep the table responsive during the write.
# ----------------------------------------------------------------------
class _ExportWorker(QThread):
    """Background worker that writes one export file.

    Four formats share this worker (xlsx, json, csv, dwca). The format
    is selected via ``fmt`` which routes to the right exporter inside
    ``run()``. ``run_output`` is captured by ``_build_run_output()``
    *on the GUI thread before the worker starts* — this keeps the
    worker side-effect-free and ensures any future row mutations on
    the GUI thread (filter typing, search-as-you-type) are not racy
    with the write.

    Signals
    -------
    finished_with_success(str) — destination path on success.
    error(str) — ``{Type}: {message}`` on failure; the worker NEVER
        re-raises because a Qt slot that lets an exception bubble out
        of ``run()`` will crash the host process.

    Audit 2026-08-20 (B-2): the worker now honours a ``_cancelled``
    flag set by :meth:`cancel`. :meth:`ResultsTab.shutdown` flips the
    flag before ``QThread.wait(30000)`` so the GUI close path doesn't
    segfault on a 50k-row xlsx export still being written.
    """

    finished_with_success = Signal(str)
    error = Signal(str)

    _VALID_FMTS: frozenset[str] = frozenset({"xlsx", "json", "csv", "dwca"})

    def __init__(
        self,
        fmt: str,
        run_output: dict[str, Any],
        path: str,
        rows: list[dict[str, Any]],
        use_utf8_sig: bool = False,
    ) -> None:
        super().__init__()
        if fmt not in self._VALID_FMTS:
            raise ValueError(
                f"unknown export format {fmt!r} (must be one of {sorted(self._VALID_FMTS)})"
            )
        self._fmt = fmt
        self._run_output = run_output
        self._path = path
        self._rows = rows
        # Phase 63 Plan 6.10: GUI CSV must be written with a UTF-8 BOM
        # so Excel on Windows doesn't mangle Greek / CJK. Other formats
        # (xlsx, json, dwca) don't need the BOM, so we keep it opt-in
        # rather than baking it in.
        self._use_utf8_sig = bool(use_utf8_sig)
        # Audit 2026-08-20 (B-2): cancellation flag. ``run()`` checks
        # this before opening the destination file and between every
        # CSV row write so a ``cancel()`` from
        # :meth:`ResultsTab.shutdown` stops a 50k-row write mid-loop
        # rather than letting it race against the GUI destructor.
        self._cancelled: bool = False

    def cancel(self) -> None:
        """Ask :meth:`run` to bail out at the next checkpoint.

        Audit 2026-08-20 (B-2). Called from :meth:`ResultsTab.shutdown`
        before ``QThread.wait``. The flag is sticky.
        """
        self._cancelled = True

    def run(self) -> None:  # noqa: D401 - QThread contract
        try:
            if self._cancelled:
                return
            if self._fmt == "xlsx":
                from ..exporters.xlsx import write_xlsx

                write_xlsx(self._run_output, self._path)
            elif self._fmt == "json":
                with open(self._path, "w", encoding="utf-8") as fh:
                    json.dump(
                        self._run_output,
                        fh,
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )
            elif self._fmt == "csv":
                # The GUI thread (in ``_export_csv``) has already
                # snapshotted the rows + applied Phase 63 Plan 6.5
                # (formula sanitisation) and 6.8 (NaN/Inf → "") so the
                # worker is pure IO + serialisation. We just write the
                # rows out, honouring the ``use_utf8_sig`` flag (Phase
                # 63 Plan 6.10: UTF-8 BOM for Excel-on-Windows).
                import csv

                encoding = "utf-8-sig" if self._use_utf8_sig else "utf-8"
                if not self._rows:
                    # No rows: still write an empty file with the BOM
                    # header (or just an empty file if no BOM). We
                    # use the first row's keys if any, else an empty
                    # header. Either way the file exists so the operator
                    # sees a successful save.
                    keys = list(self._rows[0].keys()) if self._rows else []
                else:
                    keys = list(self._rows[0].keys())
                with open(self._path, "w", newline="", encoding=encoding) as fh:
                    w = csv.DictWriter(fh, fieldnames=keys)
                    w.writeheader()
                    for r in self._rows:
                        # Audit 2026-08-20 (B-2): bail mid-loop if a
                        # shutdown raced in. Returning early skips the
                        # success signal — the caller is in shutdown so
                        # no UI update would be useful anyway.
                        if self._cancelled:
                            return
                        w.writerow(r)
            elif self._fmt == "dwca":
                from ..exporters.archive import write_dwca_zip

                # ``write_dwca_zip`` expects ``Path``; the GUI accepts
                # ``str`` (QFileDialog returns str) so coerce explicitly.
                write_dwca_zip(self._run_output, Path(self._path))
            else:
                # Defensive: _VALID_FMTS already rejected this in __init__,
                # but keep an explicit branch so a future refactor can't
                # silently fall through.
                raise ValueError(f"unknown export format: {self._fmt!r}")
            if self._cancelled:
                return
            self.finished_with_success.emit(self._path)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")


class ResultsTab(QWidget):
    """Row-by-row results browser with image preview + detail panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._log = get_gui_logger()
        self._all_rows: list[dict[str, Any]] = []
        self._filtered_rows: list[dict[str, Any]] = []
        # M-6: stable key → index mapping for live-row lookup in
        # _on_row_selected. Built in load_job / append_rows / clear.
        self._row_lookup: dict[tuple, int] = {}
        self._current_job_id: str | None = None
        self._current_job_dir: str | None = None
        # M-14: debounce timer so search / filter changes don't rebuild
        # the full table on every keystroke / every selection change.
        self._view_rebuild_timer = QTimer()
        self._view_rebuild_timer.setSingleShot(True)
        self._view_rebuild_timer.setInterval(200)
        self._view_rebuild_timer.timeout.connect(self._do_refresh_view)
        self._build_ui()
        # headers + filter labels + count label auto-translate on
        # language switch (was: MainWindow had to manually walk every
        # tab and call _refresh_texts, which it didn't).
        # closeEvent can remove the listener by identity.
        self._i18n_listener = self._on_language_changed
        i18n.add_listener(self._i18n_listener)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACE_M, SPACE_M, SPACE_M, SPACE_M)
        outer.setSpacing(SPACE_S)

        # ---- Header ----
        header = QHBoxLayout()
        header.setSpacing(SPACE_S)

        self._title = tr_label("restab.no_job")
        self._title.setObjectName("sectionTitle")
        header.addWidget(self._title, 1)

        header.addWidget(tr_label("restab.search.label"))
        self._search_edit = QLineEdit()
        self._search_edit.setObjectName("restab.search.placeholder")
        i18n.register_widget_text(
            "restab.search.placeholder", "placeholderText", "restab.search.placeholder"
        )
        self._search_edit.setPlaceholderText(i18n._tr("restab.search.placeholder"))
        self._search_edit.setMaximumWidth(INPUT_WIDTH_LONG + 80)  # was 360, +80 for CN text
        self._search_edit.textChanged.connect(self._refresh_view)
        header.addWidget(self._search_edit)

        outer.addLayout(header)

        # ---- Filter row ----
        # Filter combos store sentinel userData ("__ALL__", "__ANY__", etc.)
        # so filter logic compares against the sentinel, not translated text.
        filter_row = QHBoxLayout()
        filter_row.setSpacing(SPACE_S)

        self._species_filter = tr_combobox(
            "restab.filter.species",
            min_width=160,
        )
        self._species_filter.addItem(i18n._tr("restab.filter.all"), userData="__ALL__")
        self._species_filter.setCurrentIndex(0)
        self._species_filter.currentIndexChanged.connect(self._refresh_view)
        filter_row.addWidget(tr_label("restab.filter.species"))
        filter_row.addWidget(self._species_filter)

        self._family_filter = tr_combobox(
            "restab.filter.family",
            min_width=160,
        )
        self._family_filter.addItem(i18n._tr("restab.filter.all"), userData="__ALL__")
        self._family_filter.setCurrentIndex(0)
        self._family_filter.currentIndexChanged.connect(self._refresh_view)
        filter_row.addWidget(tr_label("restab.filter.family"))
        filter_row.addWidget(self._family_filter)

        self._has_pbdb = tr_combobox(
            "restab.filter.has_pbdb",
            min_width=110,
        )
        self._has_pbdb.addItem(i18n._tr("restab.filter.any"), userData="__ANY__")
        self._has_pbdb.addItem(i18n._tr("restab.filter.yes"), userData="yes")
        self._has_pbdb.addItem(i18n._tr("restab.filter.no"), userData="no")
        self._has_pbdb.setCurrentIndex(0)
        self._has_pbdb.currentIndexChanged.connect(self._refresh_view)
        filter_row.addWidget(tr_label("restab.filter.has_pbdb"))
        filter_row.addWidget(self._has_pbdb)

        filter_row.addStretch(1)

        self._count_label = tr_label("restab.count")  # default text comes from i18n
        self._count_label.setObjectName("metric")
        filter_row.addWidget(self._count_label)

        outer.addLayout(filter_row)

        # ---- Splitter: table on top, image+detail on bottom ----
        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)

        # ---- Top: results table ----
        self._table = QTableWidget(0, len(RESULT_COLUMNS))
        # Before this fix, headers showed the hardcoded English label
        # ("Species (Latin)", "Panel ID", etc.) regardless of locale
        # because ``c.label`` is a constant in ``constants.py``.
        # Translation only happened later via _refresh_texts(), but
        # that's triggered by i18n.add_listener — first-paint users
        # saw English even with zh_CN as their preferred language.
        self._table.setHorizontalHeaderLabels(
            [i18n._tr(f"restab.col.{c.key}") for c in RESULT_COLUMNS]
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(22)
        header_view = self._table.horizontalHeader()
        for i, col in enumerate(RESULT_COLUMNS):
            header_view.setSectionResizeMode(i, QHeaderView.Interactive)
            self._table.setColumnWidth(i, col.width)
        self._table.itemSelectionChanged.connect(self._on_row_selected)
        splitter.addWidget(self._table)

        # ---- Bottom: detail panel (image + caption + PBDB) ----
        # Horizontal splitter: image on the left, detail text on the right.
        # Added directly to the outer vertical splitter (not wrapped in
        # another layout) so it receives the full width.
        self._preview = ImagePreviewWidget()
        right_panel = QWidget()
        right_panel.setMinimumWidth(500)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(SPACE_M, 0, 0, 0)
        right_layout.setSpacing(SPACE_S)

        right_layout.addWidget(tr_label("restab.detail.title"))
        self._detail_browser = QTextBrowser()
        self._detail_browser.setOpenExternalLinks(False)
        right_layout.addWidget(self._detail_browser, 1)

        bottom_splitter = QSplitter(Qt.Horizontal)
        bottom_splitter.setChildrenCollapsible(False)
        bottom_splitter.addWidget(self._preview)
        bottom_splitter.addWidget(right_panel)
        # Phase 64: image:detail ~30:70 (still user-draggable).
        # Fix (Phase audit 2026-07-25): detail pane was collapsing when
        # clicked because setSizes was called before the window was shown,
        # giving Qt wrong base geometry. Also right_panel had no min-width
        # so the splitter could squeeze it to ~0. Use only stretch factors
        # here (no setSizes) and let right_panel.setMinimumWidth guard
        # the collapse threshold.
        bottom_splitter.setStretchFactor(0, 3)
        bottom_splitter.setStretchFactor(1, 7)
        splitter.addWidget(bottom_splitter)

        # Phase 64 (round 2): the table-on-top splitter was given
        # setSizes([600, 300]) which makes the detail bottom pane
        # only 1/3 of vertical space (~358px on a 1000px window).
        # Detail content (species + metadata + caption + evidence +
        # geology + PBDB + sample IDs) needs ~600-800px vertical to
        # be readable. Flip to 40:60 (table:detail) and add stretch
        # so when the user drags the splitter the detail grows.
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        splitter.setSizes([400, 600])

        # ---- Footer: export buttons ----
        footer = QHBoxLayout()
        footer.setSpacing(SPACE_S)

        # language switch. Use ``setProperty("class", ...)`` rather
        # than ``setObjectName`` for the primary button so the
        # i18n registry's objectName key isn't clobbered.
        # Audit 2026-08-19 (Phase 5A, M-15): keep references to the
        # export buttons on the instance so the export worker can
        # disable / re-enable them while the IO is in flight. Without
        # these references the buttons were local-only, which made
        # double-click protection impossible.
        self._btn_export_xlsx = tr_button("restab.export.xlsx")
        self._btn_export_xlsx.setProperty("class", "primary")
        self._btn_export_xlsx.clicked.connect(self._export_xlsx)
        footer.addWidget(self._btn_export_xlsx)

        self._btn_export_json = tr_button("restab.export.json")
        self._btn_export_json.clicked.connect(self._export_json)
        footer.addWidget(self._btn_export_json)

        self._btn_export_csv = tr_button("restab.export.csv")
        self._btn_export_csv.clicked.connect(self._export_csv)
        footer.addWidget(self._btn_export_csv)

        self._btn_export_dwca = tr_button("restab.export.dwca")
        self._btn_export_dwca.clicked.connect(self._export_dwca)
        footer.addWidget(self._btn_export_dwca)

        # audit 2026-08-17 (GUI-A4): "Mark verified" / "Mark
        # unverified" buttons. POST the (paper_id, figure_id,
        # panel_path) triple to /review/correction so the operator
        # can flip image_verified without leaving the desktop app.
        # The button label and accessibility tooltip refresh on
        # language switch via the standard i18n widget registry.
        self._mark_verified_btn = tr_button("restab.detail.mark_verified")
        self._mark_verified_btn.setObjectName("restab.detail.mark_verified")
        self._mark_verified_btn.setProperty(
            "class",
            "primary",
        )
        self._mark_verified_btn.clicked.connect(
            lambda: self._flip_image_verified(True),
        )
        footer.addWidget(self._mark_verified_btn)

        self._mark_unverified_btn = tr_button("restab.detail.mark_unverified")
        self._mark_unverified_btn.setObjectName("restab.detail.mark_unverified")
        self._mark_unverified_btn.clicked.connect(
            lambda: self._flip_image_verified(False),
        )
        footer.addWidget(self._mark_unverified_btn)

        footer.addStretch(1)

        self._status = QLabel("")
        self._status.setObjectName("metricLabel")
        footer.addWidget(self._status)

        outer.addLayout(footer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _reset_detail_pane(self) -> None:
        """Clear the preview image and detail browser (not the title).

        M-15 fix: called from ``load_job`` so the operator never sees
        the previous job's detail HTML after loading a new job.
        """
        self._preview.clear()
        self._detail_browser.clear()

    def load_job(
        self, job_id: str, rows: list[dict[str, Any]], output_dir: str | None = None
    ) -> None:
        """Replace the current results with a new job's rows."""
        # M-15: clear the detail pane FIRST so an empty-job load never
        # shows the previous job's HTML while the table is rebuilding.
        self._reset_detail_pane()
        # Phase 56 audit: reset search and filters so stale state doesn't leak
        self._search_edit.blockSignals(True)
        self._search_edit.clear()
        self._search_edit.blockSignals(False)
        self._species_filter.blockSignals(True)
        self._species_filter.setCurrentIndex(0)
        self._species_filter.blockSignals(False)
        self._family_filter.blockSignals(True)
        self._family_filter.setCurrentIndex(0)
        self._family_filter.blockSignals(False)
        # Phase 56 extended: also reset _has_pbdb filter (was missing)
        self._has_pbdb.blockSignals(True)
        self._has_pbdb.setCurrentIndex(0)
        self._has_pbdb.blockSignals(False)
        self._current_job_id = job_id
        self._current_job_dir = output_dir
        self._all_rows = list(rows)
        # audit 2026-08-17 (GUI-A7): use i18n key restab.done with
        # {id} / {rows} placeholders so the title localises on
        # language switch. Previously the English literal was
        # always shown.
        self._title.setText(
            i18n._tr("restab.done", "Job {id}  ·  {rows} rows").format(
                id=job_id, rows=f"{len(self._all_rows):,}"
            )
        )
        self._refresh_filter_options()
        self._do_refresh_view()

    def append_rows(self, rows: list[dict[str, Any]], job_id: str | None = None) -> None:
        """Stream-add rows from a running job (live update).

        Phase 56 audit: job_id guard prevents rows from a stale job
        accumulating under a new job's ID.
        """
        if job_id is not None and job_id != self._current_job_id:
            self._log.warning(
                "append_rows called for different job_id %s (current %s), ignoring",
                job_id,
                self._current_job_id,
            )
            return
        self._all_rows.extend(rows)
        # audit 2026-08-17 (GUI-A7): use the i18n key restab.live so
        # the live-update title localises.
        self._title.setText(
            i18n._tr("restab.live", "Job {id}  ·  {rows} rows (live)").format(
                id=self._current_job_id or "?",
                rows=f"{len(self._all_rows):,}",
            )
        )
        self._refresh_filter_options()
        self._do_refresh_view()

    def _refresh_texts(self) -> None:
        """Re-translate column headers + filter labels."""
        for i, col in enumerate(RESULT_COLUMNS):
            item = self._table.horizontalHeaderItem(i)
            if item is not None:
                item.setText(i18n._tr(f"restab.col.{col.key}"))
        # "all"/"any" labels. We update only the *displayed text*
        # at the existing index rather than clearing + rebuilding,
        # which would lose the per-species / per-family items.
        for combo, key, sentinel in (
            (self._species_filter, "restab.filter.all", "__ALL__"),
            (self._family_filter, "restab.filter.all", "__ALL__"),
            (self._has_pbdb, "restab.filter.any", "__ANY__"),
        ):
            for i in range(combo.count()):
                ud = combo.itemData(i)
                if ud == sentinel:
                    combo.setItemText(i, i18n._tr(key))
                elif ud == "yes":
                    combo.setItemText(i, i18n._tr("restab.filter.yes"))
                elif ud == "no":
                    combo.setItemText(i, i18n._tr("restab.filter.no"))
        # Phase 56 audit: refresh search placeholder on language switch
        self._search_edit.setPlaceholderText(i18n._tr("restab.search.placeholder"))

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

    def closeEvent(self, event) -> None:  # noqa: N802
        """Phase 56 audit: remove i18n listener on widget destruction.

        Audit 2026-08-20 (B-2): also call :meth:`shutdown` so any
        in-flight ``_FlipVerifiedWorker`` / ``_ExportWorker`` QThread
        gets a chance to exit before the Qt destructor walks the
        children list. Without this the GUI close path raised
        ``QThread: Destroyed while thread is still running`` (exit
        code 134) on every close after a mark-verified click or an
        export.
        """
        self.shutdown()
        self._remove_i18n_listener()
        super().closeEvent(event)

    def shutdown(self) -> None:
        """Cancel and wait for any background workers to drain.

        Audit 2026-08-20 (B-2). Both the ``_FlipVerifiedWorker`` and
        the ``_ExportWorker`` are :class:`QThread` instances captured
        on ``self`` so the GC can't reap them mid-flight (PySide6
        footgun). They must be cancelled and ``wait()``ed for before
        the host widget is destroyed, otherwise Qt raises
        ``QThread: Destroyed while thread is still running`` and the
        process exits with code 134.

        For each worker we:

        1. No-op if the attribute is missing or ``None``.
        2. Skip if ``isRunning()`` is False (worker already finished).
        3. Flip the worker's ``_cancelled`` flag via ``cancel()`` so
           ``run()`` exits at the next checkpoint instead of being
           forcibly killed (``QThread.terminate()`` orphans
           subprocesses and is forbidden by the 2026-08-01 D20
           contract).
        4. ``wait(30000)`` with a finite timeout — a 30s cap matches
           the rest of the GUI shutdown paths. We swallow the
           ``RuntimeError`` that Qt raises if the QThread C++ object
           has already been deleted under us; that's a no-op race.
        5. Log a WARNING if ``wait`` timed out — the OS will reclaim
           the thread on process exit but we want it visible in the
           troubleshooting log.
        6. Drop the reference (``self._flip_worker = None`` /
           ``self._export_worker = None``) so a subsequent start
           rebuilds a fresh worker rather than racing with the
           deleted one.

        The method is idempotent — repeated calls after the workers
        have already drained are a no-op.
        """
        for attr in ("_flip_worker", "_export_worker"):
            worker = getattr(self, attr, None)
            if worker is None:
                continue
            try:
                running = bool(worker.isRunning())
            except Exception:
                # QThread C++ object already deleted under us —
                # nothing we can wait on.
                setattr(self, attr, None)
                continue
            if not running:
                # Already finished. Drop the reference so the next
                # export / flip constructs a fresh worker.
                setattr(self, attr, None)
                continue
            try:
                worker.cancel()
            except Exception as exc:
                self._log.warning(
                    "ResultsTab.shutdown: cancel() raised on %s: %s",
                    attr,
                    exc,
                )
            try:
                finished = worker.wait(30000)
            except RuntimeError:
                # Qt: "QThread: Destroyed while thread is still
                # running" — the C++ object is gone already. We can
                # only drop our reference and hope.
                setattr(self, attr, None)
                continue
            except Exception as exc:
                self._log.warning(
                    "ResultsTab.shutdown: wait() raised on %s: %s",
                    attr,
                    exc,
                )
                setattr(self, attr, None)
                continue
            if not finished:
                # 30s wasn't enough. We refuse to call ``terminate()``
                # (D20 contract); the OS will reclaim the thread on
                # process exit. Surface a warning so the operator
                # sees why the GUI close path took longer than usual.
                self._log.warning(
                    "ResultsTab.shutdown: %s did not finish within 30s; "
                    "letting the OS reclaim on process exit",
                    attr,
                )
            setattr(self, attr, None)

    def clear(self) -> None:
        self._all_rows = []
        self._filtered_rows = []
        self._current_job_id = None
        self._current_job_dir = None
        self._title.setText(i18n._tr("restab.no_job"))
        self._table.setRowCount(0)
        self._preview.clear()
        self._detail_browser.clear()
        # Format with {shown=0, total=0} so the empty-state shows
        # "0 / 0 rows" / "0 / 0 行" in the right language.
        self._count_label.setText(i18n._tr("restab.count").format(shown=0, total=0))

    # ------------------------------------------------------------------
    # Internal — filter + view refresh
    # ------------------------------------------------------------------
    def _refresh_filter_options(self) -> None:
        # Rebuild species + family dropdowns from current rows
        species = sorted({r.get("species", "") for r in self._all_rows if r.get("species")})
        families = sorted(
            {
                (((r.get("metadata") or {}).get("paleodb") or {}).get("taxonomy") or {}).get(
                    "family"
                )
                for r in self._all_rows
                if ((r.get("metadata") or {}).get("paleodb") or {}).get("taxonomy")
            }
        )
        families = [f for f in families if f]
        self._species_filter.blockSignals(True)
        self._species_filter.clear()
        # filter logic compares against "__ALL__" instead of
        # the translated label text.
        self._species_filter.addItem(i18n._tr("restab.filter.all"), userData="__ALL__")
        self._species_filter.addItems(species)
        self._species_filter.setCurrentIndex(0)
        self._species_filter.blockSignals(False)

        self._family_filter.blockSignals(True)
        self._family_filter.clear()
        self._family_filter.addItem(i18n._tr("restab.filter.all"), userData="__ALL__")
        self._family_filter.addItems(families)
        self._family_filter.setCurrentIndex(0)
        self._family_filter.blockSignals(False)

    def _filter_rows(self) -> list[dict[str, Any]]:
        search = self._search_edit.text().lower().strip()
        # ("__ALL__", "__ANY__", "yes", "no") instead of the
        # translated label text. Previously the comparison was
        # against the literal English "(all)"/"(any)"/"yes"/"no"
        # strings, which silently dropped every row after a
        # language switch.
        species_filter = self._species_filter.currentData() or ""
        if species_filter == "__ALL__":
            species_filter = ""
        family_filter = self._family_filter.currentData() or ""
        if family_filter == "__ALL__":
            family_filter = ""
        has_pbdb = self._has_pbdb.currentData() or "__ANY__"
        out: list[dict[str, Any]] = []
        for r in self._all_rows:
            if search:
                blob = " ".join(
                    [
                        str(r.get("species") or ""),
                        str(r.get("panel_id") or ""),
                        str(r.get("caption_snippet") or ""),
                        str(r.get("label_text") or ""),
                        (
                            ((r.get("metadata") or {}).get("paleodb") or {}).get("taxonomy") or {}
                        ).get("family")
                        or "",
                    ]
                ).lower()
                if search not in blob:
                    continue
            if species_filter and r.get("species") != species_filter:
                continue
            if family_filter:
                fam = (((r.get("metadata") or {}).get("paleodb") or {}).get("taxonomy") or {}).get(
                    "family"
                )
                if fam != family_filter:
                    continue
            if has_pbdb != "__ANY__":
                pbdb = (r.get("metadata") or {}).get("paleodb")
                want = has_pbdb == "yes"
                # the truthy dict from pbdb.get("taxonomy"). Without
                # bool(), `True != {'family': 'F1'}` is True and the
                # row is dropped even when it has PBDB data.
                have = bool(pbdb is not None and pbdb.get("looked_up") and pbdb.get("taxonomy"))
                if want != have:
                    continue
            out.append(r)
        return out

    def _refresh_view(self) -> None:
        """Debounced entry point for user-driven view rebuilds.

        M-14 fix: instead of rebuilding the full table synchronously on
        every keystroke or filter change, fire the 200 ms single-shot
        timer. The timer callback runs the actual expensive work. This
        keeps the GUI responsive at 20,000 rows.
        """
        if not self._view_rebuild_timer.isActive():
            self._view_rebuild_timer.start()

    def _do_refresh_view(self) -> None:
        """Actual table refresh — called either directly (load_job) or
        after the 200 ms debounce (user interactions).

        M-14 fix: the entry-point wrapper debounces; this method
        contains the synchronous work that was previously in the entry
        point.
        """
        # M-6: rebuild the stable key → index lookup first so
        # _on_row_selected can find live rows immediately after.
        self._build_row_lookup()
        rows = self._filter_rows()
        self._filtered_rows = rows
        # Phase 56 audit: use i18n template with placeholders so the
        # count label translates on language switch (previously a bare
        # English f-string).
        self._count_label.setText(
            i18n._tr("restab.count").format(shown=len(rows), total=len(self._all_rows))
        )
        # Populate table
        self._table.setRowCount(len(rows))
        for r_idx, row in enumerate(rows):
            for c_idx, col in enumerate(RESULT_COLUMNS):
                value = self._extract_column(row, col.key)
                item = QTableWidgetItem(str(value) if value is not None else "—")
                # M-6: store stable identity keys ONLY on the first
                # column's item so _on_row_selected can do a live-row
                # lookup instead of reading the stale QVariant copy.
                if c_idx == 0:
                    item.setData(Qt.UserRole + 1, row.get("paper_id") or "")
                    item.setData(Qt.UserRole + 2, row.get("figure_id") or "")
                    item.setData(Qt.UserRole + 3, row.get("panel_path") or "")
                    item.setData(Qt.UserRole, row)
                if col.key == "confidence" and isinstance(value, (int, float)):
                    item.setText(f"{value:.2f}")
                if col.key == "species":
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self._table.setItem(r_idx, c_idx, item)
        if rows:
            self._table.selectRow(0)

    def _extract_column(self, row: dict[str, Any], key: str) -> Any:
        """Pull a display value out of a result row dict."""
        if key == "family":
            # "or {}" at each hop: dict.get(k, default) only applies the
            # default when the key is MISSING — real rows carry
            # "taxonomy": null (PBDB reverse-fallback miss), which made
            # this chain raise AttributeError and kill the whole
            # results-table refresh (2026-09-04 user report).
            return (((row.get("metadata") or {}).get("paleodb") or {}).get("taxonomy") or {}).get(
                "family"
            )
        if key == "country":
            geo = (row.get("metadata") or {}).get("geology_links") or []
            for g in geo:
                if g.get("country"):
                    return g["country"]
            return None
        if key == "biozone":
            geo = (row.get("metadata") or {}).get("geology_links") or []
            for g in geo:
                if g.get("biozone"):
                    return g["biozone"]
            return None
        if key == "coord":
            geo = (row.get("metadata") or {}).get("geology_links") or []
            for g in geo:
                if g.get("latitude") is not None and g.get("longitude") is not None:
                    # audit 2026-07-26 M10: lat/lon may arrive as strings
                    # (e.g. "12.345N") from some extractors; :.3f on a str
                    # raises TypeError. Coerce to float and fall back to
                    # the raw value on failure.
                    try:
                        return f"{float(g['latitude']):.3f}, {float(g['longitude']):.3f}"
                    except (TypeError, ValueError):
                        return f"{g['latitude']}, {g['longitude']}"
            return None
        # pipeline output, NOT at the row top level (which is None).
        # Before the fix, the Page column always showed "—" because
        # the code fell through to ``row.get("page_index")`` which
        # returned None for every pipeline-produced row.
        if key == "page_index":
            return (row.get("metadata") or {}).get("page_index")
        return row.get(key)

    # ------------------------------------------------------------------
    # Row selection → detail panel
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_figure_image_path(row: dict[str, Any]) -> Path | None:
        """Pick the figure-level image path for a row, preferring top-level.

        Audit 2026-08-20 (M-3): the previous lookup only checked
        ``row["metadata"]["figure_image_path"]``. :class:`PanelRecord`
        and :class:`PipelineWorker` (Phase 5D / 6A) emit
        ``figure_image_path`` at the row top-level too — a real-data
        regression: 100% of post-Phase-5D rows had the field at the
        top level only, so the preview silently rendered nothing.

        Resolution order:

        1. ``row["figure_image_path"]`` — top-level (canonical in v1.1.0).
        2. ``row["metadata"]["figure_image_path"]`` — pre-v1.1.0 fallback.
        3. ``row["metadata"]["primary_image"]`` — older fallback.
        4. ``row["metadata"]["image_path"]`` — legacy schema field.

        Returns the first path whose ``Path(...).exists()`` is True,
        or ``None`` if none of the candidates point at a real file.
        """
        if not isinstance(row, dict):
            return None
        md = row.get("metadata") or {}
        candidates: list[Any] = [
            row.get("figure_image_path"),
            md.get("figure_image_path"),
            md.get("primary_image"),
            md.get("image_path"),
        ]
        for v in candidates:
            if v is None:
                continue
            if isinstance(v, (str, os.PathLike)):
                try:
                    p = Path(str(v))
                except Exception:
                    continue
                if p.exists():
                    return p
        return None

    def _build_row_lookup(self) -> None:
        """Rebuild ``self._row_lookup`` from ``self._all_rows``.

        M-6 fix: used to look up live row dicts in ``_on_row_selected``
        instead of reading a stale QVariant copy from the
        QTableWidgetItem. The lookup key is ``(paper_id, figure_id,
        panel_path)`` — the same triple the flip worker matches on.
        """
        self._row_lookup.clear()
        for idx, r in enumerate(self._all_rows):
            key = (
                r.get("paper_id") or "",
                r.get("figure_id") or "",
                r.get("panel_path") or "",
            )
            self._row_lookup[key] = idx

    def _on_row_selected(self) -> None:
        items = self._table.selectedItems()
        if not items:
            return
        # M-6 fix: read stable keys from the item, look up the live
        # row in _row_lookup, and render that — never the QVariant
        # copy stored via setData(UserRole, row) which goes stale
        # after _flip_image_verified mutates _all_rows in place.
        item = items[0]
        paper_id = item.data(Qt.UserRole + 1) or ""
        figure_id = item.data(Qt.UserRole + 2) or ""
        panel_path = item.data(Qt.UserRole + 3) or ""
        key = (paper_id, figure_id, panel_path)
        idx = self._row_lookup.get(key)
        if idx is None:
            return
        row = self._all_rows[idx]
        self._render_detail(row)
        # Load image preview. audit 2026-07-31: the row bbox is in
        # PAGE/FIURE coordinates but panel_path is a PANEL CROP —
        # overlaying page coords on the crop drew rectangles far
        # outside the image (real data: 4/5 rows invisible). Prefer
        # the figure-level image for bbox overlays; a crop is shown
        # without overlays.
        #
        # Audit 2026-08-20 (M-3): delegate to the helper which also
        # honours the top-level ``figure_image_path`` key the pipeline
        # emits (the metadata-only lookup missed those rows).
        figure_img = self._resolve_figure_image_path(row)
        if figure_img is not None:
            self._preview.set_image(figure_img)
            self._preview.set_bboxes([row])  # bbox is figure-level
            return
        panel_path = row.get("panel_path")
        if panel_path:
            p = Path(panel_path)
            if p.exists():
                self._preview.set_image(p)
                # crop coords are NOT page coords — no overlay
                self._preview.set_bboxes([])
                return
        self._preview.clear()
        self._preview.set_bboxes([])

    def _render_detail(self, row: dict[str, Any]) -> None:
        """Render the right-side detail panel as HTML.

        Mirrors the Web UI's openImageModal() fields as closely as possible:
        paper_id / figure_id / panel_id / extraction_source badge /
        species / confidence / geology_scope badge / old_panel_id record /
        BBox / paper_metadata (journal/year/review_reasons) /
        caption_snippet / sample IDs / geology_links with full
        age · Ma range · lithology · formation · member · group ·
        biozone · locality · country · modern coords · paleo coords /
        evidence_text.
        """
        md = row.get("metadata") or {}
        pm = row.get("paper_metadata") or {}
        paleodb = md.get("paleodb") or {}
        tax = paleodb.get("taxonomy") or {}
        geo_links = md.get("geology_links") or []
        html = []
        html.append(
            f"<html><head><style>{_DETAIL_BADGE_CSS}</style>"
            "</head><body style='font-family:sans-serif;padding:0;margin:0'>"
        )

        # ── Heading ──────────────────────────────────────────────
        html.append(
            f"<h2 style='color:{_DETAIL_HEADING_COLOR};margin:8px 8px 2px'>"
            f"{html_escape(row.get('species') or '(no species)')}</h2>"
        )
        panel_id = row.get("panel_id") or ""
        old_panel_id = md.get("old_panel_id")
        if old_panel_id and old_panel_id != panel_id:
            html.append(
                f"<div style='color:#888;font-size:11px;margin:0 8px 6px'>"
                f"&#8594; was <code>{html_escape(str(old_panel_id))}</code></div>"
            )
        elif panel_id:
            html.append(
                f"<div style='color:#666;font-size:12px;margin:0 8px 6px'>"
                f"{html_escape(panel_id)}</div>"
            )

        # ── ID / metadata grid ─────────────────────────────────
        geo_scope = md.get("geology_scope") or "none"
        # Class-level dicts (defined once, not per-call).
        scope_key = _SCOPE_KEYS.get(geo_scope, "restab.detail.scope.none")
        scope_label = i18n._tr(scope_key)
        cls = _SCOPE_CLASSES.get(geo_scope, "badge-muted")
        ocr_src = md.get("extraction_source") or ""
        ocr_key = _OCR_KEYS.get(ocr_src)
        ocr_label = i18n._tr(ocr_key) if ocr_key else (ocr_src or "—")
        ocr_cls = _OCR_CLASSES.get(ocr_src, "badge-muted")
        conf = row.get("confidence")
        conf_str = f"{conf * 100:.0f}%" if isinstance(conf, (int, float)) else "—"

        # audit 2026-08-17 (GUI-A3): build the v1.1.0 confidence
        # interval display string from the Wilson 95% CI bounds the
        # pipeline stamps on every PanelRecord. We tolerate either
        # floats stored on the row dict or nested under ``metadata``;
        # the schema contract keeps them on the top level so the row
        # path is the canonical one.
        ci_low = row.get("confidence_interval_low")
        ci_high = row.get("confidence_interval_high")
        ci_str = None
        if isinstance(ci_low, (int, float)) and isinstance(ci_high, (int, float)):
            ci_str = f"[{ci_low * 100:.0f}%, {ci_high * 100:.0f}%]"

        # audit 2026-08-17 (GUI-A3): image_verified + review_priority
        # badges. The pipeline computes review_priority in
        # [0, 1, 2] (low/med/high); we colour-code the badge so
        # operators can scan the table for rows needing urgent
        # review. ``image_verified`` is a checkmark badge so the
        # operator's manual-review state survives a GUI restart.
        image_verified = bool(row.get("image_verified", False))
        if image_verified:
            verified_badge = (
                "<span class='badge-info' "
                "style='padding:1px 6px;border-radius:3px;font-size:11px;"
                "background:#d6f5d6;color:#1c7a1c'>"
                f"✓ {html_escape(i18n._tr('restab.detail.image_verified'))}"
                "</span>"
            )
        else:
            verified_badge = (
                "<span class='badge-muted' "
                "style='padding:1px 6px;border-radius:3px;font-size:11px'>"
                f"{html_escape(i18n._tr('restab.detail.image_unverified'))}"
                "</span>"
            )

        review_priority = row.get("review_priority", 0)
        try:
            review_priority = int(review_priority)
        except (TypeError, ValueError):
            review_priority = 0
        if review_priority < 0:
            review_priority = 0
        elif review_priority > 2:
            review_priority = 2
        priority_styles = {
            0: ("badge-muted", "#eee", "#888"),
            1: ("badge-warn", "#ffe0a0", "#c07800"),
            2: ("badge-error", "#ffd6d6", "#a01818"),
        }
        pcls, pbg, pfg = priority_styles[review_priority]
        priority_badge = (
            f"<span class='{pcls}' "
            f"style='padding:1px 6px;border-radius:3px;font-size:11px;"
            f"background:{pbg};color:{pfg}'>"
            f"{html_escape(i18n._tr('restab.detail.review_priority'))}: "
            f"{review_priority}</span>"
        )

        # audit 2026-08-17 (GUI-A3): render the scale bar metadata
        # the panel metadata carries. The ScaleBarRecord schema
        # field carries ``value`` (the number) + ``unit`` (e.g.
        # ``"μm"``) + optional ``pixel_length`` (px) so the
        # operator can sanity-check μm/px ratio.
        scale_bar = md.get("scale_bar") or {}
        scale_bar_str = None
        if isinstance(scale_bar, dict):
            sb_value = scale_bar.get("value")
            sb_unit = scale_bar.get("unit")
            sb_px = scale_bar.get("pixel_length")
            if sb_value is not None or sb_unit or sb_px:
                bits: list[str] = []
                if sb_value is not None:
                    unit_str = f" {html_escape(str(sb_unit))}" if sb_unit else ""
                    bits.append(f"{sb_value}{unit_str}")
                if sb_px is not None:
                    bits.append(f"({sb_px}px)")
                if bits:
                    scale_bar_str = f"{' '.join(bits)}"

        html.append(
            "<table style='font-size:12px;border-collapse:collapse;width:100%;margin-bottom:8px'>"
        )
        meta_pairs = [
            (i18n._tr("restab.detail.paper_id"), html_escape(row.get("paper_id") or "—")),
            (i18n._tr("restab.detail.figure_id"), html_escape(row.get("figure_id") or "—")),
            (i18n._tr("restab.detail.panel_label"), html_escape(panel_id or "—")),
            (
                i18n._tr("restab.detail.page"),
                md.get("page_index") if md.get("page_index") is not None else "—",
            ),
            (
                i18n._tr("restab.detail.source"),
                f"<span class='{ocr_cls}' style='padding:1px 5px;border-radius:3px;font-size:11px'>{html_escape(ocr_label)}</span>",
            ),
            (i18n._tr("restab.detail.confidence"), conf_str),
        ]
        # Confidence interval — append as a parenthetical next to
        # confidence if available, otherwise include it as its own
        # row. Either way, the operator sees the Wilson bounds.
        if ci_str is not None:
            meta_pairs.append(
                (
                    i18n._tr("restab.detail.ci"),
                    ci_str,
                )
            )
        meta_pairs.append(
            (
                i18n._tr("restab.detail.geo_scope"),
                f"<span class='{cls}' style='padding:1px 5px;border-radius:3px;font-size:11px'>{html_escape(scope_label)}</span>",
            ),
        )
        # audit 2026-08-17 (GUI-A3): surface image_verified + review
        # priority as their own meta rows.
        meta_pairs.append((i18n._tr("restab.detail.image_verified"), verified_badge))
        meta_pairs.append((i18n._tr("restab.detail.review_priority"), priority_badge))
        if scale_bar_str:
            meta_pairs.append((i18n._tr("restab.detail.scale_bar"), scale_bar_str))
        # Phase 56 audit: guard against non-numeric bbox elements (None, str)
        bbox = row.get("bbox")
        if (
            isinstance(bbox, list)
            and len(bbox) == 4
            and any(v > 0 for v in bbox if isinstance(v, (int, float)))
        ):
            html.append(
                f"<tr><td style='padding:2px 8px 2px 0;color:#888'>{i18n._tr('restab.detail.bbox')}</td>"
                f"<td style='padding:2px 0;font-family:monospace;font-size:11px'>"
                f"[{', '.join(str(v) for v in bbox)}]</td></tr>"
            )
        for k, v in meta_pairs:
            html.append(
                f"<tr><td style='padding:2px 8px 2px 0;color:#888;white-space:nowrap'>{k}</td>"
                f"<td style='padding:2px 0'>{v}</td></tr>"
            )
        html.append("</table>")

        # ── Paper metadata ──────────────────────────────────────
        title = pm.get("title") or ""
        authors = pm.get("authors") or []
        journal = pm.get("journal") or ""
        year = pm.get("year") or ""
        doi = pm.get("doi") or ""
        review_reasons = (pm.get("review_reasons") or [])[:3]
        if title:
            parts = [f"<b>{html_escape(title)}</b>"]
            if authors:
                parts.append(f"<span style='color:#666'>{html_escape(str(authors)[:120])}</span>")
            if journal:
                parts.append(f"<span style='color:#666'>{html_escape(journal)}</span>")
            if year:
                parts.append(f"<span style='color:#666'>({html_escape(str(year))})</span>")
            html.append(
                "<div style='padding:4px 8px;border-top:1px solid #eee'>"
                + " ".join(parts)
                + "</div>"
            )
            if doi:
                html.append(
                    f"<div style='padding:0 8px 4px;font-size:11px;color:#666'>DOI: <code>{html_escape(doi)}</code></div>"
                )
            if review_reasons:
                rw = "; ".join(str(r) for r in review_reasons)
                html.append(
                    f"<div style='padding:0 8px 4px'><span class='badge-warn' style='font-size:11px'>"
                    f"&#9888; {html_escape(rw[:80])}</span></div>"
                )

        # ── Caption snippet ───────────────────────────────────
        cap_snippet = row.get("caption_snippet") or ""
        if cap_snippet:
            display = cap_snippet[:_CAPTION_SNIPPET_MAX] + (
                "…" if len(cap_snippet) > _CAPTION_SNIPPET_MAX else ""
            )
            html.append(
                f"<div style='padding:6px 8px;border-top:1px solid #eee'>"
                f"<b style='font-size:12px'>{i18n._tr('restab.detail.caption', 'Caption')}</b>"
                f"<pre style='background:#f5f5f5;padding:6px;border-radius:4px;margin:4px 0 0;"
                f"white-space:pre-wrap;font-family:monospace;font-size:11px'>"
                f"{html_escape(display)}</pre></div>"
            )

        # ── Evidence text ─────────────────────────────────────
        ev_text = md.get("evidence_text") or ""
        if ev_text:
            html.append(
                f"<div style='padding:6px 8px;border-top:1px solid #eee'>"
                f"<b style='font-size:12px'>{html_escape(i18n._tr('restab.detail.evidence'))}</b>"
                f"<pre style='background:#fffbe5;padding:6px;border-radius:4px;margin:4px 0 0;"
                f"white-space:pre-wrap;font-family:monospace;font-size:11px;border:1px solid #ffeaa7'>"
                f"{html_escape(ev_text[:500])}"
                f"{'…' if len(ev_text) > 500 else ''}</pre></div>"
            )

        # ── PBDB taxonomy ─────────────────────────────────────
        if tax:
            html.append(
                f"<div style='padding:6px 8px;border-top:1px solid #eee'>"
                f"<b style='font-size:12px'>{html_escape(i18n._tr('restab.detail.pbdb_tax'))}</b>"
                f"<table style='font-size:12px;margin-top:4px'>"
            )
            # audit 2026-08-17 (GUI-A1): the previous version used
            # English literals ("Kingdom"/"Phylum"/...) directly,
            # which bypassed the existing zh_CN translations at
            # restab.detail.kingdom / .phylum / .class / .order /
            # restab.detail.family (restab.detail.family is the
            # legacy "family: {value}" template, so for the
            # taxonomy table we use the standalone family label).
            # The keys already existed; the table just didn't use
            # them. ``class_`` (the PBDB taxonomy dict key) maps to
            # the ``restab.detail.class`` i18n label. ``Genus`` and
            # ``Source`` have no defined i18n keys, so we add them
            # as fallbacks here and also in strings_en / strings_zh.
            tax_rows = [
                ("restab.detail.kingdom", tax.get("kingdom")),
                ("restab.detail.phylum", tax.get("phylum")),
                ("restab.detail.class", tax.get("class_")),
                ("restab.detail.order", tax.get("order")),
                ("restab.detail.family", tax.get("family")),
                ("restab.detail.genus", tax.get("genus")),
                ("restab.detail.source", tax.get("source")),
            ]
            for key, v in tax_rows:
                if not v:
                    continue
                label = i18n._tr(key)
                # ``restab.detail.family`` is the templated
                # "family: {value}" / "科：{value}" used inline next
                # to a species name. The taxonomy table cell layout
                # wants a bare label, so strip the template part.
                if "{value}" in label:
                    label = label.split(":", 1)[0].strip() or label.split("{", 1)[0].strip()
                html.append(
                    f"<tr><td style='padding:1px 8px 1px 0;color:#888'>"
                    f"{html_escape(label)}</td>"
                    f"<td style='padding:1px 0'>{html_escape(str(v))}</td></tr>"
                )
            html.append("</table></div>")

        # ── Sample IDs ─────────────────────────────────────────
        sample_ids = list({str(g.get("sample_id")) for g in geo_links if g.get("sample_id")})
        if sample_ids:
            html.append(
                f"<div style='padding:4px 8px;border-top:1px solid #eee;font-size:12px'>"
                f"<b>{i18n._tr('restab.detail.sample_ids', 'Sample IDs')}:</b> "
                f"<code style='font-size:11px'>{html_escape(', '.join(sample_ids[:10]))}"
                f"{' …' if len(sample_ids) > 10 else ''}</code></div>"
            )

        # ── Geology links ────────────────────────────────────
        if geo_links:
            # audit 2026-08-17 (GUI-A6): honour the i18n key instead
            # of the hardcoded English literal. Previously the
            # zh_CN translation at restab.detail.geo_links was
            # silently bypassed.
            html.append(
                f"<div style='padding:6px 8px;border-top:1px solid #eee'>"
                f"<b style='font-size:12px'>"
                f"{i18n._tr('restab.detail.geo_links', '{count}').format(count=len(geo_links))}</b>"
            )
            for g in geo_links[:8]:
                bits = []
                # Phase 65 Plan A.6: link source badge per geology link.
                # The cross-figure linker tags each appended entry with
                # ``coord_source = cross_figure_linker:<source>`` so we
                # can render a coloured chip without parsing the rest
                # of the link.
                _emit_link_source_badge(html, g.get("coord_source") or "")
                # Age / chronostratigraphy
                age = g.get("age") or g.get("chronostratigraphy") or ""
                if age:
                    bits.append(f"<strong>{html_escape(age)}</strong>")
                # Ma range
                ma_top = g.get("ma_top")
                ma_base = g.get("ma_base")
                ma_fmt = _fmt_float(ma_top, "{:.2f}")
                base_fmt = _fmt_float(ma_base, "{:.2f}")
                if ma_fmt is not None and base_fmt is not None:
                    bits.append(f"<span style='color:#555'>{ma_fmt}–{base_fmt} Ma</span>")
                # Lithology / formation / member / group
                for k in ("lithology", "formation", "member", "group"):
                    v = g.get(k)
                    if v:
                        tag = (
                            f"<em>{html_escape(str(v))}</em>"
                            if k == "formation"
                            else html_escape(str(v))
                        )
                        bits.append(f"<span style='font-size:11px'>{tag}</span>")
                # Biozone
                bz = g.get("biozone")
                if bz:
                    bits.append(f"<span style='font-size:11px'>{html_escape(bz)}</span>")
                # Locality / country
                loc = g.get("locality") or ""
                ctry = g.get("country") or ""
                if loc:
                    bits.append(f"<span style='font-size:11px'>{html_escape(loc)}</span>")
                if ctry:
                    bits.append(
                        f"<span style='font-size:11px;color:#666'>{html_escape(ctry)}</span>"
                    )
                # Modern coords
                mlat = g.get("modern_latitude")
                mlon = g.get("modern_longitude")
                mlat_fmt = _fmt_float(mlat, "{:.3f}")
                mlon_fmt = _fmt_float(mlon, "{:.3f}")
                if mlat_fmt is not None and mlon_fmt is not None:
                    bits.append(
                        f"<span style='font-size:11px;color:#27ae60'>now "
                        f"{mlat_fmt}, {mlon_fmt}</span>"
                    )
                # Paleo coords
                plat = g.get("paleo_latitude")
                plon = g.get("paleo_longitude")
                plate_id = g.get("plate_id")
                recon_age = g.get("reconstruction_age_ma")
                plat_fmt = _fmt_float(plat, "{:.3f}")
                plon_fmt = _fmt_float(plon, "{:.3f}")
                if plat_fmt is not None and plon_fmt is not None:
                    paleo_bits = [f"<span style='color:#e67e22'>@ {plat_fmt}, {plon_fmt}</span>"]
                    if plate_id:
                        paleo_bits.append(f"<span style='color:#e67e22'>plate={plate_id}</span>")
                    recon_fmt = _fmt_float(recon_age, "{:.1f}")
                    if recon_fmt is not None:
                        paleo_bits.append(f"<span style='color:#e67e22'>{recon_fmt} Ma</span>")
                    bits.append(" ".join(paleo_bits))
                # Confidence
                gc = g.get("geology_confidence") or g.get("confidence")
                gc_fmt = _fmt_float(gc, "{:.0f}")
                if gc_fmt is not None:
                    bits.append(f"<span style='color:#888;font-size:10px'>({gc_fmt}%)</span>")
                if bits:
                    html.append(
                        "<div style='font-size:11px;background:#f7f9fc;padding:4px 6px;"
                        "border-radius:3px;margin-top:4px'>" + " · ".join(bits) + "</div>"
                    )
            if len(geo_links) > 8:
                # audit 2026-08-17 (GUI-A6): use the i18n key for the
                # overflow badge ("… and N more" / "… 还有 N 条").
                html.append(
                    f"<div style='color:#888;font-size:11px;margin-top:4px'>"
                    f"{i18n._tr('restab.detail.geo_links_more', '{n}').format(n=len(geo_links) - 8)}</div>"
                )
            html.append("</div>")

        # ── Cross-figure linker provenance (Phase 65 Plan A.6) ────
        # If the linker ran, surface a single summary chip near the
        # geology links block so operators can spot "unlinked" rows at
        # a glance without scanning the whole link list.
        link_src = md.get("link_source")
        if link_src:
            _emit_link_summary_badge(html, link_src, md.get("link_confidence") or 0.0)

        # ── Visual-coordinate links (Phase 66 Plan C.6) ──────────
        # Phase C fires only when Phase A Strategy-1 didn't reach
        # confidence 1.0 AND the paper has both a plate and a strat
        # column / paleogeographic map. Surface the visual links in
        # a dedicated section so operators can audit Phase C's
        # precision refinements next to Phase A's text-only chips.
        visual_links = md.get("cross_figure_visual_links")
        if isinstance(visual_links, list) and visual_links:
            html.append(
                f"<div style='padding:6px 8px;border-top:1px solid #eee'>"
                f"<b style='font-size:12px'>"
                f"{i18n._tr('restab.detail.visual_links')}</b>"
            )
            html.append("<table style='font-size:12px;margin-top:4px;border-collapse:collapse'>")
            for vl in visual_links:
                if not isinstance(vl, dict):
                    continue
                target = vl.get("target_figure_id") or "—"
                layer = vl.get("target_layer")
                layer_str = str(layer) if layer is not None else "—"
                age = vl.get("target_age") or "—"
                formation = vl.get("target_formation") or "—"
                try:
                    conf_val = float(vl.get("confidence", 0.0) or 0.0)
                except (TypeError, ValueError):
                    conf_val = 0.0
                conf_val = max(0.0, min(1.0, conf_val))
                conf_pct = f"{conf_val * 100:.0f}%"
                row_html = (
                    "<tr>"
                    f"<td style='padding:1px 6px;color:#666'>"
                    f"{html_escape(i18n._tr('restab.detail.visual_target'))}</td>"
                    f"<td style='padding:1px 6px'>"
                    f"{html_escape(str(target))}</td>"
                    "</tr>"
                    "<tr>"
                    f"<td style='padding:1px 6px;color:#666'>"
                    f"{html_escape(i18n._tr('restab.detail.visual_layer'))}</td>"
                    f"<td style='padding:1px 6px'>{html_escape(layer_str)}</td>"
                    "</tr>"
                    "<tr>"
                    f"<td style='padding:1px 6px;color:#666'>"
                    f"{html_escape(i18n._tr('restab.detail.visual_age'))}</td>"
                    f"<td style='padding:1px 6px'>{html_escape(str(age))}</td>"
                    "</tr>"
                    "<tr>"
                    f"<td style='padding:1px 6px;color:#666'>"
                    f"{html_escape(i18n._tr('restab.detail.visual_formation'))}</td>"
                    f"<td style='padding:1px 6px'>{html_escape(str(formation))}</td>"
                    "</tr>"
                    "<tr>"
                    f"<td style='padding:1px 6px;color:#666'>"
                    f"{html_escape(i18n._tr('restab.detail.visual_confidence'))}</td>"
                    f"<td style='padding:1px 6px'>"
                    f"<span class='badge-info' style='padding:1px 5px;"
                    f"border-radius:3px;font-size:11px'>{conf_pct}</span></td>"
                    "</tr>"
                )
                html.append(row_html)
            html.append("</table></div>")
        elif isinstance(visual_links, list) and link_src not in (
            "sample_match",
            None,
        ):
            # Empty visual_links on a panel whose Phase A didn't nail
            # it via Strategy 1 — operators want a one-liner confirming
            # Phase C considered and rejected the visual link.
            html.append(
                f"<div style='padding:4px 8px;border-top:1px solid #eee;"
                f"font-size:11px;color:#888'>"
                f"{i18n._tr('restab.detail.visual_empty')}</div>"
            )

        # ── Schematic content (Phase 64 Plan B Task B.7) ─────────
        # When the figure was classified as schematic / diagram /
        # reconstruction / phylogenetic, the M3 ``extract_schematic``
        # result lives on ``metadata.figure_schematic_data``. Render
        # a compact summary so the operator can confirm the
        # extraction worked without opening the raw JSON.
        sch = md.get("figure_schematic_data")
        if isinstance(sch, dict):
            html.append(
                f"<div style='padding:6px 8px;border-top:1px solid #eee'>"
                f"<b style='font-size:12px'>{i18n._tr('restab.detail.schematic')}</b>"
            )
            sch_type = sch.get("figure_type") or ""
            text_elements = sch.get("text_elements") or []
            if not isinstance(text_elements, list):
                text_elements = []
            relationships = sch.get("relationships") or []
            if not isinstance(relationships, list):
                relationships = []
            facts = sch.get("extracted_facts") or {}
            if not isinstance(facts, dict):
                facts = {}
            try:
                sch_conf = float(sch.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                sch_conf = 0.0
            sch_conf = max(0.0, min(1.0, sch_conf))
            html.append("<table style='font-size:12px;margin-top:4px;border-collapse:collapse'>")
            sch_rows = [
                (
                    i18n._tr("restab.detail.schematic_type"),
                    f"<span class='badge-info' style='padding:1px 5px;border-radius:3px;font-size:11px'>{html_escape(sch_type or '—')}</span>",
                ),
                (
                    i18n._tr("restab.detail.schematic_text_count"),
                    str(len(text_elements)),
                ),
                (
                    i18n._tr("restab.detail.schematic_rel_count"),
                    str(len(relationships)),
                ),
                (
                    i18n._tr("restab.detail.schematic_confidence"),
                    f"{sch_conf * 100:.0f}%",
                ),
            ]
            # Sample of text elements (first 3) so the operator can
            # see what M3 read without opening the raw JSON.
            if text_elements:
                sample_parts: list[str] = []
                for el in text_elements[:3]:
                    if not isinstance(el, dict):
                        continue
                    txt = str(el.get("text", "") or "").strip()
                    typ = str(el.get("type", "") or "").strip()
                    if not txt:
                        continue
                    sample_parts.append(
                        f"<span style='background:#eef;color:#335;padding:1px 4px;"
                        f"border-radius:3px;font-size:11px;margin-right:3px'>"
                        f"{html_escape(txt)}"
                        f"<span style='color:#888;font-size:10px'> ({html_escape(typ)})</span>"
                        f"</span>"
                    )
                if sample_parts:
                    sch_rows.append(
                        (
                            i18n._tr("restab.detail.schematic"),
                            " ".join(sample_parts),
                        )
                    )
            # Extracted-facts key=value rows. We keep the same
            # compact format the operator sees on the geology panel.
            for fact_key, fact_label in (
                ("ages_mentioned", i18n._tr("restab.detail.schematic_ages")),
                ("geographic_names", i18n._tr("restab.detail.schematic_geo")),
                ("taxa_mentioned", i18n._tr("restab.detail.schematic_taxa")),
            ):
                vals = facts.get(fact_key) or []
                if isinstance(vals, list) and vals:
                    joined = ", ".join(str(v) for v in vals[:8])
                    if len(vals) > 8:
                        joined += f" (+{len(vals) - 8})"
                    sch_rows.append((fact_label, html_escape(joined)))
            for k, v in sch_rows:
                html.append(
                    f"<tr><td style='padding:2px 8px 2px 0;color:#888;white-space:nowrap'>{k}</td>"
                    f"<td style='padding:2px 0'>{v}</td></tr>"
                )
            html.append("</table>")
            html.append("</div>")

        html.append("</body></html>")
        self._detail_browser.setHtml("\n".join(html))

    # ------------------------------------------------------------------
    # Exports — Audit 2026-08-19 (M-15) initialised this section with
    # the minimum viable log-and-warn patch. Audit 2026-08-19 (Phase
    # 5A, M-15) supersedes that with the proper QThread worker rewrite
    # (mirroring B-14 ``_FlipVerifiedWorker``). See ``_ExportWorker``
    # below and ``_run_export_worker`` for the async path.
    # ------------------------------------------------------------------
    # Audit 2026-08-19 (Phase 5A, M-15): the 4 export functions below
    # used to write 50k+ rows synchronously on the GUI thread, freezing
    # the UI for 5–30 s on large jobs. Phase 1F added ERROR-level
    # logging so silent failures were at least visible; this sweep
    # moves the actual write to a :class:`_ExportWorker` ``QThread``
    # (mirroring the B-14 ``_FlipVerifiedWorker`` approach). The slot
    # now only does:
    #   1. validation (rows + file dialog),
    #   2. snapshot of the run_output dict on the GUI thread (so the
    #      worker can't race against subsequent filter / search edits),
    #   3. construct + start the worker,
    #   4. wire success / error signals back to UI updates.
    # The buttons are disabled while the worker is in flight and
    # re-enabled from a single ``_re_enable_export_buttons()`` helper
    # so success / failure paths agree.
    def _export_xlsx(self) -> None:
        if not self._filtered_rows:
            QMessageBox.information(
                self,
                i18n._tr("restab.export.xlsx"),
                i18n._tr("jobstab.export.no_rows"),
            )
            return
        default_path = str(
            Path(self._current_job_dir or ".") / f"{self._current_job_id or 'results'}.xlsx"
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            i18n._tr("restab.export.xlsx_title"),
            default_path,
            "Excel files (*.xlsx)",
        )
        if not path:
            return
        try:
            run_output = self._build_run_output()
            self._run_export_worker("xlsx", run_output, path)
        except Exception as exc:
            # Defensive net: if worker construction / dispatch fails
            # synchronously (e.g. unknown fmt from a future refactor),
            # we still want the operator to see a stack trace in the
            # troubleshooting log. The async path's failure is
            # handled separately in ``_run_export_worker._on_error``.
            self._log.error(
                "Export failed (xlsx → %s): %s",
                path,
                exc,
                exc_info=True,
            )

    def _export_json(self) -> None:
        if not self._filtered_rows:
            QMessageBox.information(
                self, i18n._tr("restab.export.json"), i18n._tr("jobstab.export.no_rows")
            )
            return
        default_path = str(
            Path(self._current_job_dir or ".") / f"{self._current_job_id or 'results'}.json"
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            i18n._tr("restab.export.json_title"),
            default_path,
            "JSON files (*.json)",
        )
        if not path:
            return
        try:
            run_output = self._build_run_output()
            self._run_export_worker("json", run_output, path)
        except Exception as exc:
            self._log.error(
                "Export failed (json → %s): %s",
                path,
                exc,
                exc_info=True,
            )

    def _export_csv(self) -> None:
        if not self._filtered_rows:
            QMessageBox.information(
                self, i18n._tr("restab.export.csv"), i18n._tr("jobstab.export.no_rows")
            )
            return
        default_path = str(
            Path(self._current_job_dir or ".") / f"{self._current_job_id or 'results'}.csv"
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            i18n._tr("restab.export.csv_title"),
            default_path,
            "CSV files (*.csv)",
        )
        if not path:
            return
        # Audit 2026-08-19 (Phase 5A, M-15): snapshot the row list
        # before handing it off to the worker. The worker can't safely
        # call ``_extract_column`` on the live ``self._filtered_rows``
        # because the GUI thread may mutate the list as the operator
        # edits the search / filter box. We snapshot the *visible*
        # column values into a plain list of dicts so the worker is
        # decoupled from any subsequent GUI mutations.
        #
        # We also do the Phase 63 Plan 6.5 / 6.8 / 6.10 sanitisation
        # HERE (formula-injection, NaN/Inf, ``_csv_cell`` wrapper) so
        # the worker only does the disk IO. The sanitiser code path
        # stays in the slot — Phase 63's source-guard tests on
        # ``_export_csv`` keep passing without the worker re-implementing
        # the sanitiser.
        from math import isinf, isnan

        from ..export import _csv_cell
        from ..exporters.analysis import _sanitise_csv_cell
        from .constants import RESULT_COLUMNS

        column_keys = [c.key for c in RESULT_COLUMNS]
        snapshot: list[dict[str, Any]] = []
        for r in self._filtered_rows:
            out: dict[str, Any] = {}
            for k in column_keys:
                v = self._extract_column(r, k)
                # NaN/Inf → empty string first (Phase 63 Plan 6.8)
                if isinstance(v, float) and (isnan(v) or isinf(v)):
                    v = ""
                # Then formula-injection sanitisation (Phase 63 Plan 6.5)
                v = _csv_cell(_sanitise_csv_cell(v))
                out[k] = v
            snapshot.append(out)
        # Worker writes the file with utf-8-sig BOM (Phase 63 Plan 6.10).
        # The sanitised snapshot above is already safe to write.
        try:
            run_output = self._build_run_output()
            self._run_export_worker(
                "csv",
                run_output,
                path,
                rows=snapshot,
                use_utf8_sig=True,
            )
        except Exception as exc:
            self._log.error(
                "Export failed (csv → %s): %s",
                path,
                exc,
                exc_info=True,
            )

    def _export_dwca(self) -> None:
        if not self._filtered_rows:
            QMessageBox.information(
                self, i18n._tr("restab.export.dwca"), i18n._tr("jobstab.export.no_rows")
            )
            return
        default_path = str(
            Path(self._current_job_dir or ".") / f"{self._current_job_id or 'results'}.zip"
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            i18n._tr("restab.export.dwca_title"),
            default_path,
            "Zip files (*.zip)",
        )
        if not path:
            return
        try:
            run_output = self._build_run_output()
            self._run_export_worker("dwca", run_output, path)
        except Exception as exc:
            self._log.error(
                "Export failed (dwca → %s): %s",
                path,
                exc,
                exc_info=True,
            )

    def _run_export_worker(
        self,
        fmt: str,
        run_output: dict[str, Any],
        path: str,
        rows: list[dict[str, Any]] | None = None,
        use_utf8_sig: bool = False,
    ) -> None:
        """Spin up an :class:`_ExportWorker` for ``fmt`` and wire it up.

        The ``fmt`` argument picks one of ``"xlsx"``, ``"json"``,
        ``"csv"`` or ``"dwca"``. ``run_output`` is the snapshot of
        :meth:`_build_run_output` taken on the GUI thread (so the
        worker doesn't race with subsequent filter edits). ``rows`` is
        optional; only the CSV export uses it (the worker reads
        ``r.get(k, "")`` directly, so the GUI thread pre-extracts the
        column values into a plain dict for each row). ``use_utf8_sig``
        is the Phase 63 Plan 6.10 BOM flag — CSV passes it as
        ``True``, other formats pass ``False``.

        Audit 2026-08-19 (Phase 5A, M-15): the export buttons are
        disabled while the worker is in flight (double-click guard,
        same idea as B-14 mark-verified) and re-enabled from a
        single helper so success / failure paths agree.
        """
        if rows is None:
            rows = self._filtered_rows
        worker = _ExportWorker(fmt, run_output, path, rows, use_utf8_sig=use_utf8_sig)
        # Capture the worker on the instance so the QThread isn't
        # garbage-collected mid-flight (a known PySide6 footgun).
        self._export_worker = worker

        # Disable every export button so a single click can't kick
        # off two concurrent writers to the same destination path.
        self._disable_export_buttons()

        def _on_success(saved_path: str) -> None:
            self._re_enable_export_buttons()
            # Friendly status message — match the previous sync
            # implementation's wording so the user's muscle memory
            # still works.
            count = len(self._filtered_rows)
            basename = Path(saved_path).name
            if fmt == "xlsx":
                self._set_status(
                    i18n._tr("jobstab.export.saved").format(count=count, path=basename)
                )
            else:
                self._set_status(i18n._tr("jobstab.export.saved_short").format(path=basename))

        def _on_error(msg: str) -> None:
            self._re_enable_export_buttons()
            # Audit 2026-08-19 (Phase 5A, M-15): log the failure at
            # ERROR with exc_info-style detail so a missing module or
            # disk-full still shows up in the troubleshooting log.
            # The worker already formatted the exception as
            # ``{Type}: {msg}``; we keep it as-is for the log but
            # also surface a popup so the operator knows immediately.
            self._log.error(
                "Export failed (%s → %s): %s",
                fmt,
                path,
                msg,
            )
            label_key = f"restab.export.{fmt}"
            QMessageBox.warning(
                self,
                i18n._tr(label_key),
                i18n._tr("jobstab.export.failed").format(error=msg),
            )

        worker.finished_with_success.connect(_on_success)
        worker.error.connect(_on_error)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _disable_export_buttons(self) -> None:
        """Disable every export button while a worker is in flight.

        Audit 2026-08-19 (Phase 5A, M-15): extracted helper so the
        success and error callbacks agree on the disable / re-enable
        policy. Failures re-enable so the operator can retry; the
        success path re-enables so the next export works immediately.
        """
        for attr in (
            "_btn_export_xlsx",
            "_btn_export_json",
            "_btn_export_csv",
            "_btn_export_dwca",
        ):
            btn = getattr(self, attr, None)
            if btn is not None:
                try:
                    btn.setEnabled(False)
                except Exception:
                    pass

    def _re_enable_export_buttons(self) -> None:
        """Re-enable the export buttons after a worker finishes.

        Audit 2026-08-19 (Phase 5A, M-15): companion to
        :meth:`_disable_export_buttons`. Called from both the success
        and error callbacks so a network / IO failure still leaves the
        buttons clickable for the next attempt.
        """
        for attr in (
            "_btn_export_xlsx",
            "_btn_export_json",
            "_btn_export_csv",
            "_btn_export_dwca",
        ):
            btn = getattr(self, attr, None)
            if btn is not None:
                try:
                    btn.setEnabled(True)
                except Exception:
                    pass

    def _build_run_output(self) -> dict[str, Any]:
        panels = self._filtered_rows
        return {
            "schema_version": "1.0.0",
            "provenance": {"job_id": self._current_job_id, "source": "rlpe-gui"},
            "papers": [],
            "figures": [],
            "panels": panels,
            "taxa": [],
            "samples": [],
            "geology_contexts": [
                g for r in panels for g in ((r.get("metadata") or {}).get("geology_links") or [])
            ],
            "localities": [
                {"country": g.get("country"), "locality": g.get("locality")}
                for r in panels
                for g in ((r.get("metadata") or {}).get("geology_links") or [])
                if g.get("country") or g.get("locality")
            ],
            "paleo_coordinates": [],
            "warnings": [],
        }

    def _set_status(self, text: str) -> None:
        self._status.setText(text)

    # ------------------------------------------------------------------
    # API URL setter (Audit 2026-08-20, M-5)
    # ------------------------------------------------------------------
    def _set_api_url(self, url: str) -> bool:
        """Validate ``url`` and persist to QSettings on success.

        Audit 2026-08-20 (M-5). This is the single write path for
        ``io/api_url`` so any future caller (Settings tab, command-line
        flag, etc.) can update the override safely:

        * ``file:///...``, ``javascript:...``, ``data:...`` and
          missing-scheme URLs are rejected at the door.
        * Loopback hosts (``localhost`` / ``127.0.0.1`` / ``0.0.0.0``
          / ``::1``) are rejected when ``allow_local=False`` (the
          default) because a hostile override would otherwise be
          able to redirect the POST to the operator's machine. The
          dev default uses :data:`constants.DEFAULT_API_URL` which
          is itself loopback — that's fine because the default
          bypasses QSettings entirely.
        * On rejection the QSettings key is left untouched, the
          status bar surfaces a translated error, and the bearer
          token is never logged.

        Returns ``True`` iff the URL was accepted and persisted.
        """
        validated = _validate_api_url(url)
        if validated is None:
            self._log.warning("ResultsTab._set_api_url rejected %r (M-5)", url)
            self._set_status(
                i18n._tr(
                    "restab.api_url.invalid",
                    "Invalid API URL; using default",
                )
            )
            return False
        try:
            from PySide6.QtCore import QSettings

            from .constants import APP_AUTHOR, APP_NAME, QS_KEY_API_URL

            settings = QSettings(APP_AUTHOR, APP_NAME)
            settings.setValue(QS_KEY_API_URL, validated)
        except Exception as exc:
            self._log.warning(
                "ResultsTab._set_api_url could not persist %r: %s",
                validated,
                exc,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Image-verified flip (GUI-A4)
    # ------------------------------------------------------------------
    def _flip_image_verified(self, verified: bool) -> None:
        """POST the selected row's identity to /review/correction.

        The endpoint (:func:`api.app.submit_correction`) flips
        ``PanelRecord.image_verified`` on every (paper_id, figure_id,
        panel_path) tuple in the API's ``RESULT_CACHE`` AND appends a
        corrections.jsonl row for replay. After a successful POST we
        update the in-memory row in ``self._all_rows`` /
        ``self._filtered_rows`` so the detail panel + table row
        reflect the new state without needing a full reload.

        Audit 2026-08-19 (B-14): the POST used to run SYNCHRONOUSLY on
        the GUI thread, blocking the event loop for up to the full
        request timeout (10 s). We now spin up a
        :class:`_FlipVerifiedWorker` ``QThread`` to do the POST off
        the main thread and emit success / failure back to the GUI
        via Qt signals. The triggering button is disabled while the
        request is in flight so the operator can never double-fire a
        flip.

        Errors are surfaced via ``QMessageBox.warning``; we never
        crash the GUI on a network failure.
        """
        # Pull the currently selected row's identity triple. Without
        # at least one selected row the action is a no-op so the
        # button never accidentally flips an unrelated panel.
        items = self._table.selectedItems()
        if not items:
            QMessageBox.information(
                self,
                i18n._tr("restab.detail.mark_verified"),
                i18n._tr("restab.detail.verify_failed").format(
                    error=i18n._tr("restab.detail.no_selection"),
                ),
            )
            return
        row = items[0].data(Qt.UserRole) or {}
        paper_id = row.get("paper_id")
        figure_id = row.get("figure_id")
        # The API matches on the trailing path component of
        # panel_path, so the basename is sufficient and avoids
        # any platform-specific absolute-path mismatch.
        panel_path = row.get("panel_path") or ""
        if not (paper_id and figure_id):
            QMessageBox.warning(
                self,
                i18n._tr("restab.detail.mark_verified"),
                i18n._tr("restab.detail.verify_failed").format(
                    error="missing paper_id / figure_id on the selected row",
                ),
            )
            return

        # Resolve the API URL. We honour a QSettings override at
        # ``io/api_url`` (settable from the Settings tab) and fall
        # back to the local-loopback default. Using QSettings keeps
        # operators on a remote API box happy without us needing to
        # add a new GUI tab.
        #
        # Audit 2026-08-20 (M-5): validate the URL via
        # :func:`_validate_api_url` before using it. The previous
        # code accepted any string from QSettings, so a hostile /
        # corrupt override (``file:///etc/passwd``,
        # ``javascript:...``, etc.) could redirect the POST and
        # leak the request body. On rejection we fall back to the
        # default and surface a status-bar warning so the operator
        # can fix the override.
        api_url = DEFAULT_API_URL
        try:
            from PySide6.QtCore import QSettings

            from .constants import APP_AUTHOR, APP_NAME, QS_KEY_API_URL

            settings = QSettings(APP_AUTHOR, APP_NAME)
            v = settings.value(QS_KEY_API_URL, DEFAULT_API_URL)
            if isinstance(v, str) and v.strip():
                validated = _validate_api_url(v, allow_local=True)
                if validated is not None:
                    api_url = validated
                else:
                    self._log.warning(
                        "ignoring invalid API URL override %r (M-5); falling back to %s",
                        v,
                        DEFAULT_API_URL,
                    )
                    self._set_status(
                        i18n._tr(
                            "restab.api_url.invalid",
                            "Invalid API URL; using default",
                        )
                    )
        except Exception:
            pass
        endpoint = f"{api_url.rstrip('/')}/review/correction"

        payload = {
            "paper_id": str(paper_id),
            "figure_id": str(figure_id),
            "panel_path": panel_path or None,
            "image_verified": bool(verified),
        }

        # Audit 2026-08-19 (B-14): disable the mark-verified buttons
        # while the worker is in flight so the operator can never
        # double-click a flip, and the UI stays responsive.
        for btn in (
            getattr(self, "_mark_verified_btn", None),
            getattr(self, "_mark_unverified_btn", None),
        ):
            if btn is not None:
                try:
                    btn.setEnabled(False)
                except Exception:
                    pass

        # Capture the worker on the instance so the QThread isn't
        # garbage-collected mid-flight (a known PySide6 footgun).
        worker = _FlipVerifiedWorker(endpoint, payload)
        self._flip_worker = worker

        def _on_success(_ok: bool) -> None:
            # Mutate the in-memory rows so the table + detail panel
            # refresh without a full reload.
            n = 0
            for r in self._all_rows:
                if (
                    r.get("paper_id") == paper_id
                    and r.get("figure_id") == figure_id
                    and (not panel_path or r.get("panel_path") == panel_path)
                ):
                    r["image_verified"] = bool(verified)
                    n += 1
            for r in self._filtered_rows:
                if (
                    r.get("paper_id") == paper_id
                    and r.get("figure_id") == figure_id
                    and (not panel_path or r.get("panel_path") == panel_path)
                ):
                    r["image_verified"] = bool(verified)
            # Re-render the detail panel so the badge updates.
            try:
                self._render_detail(row)
            except Exception:
                pass
            state_label = (
                i18n._tr("restab.detail.image_verified")
                if verified
                else i18n._tr("restab.detail.image_unverified")
            )
            self._set_status(
                i18n._tr("restab.detail.verify_success").format(
                    n=n,
                    state=state_label,
                )
            )
            self._re_enable_flip_buttons()

        def _on_error(msg: str) -> None:
            self._log.error("image_verified flip failed: %s", msg, exc_info=True)
            QMessageBox.warning(
                self,
                i18n._tr("restab.detail.mark_verified"),
                i18n._tr("restab.detail.verify_failed").format(
                    error=msg,
                ),
            )
            self._re_enable_flip_buttons()

        worker.finished_with_success.connect(_on_success)
        worker.error.connect(_on_error)
        # Make sure the worker is released once it finishes.
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _re_enable_flip_buttons(self) -> None:
        """Re-enable the mark-verified buttons after a flip completes.

        Audit 2026-08-19 (B-14): helper extracted from the worker
        callbacks so success / failure paths agree on the re-enable
        policy. Failures (network errors) must also re-enable so the
        operator can retry.
        """
        for btn in (
            getattr(self, "_mark_verified_btn", None),
            getattr(self, "_mark_unverified_btn", None),
        ):
            if btn is not None:
                try:
                    btn.setEnabled(True)
                except Exception:
                    pass


def _format_bbox(bbox) -> str:
    """Format a bounding box ``[x, y, w, h]`` for the detail pane.

    Phase F-3 MINOR: this helper is currently dead code — the bbox
    rendering in ``_render_detail`` builds HTML directly and never
    imports it. Kept (with an explicit docstring) because the helper
    is small, useful, and downstream tooling may want it. If it
    stays unused for two more release cycles, remove it.
    """
    if not bbox or len(bbox) != 4:
        return "—"
    x, y, w, h = bbox
    return f"x={x:.0f}, y={y:.0f}, w={w:.0f}, h={h:.0f}"
