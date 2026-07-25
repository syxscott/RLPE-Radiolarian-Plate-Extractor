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
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .constants import INPUT_WIDTH_LONG, RESULT_COLUMNS
from .i18n_widgets import tr_button, tr_combobox, tr_label
from .styles import SPACE_M, SPACE_S
from . import i18n
from .image_preview import ImagePreviewWidget
from .utils import (
    fmt_count,
    fmt_coord,
    fmt_percent,
    get_gui_logger,
    html_escape,
    short_path,
    truncate,
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
    "sample_match": "badge-info",       # blue chip
    "locality_match": "badge-info",     # blue chip
    "m3_inference": "badge-warn",        # amber chip
    "unlinked": "badge-muted",           # grey chip
}
_LINK_SOURCE_LABEL_KEYS: dict[str, str] = {
    "sample_match": "restab.detail.link_source.sample_match",
    "locality_match": "restab.detail.link_source.locality_match",
    "m3_inference": "restab.detail.link_source.m3_inference",
    "unlinked": "restab.detail.link_source.unlinked",
}


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
    raw = coord_source[len(_LINK_SOURCE_PREFIX):]
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
        f"<b>Cross-figure link:</b> "
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


class ResultsTab(QWidget):
    """Row-by-row results browser with image preview + detail panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._log = get_gui_logger()
        self._all_rows: list[dict[str, Any]] = []
        self._filtered_rows: list[dict[str, Any]] = []
        self._current_job_id: str | None = None
        self._current_job_dir: str | None = None
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
        from .i18n_widgets import tr_combobox
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
        bottom_splitter.setStretchFactor(0, 3)
        bottom_splitter.setStretchFactor(1, 7)
        bottom_splitter.setSizes([360, 840])
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
        export_xlsx_btn = tr_button("restab.export.xlsx")
        export_xlsx_btn.setProperty("class", "primary")
        export_xlsx_btn.clicked.connect(self._export_xlsx)
        footer.addWidget(export_xlsx_btn)

        export_json_btn = tr_button("restab.export.json")
        export_json_btn.clicked.connect(self._export_json)
        footer.addWidget(export_json_btn)

        export_csv_btn = tr_button("restab.export.csv")
        export_csv_btn.clicked.connect(self._export_csv)
        footer.addWidget(export_csv_btn)

        export_dwca_btn = tr_button("restab.export.dwca")
        export_dwca_btn.clicked.connect(self._export_dwca)
        footer.addWidget(export_dwca_btn)

        footer.addStretch(1)

        self._status = QLabel("")
        self._status.setObjectName("metricLabel")
        footer.addWidget(self._status)

        outer.addLayout(footer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_job(self, job_id: str, rows: list[dict[str, Any]], output_dir: str | None = None) -> None:
        """Replace the current results with a new job's rows."""
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
        self._title.setText(f"📊 Job {job_id}  ·  {len(self._all_rows):,} rows")
        self._refresh_filter_options()
        self._refresh_view()

    def append_rows(self, rows: list[dict[str, Any]], job_id: str | None = None) -> None:
        """Stream-add rows from a running job (live update).

        Phase 56 audit: job_id guard prevents rows from a stale job
        accumulating under a new job's ID.
        """
        if job_id is not None and job_id != self._current_job_id:
            self._log.warning("append_rows called for different job_id %s (current %s), ignoring", job_id, self._current_job_id)
            return
        self._all_rows.extend(rows)
        self._title.setText(
            f"📊 Job {self._current_job_id or '?'}  ·  {len(self._all_rows):,} rows (live)"
        )
        self._refresh_filter_options()
        self._refresh_view()

    def _refresh_texts(self) -> None:
        """Re-translate column headers + filter labels."""
        for i, col in enumerate(RESULT_COLUMNS):
            self._table.horizontalHeaderItem(i).setText(i18n._tr(f"restab.col.{col.key}"))
        # "all"/"any" labels. We update only the *displayed text*
        # at the existing index rather than clearing + rebuilding,
        # which would lose the per-species / per-family items.
        for combo, key, sentinel in (
            (self._species_filter, "restab.filter.all", "__ALL__"),
            (self._family_filter,   "restab.filter.all", "__ALL__"),
            (self._has_pbdb,        "restab.filter.any", "__ANY__"),
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
        """Phase 56 audit: remove i18n listener on widget destruction."""
        self._remove_i18n_listener()
        super().closeEvent(event)

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
        self._count_label.setText(
            i18n._tr("restab.count").format(shown=0, total=0)
        )

    # ------------------------------------------------------------------
    # Internal — filter + view refresh
    # ------------------------------------------------------------------
    def _refresh_filter_options(self) -> None:
        # Rebuild species + family dropdowns from current rows
        species = sorted({r.get("species", "") for r in self._all_rows if r.get("species")})
        families = sorted({
            ((r.get("metadata") or {}).get("paleodb") or {}).get("taxonomy", {}).get("family")
            for r in self._all_rows
            if ((r.get("metadata") or {}).get("paleodb") or {}).get("taxonomy")
        })
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
                blob = " ".join([
                    str(r.get("species") or ""),
                    str(r.get("panel_id") or ""),
                    str(r.get("caption_snippet") or ""),
                    str(r.get("label_text") or ""),
                    ((r.get("metadata") or {}).get("paleodb") or {}).get("taxonomy", {}).get("family") or "",
                ]).lower()
                if search not in blob:
                    continue
            if species_filter and r.get("species") != species_filter:
                continue
            if family_filter:
                fam = ((r.get("metadata") or {}).get("paleodb") or {}).get("taxonomy", {}).get("family")
                if fam != family_filter:
                    continue
            if has_pbdb != "__ANY__":
                pbdb = (r.get("metadata") or {}).get("paleodb")
                want = has_pbdb == "yes"
                # the truthy dict from pbdb.get("taxonomy"). Without
                # bool(), `True != {'family': 'F1'}` is True and the
                # row is dropped even when it has PBDB data.
                have = bool(
                    pbdb is not None
                    and pbdb.get("looked_up")
                    and pbdb.get("taxonomy")
                )
                if want != have:
                    continue
            out.append(r)
        return out

    def _refresh_view(self) -> None:
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
            return ((row.get("metadata") or {}).get("paleodb") or {}).get("taxonomy", {}).get("family")
        if key == "country":
            geo = ((row.get("metadata") or {}).get("geology_links") or [])
            for g in geo:
                if g.get("country"):
                    return g["country"]
            return None
        if key == "biozone":
            geo = ((row.get("metadata") or {}).get("geology_links") or [])
            for g in geo:
                if g.get("biozone"):
                    return g["biozone"]
            return None
        if key == "coord":
            geo = ((row.get("metadata") or {}).get("geology_links") or [])
            for g in geo:
                if g.get("latitude") is not None and g.get("longitude") is not None:
                    return f"{g['latitude']:.3f}, {g['longitude']:.3f}"
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
    def _on_row_selected(self) -> None:
        items = self._table.selectedItems()
        if not items:
            return
        row = items[0].data(Qt.UserRole)
        if not row:
            return
        self._render_detail(row)
        # Load image preview
        panel_path = row.get("panel_path")
        if panel_path:
            p = Path(panel_path)
            if p.exists():
                self._preview.set_image(p)
                bbox_list = [row]  # the row's bbox is one rectangle
                self._preview.set_bboxes(bbox_list)
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
        html.append("<html><head><style>"
            ".badge-info{padding:1px 5px;border-radius:3px;font-size:11px;background:#d6e4ff;color:#1f77b4}"
            ".badge-warn{padding:1px 5px;border-radius:3px;font-size:11px;background:#ffe0a0;color:#c07800}"
            ".badge-muted{padding:1px 5px;border-radius:3px;font-size:11px;background:#eee;color:#888}"
            "</style></head><body style='font-family:sans-serif;padding:0;margin:0'>")

        # ── Heading ──────────────────────────────────────────────
        html.append(
            f"<h2 style='color:#1f77b4;margin:8px 8px 2px'>"
            f"{html_escape(row.get('species') or '(no species)')}</h2>"
        )
        panel_id = row.get('panel_id') or ''
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

        html.append("<table style='font-size:12px;border-collapse:collapse;width:100%;margin-bottom:8px'>")
        meta_pairs = [
            (i18n._tr("restab.detail.paper_id"), html_escape(row.get("paper_id") or "—")),
            (i18n._tr("restab.detail.figure_id"), html_escape(row.get("figure_id") or "—")),
            (i18n._tr("restab.detail.panel_label"), html_escape(panel_id or "—")),
            (i18n._tr("restab.detail.page"), md.get("page_index") if md.get("page_index") is not None else "—"),
            (i18n._tr("restab.detail.source"), f"<span class='{ocr_cls}' style='padding:1px 5px;border-radius:3px;font-size:11px'>{html_escape(ocr_label)}</span>"),
            (i18n._tr("restab.detail.confidence"), conf_str),
            (i18n._tr("restab.detail.geo_scope"), f"<span class='{cls}' style='padding:1px 5px;border-radius:3px;font-size:11px'>{html_escape(scope_label)}</span>"),
        ]
        # Phase 56 audit: guard against non-numeric bbox elements (None, str)
        bbox = row.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4 and any(
            v > 0 for v in bbox if isinstance(v, (int, float))
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
                f"<div style='padding:4px 8px;border-top:1px solid #eee'>"
                + " ".join(parts) + "</div>"
            )
            if doi:
                html.append(f"<div style='padding:0 8px 4px;font-size:11px;color:#666'>DOI: <code>{html_escape(doi)}</code></div>")
            if review_reasons:
                rw = "; ".join(str(r) for r in review_reasons)
                html.append(
                    f"<div style='padding:0 8px 4px'><span class='badge-warn' style='font-size:11px'>"
                    f"&#9888; {html_escape(rw[:80])}</span></div>"
                )

        # ── Caption snippet ───────────────────────────────────
        cap_snippet = row.get("caption_snippet") or ""
        if cap_snippet:
            display = cap_snippet[:280] + ("…" if len(cap_snippet) > 280 else "")
            html.append(
                f"<div style='padding:6px 8px;border-top:1px solid #eee'>"
                f"<b style='font-size:12px'>Caption</b>"
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
            tax_rows = [
                ("Kingdom", tax.get("kingdom")),
                ("Phylum",  tax.get("phylum")),
                ("Class",   tax.get("class_")),
                ("Order",   tax.get("order")),
                ("Family",  tax.get("family")),
                ("Genus",   tax.get("genus")),
                ("Source",  tax.get("source")),
            ]
            for k, v in tax_rows:
                if not v:
                    continue
                html.append(
                    f"<tr><td style='padding:1px 8px 1px 0;color:#888'>{k}</td>"
                    f"<td style='padding:1px 0'>{html_escape(str(v))}</td></tr>"
                )
            html.append("</table></div>")

        # ── Sample IDs ─────────────────────────────────────────
        sample_ids = list({
            str(g.get("sample_id"))
            for g in geo_links
            if g.get("sample_id")
        })
        if sample_ids:
            html.append(
                f"<div style='padding:4px 8px;border-top:1px solid #eee;font-size:12px'>"
                f"<b>Sample IDs</b>: "
                f"<code style='font-size:11px'>{html_escape(', '.join(sample_ids[:10]))}"
                f"{' …' if len(sample_ids) > 10 else ''}</code></div>"
            )

        # ── Geology links ────────────────────────────────────
        if geo_links:
            html.append(
                f"<div style='padding:6px 8px;border-top:1px solid #eee'>"
                f"<b style='font-size:12px'>Geology links ({len(geo_links)})</b>"
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
                if ma_top is not None and ma_base is not None:
                    bits.append(f"<span style='color:#555'>{float(ma_top):.2f}–{float(ma_base):.2f} Ma</span>")
                # Lithology / formation / member / group
                for k in ("lithology", "formation", "member", "group"):
                    v = g.get(k)
                    if v:
                        tag = f"<em>{html_escape(str(v))}</em>" if k == "formation" else html_escape(str(v))
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
                    bits.append(f"<span style='font-size:11px;color:#666'>{html_escape(ctry)}</span>")
                # Modern coords
                mlat = g.get("modern_latitude")
                mlon = g.get("modern_longitude")
                if mlat is not None and mlon is not None:
                    bits.append(
                        f"<span style='font-size:11px;color:#27ae60'>now "
                        f"{float(mlat):.3f}, {float(mlon):.3f}</span>"
                    )
                # Paleo coords
                plat = g.get("paleo_latitude")
                plon = g.get("paleo_longitude")
                plate_id = g.get("plate_id")
                recon_age = g.get("reconstruction_age_ma")
                if plat is not None and plon is not None:
                    paleo_bits = [f"<span style='color:#e67e22'>@ {float(plat):.3f}, {float(plon):.3f}</span>"]
                    if plate_id:
                        paleo_bits.append(f"<span style='color:#e67e22'>plate={plate_id}</span>")
                    if recon_age is not None:
                        paleo_bits.append(f"<span style='color:#e67e22'>{float(recon_age):.1f} Ma</span>")
                    bits.append(" ".join(paleo_bits))
                # Confidence
                gc = g.get("geology_confidence") or g.get("confidence")
                if gc is not None:
                    bits.append(f"<span style='color:#888;font-size:10px'>({float(gc)*100:.0f}%)</span>")
                if bits:
                    html.append(
                        f"<div style='font-size:11px;background:#f7f9fc;padding:4px 6px;"
                        f"border-radius:3px;margin-top:4px'>" + " · ".join(bits) + "</div>"
                    )
            if len(geo_links) > 8:
                html.append(f"<div style='color:#888;font-size:11px;margin-top:4px'>… and {len(geo_links)-8} more</div>")
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
            html.append(
                "<table style='font-size:12px;margin-top:4px;border-collapse:collapse'>"
            )
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
            "sample_match", None,
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
    # Exports
    # ------------------------------------------------------------------
    def _export_xlsx(self) -> None:
        if not self._filtered_rows:
            QMessageBox.information(
                self,
                i18n._tr("restab.export.xlsx"),
                i18n._tr("jobstab.export.no_rows"),
            )
            return
        default_path = str(Path(self._current_job_dir or ".") / f"{self._current_job_id or 'results'}.xlsx")
        path, _ = QFileDialog.getSaveFileName(
            self,
            i18n._tr("restab.export.xlsx_title"),
            default_path,
            "Excel files (*.xlsx)",
        )
        if not path:
            return
        try:
            from ..exporters.xlsx import write_xlsx
            run_output = self._build_run_output()
            write_xlsx(run_output, path)
            self._set_status(i18n._tr("jobstab.export.saved").format(
                count=len(self._filtered_rows), path=Path(path).name,
            ))
        except Exception as exc:
            QMessageBox.warning(
                self,
                i18n._tr("restab.export.xlsx"),
                i18n._tr("jobstab.export.failed").format(
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )

    def _export_json(self) -> None:
        if not self._filtered_rows:
            QMessageBox.information(self, i18n._tr("restab.export.json"), i18n._tr("jobstab.export.no_rows"))
            return
        default_path = str(Path(self._current_job_dir or ".") / f"{self._current_job_id or 'results'}.json")
        path, _ = QFileDialog.getSaveFileName(
            self,
            i18n._tr("restab.export.json_title"),
            default_path,
            "JSON files (*.json)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self._build_run_output(), fh, indent=2, ensure_ascii=False, default=str)
            self._set_status(i18n._tr("jobstab.export.saved_short").format(path=Path(path).name))
        except Exception as exc:
            QMessageBox.warning(
                self,
                i18n._tr("restab.export.json"),
                i18n._tr("jobstab.export.failed").format(error=str(exc)),
            )

    def _export_csv(self) -> None:
        if not self._filtered_rows:
            QMessageBox.information(self, i18n._tr("restab.export.csv"), i18n._tr("jobstab.export.no_rows"))
            return
        default_path = str(Path(self._current_job_dir or ".") / f"{self._current_job_id or 'results'}.csv")
        path, _ = QFileDialog.getSaveFileName(
            self,
            i18n._tr("restab.export.csv_title"),
            default_path,
            "CSV files (*.csv)",
        )
        if not path:
            return
        try:
            # Phase 63 Plan 6.5 (Bug 6.5): route through analysis-level
            # CSV helpers instead of the bare csv.DictWriter call we had
            # here. The bare call produced three problems downstream:
            #   1) No formula-injection sanitisation (CWE-1236) —
            #      Excel/LibreOffice would treat a paper caption
            #      starting with ``=``, ``+``, ``-``, ``@`` or TAB as
            #      a formula and execute ``=cmd|'/c calc'!A1`` on open.
            #   2) No UTF-8 BOM (Phase 63 Plan 6.10) — Excel on
            #      Windows defaults to ANSI code page and mangles Greek
            #      / CJK.
            #   3) No NaN/Inf sanitisation — scale-bar or geo coord
            #      paths occasionally produced ``float("nan")`` which
            #      csv writes as the string "nan" instead of an empty
            #      cell.
            # The fix imports ``_sanitise_csv_cell`` from
            # ``rlpe.exporters.analysis`` (the same helper
            # ``analysis.write_csv`` uses), reads NaN/Inf handler from
            # ``rlpe.export._csv_cell`` (Task 6.8/6.10), and writes the
            # file with a UTF-8 BOM. The displayed column ordering
            # (RESULT_COLUMNS) is preserved so what the user sees on
            # screen is what they get in the export.
            from math import isnan, isinf
            import csv
            from ..exporters.analysis import _sanitise_csv_cell
            from ..export import _csv_cell
            column_keys = [c.key for c in RESULT_COLUMNS]
            with open(path, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.DictWriter(fh, fieldnames=column_keys)
                w.writeheader()
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
                    w.writerow(out)
            self._set_status(i18n._tr("jobstab.export.saved_short").format(path=Path(path).name))
        except Exception as exc:
            QMessageBox.warning(
                self,
                i18n._tr("restab.export.csv"),
                i18n._tr("jobstab.export.failed").format(error=str(exc)),
            )

    def _export_dwca(self) -> None:
        if not self._filtered_rows:
            QMessageBox.information(self, i18n._tr("restab.export.dwca"), i18n._tr("jobstab.export.no_rows"))
            return
        default_path = str(Path(self._current_job_dir or ".") / f"{self._current_job_id or 'results'}.zip")
        path, _ = QFileDialog.getSaveFileName(
            self,
            i18n._tr("restab.export.dwca_title"),
            default_path,
            "Zip files (*.zip)",
        )
        if not path:
            return
        try:
            from ..exporters.archive import write_dwca_zip
            run_output = self._build_run_output()
            write_dwca_zip(run_output, path)
            self._set_status(i18n._tr("jobstab.export.saved_short").format(path=Path(path).name))
        except Exception as exc:
            QMessageBox.warning(
                self,
                i18n._tr("restab.export.dwca"),
                i18n._tr("jobstab.export.failed").format(
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )

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
                for r in panels for g in ((r.get("metadata") or {}).get("geology_links") or [])
                if g.get("country") or g.get("locality")
            ],
            "paleo_coordinates": [],
            "warnings": [],
        }

    def _set_status(self, text: str) -> None:
        self._status.setText(text)


def _format_bbox(bbox) -> str:
    if not bbox or len(bbox) != 4:
        return "—"
    x, y, w, h = bbox
    return f"x={x:.0f}, y={y:.0f}, w={w:.0f}, h={h:.0f}"