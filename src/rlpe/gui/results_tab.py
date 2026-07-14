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
        # Phase 37 audit fix: register as an i18n listener so column
        # headers + filter labels + count label auto-translate on
        # language switch (was: MainWindow had to manually walk every
        # tab and call _refresh_texts, which it didn't).
        i18n.add_listener(lambda _lang: self._refresh_texts())

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
        # Phase 37 audit fix: each filter combo's "all/any" item
        # stores a sentinel ``userData`` token ("__ALL__", "__ANY__",
        # "yes", "no") so the filter logic compares against the
        # sentinel, not the translated text. Previously the code
        # compared against the literal "(all)"/"(any)" string,
        # which silently dropped every row after a language switch
        # because the translated label differs from the English
        # sentinel.
        from .i18n_widgets import tr_combobox
        filter_row = QHBoxLayout()
        filter_row.setSpacing(SPACE_S)

        self._species_filter = tr_combobox(
            "restab.filter.species",
            min_width=160,
        )
        self._species_filter.addItem(i18n._tr("restab.filter.all"), userData="__ALL__")
        # Phase 37: setCurrentIndex(0) so the default filter is the
        # "(all)" item (otherwise currentIndex=-1 and currentText=""
        # which breaks our sentinel-comparison logic).
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
        # Phase 53: use the i18n key for column headers on init too.
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
        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)
        splitter.addWidget(bottom)

        self._preview = ImagePreviewWidget()
        bottom_layout.addWidget(self._preview, 1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(SPACE_M, 0, 0, 0)
        right_layout.setSpacing(SPACE_S)

        right_layout.addWidget(tr_label("restab.detail.title"))
        self._detail_browser = QTextBrowser()
        self._detail_browser.setOpenExternalLinks(False)
        right_layout.addWidget(self._detail_browser, 1)
        bottom_layout.addWidget(right_panel, 1)

        splitter.setSizes([600, 300])

        # ---- Footer: export buttons ----
        footer = QHBoxLayout()
        footer.setSpacing(SPACE_S)

        # Phase 37 audit fix: use tr_button so these translate on
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
        self._current_job_id = job_id
        self._current_job_dir = output_dir
        self._all_rows = list(rows)
        self._title.setText(f"📊 Job {job_id}  ·  {len(self._all_rows):,} rows")
        self._refresh_filter_options()
        self._refresh_view()

    def append_rows(self, rows: list[dict[str, Any]]) -> None:
        """Stream-add rows from a running job (live update)."""
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
        # Phase 37 audit fix: preserve userData when re-setting the
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

    def clear(self) -> None:
        self._all_rows = []
        self._filtered_rows = []
        self._current_job_id = None
        self._current_job_dir = None
        # Phase 37: use i18n keys for the reset state text.
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
        # Phase 37 audit fix: store sentinel userData so the
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
        # Phase 37 audit fix: compare against sentinel userData
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
                # Phase 53: wrap in bool() so have is True/False, not
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
        self._count_label.setText(f"{len(rows):,} / {len(self._all_rows):,} rows")
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
        # Phase 53: page_index lives at metadata.page_index in the
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
        """Render the right-side detail panel as HTML."""
        md = row.get("metadata") or {}
        pm = row.get("paper_metadata") or {}
        paleodb = md.get("paleodb") or {}
        tax = paleodb.get("taxonomy") or {}
        geo_links = md.get("geology_links") or []
        html = []
        html.append("<html><body style='font-family: sans-serif;'>")
        # Heading
        html.append(
            f"<h2 style='color:#1f77b4;margin-bottom:4px'>{html_escape(row.get('species') or '(no species)')}</h2>"
        )
        html.append(
            f"<div style='color:#666;margin-bottom:10px'>{html_escape(row.get('panel_id') or '')}</div>"
        )
        # Metadata block
        html.append("<table style='font-size:12px;border-collapse:collapse;width:100%;margin-bottom:12px'>")
        meta_rows = [
            ("Confidence",    f"{row.get('confidence'):.2f}" if isinstance(row.get('confidence'), (int, float)) else "—"),
            ("Page",          md.get("page_index") if isinstance(md, dict) else row.get("page_index")),
            ("Figure",        row.get("figure_id")),
            ("Panel bbox",    _format_bbox(row.get("bbox"))),
            ("Source",        ((md.get("extraction_source") or "—")) if isinstance(md, dict) else "—"),
        ]
        for k, v in meta_rows:
            html.append(
                f"<tr><td style='padding:2px 8px 2px 0;color:#888;width:30%'>{k}</td>"
                f"<td style='padding:2px 0'>{html_escape(str(v))}</td></tr>"
            )
        html.append("</table>")
        # Caption snippet
        if row.get("caption_snippet"):
            html.append("<h3 style='font-size:13px;margin:8px 0 4px'>Caption</h3>")
            html.append(
                f"<pre style='background:#f5f5f5;padding:8px;border-radius:4px;"
                f"white-space:pre-wrap;font-family:monospace;font-size:11px'>"
                f"{html_escape(row['caption_snippet'])}</pre>"
            )
        # Paper metadata
        if pm.get("title"):
            html.append("<h3 style='font-size:13px;margin:8px 0 4px'>Paper</h3>")
            html.append(f"<div><b>{html_escape(pm['title'])}</b></div>")
            authors = pm.get("authors")
            if authors:
                html.append(f"<div style='color:#666'>{html_escape(str(authors)[:200])}</div>")
            if pm.get("doi"):
                html.append(f"<div>DOI: <code>{html_escape(pm['doi'])}</code></div>")
            if pm.get("year"):
                html.append(f"<div>Year: {html_escape(str(pm['year']))}</div>")
        # PBDB taxonomy
        if tax:
            html.append("<h3 style='font-size:13px;margin:8px 0 4px'>PBDB taxonomy</h3>")
            html.append("<table style='font-size:12px'>")
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
                    f"<tr><td style='padding:2px 8px 2px 0;color:#888;width:30%'>{k}</td>"
                    f"<td style='padding:2px 0'>{html_escape(str(v))}</td></tr>"
                )
            html.append("</table>")
        # Geology links (first 5)
        if geo_links:
            html.append(f"<h3 style='font-size:13px;margin:8px 0 4px'>Geology links ({len(geo_links)})</h3>")
            for g in geo_links[:5]:
                bits = []
                for k in ("age", "chronostratigraphy", "biozone", "country", "locality"):
                    if g.get(k):
                        bits.append(html_escape(f"{k}={g[k]}"))
                if g.get("latitude") is not None:
                    bits.append(f"lat={g['latitude']:.3f},lon={g['longitude']:.3f}")
                html.append(
                    f"<div style='font-family:monospace;font-size:11px;background:#f7f9fc;"
                    f"padding:4px 8px;border-radius:3px;margin-bottom:4px'>"
                    + " · ".join(bits)
                    + "</div>"
                )
            if len(geo_links) > 5:
                html.append(f"<div style='color:#888'>… and {len(geo_links)-5} more</div>")
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
            import csv
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=[c.key for c in RESULT_COLUMNS])
                w.writeheader()
                for r in self._filtered_rows:
                    w.writerow({k: self._extract_column(r, k) for k in (c.key for c in RESULT_COLUMNS)})
            self._set_status(i18n._tr("jobstab.export.saved_short").format(path=Path(path).name))
        except Exception as exc:
            QMessageBox.warning(
                self,
                i18n._tr("restab.export.csv"),
                i18n._tr("jobstab.export.failed").format(error=str(exc)),
            )

    def _export_dwca(self) -> None:
        if not self._filtered_rows:
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