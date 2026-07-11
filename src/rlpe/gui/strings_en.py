"""English strings for the RLPE GUI.

Format: flat dict with hierarchical keys (dot-separated).
Each key is what ``register_widget_text(widget_id, attr, key)`` will
use. When adding a new string:
  1. Add the key here.
  2. Mirror it in ``strings_zh_CN.py`` (translation is mandatory).
  3. In the widget constructor call ``register_widget_text`` with the
     same key, then the helper handles the rest.
"""

STRINGS = {
    # ============================================================
    # App-level
    # ============================================================
    "app.title": "RLPE - Radiolarian Plate Extractor",
    "app.about.title": "About RLPE",
    "app.about.body": (
        "<h3>RLPE - Radiolarian Plate Extractor</h3>"
        "<p><b>Version:</b> {version}<br>"
        "<b>Author:</b> {author}</p>"
        "<p>Native Qt6 desktop GUI for radiolarian plate extraction.</p>"
        "<p>Powered by PySide6, the RLPE pipeline, FastAPI, and PBDB.</p>"
    ),

    # ============================================================
    # Main window menus
    # ============================================================
    "menu.file": "&File",
    "menu.view": "&View",
    "menu.tools": "&Tools",
    "menu.help": "&Help",
    "menu.theme": "🎨  &Theme",
    "menu.theme.light": "&Light",
    "menu.theme.dark": "&Dark",
    "menu.theme.system": "&System default",

    "menu.file.open": "📂  &Open PDF…",
    "menu.file.batch": "📚  &Batch…",
    "menu.file.outdir": "📁  Open output &directory…",
    "menu.file.quit": "&Quit",
    "menu.tools.log": "📂  Open &log file",
    "menu.help.about": "&About RLPE",
    "menu.view.run": "&Run tab",
    "menu.view.jobs": "&Jobs tab",
    "menu.view.results": "&Results tab",
    "menu.view.settings": "&Settings tab",

    "toolbar.open": "📂  Open PDF",
    "toolbar.batch": "📚  Batch…",
    "toolbar.about": "ℹ️  About",
    "toolbar.run": "▶  Run",
    "toolbar.jobs": "📋  Jobs",
    "toolbar.results": "📊  Results",
    "toolbar.settings": "⚙️  Settings",

    # ============================================================
    # Tab labels
    # ============================================================
    "tab.run": "▶  Run",
    "tab.jobs": "📋  Jobs",
    "tab.results": "📊  Results",
    "tab.settings": "⚙️  Settings",

    # ============================================================
    # Run tab
    # ============================================================
    "runtab.input_group": "📄 Input PDF",
    "runtab.input.placeholder": "Pick a radiolarian paper PDF to extract plates from…",
    "runtab.browse": "Browse…",
    "runtab.clear": "Clear",
    "runtab.out_group": "💾 Output directory",
    "runtab.out.placeholder": "Where to write manifests / figures / xlsx…",
    "runtab.out.choose": "Choose…",
    "runtab.out.open": "Open",

    "runtab.basic_group": "⚙️ Basic configuration",
    "runtab.adv_group": "🔬 Advanced (LLM / M3 / PBDB)",

    "runtab.label.ocr_backend": "OCR backend:",
    "runtab.label.ocr_lang": "OCR language(s):",
    "runtab.label.grobid_url": "GROBID URL:",
    "runtab.label.grobid_retries": "GROBID retries:",
    "runtab.label.grobid_timeout": "GROBID timeout (s):",
    "runtab.label.caption_window": "Caption window (GROBID):",
    "runtab.label.od_caption_window": "OD caption window:",
    "runtab.label.workers": "Workers:",
    "runtab.label.panel_score": "Panel score:",
    "runtab.label.use_gpu": "Use GPU:",
    "runtab.gpu_check": "Auto-detect CUDA at startup",
    "runtab.ocr_lang.placeholder": "e.g. en, ja, ch_sim",

    "runtab.label.llm_backend": "LLM backend:",
    "runtab.label.m3_lang": "M3 prompt lang:",
    "runtab.label.m3_model": "M3 model:",
    "runtab.label.m3_budget": "M3 thinking budget:",
    "runtab.label.m3_output": "M3 max output tokens:",
    "runtab.label.m3_timeout": "M3 timeout (s):",
    "runtab.label.m3_max_retries": "M3 max retries:",
    "runtab.label.paleodb_occ": "PBDB max occurrences:",

    "runtab.use_pbdb": "Use Paleobiology Database for taxonomy + occurrence enrichment",
    "runtab.geo_vision": "Multi-modal geology vision (Round 6)",
    "runtab.m3_stage3": "M3 stage 3 (panel refinement)",
    "runtab.m3_multi_plate": "M3 multi-plate enrichment (Round 7)",
    "runtab.od_fallback": "Allow OpenDataLoader fallback when GROBID fails (Phase 29)",
    "runtab.save_intermediate": "Save intermediate panels (large disk usage)",
    "runtab.label.dpi": "Render DPI:",

    "runtab.start": "▶  Start extraction",
    "runtab.cancel": "⏹  Cancel",
    "runtab.status.idle": "Idle",
    "runtab.status.starting": "Starting…",
    "runtab.status.running": "Running",
    "runtab.status.cancelling": "Cancelling…",
    "runtab.status.done": "Done",
    "runtab.status.failed": "Failed",
    "runtab.progress.idle": "Idle  (%v / %m)",
    "runtab.progress.working": "Working…",
    "runtab.progress.starting": "Starting…",
    "runtab.progress.init": "Pipeline initialising…",
    "runtab.progress.done": "Pipeline finished. See Results tab.",
    "runtab.prompt.no_pdf.title": "Missing PDF",
    "runtab.prompt.no_pdf.body": "Please choose a valid PDF file first.",
    "runtab.prompt.no_outdir.title": "Missing output dir",
    "runtab.prompt.no_outdir.body": "Please choose an output directory.",
    "runtab.prompt.cancelled": "Cancellation requested.",
    "runtab.prompt.error.title": "Pipeline error",

    # ============================================================
    # Jobs tab
    # ============================================================
    "jobstab.clear_finished": "Clear finished",
    "jobstab.clear_all": "Clear all",
    "jobstab.no_jobs": "0 jobs",
    "jobstab.col.id": "Job ID",
    "jobstab.col.pdf": "PDF",
    "jobstab.col.status": "Status",
    "jobstab.col.progress": "Progress",
    "jobstab.col.rows": "Rows",
    "jobstab.col.elapsed": "Elapsed",
    "jobstab.col.out": "Output",
    "jobstab.status.queued": "queued",
    "jobstab.status.running": "running",
    "jobstab.status.done": "done",
    "jobstab.status.failed": "failed",
    "jobstab.status.cancelled": "cancelled",
    "jobstab.menu.open_results": "📊  Open in Results tab",
    "jobstab.menu.open_out": "📁  Open output directory",
    "jobstab.menu.export_xlsx": "📤  Export xlsx (Round 24)",
    "jobstab.menu.export_json": "📤  Export JSON",
    "jobstab.menu.remove": "🗑  Remove from list",
    "jobstab.export.no_rows": "No rows to export.",
    "jobstab.export.failed": "Export failed: {error}",
    "jobstab.export.saved": "Saved {count} rows → {path}",
    "jobstab.export.saved_short": "Saved → {path}",

    # ============================================================
    # Results tab
    # ============================================================
    "restab.no_job": "No job loaded",
    "restab.live": "Job {id}  ·  {rows} rows (live)",
    "restab.done": "Job {id}  ·  {rows} rows",
    "restab.search.placeholder": "Filter rows by species / caption / panel id / family…",
    "restab.search.label": "Search:",
    "restab.filter.species": "Species:",
    "restab.filter.family": "PBDB family:",
    "restab.filter.has_pbdb": "Has PBDB:",
    "restab.filter.all": "(all)",
    "restab.filter.any": "(any)",
    "restab.filter.yes": "yes",
    "restab.filter.no": "no",
    "restab.count": "{shown} / {total} rows",
    "restab.detail.title": "Row detail",
    "restab.export.xlsx": "📤 Export xlsx",
    "restab.export.json": "📤 Export JSON",
    "restab.export.csv": "📤 Export CSV",
    "restab.export.dwca": "📤 Export DwCA",
    "restab.col.species": "Species (Latin)",
    "restab.col.panel_id": "Panel ID",
    "restab.col.confidence": "Confidence",
    "restab.col.caption_snip": "Caption snippet",
    "restab.col.page_index": "Page",
    "restab.col.family": "PBDB Family",
    "restab.col.country": "Country",
    "restab.col.biozone": "Biozone",
    "restab.col.coord": "Lat / Lon",
    "restab.detail.confidence": "Confidence",
    "restab.detail.page": "Page",
    "restab.detail.figure": "Figure",
    "restab.detail.bbox": "Panel bbox",
    "restab.detail.source": "Source",
    "restab.detail.caption": "Caption",
    "restab.detail.paper": "Paper",
    "restab.detail.pbdb_tax": "PBDB taxonomy",
    "restab.detail.geo_links": "Geology links ({count})",
    "restab.detail.geo_links_more": "… and {n} more",
    "restab.detail.family": "family: {value}",
    "restab.detail.kingdom": "Kingdom",
    "restab.detail.phylum": "Phylum",
    "restab.detail.class": "Class",
    "restab.detail.order": "Order",

    # ============================================================
    # Settings tab
    # ============================================================
    "settab.title": "⚙️ {app}  ·  v{version}",
    "settab.appearance": "🎨 Appearance",
    "settab.theme": "Theme:",
    "settab.dirs": "📁 Default directories",
    "settab.dir.pdf": "Default PDF directory:",
    "settab.dir.out": "Default output directory:",
    "settab.grobid": "🌐 GROBID defaults",
    "settab.grobid.url": "GROBID URL:",
    "settab.grobid.retries": "Max retries:",
    "settab.grobid.timeout": "Timeout (s):",
    "settab.ocr": "🔡 OCR defaults",
    "settab.ocr.backend": "OCR backend:",
    "settab.ocr.lang": "OCR language(s):",
    "settab.ocr.lang.placeholder": "en, en,ja, en,ch_sim…",
    "settab.ocr.caption_window": "Caption window (GROBID):",
    "settab.ocr.od_caption_window": "OD caption window:",
    "settab.llm": "🧠 LLM / M3 defaults",
    "settab.llm.backend": "LLM backend:",
    "settab.m3.model": "M3 model:",
    "settab.m3.lang": "M3 prompt lang:",
    "settab.m3.budget": "M3 thinking budget:",
    "settab.m3.output": "M3 max output tokens:",
    "settab.m3.timeout": "M3 timeout (s):",
    "settab.m3.max_retries": "M3 max retries:",
    "settab.pbdb": "🦴 Paleobiology Database defaults",
    "settab.pbdb.use": "Enable PBDB enrichment (taxonomy + occurrences)",
    "settab.pbdb.occ": "Max occurrences per species:",
    "settab.pbdb.endpoint": "PBDB endpoint:",
    "settab.pbdb.endpoint.placeholder": "(leave blank for default)",
    "settab.diag": "🛠️ Diagnostics",
    "settab.diag.dpi": "Render DPI:",
    "settab.diag.save_intermediate": "Save intermediate panels (large disk usage)",
    "settab.diag.log_btn": "📂 Open log file",
    "settab.diag.log_label": "Logs:",
    "settab.save": "💾 Save settings",
    "settab.save.done": "Settings saved.",
    "settab.reset": "Reset to defaults",
    "settab.reset.confirm.title": "Reset settings?",
    "settab.reset.confirm.body": "This will reset all settings to their defaults. Continue?",
    "settab.reset.done": "Settings reset to defaults.",
    "settab.log.open_fail": "Could not open: {error}\\n\\nPath: {path}",
    "settab.log.path": "Log file: {path}",

    # ============================================================
    # Batch dialog
    # ============================================================
    "batch.title": "Batch manager — Queue multiple PDFs",
    "batch.files": "📚 PDFs to process (in serial order)",
    "batch.add": "➕ Add PDFs…",
    "batch.add_dir": "📁 Add directory…",
    "batch.remove": "➖ Remove selected",
    "batch.clear": "🗑 Clear all",
    "batch.options": "⚙️ Batch options",
    "batch.outdir": "Output directory:",
    "batch.browse": "Browse…",
    "batch.stop_on_error": "Stop batch on first error",
    "batch.xlsx_at_end": "Export combined xlsx when all jobs finish",
    "batch.count.zero": "0 PDFs queued",
    "batch.count": "{n} PDFs queued",
    "batch.cancel": "Cancel",
    "batch.start": "▶  Start batch",
    "batch.no_pdfs": "No PDFs queued.",
    "batch.no_outdir.title": "Batch",
    "batch.no_outdir.body": "Please choose an output directory.",

    # ============================================================
    # Image preview
    # ============================================================
    "preview.zoom_in": "🔍+",
    "preview.zoom_out": "🔍−",
    "preview.fit": "⛶",
    "preview.actual": "1:1",
    "preview.hint": "Wheel = zoom · Drag = pan · Double-click = fit · Click a bbox to select",
    "preview.no_image": "(no image)",
    "preview.missing": "(missing) {name}",
    "preview.failed": "(failed to load) {name}",

    # ============================================================
    # Main window status / metrics
    # ============================================================
    "main.idle": "Ready",
    "main.running": "Job {id} running…",
    "main.done": "Job {id} done — {rows} rows",
    "main.failed": "Job {id} failed",
    "main.batch_complete": "Batch complete.",

    # ============================================================
    # Common dialogs
    # ============================================================
    "common.ok": "OK",
    "common.cancel": "Cancel",
    "common.save": "Save",
    "common.yes": "Yes",
    "common.no": "No",
    "common.error": "Error",
    "common.warning": "Warning",
    "common.info": "Information",
    "common.browse": "Browse…",
    "common.confirm": "Confirm",
    "common.settings_saved": "Settings saved.",
    "common.retry.title": "Retry",
    "common.retry.body": "Original file no longer exists:\\n{path}",

    # ============================================================
    # Open-file / save-file dialog filters
    # ============================================================
    "filter.pdf": "PDF files (*.pdf)",
    "filter.all": "All files (*)",
    "filter.xlsx": "Excel files (*.xlsx)",
    "filter.json": "JSON files (*.json)",
    "filter.csv": "CSV files (*.csv)",
    "filter.zip": "Zip files (*.zip)",
    "filter.dir": "Directories",
    # Added in Phase 34 for language switcher
    "settab.lang": "Language:",
    "app.name": "RLPE - Radiolarian Plate Extractor",
    "app.version": "0.1.0",
}
