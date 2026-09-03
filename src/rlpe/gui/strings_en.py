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
    "toolbar.title": "Main toolbar",
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
    "runtab.label.caption_window.tooltip": "GROBID caption→page lookup window",
    "runtab.label.od_caption_window": "OD caption window:",
    "runtab.label.od_caption_window.tooltip": "OpenDataLoader caption↔image cross-page window",
    "runtab.label.workers": "Workers:",
    "runtab.label.panel_score": "Panel score:",
    "runtab.label.use_gpu": "Use GPU:",
    "runtab.gpu_check": "Auto-detect CUDA at startup",
    "runtab.ocr_lang.placeholder": "e.g. English, 中文 (简体), 日本語",
    "runtab.ocr_lang.tooltip": "OCR language (e.g. English, 中文, 日本語).\nEditable — power users can type 'en,ja' for multi-lang.",
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
    "runtab.status.cancelled": "Cancelled",
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
    # M-19: input validation errors
    "runtab.error.invalid_pdf": "The selected file is not a valid PDF.",
    "runtab.error.invalid_grobid_url": "GROBID URL must be a valid http(s) URL.",
    # ============================================================
    # Jobs tab
    # ============================================================
    "jobstab.clear_finished": "Clear finished",
    "jobstab.clear_all": "Clear all",
    # Audit 2026-09-03 (user-reported): "Clear all" now soft-hides
    # jobs (QSettings-backed). Two new toolbar buttons let the
    # operator peek back at hidden jobs or permanently wipe the
    # on-disk data after confirmation.
    "jobstab.show_hidden": "Show hidden",
    "jobstab.hide": "Hide",
    "jobstab.unhide": "Unhide",
    "jobstab.delete_permanently": "Delete permanently...",
    "jobstab.delete_permanently_confirm": (
        "Permanently delete on-disk data for {n} jobs "
        "(matches / panels / metadata / crops / OCR cache).\n\n"
        "This action is irreversible. Continue?"
    ),
    "jobstab.hidden_count_label": "{n} hidden jobs",
    "jobstab.no_hidden_jobs": "No hidden jobs",
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
    "jobstab.action.retry": "🔁  Retry",
    "jobstab.export.no_rows": "No rows to export.",
    "jobstab.export.failed": "Export failed: {error}",
    "jobstab.loaded_from_disk": "Loaded from disk",
    "jobstab.export.saved": "Saved {count} rows → {path}",
    "jobstab.export.saved_short": "Saved → {path}",
    "jobstab.partial_no_complete_flag": "manifest exists but no complete.flag (possibly interrupted)",
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
    # Phase 55 audit fix: key renamed from caption_snip → caption_snippet
    # to match the pipeline's field name (constants.py:66).
    "restab.col.caption_snippet": "Caption snippet",
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
    "restab.detail.paper_id": "Paper ID",
    "restab.detail.figure_id": "Figure ID",
    "restab.detail.panel_label": "Panel label",
    "restab.detail.geo_scope": "Geology scope",
    "restab.detail.scope.panel": "Panel-specific",
    "restab.detail.scope.figure_anchor": "Figure-anchored",
    "restab.detail.scope.none": "No geology",
    "restab.detail.ocr.image_ocr": "OCR",
    "restab.detail.ocr.positional": "Positional",
    "restab.detail.ocr.no_image": "No image",
    "restab.detail.pbdb_tax": "PBDB taxonomy",
    "restab.detail.evidence": "Extraction evidence",
    "restab.detail.geo_links": "Geology links ({count})",
    "restab.detail.geo_links_more": "… and {n} more",
    "restab.detail.family": "family: {value}",
    "restab.detail.kingdom": "Kingdom",
    "restab.detail.phylum": "Phylum",
    "restab.detail.class": "Class",
    "restab.detail.order": "Order",
    "restab.detail.genus": "Genus",
    "restab.detail.cross_figure_link": "Cross-figure link",
    "restab.detail.sample_ids": "Sample IDs",
    # audit 2026-08-17 (GUI-A3): v1.1.0 schema fields surfaced
    # in the detail panel. Confidence interval uses the
    # Wilson 95% lower/upper bounds the pipeline stamps on every
    # PanelRecord. Image-verified / review-priority are
    # operator-visible labels for the v1.1.0 schema fields.
    "restab.detail.ci": "CI",
    "restab.detail.image_verified": "Image verified",
    "restab.detail.image_unverified": "Image not verified",
    "restab.detail.review_priority": "Review priority",
    "restab.detail.scale_bar": "Scale bar",
    # audit 2026-08-17 (GUI-A4): Results tab "Mark verified" /
    # "Mark unverified" buttons + success / failure dialog messages.
    "restab.detail.mark_verified": "✓  Mark verified",
    "restab.detail.mark_unverified": "✗  Mark unverified",
    "restab.detail.verify_success": "Marked {n} row(s) as {state}.",
    "restab.detail.verify_failed": "Could not flip image_verified: {error}",
    "restab.detail.no_selection": "Select a row in the table first.",
    # Phase 64 Plan B (Task B.7): schematic figure content.
    "restab.detail.schematic": "Schematic content",
    "restab.detail.schematic_type": "Figure type",
    "restab.detail.schematic_text_count": "Text elements",
    "restab.detail.schematic_rel_count": "Relationships",
    "restab.detail.schematic_facts": "Extracted facts",
    "restab.detail.schematic_ages": "Ages",
    "restab.detail.schematic_geo": "Geographic",
    "restab.detail.schematic_taxa": "Taxa",
    "restab.detail.schematic_confidence": "Confidence",
    # Phase 65 Plan A.6: cross-figure linker source chip labels.
    "restab.detail.link_source.sample_match": "Sample ID match",
    "restab.detail.link_source.locality_match": "Locality match",
    "restab.detail.link_source.m3_inference": "M3 inference",
    "restab.detail.link_source.unlinked": "Unlinked",
    # Phase 66 Plan C.6: visual-coordinate cross-reference section
    # labels (Phase C fires only when Phase A Strategy-1 didn't match).
    "restab.detail.visual_links": "Visual coordinate links",
    "restab.detail.visual_target": "Target figure",
    "restab.detail.visual_layer": "Strat layer",
    "restab.detail.visual_age": "Age",
    "restab.detail.visual_formation": "Formation",
    "restab.detail.visual_confidence": "Confidence",
    "restab.detail.visual_empty": "(no visual links)",
    # ============================================================
    # Settings tab
    # ============================================================
    "settab.title": "⚙️ {app}  ·  v{version}",
    # Phase 36: QGroupBox titles wrap via "\n" so long English titles
    # don't get clipped to "🎨 Appearanc..." under narrow windows.
    "settab.appearance": "Appearance",
    "settab.theme": "Theme:",
    "settab.dirs": "Default dirs",
    "settab.dir.pdf": "Default PDF directory:",
    "settab.dir.out": "Default output directory:",
    "settab.grobid": "GROBID",
    "settab.grobid.url": "GROBID URL:",
    "settab.grobid.retries": "Max retries:",
    "settab.grobid.timeout": "Timeout (s):",
    "settab.ocr": "OCR",
    "settab.ocr.backend": "OCR backend:",
    "settab.ocr.lang": "OCR language(s):",
    "settab.ocr.lang.placeholder": "English, 中文 (简体), 日本語…",
    "settab.ocr.caption_window": "Caption window (GROBID):",
    "settab.ocr.od_caption_window": "OD caption window:",
    "settab.llm": "LLM / M3",
    "settab.llm.backend": "LLM backend:",
    "settab.m3.model": "M3 model:",
    "settab.m3.lang": "M3 prompt lang:",
    "settab.m3.budget": "M3 thinking budget:",
    "settab.m3.output": "M3 max output tokens:",
    "settab.m3.timeout": "M3 timeout (s):",
    "settab.m3.max_retries": "M3 max retries:",
    "settab.pbdb": "PBDB",
    "settab.pbdb.use": "Enable PBDB enrichment (taxonomy + occurrences)",
    "settab.pbdb.occ": "Max occurrences per species:",
    "settab.pbdb.endpoint": "PBDB endpoint:",
    "settab.pbdb.endpoint.placeholder": "(leave blank for default)",
    "settab.diag": "Diagnostics",
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
    "settab.yolo": "YOLO Radiolaria Detection",
    "settab.yolo.enable": "Enable YOLO detection (replaces OpenCV)",
    "settab.yolo.model_path": "Model path:",
    "settab.yolo.conf": "Confidence threshold:",
    "settab.yolo.iou": "NMS IoU threshold:",
    "settab.yolo.browse": "Browse…",
    # audit 2026-07-27 B3: YOLO model path validation warning
    "settab.yolo.warn.title": "YOLO Model Required",
    "settab.yolo.warn.body": "Please select a YOLO model file (.pt) before enabling YOLO detection.",
    "settab.log.open_fail": "Could not open: {error}\n\nPath: {path}",
    "settab.log.path": "Log file: {path}",
    "settab.log.not_yet": "No log file yet.\n\nThe log file is created on the first pipeline run.\n\nExpected path:\n{path}",
    "settab.log.title": "Log file",
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
    "batch.outdir.not_writable": "Output directory is not writable.\n\n{path}\n\n{error}",
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
    # Phase 6A (NIT-5): bbox hover-tooltip field labels
    "preview.tooltip.confidence": "confidence: {value}",
    "preview.tooltip.coords_xy": "x: {x}  y: {y}",
    "preview.tooltip.coords_wh": "w: {w}  h: {h}",
    "preview.tooltip.family": "family: {name}",
    # ============================================================
    # Main window status / metrics
    # ============================================================
    "main.idle": "Ready",
    "main.running": "Job {id} running…",
    "main.done": "Job {id} done — {rows} rows",
    "main.failed": "Job {id} failed",
    "main.batch_complete": "Batch complete.",
    "main.batch_stopped_on_error": "Batch stopped on error: {failed} ({remaining} remaining).",
    "main.recent_loaded": "Loaded {n} recent job(s) from disk.",
    # Phase F-2 (M-8): progress messages use an i18n key so language
    # switch re-renders the status bar correctly. Previously
    # _on_job_progress wrote directly to _status_perm.setText(),
    # bypassing _status_key / _status_kwargs — after that, switching
    # language jumped back to the stale key.
    "main.progress": "{msg}  ({current}/{total})",
    "main.cancelled": "Job {id} cancelled",
    # Phase F-2 (M-25): batch export on background thread
    "main.batch_exporting": "Exporting batch xlsx: {msg}",
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
    # Phase 55 audit HIGH-7 fix: used as the directory-chooser dialog title
    # in settings_tab._pick_dir. Previously undefined → ⟦sentinel⟧.
    "common.choose_dir": "Choose {kind}",
    "common.confirm": "Confirm",
    "common.settings_saved": "Settings saved.",
    "common.retry.title": "Retry",
    "common.retry.body": "Original file no longer exists:\n{path}",
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
    # ============================================================
    # Phase 48: file-dialog titles + export dialogs + summary labels
    # ============================================================
    # Run tab file dialogs
    "runtab.browse.title": "Choose a radiolarian paper PDF",
    "runtab.out.choose.title": "Choose an output directory",
    "runtab.out.no_outdir.title": "Output",
    "runtab.out.no_outdir.body": "No output directory set yet.",
    # Batch dialog file dialogs
    "batch.add.title": "Add PDFs",
    "batch.add_dir.title": "Choose a directory of PDFs",
    "batch.outdir.title": "Choose batch output directory",
    # Jobs tab exports
    "jobstab.export.xlsx_title": "Export xlsx",
    "jobstab.export.json_title": "Export JSON",
    # Results tab exports
    "restab.export.xlsx_title": "Export xlsx",
    "restab.export.json_title": "Export JSON",
    "restab.export.csv_title": "Export CSV",
    "restab.export.dwca_title": "Export DwCA",
    # Phase F-1 (2026-08-20, M-5): API URL validation warning shown
    # in the status bar when an operator-supplied / QSettings-supplied
    # override is rejected by _validate_api_url (file:/// javascript:
    # loopback without allow_local, etc.).
    "restab.api_url.invalid": "Invalid API URL; using default",
    # Jobs tab summary
    "jobstab.summary.count": "{total} jobs · running {running} · done {done} · failed {failed}",
    "jobstab.summary.count_label": "{total} jobs  ·  running {running}  ·  done {done}  ·  failed {failed}",
    # Added in Phase 34 for language switcher
    "settab.lang": "Language:",
    "app.name": "RLPE - Radiolarian Plate Extractor",
    "app.version": "0.1.0",
}


# Phase 55 audit M3 — fail loudly on duplicate keys (same guard as
# strings_zh_CN). Python silently keeps the LAST duplicate in a
# dict literal, masking upstream bugs in the translation generator.
import re as _re_dup_en

_src_dup_en = _re_dup_en.sub(
    r"\"\"\".*?\"\"\"", "", open(__file__, encoding="utf-8").read(), flags=_re_dup_en.S
)
_keys_dup_en = _re_dup_en.findall(
    r"^\s*['\"]([a-zA-Z][\w.]*)['\"]\s*:", _src_dup_en, flags=_re_dup_en.M
)
import collections as _coll_dup_en

_dup_counts_en = _coll_dup_en.Counter(_keys_dup_en)
_dup_check_en = [k for k, n in _dup_counts_en.items() if n > 1]
if _dup_check_en:
    raise RuntimeError(
        f"strings_en has duplicate keys: {_dup_check_en!r}. "
        "Each English key MUST map to exactly one translation."
    )
del _dup_check_en, _dup_counts_en, _keys_dup_en, _src_dup_en
