"""Constants used across the RLPE GUI.

Centralised so:
* The labels, icons, and ordering stay consistent across tabs.
* Future i18n work can replace this file with a translator.
* Tests don't need to scrape string literals from widgets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final


# Application metadata
APP_NAME: Final[str] = "RLPE - Radiolarian Plate Extractor"
APP_VERSION: Final[str] = "0.1.0"
APP_ORG: Final[str] = "RLPE Project"
APP_DOMAIN: Final[str] = "rlpe.sci"
APP_AUTHOR: Final[str] = "RLPE Contributors"

# Window default sizes
MAIN_WINDOW_DEFAULT_SIZE: Final[tuple[int, int]] = (1440, 900)
MAIN_WINDOW_MIN_SIZE: Final[tuple[int, int]] = (1100, 720)
IMAGE_PREVIEW_MIN_SIZE: Final[tuple[int, int]] = (640, 480)
SETTINGS_DIALOG_DEFAULT_SIZE: Final[tuple[int, int]] = (720, 600)
BATCH_DIALOG_DEFAULT_SIZE: Final[tuple[int, int]] = (900, 600)

# Tab indices (used by QTabWidget.setCurrentIndex)
TAB_RUN: Final[int] = 0
TAB_JOBS: Final[int] = 1
TAB_RESULTS: Final[int] = 2
TAB_SETTINGS: Final[int] = 3

# Result columns shown in the results table
@dataclass(frozen=True)
class ResultColumn:
    key: str
    label: str
    width: int
    tooltip: str = ""


RESULT_COLUMNS: Final[tuple[ResultColumn, ...]] = (
    ResultColumn("species",        "Species (Latin)",        260, "Latin binomial extracted from caption"),
    ResultColumn("panel_id",       "Panel ID",              180, "Stable panel identifier (paper/figure/panel)"),
    ResultColumn("confidence",    "Confidence",              80, "Pipeline confidence (0..1)"),
    ResultColumn("caption_snip",   "Caption snippet",        360, "First 200 chars of figure caption"),
    ResultColumn("page_index",     "Page",                    60, "PDF page index (1-based)"),
    ResultColumn("family",         "PBDB Family",            180, "PaleoDB family (or genus fallback)"),
    ResultColumn("country",        "Country",                140, "Country / locality name (fallback)"),
    ResultColumn("biozone",        "Biozone",                160, "PaleoDB biozone / age interval"),
    ResultColumn("coord",          "Lat / Lon",              140, "Centroid coordinates (lat, lon)"),
)

# Job statuses (must match the API/web UI vocabulary)
STATUS_QUEUED: Final[str] = "queued"
STATUS_RUNNING: Final[str] = "running"
STATUS_DONE: Final[str] = "done"
STATUS_FAILED: Final[str] = "failed"
STATUS_CANCELLED: Final[str] = "cancelled"

# Theme constants
THEME_LIGHT: Final[str] = "light"
THEME_DARK: Final[str] = "dark"
THEME_SYSTEM: Final[str] = "system"

# Default OCR backend / LLM backend (mirror CLI defaults)
DEFAULT_OCR_BACKEND: Final[str] = "paddleocr"
DEFAULT_LLM_BACKEND: Final[str] = "minimax"
DEFAULT_MINIMAX_MODEL: Final[str] = "MiniMax-M3"
DEFAULT_GROBID_URL: Final[str] = "http://localhost:8070"
DEFAULT_OCR_LANG: Final[str] = "en"
DEFAULT_M3_PROMPT_LANG: Final[str] = "auto"
DEFAULT_CAPTION_WINDOW: Final[int] = 2
DEFAULT_OD_CAPTION_WINDOW: Final[int] = 5
DEFAULT_GROBID_MAX_RETRIES: Final[int] = 3
DEFAULT_GROBID_TIMEOUT: Final[int] = 300
DEFAULT_NUM_WORKERS: Final[int] = 1
DEFAULT_MIN_PANEL_SCORE: Final[float] = 0.80
DEFAULT_RENDER_DPI: Final[int] = 200
DEFAULT_PALEO_MAX_OCC: Final[int] = 25
DEFAULT_M3_BUDGET: Final[int] = 1024
DEFAULT_M3_THINKING_BUDGET: Final[int] = 1024
DEFAULT_M3_OUTPUT_TOKENS: Final[int] = 2048
DEFAULT_M3_TIMEOUT: Final[int] = 60
DEFAULT_M3_MAX_RETRIES: Final[int] = 3
DEFAULT_THEME: Final[str] = "light"

# File extensions accepted by the file picker
ACCEPTED_PDF_EXTS: Final[tuple[str, ...]] = (".pdf",)
ACCEPTED_XLSX_EXTS: Final[tuple[str, ...]] = (".xlsx",)
ACCEPTED_JSON_EXTS: Final[tuple[str, ...]] = (".json",)
ACCEPTED_CSV_EXTS: Final[tuple[str, ...]] = (".csv", ".tsv")
ACCEPTED_DWCA_EXTS: Final[tuple[str, ...]] = (".zip",)

# Log file location (under the user's standard cache dir)
LOG_FILE_NAME: Final[str] = "rlpe-gui.log"

# Display limits (avoid choking the UI)
MAX_LOG_LINES_IN_VIEW: Final[int] = 5000
MAX_RECENT_JOBS_IN_LIST: Final[int] = 200
PROGRESS_BAR_MIN_HEIGHT_PX: Final[int] = 18

# Hard limits on input ranges (mirrors JobOptions validators)
RANGE_GROBID_MAX_RETRIES: Final[tuple[int, int]] = (1, 10)
RANGE_GROBID_TIMEOUT: Final[tuple[int, int]] = (10, 3600)
RANGE_CAPTION_WINDOW: Final[tuple[int, int]] = (1, 50)
RANGE_OD_CAPTION_WINDOW: Final[tuple[int, int]] = (1, 200)
RANGE_NUM_WORKERS: Final[tuple[int, int]] = (1, 32)
RANGE_PANEL_SCORE: Final[tuple[float, float]] = (0.0, 1.0)
RANGE_DPI: Final[tuple[int, int]] = (50, 600)
RANGE_PALEO_OCC: Final[tuple[int, int]] = (1, 1000)
RANGE_M3_BUDGET: Final[tuple[int, int]] = (0, 32000)
RANGE_M3_OUTPUT_TOKENS: Final[tuple[int, int]] = (1, 32000)
RANGE_M3_TIMEOUT: Final[tuple[int, int]] = (1, 3600)
RANGE_M3_MAX_RETRIES: Final[tuple[int, int]] = (1, 20)

# Status colour mapping (used by QSS / QPalette)
STATUS_COLORS: Final[dict[str, str]] = {
    STATUS_QUEUED:   "#7f7f7f",  # grey
    STATUS_RUNNING:  "#1f77b4",  # blue
    STATUS_DONE:     "#2ca02c",  # green
    STATUS_FAILED:   "#d62728",  # red
    STATUS_CANCELLED:"#ff7f0e",  # orange
}

# QSettings keys
QS_KEY_LAST_DIR: Final[str] = "io/last_pdf_dir"
QS_KEY_LAST_EXPORT_DIR: Final[str] = "io/last_export_dir"
QS_KEY_THEME: Final[str] = "ui/theme"
QS_KEY_GEOMETRY: Final[str] = "ui/main_window_geometry"
QS_KEY_STATE: Final[str] = "ui/main_window_state"
QS_KEY_RECENT_DIRS: Final[str] = "io/recent_dirs"