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
# ============================================================
# Input widget sizing (Phase 33)
# ============================================================
# Centralised so all tabs use consistent widths. These were chosen
# to fit:
#   - Default system font (13px)
#   - Most common values (e.g. PBDB endpoint URL, GROBID URL,
#     GROBID timeout numbers)
#   - Chinese placeholder text which is often longer than the
#     English equivalent (Phase 33 bug fix: results_tab search box
#     was hard-coded at 360px which clipped the Chinese placeholder)

# Editable text fields
INPUT_WIDTH_SHORT: Final[int] = 100      # SpinBox for small ints (0-99)
INPUT_WIDTH_MEDIUM: Final[int] = 180     # Single-line text (URLs, paths)
INPUT_WIDTH_LONG: Final[int] = 320      # Long URLs, search boxes
INPUT_WIDTH_PATH: Final[int] = 420      # File paths (PDF, output dir)
INPUT_WIDTH_OCR_LANG: Final[int] = 160   # OCR language code lists
INPUT_WIDTH_FULL: Final[int] = 600      # Long-form comments / descriptors

# Buttons
BUTTON_MIN_WIDTH: Final[int] = 90
BUTTON_MIN_HEIGHT: Final[int] = 32     # Phase 36: bumped 30 → 32 for parity with input rows
BUTTON_PRIMARY_HEIGHT: Final[int] = 34     # Main action button (e.g. Start)

# QComboBox
COMBO_MIN_WIDTH: Final[int] = 130


# ============================================================
# Phase 42: range / constant validation
# ============================================================
# All RANGE_* tuples must satisfy (min < max) and both bounds must
# be non-negative. Validate at import time so a typo in a constant
# (e.g. swapping (1, 10) to (10, 1)) fails loudly here instead of
# producing a silent zero-row run later.
def _validate_ranges() -> None:
    import sys as _sys
    for _name, _val in list(globals().items()):
        if not _name.startswith("RANGE_"):
            continue
        if not isinstance(_val, tuple) or len(_val) != 2:
            raise ValueError(
                f"constants.{_name} must be a 2-tuple, got {_val!r}"
            )
        _lo, _hi = _val
        if not isinstance(_lo, (int, float)) or not isinstance(_hi, (int, float)):
            raise ValueError(
                f"constants.{_name} bounds must be numeric, got {_val!r}"
            )
        if type(_lo) is not type(_hi):
            raise ValueError(
                f"constants.{_name} bounds must be same type, got {_val!r}"
            )
        if _lo >= _hi:
            raise ValueError(
                f"constants.{_name} must satisfy min < max, got {_val!r}"
            )


_validate_ranges()

# Phase 42: also validate that DEFAULT_LLM_BACKEND matches what
# pipeline_worker._build_config defaults to (the bug: the constant
# was "minimax" but the worker defaulted to "rules"). We can only
# check at import time that the constant is one of the known
# backends, not that pipeline_worker actually uses it.
_KNOWN_LLM_BACKENDS: Final[tuple[str, ...]] = (
    "minimax", "minimax-m3", "minimax_api", "rules",
    "transformers", "ollama", "llamacpp",
)
if DEFAULT_LLM_BACKEND not in _KNOWN_LLM_BACKENDS:
    raise ValueError(
        f"DEFAULT_LLM_BACKEND={DEFAULT_LLM_BACKEND!r} is not in the "
        f"known backends list: {_KNOWN_LLM_BACKENDS}"
    )


# Phase 46: OCR language friendly names.
# The OCR backend accepts ISO codes (en, ja, ch_sim, ch_tra, ...) but
# these are unfriendly for end users. We map them to human-readable
# labels and let the user pick from a QComboBox. The ISO code is
# stored in the userData so the worker still receives the right
# value.
#
# Each entry: (iso_code, english_name, chinese_name)
OCR_LANGUAGE_OPTIONS: Final[tuple[tuple[str, str, str], ...]] = (
    ("en", "English", "英语"),
    ("ch_sim", "Chinese (Simplified)", "中文 (简体)"),
    ("ch_tra", "Chinese (Traditional)", "中文 (繁體)"),
    ("ja", "Japanese", "日语"),
    ("ko", "Korean", "韩语"),
    ("fr", "French", "法语"),
    ("de", "German", "德语"),
    ("ru", "Russian", "俄语"),
)


def ocr_lang_to_friendly_name(iso_code: str) -> str:
    """Convert an ISO OCR code (e.g. ``"ch_sim"``) to a human-readable
    name. Falls back to the ISO code if it's not in the table."""
    for code, en_name, zh_name in OCR_LANGUAGE_OPTIONS:
        if code == iso_code:
            # Use the current language to pick the right name.
            from . import i18n as _i18n
            lang = _i18n.current_language()
            return zh_name if lang == "zh_CN" else en_name
    return iso_code


def ocr_lang_friendly_options() -> list[tuple[str, str]]:
    """Return ``[(iso_code, friendly_name), ...]`` in the current
    language. Used by the GUI to populate the OCR language
    QComboBox so users see ``"English"`` / ``"中文 (简体)"`` instead
    of ``"en"`` / ``"ch_sim"``.
    """
    from . import i18n as _i18n
    lang = _i18n.current_language()
    return [
        (code, zh_name if lang == "zh_CN" else en_name)
        for code, en_name, zh_name in OCR_LANGUAGE_OPTIONS
    ]


# ============================================================
# Phase 47: friendly-name mappings for other technical tokens
# ============================================================
# OCR backend, LLM backend, M3 prompt language, and theme all show
# raw technical strings to the user (e.g. "paddleocr",
# "minimax", "auto", "dark"). Phase 47 maps them to friendly
# Chinese / English names. Each entry: (raw_code, en_name, zh_name)

# OCR backend options
OCR_BACKEND_OPTIONS: Final[tuple[tuple[str, str, str], ...]] = (
    ("paddleocr", "PaddleOCR (Recommended)", "PaddleOCR (推荐)"),
    ("easyocr", "EasyOCR (Multi-language)", "EasyOCR (多语言)"),
)


def ocr_backend_friendly_options() -> list[tuple[str, str]]:
    """``[(code, friendly_name), ...]`` in the current language."""
    from . import i18n as _i18n
    lang = _i18n.current_language()
    return [
        (code, zh_name if lang == "zh_CN" else en_name)
        for code, en_name, zh_name in OCR_BACKEND_OPTIONS
    ]


# LLM backend options
LLM_BACKEND_OPTIONS: Final[tuple[tuple[str, str, str], ...]] = (
    ("minimax", "MiniMax-M3 (Recommended)", "MiniMax-M3 (推荐)"),
    ("minimax-m3", "MiniMax-M3 (alt endpoint)", "MiniMax-M3 (备用接入点)"),
    ("minimax_api", "MiniMax API (legacy)", "MiniMax API (旧版)"),
    ("transformers", "Local Transformers", "本地 Transformers"),
    ("ollama", "Ollama (Local Server)", "Ollama (本地服务器)"),
    ("llamacpp", "llama.cpp (Local)", "llama.cpp (本地)"),
    ("rules", "Rules only (no LLM)", "仅规则匹配 (无 LLM)"),
)


def llm_backend_friendly_options() -> list[tuple[str, str]]:
    """``[(code, friendly_name), ...]`` in the current language."""
    from . import i18n as _i18n
    lang = _i18n.current_language()
    return [
        (code, zh_name if lang == "zh_CN" else en_name)
        for code, en_name, zh_name in LLM_BACKEND_OPTIONS
    ]


# M3 prompt language options
M3_PROMPT_LANG_OPTIONS: Final[tuple[tuple[str, str, str], ...]] = (
    ("auto", "Auto-detect", "自动检测"),
    ("zh", "Chinese (中文)", "中文 (中文)"),
    ("en", "English", "英语"),
    ("ja", "Japanese (日本語)", "日语 (日本語)"),
)


def m3_prompt_lang_friendly_options() -> list[tuple[str, str]]:
    """``[(code, friendly_name), ...]`` in the current language."""
    from . import i18n as _i18n
    lang = _i18n.current_language()
    return [
        (code, zh_name if lang == "zh_CN" else en_name)
        for code, en_name, zh_name in M3_PROMPT_LANG_OPTIONS
    ]


# Theme options (already uses THEME_LIGHT/THEME_DARK/THEME_SYSTEM)
THEME_OPTIONS: Final[tuple[tuple[str, str, str], ...]] = (
    ("light", "Light", "浅色"),
    ("dark", "Dark", "深色"),
    ("system", "System default", "跟随系统"),
)


def theme_friendly_options() -> list[tuple[str, str]]:
    """``[(code, friendly_name), ...]`` in the current language."""
    from . import i18n as _i18n
    lang = _i18n.current_language()
    return [
        (code, zh_name if lang == "zh_CN" else en_name)
        for code, en_name, zh_name in THEME_OPTIONS
    ]


def friendly_name_for(code: str, options) -> str:
    """Generic helper: find ``code`` in ``options`` and return the
    friendly name in the current language. Falls back to the
    raw code if not found."""
    for entry in options:
        if len(entry) == 3:
            raw, en_name, zh_name = entry
            if raw == code:
                from . import i18n as _i18n
                return zh_name if _i18n.current_language() == "zh_CN" else en_name
    return code
