"""Utility helpers for the RLPE GUI."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .constants import LOG_FILE_NAME


# ============================================================
# Logging
# ============================================================

_GUI_LOGGER_NAME = "rlpe.gui"


def get_gui_logger() -> logging.Logger:
    """Return the shared GUI logger, attaching a file handler on first call.

    The file handler writes to ``~/.cache/rlpe/gui/rlpe-gui.log``
    (Linux) / ``%LOCALAPPDATA%\\rlpe\\gui\\rlpe-gui.log`` (Windows)
    / ``~/Library/Logs/rlpe/gui/rlpe-gui.log`` (macOS) — anything
    that ``Path.home() / ".cache" / "rlpe" / "gui"`` maps to.
    """
    logger = logging.getLogger(_GUI_LOGGER_NAME)
    if getattr(get_gui_logger, "_initialised", False):
        return logger
    logger.setLevel(logging.INFO)
    log_dir = Path.home() / ".cache" / "rlpe" / "gui"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / LOG_FILE_NAME
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)
    logger.info("RLPE GUI logger initialised at %s", log_path)
    get_gui_logger._initialised = True  # type: ignore[attr-defined]
    return logger


def log_call(func):
    """Decorator: log function entry + exit at INFO; exceptions at ERROR."""

    def wrapper(*args, **kwargs):
        log = get_gui_logger()
        try:
            log.info("CALL %s", func.__qualname__)
            result = func(*args, **kwargs)
            log.info("DONE %s", func.__qualname__)
            return result
        except Exception as exc:
            log.exception("FAIL %s: %s", func.__qualname__, exc)
            raise

    return wrapper


# ============================================================
# Path / file helpers
# ============================================================


def file_size_human(path: Path | None) -> str:
    """Format file size like ``12.4 MB``. Returns ``"—"`` if no path.

    Phase 42: previous implementation rounded 999 B to "1.0 KB" because
    it divided BEFORE the comparison. Fixed: pick the unit based on
    the size FIRST, then format with at most 1 decimal place.
    """
    if path is None or not path.exists():
        return "—"
    size = path.stat().st_size
    if size < 0:
        return "—"
    # Pick the right unit (1024-based for binary, but readable).
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            # Integer B for byte units, 1-decimal for everything else.
            return f"{size} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def short_path(path: Path | None, max_chars: int = 60) -> str:
    """Shorten a path for display in narrow widgets."""
    if path is None:
        return "—"
    s = str(path)
    if len(s) <= max_chars:
        return s
    # Keep filename + 1 prefix dir
    parts = path.parts
    if len(parts) >= 2:
        tail = str(Path(*parts[-2:]))
        return "…" + tail if len(tail) < max_chars else "…" + s[-max_chars + 1:]
    return "…" + s[-max_chars + 1:]


def truncate(text: str, max_chars: int = 120, ellipsis: str = "…") -> str:
    """Truncate text with an ellipsis. Used for table cells."""
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(ellipsis)] + ellipsis


# ============================================================
# ID / hash helpers
# ============================================================


def paper_id_from_path(pdf_path: Path) -> str:
    """SHA1-based stable paper ID, matching ``rlpe.utils.stable_id``.

    We re-implement here to avoid a circular import (gui imports
    pipeline which imports utils — keeping gui self-contained).
    """
    h = hashlib.sha1(str(pdf_path).encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


def short_hash(text: str, length: int = 8) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


# ============================================================
# Format helpers
# ============================================================

_HTML_ESCAPE = re.compile(r"[&<>\"]")
_HTML_ESCAPE_MAP = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
}


def html_escape(text: str) -> str:
    """Lightweight HTML escaper for ``QTextBrowser`` content."""
    if text is None:
        return ""
    return _HTML_ESCAPE.sub(lambda m: _HTML_ESCAPE_MAP[m.group(0)], text)


def fmt_duration(seconds: float | None) -> str:
    """Format a duration like ``00:01:23``. Negative/None → ``"—"``."""
    if seconds is None or seconds < 0:
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_count(value: int | None) -> str:
    if value is None:
        return "—"
    return f"{value:,}"


def fmt_percent(value: float | None, digits: int = 0) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def fmt_confidence(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


def fmt_coord(lat: float | None, lon: float | None) -> str:
    if lat is None or lon is None:
        return "—"
    return f"{lat:.3f}, {lon:.3f}"


# ============================================================
# Safe QObject slot wrapper
# ============================================================


def safe_slot(slot):
    """Decorator: catch and log exceptions inside a slot.

    Qt slots that raise break the signal/slot connection. This
    decorator ensures an unhandled exception is logged and the GUI
    stays responsive instead of dying silently.
    """

    def wrapper(*args, **kwargs):
        try:
            return slot(*args, **kwargs)
        except Exception:
            get_gui_logger().exception("Slot %s raised", slot.__qualname__)
            return None

    return wrapper


# ============================================================
# JSON serialisation helpers (used by Settings tab)
# ============================================================


def safe_json_loads(text: str, default: Any = None) -> Any:
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return default


def to_jsonable(obj: Any) -> Any:
    """Convert dataclasses / Path / etc to JSON-safe primitives.

    Phase 42 audit fix: previously this fell through to ``str(obj)``
    for unknown types — silently corrupting ``bytes`` (became
    ``"b'\\x89PNG...'"`` Python repr) and ``datetime`` (lost
    timezone information). Now we handle the common cases
    explicitly and raise ``TypeError`` for genuinely
    un-serialisable types so callers know.
    """
    import base64
    import datetime
    import decimal
    if is_dataclass(obj):
        return to_jsonable(asdict(obj))
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        return {"__bytes__": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, datetime.timedelta):
        return obj.total_seconds()
    if isinstance(obj, decimal.Decimal):
        return str(obj)
    if isinstance(obj, set):
        return [to_jsonable(v) for v in sorted(obj, key=repr)]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    raise TypeError(
        f"to_jsonable: cannot serialize {type(obj).__name__} "
        f"(value={obj!r:.100})"
    )


# ============================================================
# Single-instance enforcement (avoid double-launch)
# ============================================================

def single_instance_lock(name: str = "rlpe-gui") -> bool:
    """Try to acquire a named lock file under ``~/.cache/rlpe/gui/``.

    Returns True if this is the only running instance, False if
    another RLPE GUI is already running (in which case the caller
    should warn the user and exit).

    Phase 42: previously this used ``fcntl.flock`` which is
    Unix-only — on Windows the ``import fcntl`` would raise
    ImportError. Fixed: try ``msvcrt.locking`` on Windows, fall
    back to a TCP port lock as a cross-platform last resort.
    """
    import socket

    lock_dir = Path.home() / ".cache" / "rlpe" / "gui"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{name}.lock"

    if sys.platform.startswith("win"):
        # Windows: use msvcrt.locking (lock the first byte of the file).
        try:
            import msvcrt  # type: ignore[import-not-found]
        except ImportError:
            return _tcp_port_lock(name)
        fp = open(lock_path, "w")
        try:
            # LK_NBLCK = 2 (non-blocking), LK_LOCK = 1. The file size
            # must be >= 1 byte for locking to work.
            fp.write("0")
            fp.flush()
            msvcrt.locking(fp.fileno(), msvcrt.LK_NBLCK, 1)
        except (OSError, IOError):
            fp.close()
            return False
        fp.flush()
        globals()["_SINGLE_INSTANCE_FP"] = fp  # type: ignore[assignment]
        return True

    # Unix / macOS: use fcntl.flock with LOCK_EX | LOCK_NB.
    try:
        import fcntl
    except ImportError:
        return _tcp_port_lock(name)
    fp = open(lock_path, "w")
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fp.close()
        return False
    fp.write(str(Path.home()))
    fp.flush()
    # Hold the lock by keeping the file descriptor open on the
    # singleton module — this module persists for the GUI's lifetime.
    globals()["_SINGLE_INSTANCE_FP"] = fp  # type: ignore[assignment]
    return True


def _tcp_port_lock(name: str) -> bool:
    """Cross-platform fallback for ``single_instance_lock``.

    Phase 42: try to bind a TCP socket to a deterministic port
    derived from the lock name. If the bind fails, another
    instance is running. Not as robust as a real file lock (no
    cleanup on crash) but works on every OS.
    """
    import socket
    # Hash the name to a port in the IANA private range (49152+).
    port = 49152 + (hash(name) & 0x3FFF)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        sock.close()
        return False
    sock.listen(1)
    globals()["_SINGLE_INSTANCE_FP"] = sock  # type: ignore[assignment]
    return True


# ============================================================
# Misc
# ============================================================


def is_windows() -> bool:
    return sys.platform.startswith("win")


def is_macos() -> bool:
    return sys.platform == "darwin"


def is_linux() -> bool:
    return sys.platform.startswith("linux")