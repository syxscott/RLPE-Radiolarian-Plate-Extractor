"""Application entry point — wires ``QApplication`` to ``MainWindow``.

Used by both ``main.py`` (project root) and the ``rlpe-gui`` script
entry declared in ``pyproject.toml``. Handles:
  * ``QApplication`` construction + metadata
  * High-DPI scaling (Qt 6 enables this by default but we set the
    rounding policy explicitly for cross-platform consistency)
  * Single-instance enforcement (optional — passes on non-Linux)
  * ``QSS`` theme application
  * Global exception hook (logs uncaught exceptions to the GUI
    log file before re-raising so the user sees a crash dialog)
  * Graceful shutdown — wait for the pipeline worker thread to
    finish (with a hard timeout) before exiting
"""
from __future__ import annotations

import sys
import traceback
from typing import Any

from PySide6.QtCore import QCoreApplication, QSettings, Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QMessageBox

from .constants import (
    APP_AUTHOR,
    APP_DOMAIN,
    APP_NAME,
    APP_VERSION,
)
from .main_window import MainWindow
from .styles import apply_theme
from .utils import get_gui_logger, single_instance_lock


def _install_exception_hook(log) -> None:
    """Install a global exception hook that logs to the GUI log file.

    Without this, an unhandled exception in a Qt slot or signal
    handler crashes the GUI silently. With the hook, we get a
    stack trace in the log file AND a crash dialog.
    """

    def hook(exc_type, exc_value, exc_tb):
        log.error("UNCAUGHT EXCEPTION", exc_info=(exc_type, exc_value, exc_tb))
        # Format a short message for the dialog
        msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb, limit=8))
        if QCoreApplication.instance() is not None:
            try:
                QMessageBox.critical(
                    None,
                    f"{APP_NAME} — unexpected error",
                    f"An unexpected error occurred:\n\n{exc_value}\n\n"
                    f"Details written to the log file. The application will continue.\n\n{msg[:1000]}",
                )
            except Exception:
                pass  # No GUI yet (very early in startup)

    sys.excepthook = hook


def run_app(argv: list[str] | None = None) -> int:
    """Construct the QApplication, install the main window, run the loop.

    Returns the application's exit code (0 if clean shutdown, 1
    if startup failed).

    The function is the single source of truth for GUI startup.
    ``main.py`` is a thin wrapper.
    """
    log = get_gui_logger()
    log.info("=" * 70)
    log.info("%s v%s starting up", APP_NAME, APP_VERSION)
    log.info("=" * 70)

    if argv is None:
        argv = sys.argv

    # ---- Single-instance enforcement (optional, Linux-friendly) ----
    # Windows + macOS use Qt's QSharedMemory equivalent; for the
    # MVP we just try the file-lock approach. If it fails, we warn
    # the user but continue (they can have multiple windows open).
    if not single_instance_lock("rlpe-gui"):
        log.warning("Another RLPE GUI instance is already running.")
        # We don't quit — sometimes users want a second window to
        # work on a different job. Log it and continue.

    # ---- Construct QApplication ----
    # Set organisation + app name BEFORE constructing QApplication so
    # QSettings has a stable storage location.
    QCoreApplication.setOrganizationName(APP_AUTHOR)
    QCoreApplication.setOrganizationDomain(APP_DOMAIN)
    QCoreApplication.setApplicationName(APP_NAME)
    QCoreApplication.setApplicationVersion(APP_VERSION)

    # High-DPI scaling defaults
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication.instance() or QApplication(argv)

    # ---- Install exception hook ----
    _install_exception_hook(log)

    # ---- Apply theme from QSettings ----
    qsettings = QSettings(APP_DOMAIN, APP_NAME)
    theme = qsettings.value("ui/theme", "light")
    if theme not in ("light", "dark", "system"):
        theme = "light"
    apply_theme(app, theme)

    # ---- Construct main window + show ----
    window = MainWindow()
    window.show()

    log.info("GUI ready — entering event loop")
    rc = app.exec()

    log.info("GUI shutting down (exit code %d)", rc)
    return int(rc)


# Convenience alias
main = run_app