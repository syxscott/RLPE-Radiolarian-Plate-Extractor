#!/usr/bin/env python3
"""RLPE GUI entry point.

Usage
-----
::

    # From the project root:
    python main.py

    # Or as a script entry (after ``pip install -e .``):
    rlpe-gui

The entry point itself is intentionally tiny: all real work lives
in ``rlpe.gui.app.run_app``. This file exists so the user can launch
the GUI without installing the package — useful during development.

Qt platform notes
----------------
PySide6 6.5+ requires ``libxcb-cursor0`` for the XCB platform
plugin. If the system is missing it (common on minimal Ubuntu
Docker images), the user sees::

    qt.qpa.plugin: Could not load the Qt platform plugin "xcb" ...
    This application failed to start ...

This entry point works around the gap by preloading
``libxcb-cursor.so.0`` from the local anaconda installation (or
the system path) via ``ctypes.CDLL`` *before* the Qt plugin loader
runs. Setting ``LD_PRELOAD`` from inside Python is too late
(the dynamic loader has already started before Python executes
user code), but ``ctypes.CDLL(..., mode=RTLD_GLOBAL)`` succeeds
because Qt's plugin loader calls ``dlopen`` on the cursor lib on
demand, by which time the symbol is already in the global
namespace.
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path


def _preload_xcb_cursor_if_available() -> None:
    """Best-effort preload of libxcb-cursor.so.0 from common paths.

    Idempotent: does nothing if the lib is absent or already loaded.
    Catches + swallows all exceptions — a missing xcb-cursor is a
    recoverable error (the offscreen / vnc / wayland plugins still
    work without it).
    """
    candidates = [
        # Anaconda's own Qt6 / libxcb-cursor bundle (very common in
        # our CV env). Symlink chain:
        #   libxcb-cursor.so → libxcb-cursor.so.0 → libxcb-cursor.so.0.0.0
        "/home/user/anaconda3/lib/libxcb-cursor.so.0",
        # System path on Debian / Ubuntu
        "/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0",
        # Conda envs
        "/opt/conda/lib/libxcb-cursor.so.0",
    ]
    for path in candidates:
        if not Path(path).exists():
            continue
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass  # Best effort — Qt's plugin loader will report a
                  # specific error if the .so is incompatible.
        return


def main() -> int:
    # Workaround for missing libxcb-cursor0 on systems that don't
    # have the apt package installed. Must run before any Qt import
    # (PySide6.QtGui, PySide6.QtWidgets, etc.) so the symbol is
    # in the global namespace by the time the plugin loader runs.
    _preload_xcb_cursor_if_available()

    # Make ``src/`` importable when running from a checkout where the
    # package is not yet ``pip install``-ed. This mirrors the
    # ``run_web_server.py`` pattern in this repo.
    src = Path(__file__).resolve().parent / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from rlpe.gui.app import run_app

    return run_app(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())