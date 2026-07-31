"""Writability probe for user-selected output directories.

Extracted from ``batch_dialog.py`` (audit 2026-07-31) so the probe
can be unit-tested WITHOUT importing PySide6 — the GUI modules are
skipped by the test suite when PySide6 is absent, which let the
batch-dialog crash regressions (missing ``import os``, ``str / str``)
through. This module has no Qt imports.
"""

from __future__ import annotations

import os
from pathlib import Path

# audit 2026-07-31: the previous probe did ``out_dir / ...`` on the
# raw ``str`` from ``QLineEdit.text()`` (TypeError) and referenced
# ``os.getpid()`` while only ``_os_probe`` was imported (NameError).
# Both crashed the batch dialog before ``batch_started`` could emit.
# The probe now lives here as a pure function operating on a ``str``.


def probe_output_dir_writable(out_dir: str | os.PathLike[str]) -> str | None:
    """Return ``None`` if ``out_dir`` exists and is writable, else an
    error message suitable for a UI dialog.

    Verifies actual writability by creating a temp file (permission
    checks can lie on network / overlay filesystems). Uses
    ``os.fsync`` on the open handle rather than ``os.sync()`` to flush
    only this file's kernel buffers, not the entire system.
    """
    probe_dir = Path(out_dir)
    probe_path: Path | None = None
    try:
        # Probe actual writability by creating a temp file.
        probe_path = probe_dir / f".rlpe_probe_{os.getpid()}.tmp"
        fd = os.open(probe_path, os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, b"ok")
            os.fsync(fd)  # flush this file's buffers to disk only
        finally:
            os.close(fd)
        if not os.access(probe_dir, os.W_OK):
            return f"Directory is not writable: {out_dir}"
        return None
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"
    finally:
        if probe_path is not None:
            try:
                probe_path.unlink()
            except OSError:
                pass
