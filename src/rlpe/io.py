from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from .utils import ensure_dir

logger = logging.getLogger(__name__)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically write a CSV file.

    Audit 2026-09-01 CR-1 / BL-34: the previous implementation wrote
    directly via ``path.open("w")`` — a mid-write SIGKILL / OOM /
    ctrl-C left a half-written CSV with only the header (or a
    truncated subset of rows). Downstream consumers then loaded the
    zero-row "header-only" file and reported empty results without
    any error. Now we write to a sibling temp file in the same
    directory and ``os.replace`` atomically over the destination
    (POSIX guarantees the rename is atomic on the same filesystem).
    """
    ensure_dir(path.parent)
    if not rows:
        # Empty result: still atomic, but use the same temp+rename path
        # so an interrupted write never leaves a half-empty file.
        import os as _os
        import tempfile as _tempfile

        fd, tmp_path = _tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with _os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
                pass
            _os.replace(tmp_path, path)
        except Exception:
            try:
                _os.unlink(tmp_path)
            except OSError:
                pass
            raise
        return
    import os as _os
    import tempfile as _tempfile

    fd, tmp_path = _tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with _os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            try:
                _os.fsync(f.fileno())
            except OSError:
                # fsync may fail on some network filesystems; the
                # rename is still atomic so we don't lose data.
                pass
        _os.replace(tmp_path, path)
    except Exception:
        try:
            _os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file, skipping malformed lines with a warning.

    A single corrupt line in a multi-thousand-row JSONL used to crash
    the entire loader (and every downstream consumer). Now we log
    and skip — the caller sees a short list and a warning, and the
    pipeline can finish what it can. This matches the behaviour the
    eval and export CLIs were always assumed to have.
    """
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed JSONL line %d in %s: %s",
                    line_no,
                    path,
                    exc,
                )
    return rows
