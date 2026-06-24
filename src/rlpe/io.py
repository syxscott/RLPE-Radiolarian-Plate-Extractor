from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from .utils import ensure_dir

logger = logging.getLogger(__name__)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
