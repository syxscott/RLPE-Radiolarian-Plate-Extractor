from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(text: str, fallback: str = "item") -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text.lower() or fallback


def stable_id(path: Path | str) -> str:
    """Stable identifier for a PDF — derived from file content (size + SHA1
    of the bytes), not from the path. The same PDF processed from
    different directories must produce the same paper_id, otherwise
    downstream eval/gold matching breaks on re-runs.

    Falls back to a path-based hash when the file does not exist (e.g.
    unit tests passing a placeholder path).

    The hash is computed in streaming chunks so a 100MB PDF doesn't
    allocate 100MB of RAM per call (and so the worker pool doesn't
    cumulatively hold many full PDFs in memory at once).
    """
    p = Path(path)
    try:
        if p.is_file():
            size = p.stat().st_size
            h = hashlib.sha1()
            # size prefix prevents the (vanishingly rare) SHA1 collision
            # between two PDFs of the same content length, and makes
            # the id clearly content-derived.
            h.update(f"{size}:".encode("ascii"))
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()[:16]
    except OSError:
        pass
    return hashlib.sha1(str(path).encode("utf-8", errors="ignore")).hexdigest()[:16]


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default
