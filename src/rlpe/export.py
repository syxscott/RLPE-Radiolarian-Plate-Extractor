from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from .utils import ensure_dir


def export_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_json(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_assets(rows: list[dict[str, Any]], dst_dir: Path, key: str = "panel_path") -> list[dict[str, Any]]:
    ensure_dir(dst_dir)
    copied: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        src = row.get(key)
        if src:
            src_path = Path(src)
            if src_path.exists():
                dst_path = dst_dir / src_path.name
                shutil.copy2(src_path, dst_path)
                new_row[key] = str(dst_path)
        copied.append(new_row)
    return copied
