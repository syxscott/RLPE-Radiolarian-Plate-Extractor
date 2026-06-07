from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

from .utils import ensure_dir

# ---------------------------------------------------------------------------
# JSON sanitiser — make rows safe for json.dumps and pd.read_json
# ---------------------------------------------------------------------------


def _sanitize(obj: Any) -> Any:
    """Recursively coerce ``obj`` into a JSON-encodable value.

    - ``float('nan')`` / ``float('inf')``  → ``None``
    - ``numpy`` scalars (float32/int64/...) → ``float()`` / ``int()``
    - ``pathlib.Path`` → ``str(path)``
    - ``set`` / ``frozenset`` → ``sorted(list(...))`` (stable order)
    - unknown objects → ``repr(obj)`` (best-effort string fallback)
    """
    # None / bool / int / str — already JSON-safe
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    # NaN / Inf — must be checked BEFORE float, because isinstance(float('nan'), float) is True
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    # numpy scalars (optional dep)
    try:
        import numpy as np  # type: ignore
        if isinstance(obj, np.floating):
            v = float(obj)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return _sanitize(obj.tolist())
    except ImportError:
        pass
    # pathlib
    try:
        from pathlib import Path as _Path
        if isinstance(obj, _Path):
            return str(obj)
    except ImportError:
        pass
    # dict / list / tuple — recurse
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return _sanitize(sorted(obj, key=lambda x: str(x)))
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return repr(obj)
    # Dataclass with to_dict / asdict
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return _sanitize(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return _sanitize({k: v for k, v in vars(obj).items()})
        except Exception:
            pass
    return repr(obj)


# ---------------------------------------------------------------------------
# Flatten — lift commonly-used metadata.* keys to top-level CSV columns
# ---------------------------------------------------------------------------


# Mapping: metadata key → top-level column name. If a top-level field with
# this name already exists, the metadata value is merged under "_md__{name}"
# to avoid collision.
_MD_LIFT_KEYS: tuple[tuple[str, str], ...] = (
    ("knowledge_graph", "knowledge_graph"),
    ("cross_refs", "cross_refs"),
    ("coordinates", "coordinates"),
    ("paleodb", "paleodb"),
    ("stratigraphy", "stratigraphy"),
    ("chronostratigraphy", "chronostratigraphy"),
    ("chronostratigraphy_rank", "chronostratigraphy_rank"),
    ("latitude", "latitude"),
    ("longitude", "longitude"),
    ("paper_metadata", "paper_metadata"),
)


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    """Lift selected ``metadata.*`` keys to top-level CSV columns.

    - Primitive lifts (str/int/float/bool) become native columns.
    - Compound lifts (dict/list) are JSON-encoded as strings (so csv.DictWriter
      can write them without repr-truncation).
    - The original ``metadata`` dict is preserved untouched.
    """
    md = row.get("metadata")
    if not isinstance(md, dict) or not md:
        return row
    out = dict(row)
    for src, dst in _MD_LIFT_KEYS:
        if src not in md:
            continue
        val = md[src]
        if dst in out and out[dst] not in (None, "", []):
            # Don't clobber an existing top-level value
            out[f"_md__{dst}"] = _jsonify(val)
            continue
        if isinstance(val, (str, int, float, bool)) or val is None:
            out[dst] = val
        else:
            out[dst] = _jsonify(val)
    return out


def _jsonify(val: Any) -> str:
    """Sanitize then json.dumps a compound value into a compact string."""
    return json.dumps(_sanitize(val), ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Public export API
# ---------------------------------------------------------------------------


def export_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            sanitized = _sanitize(flatten_for_csv(row))
            f.write(json.dumps(sanitized, ensure_ascii=False) + "\n")


def export_json(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    sanitized = [_sanitize(flatten_for_csv(r)) for r in rows]
    path.write_text(json.dumps(sanitized, ensure_ascii=False, indent=2), encoding="utf-8")


def export_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flat = [flatten_for_csv(r) for r in rows]
    fieldnames = sorted({k for row in flat for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in flat:
            writer.writerow({k: _csv_cell(v) for k, v in row.items()})


def _csv_cell(v: Any) -> str:
    """Convert a Python value into a string cell that csv.DictWriter accepts."""
    if v is None:
        return ""
    if isinstance(v, (bool, int, float, str)):
        return v if isinstance(v, str) else str(v)
    return _jsonify(v)


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
