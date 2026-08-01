from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .utils import ensure_dir

# ---------------------------------------------------------------------------
# Atomic text-write helper — audit 2026-08-01 W1 / D6
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically (tmp file + fsync + os.replace).

    A crash or signal mid-write leaves the previous good ``path`` untouched
    and only a ``.tmp`` scratch file behind, which we clean up on error.
    This mirrors the pattern already used by ``exporters/xlsx.py`` (Phase 38).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


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
        except Exception as exc:
            # The to_dict method is the contract for "I know how to
            # serialise myself". A failure here usually means the
            # object's internal state is broken (a malformed metadata
            # field, a numpy scalar in a non-numpy-aware path, etc.).
            # Logging at debug means operators investigating a bad
            # export can see which object tripped, without spamming
            # warnings on the happy path.
            import logging

            logging.getLogger(__name__).debug(
                "export._sanitize: to_dict() failed for %s: %s; falling back to vars()",
                type(obj).__name__,
                exc,
            )
    if hasattr(obj, "__dict__"):
        try:
            return _sanitize({k: v for k, v in vars(obj).items()})
        except Exception as exc:
            import logging

            logging.getLogger(__name__).debug(
                "export._sanitize: vars() failed for %s: %s; falling back to repr()",
                type(obj).__name__,
                exc,
            )
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
    lines = [json.dumps(_sanitize(flatten_for_csv(row)), ensure_ascii=False) for row in rows]
    _atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def export_json(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    sanitized = [_sanitize(flatten_for_csv(r)) for r in rows]
    _atomic_write_text(
        path,
        json.dumps(sanitized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def export_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        # Phase 63 Plan 6.10 (Bug 6.10): still emit a 3-byte UTF-8 BOM
        # so the empty file isn't misinterpreted as ANSI when the
        # operator opens it later in Excel. Route through the atomic
        # helper so the BOM-only file is also written via tmp + os.replace.
        _atomic_write_text(path, "﻿", encoding="utf-8")
        return
    flat = [flatten_for_csv(r) for r in rows]
    fieldnames = sorted({k for row in flat for k in row.keys()})
    # Build the CSV in memory so we can hand the whole string to the
    # atomic helper. utf-8-sig adds a single BOM at file start.
    import io

    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in flat:
        writer.writerow({k: _csv_cell(v) for k, v in row.items()})
    _atomic_write_text(path, buf.getvalue(), encoding="utf-8-sig")


def _csv_cell(v: Any) -> str:
    """Convert a Python value into a string cell that csv.DictWriter accepts."""
    if v is None:
        return ""
    if isinstance(v, (bool, int, float, str)):
        return v if isinstance(v, str) else str(v)
    return _jsonify(v)


def copy_assets(
    rows: list[dict[str, Any]], dst_dir: Path, key: str = "panel_path"
) -> list[dict[str, Any]]:
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
