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
            # Phase 54 audit m17 — switch from SHA1 to SHA256. SHA1 is
            # broken for adversarial collisions (SHAttered, 2017); for
            # paper-dedup it's harmless in practice (an attacker can't
            # craft a colliding radiolarian PDF), but SHA256 costs
            # ~negligible extra and removes a misleading "vanishingly
            # rare" comment. Also widen the truncation prefix from 12
            # to 16 hex chars (64 bits) so an accidental collision on
            # two distinct PDFs is even more unlikely.
            h = hashlib.sha256()
            # size prefix prevents the (vanishingly rare) SHA256 collision
            # between two PDFs of the same content length, and makes
            # the id clearly content-derived.
            h.update(f"{size}:".encode("ascii"))
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()[:16]
    except OSError:
        pass
    # Phase 54 audit m17 — match the SHA256 change above.
    return hashlib.sha256(str(path).encode("utf-8", errors="ignore")).hexdigest()[:16]


def _json_default(obj: Any) -> Any:
    """Best-effort JSON encoder for objects ``json.dumps`` cannot serialise.

    Covers:
      - ``Path``    → ``str(path)``
      - dataclass / objects with ``to_dict`` → recurse on the dict
      - numpy scalars / arrays → Python equivalents
      - ``set`` / ``frozenset`` / ``tuple`` → list
      - ``datetime`` / ``date`` → ISO 8601 string
      - ``bytes`` → utf-8 with replacement
      - everything else → ``str(obj)`` so the call never raises.
    """
    # Path
    if isinstance(obj, Path):
        return str(obj)
    # numpy — guard specifically against import failure (no numpy installed)
    # and AttributeError (object has an attribute that raises during isinstance
    # checks, e.g. a lazy numpy wrapper that proxies attribute access).
    # NOT bare Exception: we explicitly want RecursionError and
    # MemoryError to propagate so the caller knows the encoding failed.
    try:
        import numpy as _np  # type: ignore[import-not-found]

        if isinstance(obj, _np.integer):
            return int(obj)
        if isinstance(obj, _np.floating):
            return float(obj)
        if isinstance(obj, _np.bool_):
            return bool(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except (ImportError, ModuleNotFoundError, AttributeError):
        pass
    # to_dict / dataclass — guard against to_dict raising (e.g. a
    # malformed dataclass with a broken to_dict method). AttributeError
    # covers the case where the object has a to_dict attribute that
    # isn't callable. NOT bare Exception: a real bug in to_dict should
    # surface, not silently produce wrong JSON.
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return obj.to_dict()
        except (AttributeError, TypeError):
            pass
    # collections
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=lambda x: str(x))
    if isinstance(obj, tuple):
        return list(obj)
    # datetime — guard against datetime module import failure (already
    # imported above but guard against it not being available) and
    # AttributeError from the isinstance check.
    try:
        import datetime as _dt

        if isinstance(obj, (_dt.datetime, _dt.date)):
            return obj.isoformat()
    except (ImportError, ModuleNotFoundError, AttributeError):
        pass
    # bytes — decode with replacement, falling back to repr only if
    # the decode itself fails (e.g. a bytes subclass that raises on decode).
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8", errors="replace")
        except (UnicodeDecodeError, TypeError):
            return repr(obj)
    # last resort
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def write_json(path: Path, data: Any) -> None:
    """Atomically write JSON to ``path``.

    Writes to a sibling temp file first, then ``os.replace`` swaps it
    into place. Audit H6: a naive ``path.write_text`` is not atomic
    — a concurrent reader (or another writer) can see a partial file.
    On Linux/macOS ``os.replace`` is atomic at the inode level.
    """
    import os as _os
    import tempfile as _tempfile

    ensure_dir(path.parent)
    fd, tmp_name = _tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with _os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default))
            f.flush()
            # fsync so the file content is durable before the rename.
            try:
                _os.fsync(f.fileno())
            except OSError:
                # Some filesystems (e.g. some FUSE mounts) don't support
                # fsync. Fall back to best-effort: the rename still
                # ensures the reader never sees a torn write.
                pass
        _os.replace(tmp_name, path)
    except Exception:
        # Clean up the temp file on any failure so we don't litter.
        try:
            _os.unlink(tmp_name)
        except OSError:
            pass
        raise


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    # Same atomic-rename pattern as ``write_json`` so a reader never
    # sees a half-written matches.jsonl during a concurrent scan.
    import os as _os
    import tempfile as _tempfile

    ensure_dir(path.parent)
    fd, tmp_name = _tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with _os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
            try:
                _os.fsync(f.fileno())
            except OSError:
                pass
        _os.replace(tmp_name, path)
    except Exception:
        try:
            _os.unlink(tmp_name)
        except OSError:
            pass
        raise


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default
