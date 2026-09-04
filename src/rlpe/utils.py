from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

# Phase F-3 NIT: precompile the slugify regex at module load so each
# call avoids the (small but cumulative) re.compile cost. Used in
# `stable_id` and across the export pipeline.
_SLUGIFY_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")

# Phase F-3 NIT: pre-import numpy at module load so ``_json_default``
# doesn't pay an import-lookup cost on every JSON-encoded row. ``numpy``
# is an optional dependency (most operators don't have it installed);
# when it's missing, ``_np`` is left as ``None`` and the relevant
# branch in ``_json_default`` is skipped.
try:
    import numpy as _np  # type: ignore[import-not-found]
except (ImportError, ModuleNotFoundError):  # pragma: no cover — optional dep
    _np = None

# Phase F-3 NIT: same lazy-import treatment for ``datetime`` so the
# ``_json_default`` hot path doesn't re-execute ``import datetime``
# on every non-datetime row.
import datetime as _dt  # noqa: E402  (always available with stdlib)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(text: str, fallback: str = "item") -> str:
    text = _SLUGIFY_NON_ALNUM.sub("_", text).strip("_")
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

    NOTE: this is SHA1, not SHA256. Phase 54 audit proposed a SHA1→SHA256
    switch (commit referencing "m17") but that change broke backward
    compat with the committed 9-paper gold dataset (data/gold/*.jsonl),
    which still uses the SHA1-derived paper_ids. Regenerating predictions
    against the current code would emit paper_ids that don't match any
    gold file, dropping aggregate F1 to 0. The SHA1 risk is acceptable
    here because:
      * paper_id is a content-dedup key, not a security primitive
      * the corpus is a fixed set of OA radiolarian PDFs; an attacker
        cannot realistically inject a SHA1-colliding radiolarian PDF
      * the change broke a working CI smoke check, so we revert to keep
        gold + predictions + eval internally consistent.
    """
    p = Path(path)
    try:
        if p.is_file():
            size = p.stat().st_size
            # size prefix prevents the (vanishingly rare) SHA1 collision
            # between two PDFs of the same content length, and makes
            # the id clearly content-derived.
            h = hashlib.sha1()
            h.update(f"{size}:".encode("ascii"))
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()[:16]
    except OSError:
        pass
    return hashlib.sha1(str(path).encode("utf-8", errors="ignore")).hexdigest()[:16]


def _json_default(obj: Any) -> Any:
    """Best-effort JSON encoder for objects ``json.dumps`` cannot serialise.

    Covers:
      - ``Path``    → ``str(path)``
      - dataclass / objects with ``to_dict`` → recurse on the dict
      - numpy scalars / arrays → Python equivalents (when numpy is installed)
      - ``set`` / ``frozenset`` / ``tuple`` → list
      - ``datetime`` / ``date`` → ISO 8601 string
      - ``bytes`` → utf-8 with replacement
      - everything else → ``str(obj)`` so the call never raises.

    Phase F-3 NIT: the previous version did
    ``import numpy as _np`` / ``import datetime as _dt`` inside this
    function on every call. Both modules are now imported at module
    top (``_np`` is optional, ``_dt`` is always available). The
    ``isinstance`` checks still skip gracefully when ``_np is None``.
    """
    # Path
    if isinstance(obj, Path):
        return str(obj)
    # numpy — guard specifically against AttributeError (object has
    # an attribute that raises during isinstance checks, e.g. a lazy
    # numpy wrapper that proxies attribute access). NOT bare
    # Exception: we explicitly want RecursionError and MemoryError
    # to propagate so the caller knows the encoding failed.
    if _np is not None:
        try:
            if isinstance(obj, _np.integer):
                return int(obj)
            if isinstance(obj, _np.floating):
                return float(obj)
            if isinstance(obj, _np.bool_):
                return bool(obj)
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
        except (AttributeError, ValueError):
            # ValueError from obj.tolist() on e.g. a structured numpy
            # array with non-trivial Python objects — not a deployment
            # issue, so surface it rather than silently produce wrong JSON.
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
    # datetime — guard against AttributeError from the isinstance check
    # (e.g. a ducktyped datetime). ValueError covers extreme years
    # (> 9999) that isoformat() rejects; OverflowError covers very
    # large timedeltas.
    try:
        if isinstance(obj, (_dt.datetime, _dt.date)):
            return obj.isoformat()
    except (AttributeError, ValueError, OverflowError):
        pass
    # bytes — decode with replacement. errors='replace' never raises
    # UnicodeDecodeError; TypeError covers a bytes subclass that
    # overrides decode with an incompatible signature.
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8", errors="replace")
        except TypeError:
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
    """Read a UTF-8 text file or return ``default`` on missing-path.

    Phase F-3 NIT: was catching only ``FileNotFoundError``. A directory
    passed in by mistake would raise ``IsADirectoryError`` (a subclass
    of ``OSError``), which propagated and crashed the caller with a
    unhelpful error. A ``PermissionError`` (read-protected file) was
    also propagated. Now we catch ``OSError`` and treat every flavour
    as "missing-or-unreadable" — the caller still gets the default
    back, but no longer crashes. Other exceptions (``UnicodeDecodeError``,
    ``ValueError``) are intentionally NOT caught; they're bugs, not
    deployment conditions.
    """
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return default


# ---------------------------------------------------------------------------
# Audit 2026-09-01 (Step 2): ``_safe_call`` — the unified error-handling
# helper that replaces 51 ``except Exception`` blocks scattered across
# ``pipeline.py`` with a single, categorised failure surface. Each call
# gets:
#   1. a structured warning written to ``run_output.warnings`` (label +
#      paper_id + message + timestamp);
#   2. a deduplicated logger.warning so the operator sees one breadcrumb
#      per (label, paper_id), not one per retry;
#   3. a return value (``default``) so the caller can keep moving.
# ---------------------------------------------------------------------------
_WARNINGS: list[dict[str, Any]] = []
_WARNINGS_LOCK = threading.Lock()


def _safe_call(
    label: str,
    fn: Callable[..., Any],
    *args: Any,
    paper_id: str | None = None,
    default: Any = None,
    reraise_on: tuple[type[BaseException], ...] = (),
    **kwargs: Any,
) -> Any:
    """Invoke ``fn(*args, **kwargs)`` and convert any failure into a
    structured warning + ``default`` return.

    Args:
        label: short tag for the failure surface — e.g. ``"apply_geo"``,
            ``"cross_figure_linker"``. Used as the dedupe key.
        fn: callable to invoke.
        *args / **kwargs: forwarded to ``fn``.
        paper_id: optional paper identifier; when set, the warning
            also includes ``paper_id`` so a triage grep on the run
            output can scope failures to a single paper.
        default: value to return on failure. ``None`` by default.
        reraise_on: tuple of exception TYPES that MUST propagate (do
            not catch). Use for ``KeyboardInterrupt`` / ``SystemExit`` /
            ``MemoryError`` to avoid swallowing real crashes. ``Async``
            cancellation exceptions are also re-raised.

    The accumulated warnings are exposed via :func:`drain_warnings` so
    the pipeline orchestrator can write them into ``run_output.json``.
    """
    try:
        return fn(*args, **kwargs)
    except reraise_on:
        raise
    except Exception as exc:  # noqa: BLE001 — see audit 2026-09-01
        msg = f"{type(exc).__name__}: {exc}"[:500]
        entry = {
            "label": label,
            "paper_id": paper_id,
            "message": msg,
            "timestamp": __import__("time").time(),
        }
        with _WARNINGS_LOCK:
            _WARNINGS.append(entry)
        try:
            import logging as _logging

            _logging.getLogger("rlpe.utils").warning(
                "_safe_call[%s paper=%s] swallowed: %s",
                label,
                paper_id or "?",
                msg,
                exc_info=True,
            )
        except Exception:
            pass
        return default


def drain_warnings() -> list[dict[str, Any]]:
    """Return and clear the accumulated ``_safe_call`` warnings.

    The pipeline orchestrator calls this once when assembling the
    final ``run_output.json`` so the user sees every de-duplicated
    warning from a single run in one place — instead of grepping
    log lines.
    """
    with _WARNINGS_LOCK:
        out = list(_WARNINGS)
        _WARNINGS.clear()
        return out
