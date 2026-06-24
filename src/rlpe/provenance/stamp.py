"""Provenance stamping: collect metadata that proves how an output was produced.

Every pipeline run writes a ``provenance`` block at the top of its output
JSONL/JSON. This block lets a reviewer reconstruct exactly which code,
configuration, and input bytes produced the result, and on which host.

The block contains:
    - ``pipeline_version``: semver of the rlpe package
    - ``git_commit``: short SHA of the running commit (or "unknown")
    - ``git_dirty``: True if there were uncommitted changes
    - ``config_snapshot``: the resolved PipelineConfig as a JSON-safe dict
    - ``input_sha256``: {filename: hex digest} for every input PDF
    - ``timestamp_utc``: ISO 8601, UTC
    - ``host``: platform/release/machine triple

The function is pure-ish: it inspects the filesystem and ``git`` binary
but does not modify state. Calling it twice in the same second produces
the same timestamp.
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

# Compat shim: ``datetime.UTC`` was added in Python 3.11; on 3.10
# (and earlier) the canonical spelling is ``datetime.timezone.utc``.
# Using ``timezone.utc`` keeps this module importable on Python 3.10
# even though pyproject.toml formally requires >=3.11, so users who
# run the pipeline in an older conda env aren't blocked at import time.
UTC = timezone.utc  # noqa: UP017
from pathlib import Path
from typing import Any


def _resolve_pipeline_version() -> str:
    """Read the rlpe package version from installed metadata, falling
    back to the project's pyproject.toml in source-checkout runs.

    The previous code hard-coded ``"1.1.0"`` here AND in the API
    layer; bumping the version in pyproject.toml left the provenance
    stamp lying about which release produced an output. Reading
    ``importlib.metadata`` keeps the two in lock-step.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("rlpe")
        except PackageNotFoundError:
            pass
    except Exception:
        pass
    # source-checkout fallback
    try:
        try:
            import tomllib  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[import-not-found,no-redef]
        # provenance/stamp.py -> provenance/ -> rlpe/ -> src/ -> repo root
        pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
        if pyproject.exists():
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
            v = (data.get("project") or {}).get("version")
            if v:
                return str(v)
    except Exception:
        pass
    return "unknown"


PIPELINE_VERSION = _resolve_pipeline_version()
SCHEMA_VERSION = "1.0.0"


@dataclass(slots=True)
class Provenance:
    pipeline_version: str
    schema_version: str
    git_commit: str
    git_dirty: bool
    config_snapshot: dict[str, Any]
    input_sha256: dict[str, str]
    timestamp_utc: str
    host: str
    python_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_run_git(*args: str, cwd: Path) -> str:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return ""


def _git_commit_and_dirty(repo_root: Path) -> tuple[str, bool]:
    commit = _safe_run_git("rev-parse", "--short", "HEAD", cwd=repo_root)
    if not commit:
        # Not a git repo or git not installed; return a stable sentinel
        return "unknown", False
    # "git status --porcelain" returns empty stdout when the tree is clean
    porcelain = _safe_run_git("status", "--porcelain", cwd=repo_root)
    return commit, bool(porcelain)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _input_sha256(pdf_paths: list[Path]) -> dict[str, str]:
    digests: dict[str, str] = {}
    for p in pdf_paths:
        if p.exists() and p.is_file():
            try:
                digests[p.name] = _sha256_file(p)
            except OSError:
                digests[p.name] = "unreadable"
        else:
            digests[p.name] = "missing"
    return digests


def _host_string() -> str:
    try:
        node = socket.gethostname() or "unknown"
    except OSError:
        node = "unknown"
    return f"{platform.system().lower()}/{platform.release()}/{platform.machine()}/{node}"


def _config_snapshot(config: Any) -> dict[str, Any]:
    """Best-effort JSON-safe dump of a PipelineConfig-like object."""
    if config is None:
        return {}
    if isinstance(config, dict):
        return _json_safe(config)
    if hasattr(config, "to_dict") and callable(config.to_dict):
        try:
            snap = config.to_dict()
        except Exception:
            snap = _fallback_walk(config)
    else:
        snap = _fallback_walk(config)
    return _json_safe(snap)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _fallback_walk(config: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in dir(config):
        if k.startswith("_"):
            continue
        try:
            v = getattr(config, k)
        except AttributeError:
            continue
        if callable(v):
            continue
        out[k] = v
    return out


def build_provenance(
    config: Any = None,
    pdf_paths: list[Path] | None = None,
    repo_root: Path | None = None,
) -> Provenance:
    """Construct a Provenance object for the current run.

    Args:
        config: PipelineConfig (or any object exposing to_dict). Used to
            snapshot the resolved configuration.
        pdf_paths: Input PDFs to checksum. Order is preserved.
        repo_root: Git working directory. Defaults to the rlpe package root.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    commit, dirty = _git_commit_and_dirty(repo_root)
    return Provenance(
        pipeline_version=PIPELINE_VERSION,
        schema_version=SCHEMA_VERSION,
        git_commit=commit,
        git_dirty=dirty,
        config_snapshot=_config_snapshot(config),
        input_sha256=_input_sha256(pdf_paths or []),
        timestamp_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        host=_host_string(),
        python_version=platform.python_version(),
    )


def write_provenance_sidecar(
    provenance: Provenance,
    output_path: Path,
) -> Path:
    """Write the provenance block as a standalone JSON file.

    Sidecar location: ``<output>.provenance.json``. Reviewers use this
    to verify reproducibility without parsing the full output.
    """
    sidecar = output_path.with_suffix(output_path.suffix + ".provenance.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(provenance.to_dict(), f, indent=2, sort_keys=True, ensure_ascii=False)
    return sidecar
