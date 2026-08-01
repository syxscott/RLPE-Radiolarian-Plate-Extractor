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
import re as _re

# Compat shim: ``datetime.UTC`` was added in Python 3.11; on 3.10
# (and earlier) the canonical spelling is ``datetime.timezone.utc``.
# Using ``timezone.utc`` keeps this module importable on Python 3.10
# even though pyproject.toml formally requires >=3.11, so users who
# run the pipeline in an older conda env aren't blocked at import time.
UTC = timezone.utc  # noqa: UP017
from pathlib import Path
from typing import Any

from ..schema_models import SCHEMA_VERSION


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
    """Hash every input PDF and return ``{key: sha256}`` where ``key`` is a
    disambiguated identifier for the file.

    Phase 63 / audit 2026-08-01 D13: previously the key was just
    ``path.name`` (basename), so two PDFs with the same basename from
    different sub-directories could collide and the second would silently
    overwrite the first's hash. We now build a list of
    ``(candidate_key, hash)`` tuples — using ``parent/name`` when the
    basenames collide or when relative-to-root cannot be computed — and
    then deduplicate with a ``[N]`` suffix appended to the basename so
    every input is always represented in the returned dict.
    """
    digests: dict[str, str] = {}
    missing_or_unreadable: list[str] = []
    # First pass: produce (candidate_key, hash) pairs.
    pairs: list[tuple[str, str]] = []
    for p in pdf_paths:
        if p.exists() and p.is_file():
            try:
                digest = _sha256_file(p)
                status_msg = None
            except OSError:
                digest = "unreadable"
                status_msg = f"{p.name}: unreadable"
        else:
            digest = "missing"
            status_msg = f"{p.name}: missing"
        # Build a path-aware key so two PDFs with the same basename under
        # different parents are distinguishable. Fall back to parent/name
        # when relative_to has no usable root.
        name = p.name
        parent_name = p.parent.name if p.parent and p.parent.name else ""
        if parent_name:
            candidate = f"{parent_name}/{name}"
        else:
            candidate = name
        pairs.append((candidate, digest))
        if status_msg:
            missing_or_unreadable.append(status_msg)
    # Second pass: deduplicate by appending [N] suffix on collision.
    counts: dict[str, int] = {}
    for candidate, _digest in pairs:
        counts[candidate] = counts.get(candidate, 0) + 1
    seen: dict[str, int] = {}
    for candidate, digest in pairs:
        if counts[candidate] > 1:
            idx = seen.get(candidate, 0)
            seen[candidate] = idx + 1
            stem, dot, ext = candidate.rpartition(".")
            if dot:
                key = f"{stem}[{idx}].{ext}"
            else:
                key = f"{candidate}[{idx}]"
        else:
            key = candidate
        digests[key] = digest
    if missing_or_unreadable:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Some input PDFs could not be hashed: %s", "; ".join(missing_or_unreadable))
    return digests


def _host_string() -> str:
    try:
        node = socket.gethostname() or "unknown"
    except OSError:
        node = "unknown"
    return f"{platform.system().lower()}/{platform.release()}/{platform.machine()}/{node}"


def _config_snapshot(config: Any) -> dict[str, Any]:
    """Best-effort JSON-safe dump of a PipelineConfig-like object.

    Phase 63 Plan 6.9 (Bug 6.9): also strips known API-key fields and
    applies ``rlpe.llm_backends._redact_api_keys`` to every string
    value so a stray ``sk-...`` token embedded in a prompt, endpoint,
    or header can't leak into ``run_output.json`` /
    ``matches.jsonl``. The fallback walker already filters keys
    starting with ``_`` (which is why ``_MiniMax_external_handler``
    was never exposed), but the public ``MiniMax_api_key`` field
    *was* exposed — the fix removes it by name before applying the
    string-level redaction to every remaining value.
    """
    if config is None:
        return {}
    if isinstance(config, dict):
        snap = _json_safe(config)
    elif hasattr(config, "to_dict") and callable(config.to_dict):
        try:
            snap = config.to_dict()
        except Exception:
            snap = _fallback_walk(config)
        snap = _json_safe(snap)
    else:
        snap = _json_safe(_fallback_walk(config))
    return _redact_secrets(snap)


# Known field names that carry credentials. Removing them by name is
# safer than relying on regex alone because operators occasionally
# store keys under domain-specific names (``MiniMax_api_key``,
# ``anthropic_api_key``, ``openai_api_key``, ...) and the canonical
# ``MiniMax_api_key`` pattern needs an exact match. The regex layer
# below still catches stray tokens embedded in any other string.
_API_KEY_FIELD_NAMES = {
    "MiniMax_api_key",
    "minimax_api_key",
    "MiniMax_api_token",
    "minimax_api_token",
    "anthropic_api_key",
    "openai_api_key",
    "google_api_key",
    "minimax_api_key",
    "minimax_secret_key",
    "gemma_api_key",
    "huggingface_token",
    "hf_token",
    "wandb_api_key",
    "pbdb_api_key",
}

# Re-exported here for the recursive walker. We import the helper
# lazily to avoid an import cycle (rlpe.llm_backends imports a number
# of heavy ML client libraries).
_API_KEY_REGEXES: tuple[_re.Pattern[str], ...] = (
    _re.compile(r"(?<![A-Za-z0-9_])sk-(?=[A-Za-z0-9]{16})[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+){0,3}"),
    _re.compile(r"(?<![A-Za-z0-9_])sk-ant-api03-[A-Za-z0-9]{20,}"),
    _re.compile(r"(?<![A-Za-z0-9_])sk-ant-(?!api03-)[A-Za-z0-9]{16,}"),
    _re.compile(r"(?<![A-Za-z0-9_])sk-proj-[A-Za-z0-9]{16,}"),
    _re.compile(r"(?<![A-Za-z0-9_])sk-cp-[A-Za-z0-9]{16,}"),
)


def _redact_api_keys(text: str) -> str:
    """Replace any API-key-looking substrings with ``[REDACTED]``.

    Mirrors ``rlpe.llm_backends._redact_api_keys``; defined here so
    we don't need an import cycle through llm_backends (which pulls
    in heavy ML client libraries at import time).
    """
    if not text:
        return text
    for pat in _API_KEY_REGEXES:
        text = pat.sub("[REDACTED]", text)
    return text


def _redact_secrets(obj: Any) -> Any:
    """Walk ``obj`` recursively; remove known API-key fields and
    redact stray ``sk-...`` tokens in every string value."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if str(k) in _API_KEY_FIELD_NAMES:
                continue
            out[k] = _redact_secrets(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_redact_secrets(v) for v in obj]
    if isinstance(obj, str):
        return _redact_api_keys(obj)
    return obj


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
        # Phase 54 audit: M9 — the previous ``parents[2]`` walked up
        # to ``src/`` (provenance/stamp.py → provenance → rlpe → src),
        # not the repo root. ``_git_commit_and_dirty`` would then run
        # ``git rev-parse`` in ``src/`` (not a git repo) and always
        # return ``("unknown", False)``, silently breaking the
        # reproducibility story in every ``run_output.json``.
        # This is the SAME off-by-one family Phase 50 fixed in
        # ``src/rlpe/gui/constants.py:PROJECT_ROOT``; match the
        # depth here so the build_provenance call gets the actual
        # git working directory.
        repo_root = Path(__file__).resolve().parents[3]
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
