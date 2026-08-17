"""Shared .env loading with project-key override semantics.

Both ``run_web_server.py`` and ``src/rlpe/cli.py`` used to carry
their own copies of this logic — and the key sets drifted (the CLI
copy missed ``MINIMAX_API_KEY``). Centralising here also makes the
behaviour unit-testable: the old test re-implemented the loader
inline and asserted on its own copy (tautology).

Override rule (audit 2026-07-31):
  1. Keys that are NOT set in the OS environment are always loaded.
  2. The project's reserved MiniMax keys (.env wins over an OS env
     var) — tools like Claude Code set ``ANTHROPIC_BASE_URL``
     globally; without the override RLPE would silently connect to
     the wrong endpoint.
  3. ``RLPE_FORCE_ENV_OVERRIDE=1`` forces .env to win for ALL keys
     (escape hatch).
"""

from __future__ import annotations

import os
from pathlib import Path

_RLPE_PROJECT_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "MiniMax_API_KEY",
        "MiniMax_MODEL",
        "MiniMax_BASE_URL",
        "MINIMAX_API_KEY",
    }
)


def load_env_file(
    env_path: str | Path,
    *,
    force_override: bool | None = None,
    override_keys: set[str] | frozenset[str] | None = None,
) -> int:
    """Load ``env_path`` into ``os.environ`` with the override rules.

    Returns the number of keys actually set. ``force_override``
    defaults to the ``RLPE_FORCE_ENV_OVERRIDE`` env var; when None is
    passed explicitly the caller controls it (tests).
    """
    path = Path(env_path)
    if not path.exists():
        return 0
    if force_override is None:
        force_override = os.environ.get("RLPE_FORCE_ENV_OVERRIDE") == "1"
    project_keys = override_keys if override_keys is not None else _RLPE_PROJECT_OVERRIDE_KEYS
    set_count = 0
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if not key:
                    continue
                should_override = force_override or key in project_keys or key not in os.environ
                if should_override:
                    os.environ[key] = value
                    set_count += 1
    except OSError:
        return 0
    return set_count
