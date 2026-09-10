"""Persisted LLM API settings shared by the CLI, Web server and GUI.

Storage design (F17): a single JSON file at ``~/.rlpe/llm_api.json``
holds the user's *current* Anthropic-compatible provider configuration
— base_url + api_key + model. Multi-provider presets are deliberately
out of scope; this file always stores the last configuration the user
saved from either the Web settings tab or the desktop GUI.

Resolution priority used everywhere (see
``llm_backends.resolve_llm_api_key`` / ``build_anthropic_compat_backend``):

    per-run explicit option  >  this file  >  environment variables

Security: the file is written with mode 0600 via an atomic
tempfile+``os.replace`` so a concurrent reader never sees a truncated
payload. Corrupt or missing files are treated as "nothing saved"
(load never raises); the API key is only ever echoed back in masked
preview form by the API layer.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SETTINGS_DIRNAME = ".rlpe"
SETTINGS_FILENAME = "llm_api.json"


@dataclass(slots=True)
class LLMApiSettings:
    """The single persisted provider configuration."""

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    updated_at: str | None = None

    @property
    def is_empty(self) -> bool:
        return not (self.base_url or self.api_key or self.model)


def settings_path(home: Path | None = None) -> Path:
    """Absolute path of the settings file (overridable for tests)."""
    base = home if home is not None else Path.home()
    return base / SETTINGS_DIRNAME / SETTINGS_FILENAME


def load_llm_settings(path: Path | None = None) -> LLMApiSettings:
    """Load the persisted settings; missing/corrupt file → defaults."""
    p = path or settings_path()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return LLMApiSettings()
    if not isinstance(raw, dict):
        return LLMApiSettings()

    def _field(key: str) -> str:
        value = raw.get(key)
        return value.strip() if isinstance(value, str) else ""

    return LLMApiSettings(
        base_url=_field("base_url"),
        api_key=_field("api_key"),
        model=_field("model"),
        updated_at=_field("updated_at") or None,
    )


def save_llm_settings(
    settings: LLMApiSettings,
    path: Path | None = None,
) -> Path:
    """Persist *settings* atomically with mode 0600; returns the path.

    ``updated_at`` is stamped here so callers never need to set it.
    """
    p = path or settings_path()
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = {
        "base_url": settings.base_url.strip(),
        "api_key": settings.api_key.strip(),
        "model": settings.model.strip(),
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    fd, tmp_name = tempfile.mkstemp(dir=str(p.parent), prefix=".llm_api_", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        tmp.chmod(0o600)
        os.replace(tmp, p)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    try:
        p.chmod(0o600)
    except OSError:
        pass  # e.g. exotic filesystems; tempfile already set 0600
    return p


def clear_llm_settings(path: Path | None = None) -> None:
    """Forget the persisted configuration (used by "clear key" flows)."""
    p = path or settings_path()
    try:
        p.unlink()
    except OSError:
        pass
