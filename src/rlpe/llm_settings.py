"""Persisted LLM API settings shared by the CLI, Web server and GUI.

Storage design (F17, extended to multi-provider presets in F18): a
single JSON file at ``~/.rlpe/llm_api.json`` holds N named provider
presets plus a pointer to the active one — the cc-switch model::

    {"version": 2, "current": "<provider-id>",
     "providers": [{"id": "...", "name": "MiniMax", "base_url": "...",
                    "api_key": "...", "model": "...", "updated_at": "..."}]}

Switching providers = flipping ``current``; nothing else needs to
change because ``load_llm_settings()`` keeps returning the flat
``{base_url, api_key, model}`` of the ACTIVE provider, which is all
the resolution chain in ``llm_backends`` consumes.

A pre-F18 flat file (``{"base_url", "api_key", "model", ...}``) is
migrated transparently on load into a single preset named "默认"
which becomes current; the v2 layout is written back on the next save.

Resolution priority used everywhere (see
``llm_backends.resolve_llm_api_key`` / ``build_anthropic_compat_backend``):

    per-run explicit option  >  this file (active preset)  >  environment

Security: the file is written with mode 0600 via an atomic
tempfile+``os.replace`` so a concurrent reader never sees a truncated
payload. Corrupt or missing files are treated as "nothing saved"
(load never raises); the API key is only ever echoed back in masked
preview form by the API layer.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SETTINGS_DIRNAME = ".rlpe"
SETTINGS_FILENAME = "llm_api.json"
FILE_FORMAT_VERSION = 2
DEFAULT_PROFILE_NAME = "默认"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(slots=True)
class ProviderConfig:
    """One named API preset (address + key + model)."""

    name: str
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    id: str = ""  # generated on save when empty
    updated_at: str | None = None

    @property
    def is_empty(self) -> bool:
        return not (self.base_url or self.api_key or self.model)


@dataclass(slots=True)
class LLMApiSettings:
    """The flat view of the ACTIVE preset (backward-compatible shape).

    ``load_llm_settings()`` returns this so the F17 resolution chain
    keeps working unchanged across the v2 storage upgrade.
    """

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    updated_at: str | None = None
    profile_id: str | None = None
    profile_name: str | None = None

    @property
    def is_empty(self) -> bool:
        return not (self.base_url or self.api_key or self.model)


def settings_path(home: Path | None = None) -> Path:
    """Absolute path of the settings file (overridable for tests)."""
    base = home if home is not None else Path.home()
    return base / SETTINGS_DIRNAME / SETTINGS_FILENAME


# ---------------------------------------------------------------------------
# v2 multi-provider core
# ---------------------------------------------------------------------------


def _read_raw(path: Path) -> dict:
    """Read the raw file as a dict; missing/corrupt → {}."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _migrate_v1(raw: dict) -> dict:
    """Convert a pre-F18 flat payload into the v2 layout.

    Flat shape: ``{"base_url", "api_key", "model", "updated_at"}``.
    Anything that already looks like v2 (has "providers") is returned
    unchanged.
    """
    if "providers" in raw:
        return raw
    if not any(k in raw for k in ("base_url", "api_key", "model")):
        return {"version": FILE_FORMAT_VERSION, "current": None, "providers": []}
    # Stable content-derived id: migration happens on every read until a
    # save persists the v2 layout, so repeated in-memory migrations must
    # produce the SAME id or the current pointer would drift.
    digest = hashlib.sha1(
        "\x00".join(str(raw.get(k) or "") for k in ("base_url", "api_key", "model")).encode("utf-8")
    ).hexdigest()[:12]
    provider = {
        "id": digest,
        "name": DEFAULT_PROFILE_NAME,
        "base_url": str(raw.get("base_url") or "").strip(),
        "api_key": str(raw.get("api_key") or "").strip(),
        "model": str(raw.get("model") or "").strip(),
        "updated_at": raw.get("updated_at"),
    }
    return {
        "version": FILE_FORMAT_VERSION,
        "current": provider["id"],
        "providers": [provider],
    }


def _write_payload(payload: dict, path: Path) -> Path:
    """Atomically persist a v2 payload with mode 0600."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".llm_api_", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        tmp.chmod(0o600)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    try:
        path.chmod(0o600)
    except OSError:
        pass  # e.g. exotic filesystems; tempfile already set 0600
    return path


def _load_v2(path: Path) -> tuple[list[dict], str | None]:
    """Return (providers, current_id) after transparent v1 migration."""
    raw = _migrate_v1(_read_raw(path))
    providers = [p for p in raw.get("providers", []) if isinstance(p, dict)]
    current = raw.get("current")
    if current is not None and not any(p.get("id") == current for p in providers):
        current = None
    if current is None and providers:
        current = providers[0].get("id")
    return providers, current


def _normalize_provider(raw: dict) -> ProviderConfig:
    def _s(key: str) -> str:
        v = raw.get(key)
        return v.strip() if isinstance(v, str) else ""

    return ProviderConfig(
        name=_s("name") or DEFAULT_PROFILE_NAME,
        base_url=_s("base_url"),
        api_key=_s("api_key"),
        model=_s("model"),
        id=_s("id"),
        updated_at=raw.get("updated_at") if isinstance(raw.get("updated_at"), str) else None,
    )


def _provider_to_dict(p: ProviderConfig) -> dict:
    return {
        "id": p.id,
        "name": p.name,
        "base_url": p.base_url,
        "api_key": p.api_key,
        "model": p.model,
        "updated_at": p.updated_at,
    }


def _persist_providers(
    providers: list[ProviderConfig],
    current: str | None,
    path: Path,
) -> Path:
    payload = {
        "version": FILE_FORMAT_VERSION,
        "current": current,
        "providers": [_provider_to_dict(p) for p in providers],
    }
    return _write_payload(payload, path)


def list_providers(path: Path | None = None) -> list[ProviderConfig]:
    """All saved presets, in file order."""
    p = path or settings_path()
    providers, _ = _load_v2(p)
    return [_normalize_provider(raw) for raw in providers]


def current_provider_id(path: Path | None = None) -> str | None:
    """The active preset's id (falls back to the first preset)."""
    p = path or settings_path()
    _, current = _load_v2(p)
    return current


def upsert_provider(
    provider: ProviderConfig,
    path: Path | None = None,
    *,
    activate: bool | None = None,
) -> ProviderConfig:
    """Create or update a preset and persist the file.

    A provider with a matching ``id`` is updated in place; otherwise a
    new preset is appended (``id`` assigned if empty). ``activate=True``
    makes it the current preset; ``None`` (default) auto-activates a
    newly created preset when no current exists yet. Returns the stored
    provider (with id and updated_at filled in).
    """
    p = path or settings_path()
    providers_raw, current = _load_v2(p)
    providers = [_normalize_provider(raw) for raw in providers_raw]
    stored = ProviderConfig(
        name=provider.name.strip() or DEFAULT_PROFILE_NAME,
        base_url=provider.base_url.strip(),
        api_key=provider.api_key.strip(),
        model=provider.model.strip(),
        id=provider.id.strip(),
        updated_at=_now_iso(),
    )
    if not stored.id:
        stored.id = uuid.uuid4().hex[:12]
    replaced = False
    for i, existing in enumerate(providers):
        if existing.id == stored.id:
            providers[i] = stored
            replaced = True
            break
    if not replaced:
        providers.append(stored)
    if activate is True or (activate is None and current is None):
        current = stored.id
    _persist_providers(providers, current, p)
    return stored


def delete_provider(provider_id: str, path: Path | None = None) -> bool:
    """Remove a preset by id; returns True when something was deleted.

    Deleting the current preset promotes the first remaining one (or
    leaves the file with no current when empty).
    """
    p = path or settings_path()
    providers, current = _load_v2(p)
    remaining = [pr for pr in providers if pr.get("id") != provider_id]
    if len(remaining) == len(providers):
        return False
    if current == provider_id:
        current = remaining[0].get("id") if remaining else None
    _persist_providers([_normalize_provider(raw) for raw in remaining], current, p)
    return True


def set_current_provider(provider_id: str, path: Path | None = None) -> bool:
    """Activate a preset by id; returns False when the id is unknown."""
    p = path or settings_path()
    providers, _ = _load_v2(p)
    if not any(raw.get("id") == provider_id for raw in providers):
        return False
    _persist_providers([_normalize_provider(raw) for raw in providers], provider_id, p)
    return True


def get_provider(provider_id: str, path: Path | None = None) -> ProviderConfig | None:
    """Fetch one preset by id (including its api_key)."""
    p = path or settings_path()
    for pr in list_providers(p):
        if pr.id == provider_id:
            return pr
    return None


# ---------------------------------------------------------------------------
# Flat active-preset view (backward-compatible F17 surface)
# ---------------------------------------------------------------------------


def load_llm_settings(path: Path | None = None) -> LLMApiSettings:
    """Load the ACTIVE preset as a flat view; missing/corrupt → defaults."""
    p = path or settings_path()
    providers, current = _load_v2(p)
    if not providers:
        return LLMApiSettings()
    raw = next((r for r in providers if r.get("id") == current), providers[0])
    pr = _normalize_provider(raw)
    return LLMApiSettings(
        base_url=pr.base_url,
        api_key=pr.api_key,
        model=pr.model,
        updated_at=pr.updated_at,
        profile_id=pr.id,
        profile_name=pr.name,
    )


def save_llm_settings(settings: LLMApiSettings, path: Path | None = None) -> Path:
    """Persist *settings* as the CURRENT preset and return the path.

    F17 compat surface: when the file already has presets, the active
    one is updated in place; when empty (or only a legacy flat file
    existed), this creates/updates the single "默认" preset. The
    Web/GUI multi-preset editors use :func:`upsert_provider` instead.
    """
    p = path or settings_path()
    providers, current = _load_v2(p)
    target_id = current or (providers[0].get("id") if providers else uuid.uuid4().hex[:12])
    # Updating an existing preset in place must keep its user-chosen
    # name (e.g. "DeepSeek" edited via the compat llm-config endpoint
    # should not silently become "默认").
    existing_name = next(
        (raw.get("name") for raw in providers if raw.get("id") == target_id),
        None,
    )
    upsert_provider(
        ProviderConfig(
            name=str(existing_name) if existing_name else DEFAULT_PROFILE_NAME,
            base_url=settings.base_url,
            api_key=settings.api_key,
            model=settings.model,
            id=target_id,
        ),
        path=p,
    )
    return p


def clear_llm_settings(path: Path | None = None) -> None:
    """Forget the persisted configuration (used by "clear key" flows)."""
    p = path or settings_path()
    try:
        p.unlink()
    except OSError:
        pass
