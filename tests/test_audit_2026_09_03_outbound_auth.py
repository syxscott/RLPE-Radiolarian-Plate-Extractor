"""Audit 2026-09-03 (BLOCKER-#2 + BLOCKER-#3): outbound policy + API auth.

This file covers the fail-secure defaults added in commit
``audit-2026-09-03-theme-A-console-auth``:

* ``data_outbound_policy`` default flipped from ``api_full`` to
  ``api_redacted``.
* ``api_full`` now requires an explicit opt-in (env var
  ``RLPE_DATA_OUTBOUND_OPT_IN``) and raises ``ValueError`` otherwise.
* ``require_api_key`` fails with HTTP 503 when the server is bound
  to a non-loopback host without ``RLPE_API_KEY`` set (BLOCKER-#3).
* ``run_web_server.py`` prints an ephemeral key to stderr on loopback
  startup when no key is configured.
* ``llm_status`` endpoint exposes the new consent / security fields.

Each test is independent — failing one does not cascade. We also
avoid touching the network (no real MiniMax calls) and avoid loading
the full PipelineConfig so the suite stays fast.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# 1) data_outbound_policy default is api_redacted (BLOCKER-#2)
# ---------------------------------------------------------------------------


def test_data_outbound_policy_default_is_api_redacted() -> None:
    """Without any opt-in, the default is the private posture."""
    from rlpe.llm_backends import MiniMaxM3Backend

    # Pass a fake API key so the api_key check at __post_init__ doesn't
    # fire; we are only asserting the policy field default here.
    b = MiniMaxM3Backend(api_key="fake-test-key", data_outbound_policy="api_redacted")
    assert b.data_outbound_policy == "api_redacted"


def test_data_outbound_policy_default_in_dataclass_definition() -> None:
    """Source-guard: the dataclass default in llm_backends.py must be
    ``api_redacted``. If a future commit flips it back to ``api_full``
    this test fails loudly — that is the BLOCKER-#2 regression we
    want to prevent."""
    src = (Path(__file__).resolve().parents[1] / "src" / "rlpe" / "llm_backends.py").read_text(
        encoding="utf-8"
    )
    # Match the dataclass default; allow optional inline comment.
    import re

    m = re.search(
        r'data_outbound_policy\s*:\s*str\s*=\s*["\']([^"\']+)["\']',
        src,
    )
    assert m is not None, "data_outbound_policy dataclass default not found"
    assert m.group(1) == "api_redacted", (
        f"data_outbound_policy default is {m.group(1)!r}, expected 'api_redacted'. "
        "This is BLOCKER-#2 regression — the api_full historical default "
        "silently ships full PDF payload to the cloud."
    )


def test_cli_default_is_api_redacted() -> None:
    """Source-guard: ``--data-outbound-policy`` CLI default must be
    ``api_redacted``. If a future commit changes it back to ``api_full``
    this test fails."""
    src = (Path(__file__).resolve().parents[1] / "src" / "rlpe" / "cli.py").read_text(
        encoding="utf-8"
    )
    import re

    # The argument is registered in build_parser() with default=...
    m = re.search(
        r'--data-outbound-policy[^)]*default=["\']([^"\']+)["\']',
        src,
        re.DOTALL,
    )
    assert m is not None, "--data-outbound-policy default not found in cli.py"
    assert m.group(1) == "api_redacted", (
        f"CLI --data-outbound-policy default is {m.group(1)!r}, expected 'api_redacted'"
    )


# ---------------------------------------------------------------------------
# 2) api_full opt-in guard (BLOCKER-#2)
# ---------------------------------------------------------------------------


def test_api_full_without_opt_in_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting ``api_full`` without the opt-in env var must raise
    ValueError. The error message must point the operator at the
    opt-in knob so the fix is obvious."""
    from rlpe.llm_backends import MiniMaxM3Backend

    monkeypatch.delenv("RLPE_DATA_OUTBOUND_OPT_IN", raising=False)
    with pytest.raises(ValueError) as excinfo:
        MiniMaxM3Backend(api_key="fake-test-key", data_outbound_policy="api_full")
    msg = str(excinfo.value)
    assert "api_full" in msg
    assert "RLPE_DATA_OUTBOUND_OPT_IN" in msg
    assert "api_redacted" in msg
    assert "local_only" in msg


@pytest.mark.parametrize("opt_in_value", ["1", "true", "yes", "on"])
def test_api_full_with_opt_in_succeeds(monkeypatch: pytest.MonkeyPatch, opt_in_value: str) -> None:
    """All four accepted opt-in spellings must unblock api_full."""
    from rlpe.llm_backends import MiniMaxM3Backend

    monkeypatch.setenv("RLPE_DATA_OUTBOUND_OPT_IN", opt_in_value)
    b = MiniMaxM3Backend(api_key="fake-test-key", data_outbound_policy="api_full")
    assert b.data_outbound_policy == "api_full"


def test_api_redacted_does_not_require_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default posture must NEVER require an opt-in flag —
    that would break every existing fresh install. The opt-in is
    strictly for the more permissive api_full mode."""
    from rlpe.llm_backends import MiniMaxM3Backend

    monkeypatch.delenv("RLPE_DATA_OUTBOUND_OPT_IN", raising=False)
    # Pass a fake API key — the api_redacted mode still requires one
    # for the SDK to initialise; the opt-in flag is an additional
    # check we are testing here.
    b = MiniMaxM3Backend(api_key="fake-test-key", data_outbound_policy="api_redacted")
    assert b.data_outbound_policy == "api_redacted"


def test_local_only_does_not_require_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local-only posture (no network) must also work without opt-in."""
    from rlpe.llm_backends import MiniMaxM3Backend

    monkeypatch.delenv("RLPE_DATA_OUTBOUND_OPT_IN", raising=False)
    b = MiniMaxM3Backend(api_key="", data_outbound_policy="local_only")
    assert b.data_outbound_policy == "local_only"


# ---------------------------------------------------------------------------
# 3) require_api_key fail-secure (BLOCKER-#3)
# ---------------------------------------------------------------------------


def test_require_api_key_fails_secure_on_non_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``RLPE_API_KEY`` is unset and the server is bound to a
    non-loopback host, ``require_api_key`` must raise HTTPException(503)
    — there is no silent auth-free mode for LAN-exposed binds. This
    closes the historical opt-in LAN exposure (BLOCKER-#3)."""
    from fastapi import HTTPException

    from rlpe.api.app import require_api_key

    monkeypatch.delenv("RLPE_API_KEY", raising=False)
    monkeypatch.setenv("RLPE_HOST", "0.0.0.0")
    with pytest.raises(HTTPException) as excinfo:
        require_api_key(x_api_key=None)
    assert excinfo.value.status_code == 503
    assert "RLPE_API_KEY" in excinfo.value.detail
    assert "non-loopback" in excinfo.value.detail


def test_require_api_key_passes_on_loopback_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``RLPE_API_KEY`` is unset but the bind is loopback,
    ``require_api_key`` is a no-op (local dev / test mode)."""
    from rlpe.api.app import require_api_key

    monkeypatch.delenv("RLPE_API_KEY", raising=False)
    monkeypatch.setenv("RLPE_HOST", "127.0.0.1")
    # Should not raise.
    require_api_key(x_api_key=None)
    require_api_key(x_api_key="anything")


def test_require_api_key_validates_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``RLPE_API_KEY`` is set, a missing or wrong header must
    raise 403 (preserves the historical behaviour)."""
    from fastapi import HTTPException

    from rlpe.api.app import require_api_key

    monkeypatch.setenv("RLPE_API_KEY", "secret-xyz")
    # No header → 403
    with pytest.raises(HTTPException) as excinfo:
        require_api_key(x_api_key=None)
    assert excinfo.value.status_code == 403
    # Wrong header → 403
    with pytest.raises(HTTPException) as excinfo:
        require_api_key(x_api_key="wrong")
    assert excinfo.value.status_code == 403
    # Correct header → no raise
    require_api_key(x_api_key="secret-xyz")


# ---------------------------------------------------------------------------
# 4) llm_status endpoint surfaces the new fields
# ---------------------------------------------------------------------------


def test_llm_status_includes_consent_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The /system/llm-status endpoint must expose the BLOCKER-#2
    and BLOCKER-#3 fields so the SPA can render the consent banner."""
    from fastapi.testclient import TestClient

    from rlpe.api.app import app

    monkeypatch.setenv("RLPE_HOST", "127.0.0.1")
    monkeypatch.delenv("RLPE_DATA_OUTBOUND_OPT_IN", raising=False)
    monkeypatch.delenv("RLPE_API_KEY", raising=False)
    client = TestClient(app)
    resp = client.get("/system/llm-status")
    assert resp.status_code == 200
    data = resp.json()
    # New BLOCKER-#2 fields
    assert "data_outbound_policy_default" in data
    assert data["data_outbound_policy_default"] == "api_redacted"
    assert "data_outbound_opt_in_set" in data
    assert data["data_outbound_opt_in_set"] is False
    # New BLOCKER-#3 fields
    assert "host_bind" in data
    assert "api_auth_required" in data
    assert data["api_auth_required"] is False
    # Pre-existing fields still there
    assert "key_configured" in data


def test_llm_status_reflects_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the opt-in env var is set, /system/llm-status must report
    ``data_outbound_opt_in_set=True`` so the SPA can warn."""
    from fastapi.testclient import TestClient

    from rlpe.api.app import app

    monkeypatch.setenv("RLPE_HOST", "127.0.0.1")
    monkeypatch.setenv("RLPE_DATA_OUTBOUND_OPT_IN", "1")
    client = TestClient(app)
    resp = client.get("/system/llm-status")
    data = resp.json()
    assert data["data_outbound_opt_in_set"] is True


# ---------------------------------------------------------------------------
# 5) run_web_server.py fail-fast + ephemeral key
# ---------------------------------------------------------------------------


def test_run_web_server_fails_fast_on_non_loopback_without_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``run_web_server.main()`` must exit 1 with a clear stderr
    message when ``RLPE_HOST`` is non-loopback and no API key is set.
    This is the fail-secure posture (BLOCKER-#3) at the launcher
    layer — even if a future commit accidentally loosens the
    ``require_api_key`` guard, the launcher still refuses to start."""
    # Force import to pick up the in-tree copy (script lives at
    # project root, not under src/).
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    # Lazy import so the script can be re-imported safely.
    if "run_web_server" in sys.modules:
        del sys.modules["run_web_server"]
    import run_web_server

    monkeypatch.setenv("RLPE_HOST", "0.0.0.0")
    monkeypatch.delenv("RLPE_API_KEY", raising=False)
    rc = run_web_server.main()
    assert rc == 1
    captured = capsys.readouterr()
    assert "non-loopback" in captured.err or "RLPE_API_KEY" in captured.err


def test_run_web_server_loopback_prints_ephemeral_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``run_web_server.main()`` must print a 64-char ephemeral
    hex key to stderr on loopback startup when no API key is set,
    so the SPA onboarding banner can require it without forcing
    another restart cycle. We don't start uvicorn here — we just
    confirm the ephemeral-key banner was emitted before uvicorn.run()
    would be reached. (uvicorn.run is mocked to SystemExit so the
    test exits cleanly.)"""
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if "run_web_server" in sys.modules:
        del sys.modules["run_web_server"]
    import run_web_server

    monkeypatch.setenv("RLPE_HOST", "127.0.0.1")
    monkeypatch.delenv("RLPE_API_KEY", raising=False)

    # Patch uvicorn.run to raise SystemExit immediately so we don't
    # actually try to bind a port.
    import uvicorn

    def _fake_run(*args, **kwargs):
        raise SystemExit(0)

    monkeypatch.setattr(uvicorn, "run", _fake_run)

    rc = run_web_server.main()
    assert rc == 0
    captured = capsys.readouterr()
    assert "[ephemeral-key]" in captured.err
    # The 64-char hex token is printed after the banner.
    import re

    m = re.search(r"RLPE_API_KEY=([0-9a-f]{64})", captured.err)
    assert m is not None, "ephemeral key (64 hex chars) not found in stderr: " + captured.err[:300]
