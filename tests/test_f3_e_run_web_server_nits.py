"""Phase F-3-E (2026-08-20) — regression tests for the run_web_server.py
NIT fixes (NIT-1 box-banner fallback, NIT-2 numeric env validation,
NIT-3 main() return type).

The launcher is the script operators run on a headless server; the
previous version crashed with a mojibake banner on Windows cp1252
streams, raised an unhelpful ValueError on a malformed RLPE_PORT,
and returned no exit code (so a shell wrapper couldn't tell a clean
SIGINT-shutdown apart from a port-bind failure). These tests guard
the fixes.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_RUN_WEB_SERVER = _REPO / "run_web_server.py"


def _load_module():
    """Import run_web_server.py as a module without executing the
    ``if __name__ == "__main__"`` block (we don't want to actually
    start uvicorn during tests).

    The module-level code does try to import ``uvicorn`` and
    ``rlpe.api.app``; if either is missing in the test env the
    ``SystemExit(1)`` propagates here and the test skips. In a
    normal dev install both are present.
    """
    spec = importlib.util.spec_from_file_location("run_web_server", _RUN_WEB_SERVER)
    if spec is None or spec.loader is None:
        pytest.skip("cannot load run_web_server.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit as exc:
        # uvicorn not installed → module sys.exit(1)s before we get
        # to ``main()``. Skip the test instead of failing.
        pytest.skip(f"run_web_server.py self-exits at import: {exc}")
    except ImportError as exc:
        pytest.skip(f"run_web_server deps missing: {exc}")
    return mod


# ============================================================
# NIT-3: main() return type annotation
# ============================================================
class TestF3EMainReturnType:
    """F-3-E fix: ``main()`` is annotated ``-> int`` and the script's
    ``__main__`` block uses ``raise SystemExit(main())`` so a shell
    wrapper can observe the exit code."""

    def test_main_is_annotated_int(self):
        import inspect

        # Read the file and grep — annotation introspection would
        # require importing the module which then starts uvicorn.
        src = _RUN_WEB_SERVER.read_text(encoding="utf-8")
        assert "def main() -> int:" in src
        assert "raise SystemExit(main())" in src


# ============================================================
# NIT-1: ASCII banner fallback
# ============================================================
class TestF3EAsciiBanner:
    """F-3-E fix: when stdout / stderr can't be reconfigured to UTF-8,
    the launcher falls back to an ASCII-only banner instead of
    mojibake-printing the box-drawing characters."""

    def test_ascii_banner_template_defined(self):
        src = _RUN_WEB_SERVER.read_text(encoding="utf-8")
        # The ASCII banner uses ``+---+`` instead of ``╔═╗``.
        assert "_BANNER_ASCII" in src
        assert "+-" in src
        assert "|" in src
        # And it carries the same fields as the box banner.
        for needle in ("{host}", "{port}", "{workers}", "{log_level}"):
            assert needle in src

    def test_box_banner_template_defined(self):
        src = _RUN_WEB_SERVER.read_text(encoding="utf-8")
        assert "_BANNER_BOX" in src
        # The box banner keeps the original box-drawing characters.
        assert "╔" in src and "╚" in src

    def test_module_picks_ascii_when_reconfigure_fails(self, monkeypatch):
        """If ``sys.stdout.reconfigure`` raises, the module sets
        ``_USE_BOX_BANNER = False`` so the ASCII banner prints."""

        # Patch sys.stdout so reconfigure() raises. We do this BEFORE
        # importing the module so the module-load code path is
        # exercised.
        class _BoomStream:
            encoding = "cp1252"

            def reconfigure(self, **kw):
                raise OSError("reconfigure not supported")

            def write(self, *_):
                pass

        monkeypatch.setattr(sys, "stdout", _BoomStream())
        # Force a fresh import: clear any cached version.
        for name in list(sys.modules):
            if name == "run_web_server" or name.startswith("rlpe."):
                sys.modules.pop(name, None)
        mod = _load_module()
        assert mod._USE_BOX_BANNER is False, (
            "should fall back to ASCII banner when reconfigure fails"
        )


# ============================================================
# NIT-2: numeric env var validation
# ============================================================
class TestF3ENumericEnvValidation:
    """F-3-E fix: a malformed RLPE_PORT (e.g. ``abc``) returns 1 with
    a one-line error instead of an unhelpful ValueError traceback."""

    def test_invalid_port_returns_one(self, monkeypatch, capsys):
        monkeypatch.setenv("RLPE_PORT", "not-a-number")
        monkeypatch.setenv("RLPE_WORKERS", "1")
        # Force fresh import so module-level code re-runs against
        # the new env.
        for name in list(sys.modules):
            if name == "run_web_server":
                sys.modules.pop(name, None)
        mod = _load_module()
        rc = mod.main()
        assert rc == 1
        captured = capsys.readouterr()
        assert "invalid numeric env var" in captured.err

    def test_invalid_workers_returns_one(self, monkeypatch, capsys):
        monkeypatch.setenv("RLPE_PORT", "8000")
        monkeypatch.setenv("RLPE_WORKERS", "two-many")
        for name in list(sys.modules):
            if name == "run_web_server":
                sys.modules.pop(name, None)
        mod = _load_module()
        rc = mod.main()
        assert rc == 1
        captured = capsys.readouterr()
        assert "invalid numeric env var" in captured.err
