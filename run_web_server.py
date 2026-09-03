#!/usr/bin/env python3
"""
Web Server Launcher for RLPE
启动 RLPE Web 界面的服务器

环境变量:
  RLPE_HOST  - 监听地址 (默认 127.0.0.1；audit 2026-07-31 改)
  RLPE_PORT  - 监听端口 (默认 8000)
  RLPE_WORKERS - uvicorn worker 数量 (默认 1)
  RLPE_LOG_LEVEL - log level (默认 info)
  RLPE_API_KEY - 设置后敏感 endpoint 需要 X-API-Key header 验证
                 (audit 2026-08-19 phase 5b)。
                 Audit 2026-09-03 (BLOCKER-#3): 当 RLPE_HOST 非 loopback
                 (即对外暴露) 时未设置 RLPE_API_KEY 会被拒绝启动。
  RLPE_MAX_UPLOAD_MB - 单个 PDF 上传大小上限 MB (默认 100, audit 2026-08-19 phase 5b)
"""

import os
import sys
from pathlib import Path

try:
    import uvicorn
except ImportError:
    print("Error: uvicorn not installed. Install with: pip install uvicorn fastapi")
    sys.exit(1)

# Add project root to path
project_root = Path(__file__).resolve().parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Load .env if present (silently skip if missing).
#
# Precedence policy (per RLPE design):
#   1. Pre-existing OS env vars win for MOST keys (so an operator can
#      temporarily override .env from the shell).
#   2. EXCEPT for the project's MiniMax-related keys (ANTHROPIC_API_KEY,
#      ANTHROPIC_BASE_URL, ANTHROPIC_MODEL, MINIMAX_*) where the .env
#      file wins. This is because tools like Claude Code set
#      ``ANTHROPIC_BASE_URL`` globally for their own use, and that
#      value (e.g. ark.cn-beijing.volces.com) is NOT what RLPE wants —
#      RLPE wants the value the OPERATOR put in this project's .env
#      (typically https://api.minimaxi.com/anthropic).
#   3. Setting ``RLPE_FORCE_ENV_OVERRIDE=1`` in the SHELL forces .env
#      to win for ALL keys (escape hatch for unusual setups).
#
# Without rule 2, an operator would point .env at MiniMax, start the
# server, and silently get connected to the wrong endpoint because the
# Claude Code global ANTHROPIC_BASE_URL takes precedence.
# audit 2026-07-31: logic centralised in rlpe.env_loader (the CLI and
# server copies had drifted — the CLI missed MINIMAX_API_KEY).
from rlpe.env_loader import load_env_file

env_path = project_root / ".env"
try:
    load_env_file(env_path)
except Exception as e:
    print(f"Warning: failed to load .env: {e}")

from rlpe.api.app import app

# Force UTF-8 on stdout/stderr so the banner (which contains box-drawing
# characters and an emoji) can print on Windows code pages (cp936 / cp1252).
# Phase F-3-E NIT-1: also fall back to an ASCII-only banner if the
# stream still can't render the box characters (e.g. running under
# ``nohup`` redirect to a cp1252 log file where reconfigure silently
# failed, or under PyInstaller where sys.stdout has no ``encoding``).
# The previous version assumed reconfigure ALWAYS succeeded and printed
# mojibake on the failure path.
_USE_BOX_BANNER = True
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        # If reconfigure fails the stream is likely a pipe / file
        # that can't be re-encoded; fall back to ASCII-only banner
        # to keep the output usable.
        _USE_BOX_BANNER = False
        break
    else:
        # Probe: if encoding is still NOT utf-8 after reconfigure
        # (some Windows consoles keep cp1252), drop to ASCII mode.
        try:
            _enc = (getattr(_stream, "encoding", "") or "").lower().replace("-", "_")
            if _enc not in ("utf_8", "utf8"):
                _USE_BOX_BANNER = False
        except Exception:
            _USE_BOX_BANNER = False


_BANNER_BOX = """
    ╔════════════════════════════════════════════════════════════╗
    ║  🔬 RLPE Web Server                                        ║
    ║  放射虫图版提取系统 - Web 界面                               ║
    ║                                                            ║
    ║  访问地址: http://{host}:{port}                             ║
    ║  API 文档: http://{host}:{port}/docs                       ║
    ║  workers : {workers}                                       ║
    ║  log     : {log_level}                                     ║
    ╚════════════════════════════════════════════════════════════╝
"""

_BANNER_ASCII = """
    +------------------------------------------------------------+
    |  RLPE Web Server                                           |
    |  放射虫图版提取系统 - Web 界面                               |
    |                                                            |
    |  访问地址: http://{host}:{port}                             |
    |  API 文档: http://{host}:{port}/docs                       |
    |  workers : {workers}                                       |
    |  log     : {log_level}                                     |
    +------------------------------------------------------------+
"""


def main() -> int:
    """Start the RLPE web server.

    Returns 0 on a clean shutdown (e.g. SIGINT → uvicorn raises
    SystemExit(0)) and non-zero on a configuration / startup error.
    Without the explicit ``-> int`` and exit code (Phase F-3-E NIT-3),
    a shell wrapper that runs ``run_web_server.py`` cannot tell
    "server started then was killed" apart from "failed to bind port"
    — both just exit with whatever uvicorn chose.
    """
    # audit 2026-07-31: default to loopback. The server runs the user's
    # paid MiniMax key; binding 0.0.0.0 exposed it to the LAN (any local
    # webpage could drive it via a CORS simple request). Set
    # ``RLPE_HOST=0.0.0.0`` explicitly for remote use.
    # Audit 2026-09-03 (BLOCKER-#3): when ``RLPE_HOST`` is non-loopback
    # AND ``RLPE_API_KEY`` is unset, refuse to start. The historical
    # posture let any LAN user POST to ``/jobs/upload`` and trigger the
    # paid MiniMax API — fail-secure is mandatory here.
    host = os.environ.get("RLPE_HOST", "127.0.0.1")
    api_key = os.environ.get("RLPE_API_KEY", "")
    is_loopback = host.strip().lower() in ("127.0.0.1", "::1", "localhost", "[::1]")
    if not is_loopback and not api_key:
        print(
            "[fatal] RLPE_HOST=" + host + " is non-loopback but RLPE_API_KEY is not set.\n"
            "        Refusing to start (audit 2026-09-03 BLOCKER-#3: fail-secure\n"
            "        posture — a LAN-reachable bind without an API key would let\n"
            "        any peer trigger paid MiniMax calls).\n"
            "        Either:\n"
            "          export RLPE_API_KEY=\"$(openssl rand -hex 32)\"   # remote access\n"
            "        Or:\n"
            "          unset RLPE_HOST     # fall back to the loopback default",
            file=sys.stderr,
        )
        return 1
    # Audit 2026-09-03 (BLOCKER-#3) second half: when loopback + no key,
    # print an ephemeral random 32-byte-hex key to stderr ONCE so the
    # SPA onboarding banner can paste it into its auth field. The key is
    # NOT auto-loaded by the server (require_api_key stays a no-op), so
    # existing local-dev workflows keep working; the printed key is just
    # available for operators who want to enable auth without restarting.
    if is_loopback and not api_key:
        import secrets as _secrets
        ephemeral_key = _secrets.token_hex(32)
        print(
            "\n[ephemeral-key] No RLPE_API_KEY set; auth is disabled on loopback.\n"
            "                 To enable API auth on this server without restarting,\n"
            "                 set the env var to the following 64-char hex:\n"
            "                 RLPE_API_KEY=" + ephemeral_key + "\n",
            file=sys.stderr,
        )
    try:
        port = int(os.environ.get("RLPE_PORT", "8000"))
        workers = int(os.environ.get("RLPE_WORKERS", "1"))
    except ValueError as exc:
        # Phase F-3-E NIT-2: a malformed RLPE_PORT (e.g. ``abc``) used
        # to raise an unhelpful ValueError traceback and exit 1. Now
        # we catch and surface a one-line error.
        print(f"Error: invalid numeric env var ({exc})", file=sys.stderr)
        return 1
    log_level = os.environ.get("RLPE_LOG_LEVEL", "info")

    if _USE_BOX_BANNER:
        print(_BANNER_BOX.format(host=host, port=port, workers=workers, log_level=log_level))
    else:
        print(_BANNER_ASCII.format(host=host, port=port, workers=workers, log_level=log_level))

    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=workers,
            reload=False,
            log_level=log_level,
        )
    except SystemExit as exc:
        # uvicorn raises SystemExit on clean shutdown (SIGINT) or
        # port-bind failure. Forward the exit code so the shell
        # wrapper sees "server stopped cleanly" (0) vs "port in use"
        # (1).
        return int(exc.code) if exc.code is not None else 0
    except Exception as exc:
        print(f"Error: uvicorn failed to start: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
