#!/usr/bin/env python3
"""
Web Server Launcher for RLPE
启动 RLPE Web 界面的服务器

环境变量:
  RLPE_HOST  - 监听地址 (默认 0.0.0.0)
  RLPE_PORT  - 监听端口 (默认 8000)
  RLPE_WORKERS - uvicorn worker 数量 (默认 1)
  RLPE_LOG_LEVEL - log level (默认 info)
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
#      ANTHROPIC_BASE_URL, ANTHROPIC_MODEL, MiniMax_*) where the .env
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
_RLPE_PROJECT_OVERRIDE_KEYS = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "MiniMax_API_KEY",
    "MiniMax_MODEL",
    "MiniMax_BASE_URL",
    "MINIMAX_API_KEY",
}
_force_override = os.environ.get("RLPE_FORCE_ENV_OVERRIDE") == "1"

env_path = project_root / ".env"
if env_path.exists():
    try:
        with env_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if not key:
                    continue
                # Decide whether to override an existing env var.
                should_override = (
                    _force_override
                    or key in _RLPE_PROJECT_OVERRIDE_KEYS
                    or key not in os.environ
                )
                if should_override:
                    os.environ[key] = value
    except Exception as e:
        print(f"Warning: failed to load .env: {e}")

from rlpe.api.app import app

# Force UTF-8 on stdout/stderr so the banner (which contains box-drawing
# characters and an emoji) can print on Windows code pages (cp936 / cp1252).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def main() -> None:
    """Start the RLPE web server."""
    host = os.environ.get("RLPE_HOST", "0.0.0.0")
    port = int(os.environ.get("RLPE_PORT", "8000"))
    workers = int(os.environ.get("RLPE_WORKERS", "1"))
    log_level = os.environ.get("RLPE_LOG_LEVEL", "info")

    print(f"""
    ╔════════════════════════════════════════════════════════════╗
    ║  🔬 RLPE Web Server                                        ║
    ║  放射虫图版提取系统 - Web 界面                               ║
    ║                                                            ║
    ║  访问地址: http://{host}:{port}                          ║
    ║  API 文档: http://{host}:{port}/docs                    ║
    ║  workers : {workers}                                      ║
    ║  log     : {log_level}                                    ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        reload=False,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
