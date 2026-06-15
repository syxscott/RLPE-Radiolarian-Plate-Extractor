"""MiniMax M3 API integration smoke test.

Usage:
    # 1. Set your key in .env or env var
    cp .env.example .env
    # edit .env, set ANTHROPIC_API_KEY

    # 2. Run
    python scripts/test_MiniMax_api.py

    # Optional flags
    python scripts/test_MiniMax_api.py --no-thinking        # disable thinking
    python scripts/test_MiniMax_api.py --panel /path/to.png  # specific panel
    python scripts/test_MiniMax_api.py --text-only          # skip image, just text
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Load .env if available, with project-key precedence over OS env.
# See ``run_web_server.py`` / ``cli.py`` for the rationale: tools like
# Claude Code set ANTHROPIC_BASE_URL globally for their own backend, so
# we must override the project's MiniMax keys explicitly.
try:
    from dotenv import dotenv_values, load_dotenv
    _env_file = ROOT / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=False)
        _force = os.environ.get("RLPE_FORCE_ENV_OVERRIDE") == "1"
        _project_keys = {
            "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL",
            "MiniMax_API_KEY", "MiniMax_MODEL", "MiniMax_BASE_URL",
        }
        for _k, _v in (dotenv_values(_env_file) or {}).items():
            if _v is None:
                continue
            if _force or _k in _project_keys:
                os.environ[_k] = _v
except Exception:
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description="MiniMax M3 API smoke test")
    parser.add_argument("--MiniMax-api-key", type=str, default=None)
    parser.add_argument("--MiniMax-endpoint", type=str, default="https://api.minimaxi.com/anthropic")
    parser.add_argument("--MiniMax-model", type=str, default="MiniMax-M3")
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--thinking-budget", type=int, default=1024)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--panel", type=str, default=None,
                        help="Path to a single panel PNG/JPG. If omitted, auto-discover under uploads/")
    parser.add_argument("--text-only", action="store_true",
                        help="Skip image, test text-only path")
    args = parser.parse_args()

    api_key = args.MiniMax_api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: set ANTHROPIC_API_KEY (or pass --MiniMax-api-key)", file=sys.stderr)
        print("  Get a Token Plan subscription key at:", file=sys.stderr)
        print("  https://platform.minimaxi.com/user-center/payment/token-plan", file=sys.stderr)
        return 2

    from rlpe.llm_backends import MiniMaxM3Backend

    backend = MiniMaxM3Backend(
        api_key=api_key,
        base_url=args.MiniMax_endpoint,
        model=args.MiniMax_model,
        enable_thinking=not args.no_thinking,
        thinking_budget_tokens=args.thinking_budget,
        max_output_tokens=args.max_output_tokens,
    )

    # System prompt (from gemma_postprocess.py for compatibility)
    from rlpe.gemma_postprocess import GEMMA_SYSTEM_PROMPT_ZH

    # Build user prompt + image
    if args.text_only:
        panel_image = None
        panel_path = None
    else:
        panel_path = Path(args.panel) if args.panel else None
        if panel_path is None or not panel_path.exists():
            # auto-discover
            candidates = []
            for sub in ("uploads", "work", "service_work", "output"):
                d = ROOT / sub
                if d.exists():
                    candidates += list(d.glob("**/panels/panel_*.png"))
                    candidates += list(d.glob("**/panels/panel_*.jpg"))
                    candidates += list(d.glob("**/panel_*.png"))
            if not candidates:
                print("ERROR: no panel images found. Pass --panel or generate some via run_pipeline first.",
                      file=sys.stderr)
                return 3
            panel_path = candidates[0]
        print(f"[1/3] Loading panel: {panel_path}", file=sys.stderr)
        from PIL import Image
        panel_image = Image.open(panel_path).convert("RGB")
        print(f"      size={panel_image.size}", file=sys.stderr)

    user_prompt = (
        "[Caption]\n"
        "Fig. 1. (A, B) Actinomma leptodermum; (C, D) Stylosphaera hispida; "
        "(E) Cenodiscus sp.\n\n"
        "[OCR labels]\n"
        "['A', 'B', 'C', 'D', 'E']\n\n"
        "请判断该panel最可能对应的label与拉丁学名。严格输出JSON。"
    )

    print(f"[2/3] Calling MiniMax M3 (model={args.MiniMax_model}, thinking={'ON' if not args.no_thinking else 'OFF'})...",
          file=sys.stderr)
    t0 = time.time()
    result = backend.infer_panel(
        panel_image=panel_image,
        caption_text="Fig. 1. (A, B) Actinomma leptodermum...",
        ocr_labels=["A", "B", "C", "D", "E"],
        system_prompt=GEMMA_SYSTEM_PROMPT_ZH,
        user_prompt=user_prompt,
    )
    elapsed = time.time() - t0

    print(f"[3/3] Done in {elapsed:.2f}s", file=sys.stderr)
    print("=" * 70)
    print("RESPONSE:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("=" * 70)
    print("SUMMARY:")
    print(f"  request_id    = {result.get('request_id')}")
    print(f"  model_version = {result.get('model_version')}")
    print(f"  fallback_used = {result.get('fallback_used')}")
    print(f"  usage         = {result.get('usage', {})}")
    print(f"  cost_cny      = {result.get('cost_cny', 'n/a')}")
    summary = backend.cost_summary()
    print(f"  session total = calls={summary['calls']} in={summary['input_tokens']} out={summary['output_tokens']} cost_cny={summary['total_cost_cny']}")
    return 0 if not result.get("fallback_used") else 1


if __name__ == "__main__":
    raise SystemExit(main())
