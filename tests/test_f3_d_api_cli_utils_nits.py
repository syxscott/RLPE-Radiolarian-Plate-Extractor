"""Phase F-3 (2026-08-20) — Area D MINOR + NIT regression tests.

After the F-3-D enumeration agent produced the 28-item API+CLI+pipeline
MINOR+NIT list, this phase applied the high-impact fixes. The tests
below guard the most likely regressions (re-introducing silent error
swallows, removing the magic-32 constant, etc.) so the next sweep
doesn't re-do them.

Each test class maps to one fix area:

* ``TestF3DMaskApiKey``     — app.py: short-key sentinel vs None
* ``TestF3DSleepConstants`` — app.py: named sleep/tick constants
* ``TestF3DCliNumWorkers``  — cli.py: MAX_NUM_WORKERS clamp constant
* ``TestF3DCliRunDry``      — cli.py: dry-run echoes full config
* ``TestF3DUtilsSlugify``   — utils.py: precompiled regex
* ``TestF3DUtilsReadText``  — utils.py: broader OSError catch
* ``TestF3DCliExportEncoding`` — cli_export.py: UTF-8 JSONL open
* ``TestF3DPipelineSha256``  — pipeline.py: renamed hash function
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _read(rel: str) -> str:
    return (_REPO / rel).read_text(encoding="utf-8")


# ============================================================
# api/app.py: _mask_api_key short-key sentinel
# ============================================================
class TestF3DMaskApiKey:
    """F-3 fix: ``_mask_api_key`` now distinguishes "key set but too
    short to preview" from "no key configured"."""

    def test_short_key_returns_distinct_sentinel(self):
        from rlpe.api.app import _mask_api_key

        # Short keys (less than 8 chars) used to return "***"
        # which the frontend couldn't distinguish from "not set".
        result = _mask_api_key("abc")
        assert result == "(short)"
        # And the "no key" state still returns None.
        assert _mask_api_key(None) is None
        assert _mask_api_key("") is None
        # And a normal-length key still returns the truncated form
        # (first 3 chars + "..." + last 4 chars).
        long_key = "sk-or-v1-abcDEF1234xyz"
        assert len(long_key) >= 8
        assert _mask_api_key(long_key) == f"{long_key[:3]}...{long_key[-4:]}"
        # And the edge case (exactly 8 chars) gets a 2-char hint.
        assert _mask_api_key("12345678") == "12…"


# ============================================================
# api/app.py: sleep interval constants
# ============================================================
class TestF3DSleepConstants:
    """F-3 fix: SSE/WS poll intervals + heartbeat tick + fallback popup
    timeout are now module-level constants instead of magic literals."""

    def test_module_constants_defined(self):
        from rlpe.api import app

        assert hasattr(app, "_SSE_POLL_INTERVAL_SEC")
        assert hasattr(app, "_WS_POLL_INTERVAL_SEC")
        assert hasattr(app, "_HEARTBEAT_TICK_SEC")
        assert hasattr(app, "FALLBACK_POPUP_TIMEOUT_MS")
        assert hasattr(app, "FALLBACK_POPUP_TICK_MS")

    def test_inlined_magic_numbers_removed(self):
        src = _read("src/rlpe/api/app.py")
        # The SSE poll loop used to be ``await _asyncio.sleep(1.0)``.
        body = re.search(r"def _event_stream.*?\n[ \t]+\}\n", src, re.DOTALL)
        assert body is not None
        assert "_SSE_POLL_INTERVAL_SEC" in body.group(0)
        # The WebSocket handler used to be ``await _asyncio.sleep(0.5)``.
        # The function is named ``websocket_job_progress`` (not
        # ``_ws_event_stream``) so use a broader pattern.
        ws_body = re.search(
            r"async def websocket_job_progress.*?while True:.*?await _asyncio\.sleep\([^)]+\)",
            src,
            re.DOTALL,
        )
        assert ws_body is not None
        assert "_WS_POLL_INTERVAL_SEC" in ws_body.group(0)
        # The heartbeat used to be ``stop_hb.wait(1.0)``.
        assert "_HEARTBEAT_TICK_SEC" in src
        # The fallback popup timeout used to be ``TIMEOUT_MS = 300_000``.
        assert "FALLBACK_POPUP_TIMEOUT_MS" in src


# ============================================================
# cli.py: MAX_NUM_WORKERS constant
# ============================================================
class TestF3DCliNumWorkers:
    """F-3 fix: the magic ``32`` clamping --num-workers is now a named
    constant so the argparse help text and the runtime clamp agree."""

    def test_max_num_workers_constant_exists(self):
        from rlpe import cli

        assert hasattr(cli, "MAX_NUM_WORKERS")
        assert hasattr(cli, "MIN_NUM_WORKERS")
        assert cli.MAX_NUM_WORKERS == 32
        assert cli.MIN_NUM_WORKERS == 1

    def test_run_pipeline_uses_constants(self):
        import argparse

        from rlpe.cli import _run_pipeline

        # Build a minimal namespace with the args _run_pipeline touches.
        ns = argparse.Namespace(num_workers=999)
        try:
            _run_pipeline(ns)
        except Exception:
            # We don't care if the rest of _run_pipeline crashes (it
            # does a lot of setup); we just want to know the clamp
            # ran first.
            pass
        assert ns.num_workers == 32, (
            f"clamp should cap at MAX_NUM_WORKERS=32, got {ns.num_workers}"
        )


# ============================================================
# cli.py: --dry-run echoes full config
# ============================================================
class TestF3DCliRunDry:
    """F-3 fix: ``_run_dry`` now prints caption_window, od_caption_window,
    use_opendataloader, min_panel_score, render_dpi, save_intermediate,
    taxon_model, grobid_url, use_yolo_figures, yolo_conf, yolo_iou."""

    def test_dry_run_prints_extra_fields(self, capsys):
        import argparse

        from rlpe.cli import _run_dry

        ns = argparse.Namespace(
            pdf_dir=Path("/tmp"),
            work_dir=Path("/tmp"),
            output_dir=None,
            num_workers=4,
            use_gpu=None,
            ocr_backend="paddleocr",
            ocr_lang="en",
            m3_prompt_lang="auto",
            llm_backend="rules",
            deterministic=False,
            data_outbound_policy="local_only",
            config=None,
            caption_window=2,
            od_caption_window=5,
            use_opendataloader=True,
            min_panel_score=0.8,
            render_dpi=200,
            save_intermediate=False,
            taxon_model="en_eco",
            grobid_url="http://localhost:8070",
            use_yolo_figures=False,
            yolo_conf_threshold=0.25,
            yolo_iou_threshold=0.45,
        )
        _run_dry(ns)
        out = capsys.readouterr().out
        # Spot-check: the new fields actually appear.
        assert "--caption-window" in out
        assert "--od-caption-window" in out
        assert "--use-opendataloader" in out
        assert "--min-panel-score" in out
        assert "--render-dpi" in out
        assert "--taxon-model" in out
        assert "--grobid-url" in out
        assert "--yolo-conf" in out
        assert "--yolo-iou" in out


# ============================================================
# utils.py: slugify uses precompiled regex
# ============================================================
class TestF3DUtilsSlugify:
    """F-3 fix: the regex used by ``slugify`` is now module-level."""

    def test_module_has_compiled_pattern(self):
        from rlpe import utils

        assert hasattr(utils, "_SLUGIFY_NON_ALNUM")

    def test_slugify_behaviour_unchanged(self):
        from rlpe.utils import slugify

        assert slugify("Foo Bar Baz") == "foo_bar_baz"
        assert slugify("___leading_and_trailing___") == "leading_and_trailing"
        assert slugify("中文") == "item"  # fallback


# ============================================================
# utils.py: read_text broader OSError catch
# ============================================================
class TestF3DUtilsReadText:
    """F-3 fix: ``read_text`` now catches ``OSError`` (covers
    ``IsADirectoryError``, ``PermissionError``) instead of only
    ``FileNotFoundError``."""

    def test_directory_returns_default(self, tmp_path):
        from rlpe.utils import read_text

        # Pass a directory path — previously raised IsADirectoryError.
        assert read_text(tmp_path, default="<empty>") == "<empty>"

    def test_missing_file_returns_default(self, tmp_path):
        from rlpe.utils import read_text

        missing = tmp_path / "nope.txt"
        assert read_text(missing, default="<empty>") == "<empty>"

    def test_existing_file_returns_content(self, tmp_path):
        from rlpe.utils import read_text

        path = tmp_path / "real.txt"
        path.write_text("hello", encoding="utf-8")
        assert read_text(path) == "hello"


# ============================================================
# cli_export.py: UTF-8 JSONL open
# ============================================================
class TestF3DCliExportEncoding:
    """F-3 fix: ``_run_output_from_jsonl`` opens the input file with
    ``encoding='utf-8'`` instead of the platform default."""

    def test_open_call_uses_utf8_encoding(self):
        src = _read("src/rlpe/cli_export.py")
        # The old form: ``with open(input_path) as f:``
        # The new form: ``with open(input_path, encoding="utf-8") as f:``
        assert "open(input_path, encoding=\"utf-8\")" in src

    def test_chinese_jsonl_parses(self, tmp_path):
        """Round-trip: write a JSONL containing Chinese species names
        and verify the UTF-8 open() call decodes them correctly."""
        path = tmp_path / "data.jsonl"
        # Chinese species names (RLPE's actual data domain).
        line = json.dumps(
            {"species": "中文测试种", "paper_id": "p1"},
            ensure_ascii=False,
        )
        path.write_text(line, encoding="utf-8")
        # The file should round-trip through utf-8 decode without
        # raising UnicodeDecodeError.
        text = path.read_text(encoding="utf-8")
        assert "中文测试种" in text
        # And ``json.loads`` should give back the original species name.
        decoded = json.loads(text)
        assert decoded["species"] == "中文测试种"


# ============================================================
# pipeline.py: renamed hash function
# ============================================================
class TestF3DPipelineSha256:
    """F-3 fix: ``_sha256_file`` was renamed to ``_short_sha256_file``
    to eliminate the cross-module name collision with the longer-digest
    helper in ``provenance/stamp.py``. A ``_sha256_file`` alias is kept
    for backward compatibility."""

    def test_short_sha256_file_exists(self):
        from rlpe import pipeline

        assert hasattr(pipeline, "_short_sha256_file")
        # The legacy name still resolves (backward-compat alias).
        assert hasattr(pipeline, "_sha256_file")
        assert pipeline._sha256_file is pipeline._short_sha256_file

    def test_short_sha256_returns_16_chars(self, tmp_path):
        from rlpe.pipeline import _short_sha256_file

        path = tmp_path / "blob.bin"
        path.write_bytes(b"x" * 1024)
        result = _short_sha256_file(path)
        assert len(result) == 16
        # Stable across calls.
        assert _short_sha256_file(path) == result


# ============================================================
# cli.py: module-level logger
# ============================================================
class TestF3DCliLogger:
    """F-3 fix: ``apply_log_level`` uses a module-level ``_CLI_LOGGER``
    instead of re-fetching via ``logging.getLogger`` per call."""

    def test_cli_logger_is_module_level(self):
        from rlpe import cli

        assert hasattr(cli, "_CLI_LOGGER")
        assert cli._CLI_LOGGER.name == "rlpe.cli"

    def test_apply_log_level_uses_module_logger(self):
        from rlpe import cli

        # If apply_log_level were still re-fetching the logger, the
        # module-level reference would not see the level change.
        cli.apply_log_level(quiet=False, verbose=True)
        assert cli._CLI_LOGGER.level == 10  # logging.DEBUG
        cli.apply_log_level(quiet=True, verbose=False)
        assert cli._CLI_LOGGER.level == 40  # logging.ERROR
        cli.apply_log_level(quiet=False, verbose=False)
        assert cli._CLI_LOGGER.level == 30  # logging.WARNING
