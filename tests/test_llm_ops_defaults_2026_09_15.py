"""LLM ops defaults from the v4-rerun diagnosis (2026-09-15).

A cold-start batch on another machine ran 200 minutes for 1 paper.
Four code-side causes were confirmed and fixed here:

  1. the CLI forced thinking ON by default ("not args.llm_no_thinking"),
     contradicting the backend dataclass default and the GUI/API paths —
     every call paid ~1k thinking tokens and 30-60s latency;
  2. the resolved model/base_url were never logged, so a shell
     ANTHROPIC_MODEL env leak silently ran the wrong model for hours;
  3. the flat 3600s worker timeout killed 4 workers chewing on 3-7MB
     scanned books (head-of-line blocking starved the batch);
  4. the global concurrency cap defaulted to "off + warning only" and
     an 8-worker × 8-concurrent run still smashed the rate limit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.cli import build_parser
from rlpe.config import PipelineConfig
from rlpe.pipeline import RadiolarianPipeline, _worker_timeout_for


# ============================================================
# Fix 1 — thinking is opt-in from the CLI
# ============================================================
def _extra_for(argv: list[str]) -> dict:
    """Build the cli extra dict the same way _run_pipeline does.

    The thinking wiring lives in the big extra dict literal inside
    _run_pipeline; rather than re-implementing it, assert on the parsed
    args contract the dict now follows:
      llm_enable_thinking = args.llm_enable_thinking and not llm_no_thinking
    """
    args = build_parser().parse_args(argv)
    return {
        "llm_enable_thinking": (
            bool(getattr(args, "llm_enable_thinking", False)) and not args.llm_no_thinking
        )
    }


def test_thinking_off_by_default():
    assert _extra_for([])["llm_enable_thinking"] is False


def test_thinking_opt_in_flag():
    assert _extra_for(["--llm-enable-thinking"])["llm_enable_thinking"] is True


def test_llm_no_thinking_still_disables():
    # backward compatibility: the legacy flag keeps its meaning even
    # when someone also passes the new opt-in.
    assert _extra_for(["--llm-no-thinking"])["llm_enable_thinking"] is False
    assert (
        _extra_for(["--llm-enable-thinking", "--llm-no-thinking"])["llm_enable_thinking"] is False
    )


def test_no_thinking_help_contradiction_fixed():
    """The old help said 'default: OFF' while the default was ON."""
    parser = build_parser()
    flag_help = next(
        a.help for a in parser._actions if "--llm-no-thinking" in getattr(a, "option_strings", [])
    )
    assert "OFF" in (flag_help or "")


# ============================================================
# Fix 3 — adaptive worker timeout
# ============================================================
def _make_pdf(tmp_path: Path, pages: int) -> Path:
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page()
    p = tmp_path / f"book_{pages}p.pdf"
    doc.save(str(p))
    doc.close()
    return p


def test_explicit_timeout_is_authoritative(tmp_path):
    p = _make_pdf(tmp_path, 80)
    assert _worker_timeout_for(p, 5) == 5
    assert _worker_timeout_for(p, 7200) == 7200


def test_adaptive_timeout_scales_with_pages(tmp_path):
    p = _make_pdf(tmp_path, 30)
    # 3600 + 120 × 30 = 7200
    assert _worker_timeout_for(p, None) == 7200


def test_adaptive_timeout_is_capped(tmp_path):
    p = _make_pdf(tmp_path, 200)
    assert _worker_timeout_for(p, None) == 14400


def test_timeout_fallback_when_pdf_unreadable(tmp_path):
    missing = tmp_path / "nope.pdf"
    assert _worker_timeout_for(missing, None) == 3600


# ============================================================
# Fix 4 — global concurrency cap defaults to 32
# ============================================================
@pytest.fixture()
def pipe(tmp_path: Path) -> RadiolarianPipeline:
    cfg = PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "w")
    return RadiolarianPipeline(cfg)


def test_dump_default_cap_clamps_8x8(pipe):
    """The friend's run 1: 8 workers × 8 concurrent = 64 in-flight calls
    with no explicit cap. The auto cap of 32 must now clamp to 4/worker."""
    pipe.config.extra["llm_max_concurrent"] = 8
    pipe.config.extra["llm_global_max_concurrent"] = 0
    pipe.config.num_workers = 8
    path = pipe._dump_worker_config()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["extra"]["llm_max_concurrent"] == 4
    finally:
        path.unlink(missing_ok=True)


def test_dump_no_clamp_when_under_cap(pipe):
    pipe.config.extra["llm_max_concurrent"] = 8
    pipe.config.extra["llm_global_max_concurrent"] = 0
    pipe.config.num_workers = 4
    path = pipe._dump_worker_config()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["extra"]["llm_max_concurrent"] == 8
    finally:
        path.unlink(missing_ok=True)


def test_dump_explicit_cap_still_wins(pipe):
    pipe.config.extra["llm_max_concurrent"] = 8
    pipe.config.extra["llm_global_max_concurrent"] = 24
    pipe.config.num_workers = 8
    path = pipe._dump_worker_config()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["extra"]["llm_max_concurrent"] == 3
    finally:
        path.unlink(missing_ok=True)
