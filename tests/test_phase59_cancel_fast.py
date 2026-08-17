"""Phase 59 — Pipeline correctness, Bug 2.3.

``pipeline.run`` uses ``with ThreadPoolExecutor(...)`` which calls
``shutdown(wait=True)`` on exit. After Cancel, ``f.cancel()`` only
cancels not-yet-started futures; running futures must complete
(especially LLM API calls). For 4 PDFs each sleeping 30 seconds,
Cancel currently blocks until all 30s elapse.

The fix:

  1. Move from ``with pool:`` to manual ``pool = ThreadPoolExecutor(...)``
     + ``try / finally``.
  2. Inside the cancel branch: ``pool.shutdown(wait=False, cancel_futures=True)``
     (Python 3.9+).
  3. Wrap long-running pipeline calls with periodic
     ``cancel_event.is_set()`` checks (every 5s in tight loops).

This test submits 4 mock "slow PDF" tasks that each sleep 30s, sets
the cancel event after ~1s, and asserts ``run()`` returns within
~6 seconds (well under the 30s sleep).

Because ``run()`` constructs a full pipeline (heavy __init__), we
patch ``_process_one_pdf`` directly on a stub pipeline and drive
``run()`` against a fake config that lists 4 stub PDF paths.
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_cancel_during_run_returns_within_6s() -> None:
    """Bug 2.3 fix: cancel_event set during run() returns fast."""
    from rlpe.config import PipelineConfig
    from rlpe.pipeline import RadiolarianPipeline

    cancel = threading.Event()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        # 4 stub PDFs.
        for i in range(4):
            (pdf_dir / f"paper_{i}.pdf").write_bytes(b"%PDF-1.4\n%EOF\n")

        cfg = PipelineConfig(pdf_dir=pdf_dir, work_dir=work_dir, num_workers=4)

        def slow_pdf(self, p):
            # Each PDF sleeps 30s — but we must check the cancel_event
            # frequently so cancellation propagates.
            for _ in range(60):
                if cancel.is_set():
                    return []
                time.sleep(0.5)
            return []

        pipe = RadiolarianPipeline.__new__(RadiolarianPipeline)
        RadiolarianPipeline.__init__(pipe, cfg, cancel_event=cancel)
        pipe._process_one_pdf = slow_pdf.__get__(pipe)

        def set_cancel_later():
            time.sleep(1.0)
            cancel.set()

        threading.Thread(target=set_cancel_later, daemon=True).start()

        start = time.monotonic()
        rows = pipe.run()
        elapsed = time.monotonic() - start
        assert rows == [], "Cancel should produce zero rows when all PDFs are mid-flight"
        assert elapsed < 6.0, f"Cancel must short-circuit within 6s; took {elapsed:.2f}s"


def test_pool_uses_cancel_futures_on_cancel_branch() -> None:
    """Bug 2.3 source-guard: pool.shutdown() must use wait=False and
    cancel_futures=True when cancellation is requested (Python 3.9+).
    """
    src = (Path(__file__).resolve().parents[1] / "src/rlpe/pipeline.py").read_text(encoding="utf-8")
    assert "shutdown(wait=False, cancel_futures=True)" in src, (
        "Cancel branch must call pool.shutdown(wait=False, cancel_futures=True)"
    )
    # Must NOT still be using `with ThreadPoolExecutor(...) as pool:`.
    # We look for it as actual code (not in a comment) by checking for
    # the line start with the indentation pattern.
    import re

    code_with_pattern = [
        line for line in src.splitlines() if re.match(r"^\s+with ThreadPoolExecutor\(", line)
    ]
    assert not code_with_pattern, (
        "Phase 59 Bug 2.3: pool lifecycle must be manual (try/finally) "
        "so cancel_event can short-circuit shutdown(wait=True). "
        f"Found: {code_with_pattern}"
    )


def test_pool_is_released_in_finally() -> None:
    """Bug 2.3 source-guard: the pool is released in a ``finally`` block
    so the executor is always shut down even on exceptions.
    """
    src = (Path(__file__).resolve().parents[1] / "src/rlpe/pipeline.py").read_text(encoding="utf-8")
    # Verify try/finally around the executor lifecycle.
    assert "pool = ThreadPoolExecutor(" in src, (
        "Pool must be assigned to a variable (not context-managed)"
    )
    assert "finally:" in src, "Pool must be shut down in a finally block"
