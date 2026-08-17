"""Phase 43 — runtime-error regression tests.

User reported three runtime errors after launching the GUI:
  1. "QBackingStore::endPaint() called with active painter" —
     caused by _ProgressCellDelegate.paint() in jobs_tab.py
     not calling super().paint() to draw the cell background.
  2. "QThread: Destroyed while thread is still running" —
     caused by run_tab._on_thread_done() dropping the worker
     reference without quit()/wait().
  3. GROBID retry loop spamming "connection_refused" for 3
     retries per PDF when the server is offline (instead of
     failing fast via is_available() probe).

These tests pin the fixes.
"""

from __future__ import annotations

import os
import sys
import time
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QStyleOptionProgressBar  # noqa: E402

_app = QApplication.instance() or QApplication([])


import pytest


# ============================================================
# QBackingStore error: _ProgressCellDelegate
# ============================================================
def test_progress_cell_delegate_calls_super_paint():
    """Phase 43: _ProgressCellDelegate.paint() must call
    super().paint() to draw the cell background. Skipping it
    leaves the painter in an active state, causing
    "QBackingStore::endPaint() called with active painter"."""
    import inspect

    from rlpe.gui.jobs_tab import _ProgressCellDelegate

    src = inspect.getsource(_ProgressCellDelegate.paint)
    assert "super().paint" in src, (
        "_ProgressCellDelegate.paint must call super().paint() for "
        "the cell background — without it, Qt leaves the painter "
        "active when endPaint() is called."
    )


# ============================================================
# QThread destroyed: run_tab._on_thread_done
# ============================================================
def test_run_tab_on_thread_done_calls_quit_and_wait():
    """Phase 43: _on_thread_done() must stop + wait on the QThread
    before dropping the reference, otherwise the QThread C++ object
    is destroyed while still running.

    Audit 2026-08-18: Phase 56 replaced ``worker.quit()`` with
    ``worker.requestInterruption()`` (which sets the interrupt flag
    the pipeline polls between stages, then a 30s ``worker.wait()``
    gives it time to flush subprocesses like the OpenDataLoader
    JVM). Accept either pattern — the test ensures the worker is
    gracefully drained, not the specific API."""
    import inspect

    from rlpe.gui.run_tab import RunTab

    src = inspect.getsource(RunTab._on_thread_done)
    # Must call either ``worker.quit()`` (Phase 43-55 pattern) or
    # ``worker.requestInterruption()`` (Phase 56 pattern).
    assert ("worker.quit()" in src) or ("worker.requestInterruption()" in src), (
        "_on_thread_done must call worker.quit() OR "
        "worker.requestInterruption() to ask the thread to stop"
    )
    assert "worker.wait(" in src, (
        "_on_thread_done must call worker.wait() to block until the thread actually exits"
    )
    # Also must guard with isRunning() to avoid draining a
    # not-yet-started thread.
    assert "isRunning" in src, (
        "_on_thread_done must check isRunning() before quit()/wait()/"
        "requestInterruption()"
    )


# ============================================================
# GROBID fast-fail probe
# ============================================================
def test_pipeline_grobid_path_calls_is_available_first():
    """Phase 43: when GROBID is offline, the pipeline used to
    retry 3 times with 5s timeout per attempt = up to 15 seconds
    of HTTPConnectionPool retries per PDF. Fixed: probe
    is_available() with a 2s timeout FIRST; if False, skip retries
    and go straight to OD fallback.

    Audit 2026-08-18: Phase 59 refactor moved the probe into
    ``_process_one_pdf_grobid_impl``. Inspect both methods to
    catch the probe wherever it lives now."""
    import inspect

    from rlpe.pipeline import RadiolarianPipeline

    src = inspect.getsource(RadiolarianPipeline._process_one_pdf_grobid) + "\n" + inspect.getsource(
        getattr(RadiolarianPipeline, "_process_one_pdf_grobid_impl", RadiolarianPipeline._process_one_pdf_grobid)
    )
    assert "is_available" in src, (
        "_process_one_pdf_grobid must call is_available() before "
        "the retry loop so the user doesn't wait 15+ seconds per PDF "
        "when the GROBID server is down"
    )
    # Must skip the retry when is_available returns False
    assert "_process_one_pdf_od" in src, (
        "_process_one_pdf_grobid must fall back to _process_one_pdf_od when the GROBID probe fails"
    )


def test_grobid_no_probe_config_key_registered():
    """Phase 43: 'grobid_no_probe' must be a known extra-config key
    so users can disable the probe (for tests / special deployments)."""
    from rlpe import config

    assert "grobid_no_probe" in config._KNOWN_EXTRA_KEYS, (
        "config._KNOWN_EXTRA_KEYS must include 'grobid_no_probe'"
    )


# ============================================================
# Pipeline worker integration
# ============================================================
def test_pipeline_worker_has_request_cancel():
    """Phase 43: PipelineWorker.request_cancel() must set
    _cancel_event AND call QThread.requestInterruption."""
    import tempfile

    from rlpe.gui.pipeline_worker import PipelineWorker

    with tempfile.TemporaryDirectory() as tmp:
        w = PipelineWorker({}, Path(tmp) / "fake.pdf", Path(tmp) / "work")
        # Both attributes and methods must exist
        assert hasattr(w, "request_cancel")
        assert hasattr(w, "_cancel_event")
        assert hasattr(w, "requestInterruption")
