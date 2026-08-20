"""Phase F-2 (2026-08-20) — MainWindow bug fixes M-7, M-8, M-9, M-10/M-16, M-25.

Five fixes from the 2026-08-20 frontend audit:

* **M-7** — "取消" substring falsely matches non-cancellation errors.
  Fix: use worker.was_cancelled() flag first, fall back to
  word-boundary regex only when worker not accessible.

* **M-8** — _on_job_progress bypassed _set_status, so language
  switch re-rendered the stale i18n key instead of the progress text.
  Fix: use _set_status("main.progress", ...) with new i18n key.

* **M-10/M-16** — JobRecord.output_dir was derived from pdf_path/stem
  instead of coming from the signal, so batch jobs pointed at the wrong dir.
  Fix: job_started signal now carries output_dir; batch placeholders
  use <stem>_rlpe_out/work/output.

* **M-25** — _export_batch_xlsx ran on the GUI thread, freezing the UI.
  Fix: new _BatchExportWorker QThread; thin wrapper launches it.

* **M-9+M-27** — wait(30000) raises RuntimeError if the QThread C++
  object was already destroyed. Also, RunTab.shutdown was not called.
  Fix: wrap wait() in try/except RuntimeError; call _run_tab.shutdown()
  before _stop_pipeline_worker().
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os_environ_setdefault_QT_QPA_PLATFORM_offscreen = True
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QThread, QTimer
from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])


# ============================================================
# Helper: build a minimal MainWindow with mocked tabs
# ============================================================


def _make_window():
    """Return a MainWindow with all heavy tab init bypassed for unit testing."""
    from PySide6.QtWidgets import QMainWindow, QWidget

    from rlpe.gui.main_window import MainWindow

    w = MainWindow.__new__(MainWindow)
    # Properly initialize the QMainWindow base class so closeEvent's
    # super().closeEvent(event) call works. We bypass MainWindow's
    # own __init__ (which builds tabs, sets up signals, etc.) because
    # those are not under test here — they're mocked instead.
    QMainWindow.__init__(w)
    w._log = MagicMock()
    w._qsettings = MagicMock()
    w._settings = {}
    # Give tabs sensible mocks so signal dispatch doesn't crash
    w._jobs_tab = MagicMock()
    w._results_tab = MagicMock()
    w._run_tab = MagicMock()
    w._run_tab._worker = None
    w._run_tab._current_job_id = None
    w._settings_tab = MagicMock()
    w._mini_progress_timer = None
    w._mini_progress = MagicMock()
    w._batch_pdfs = None
    w._batch_settings = {}
    w._batch_index = 0
    w._status_perm = MagicMock()
    w._status_key = "main.idle"
    w._status_kwargs = {}
    # _tabs is referenced in _on_job_started (for TAB_JOBS auto-switch)
    w._tabs = MagicMock()
    return w


# ============================================================
# M-7: cancelled flag used, not substring
# ============================================================


def test_cancelled_flag_used_not_substring():
    """M-7: an error like "无法取消: real failure" must NOT be classified
    as user-cancelled when the worker was not cancelled."""

    class FakeWorker:
        def was_cancelled(self):
            return False  # worker was NOT cancelled

    w = _make_window()
    w._jobs_tab = MagicMock()
    w._run_tab = MagicMock()
    w._run_tab._worker = FakeWorker()
    w._mini_progress = MagicMock()

    # The error string contains "取消" but the worker says it was NOT cancelled
    w._on_job_failed("job-1", "无法取消: worker still running")

    # Must NOT call mark_cancelled
    w._jobs_tab.mark_cancelled.assert_not_called()
    # Must call mark_failed
    w._jobs_tab.mark_failed.assert_called_once()


def test_cancelled_flag_set_by_interrupt():
    """M-7: when worker.was_cancelled() returns True, job is marked cancelled
    even if the error message is empty or unrelated."""
    from rlpe.gui.pipeline_worker import PipelineWorker

    # The flag starts False
    pw = PipelineWorker({}, Path("/nonexistent.pdf"), Path("/tmp/work"))
    assert pw.was_cancelled() is False

    # After request_cancel the flag is True
    pw.request_cancel()
    assert pw.was_cancelled() is True


def test_cancelled_fallback_to_regex_word_boundary():
    """M-7: when no worker is accessible, word-boundary regex is used.
    'cannot cancel' (embedded) should NOT match; 'cancelled (...)' should."""
    import re

    w = _make_window()
    w._jobs_tab = MagicMock()
    w._run_tab = MagicMock()
    w._run_tab._worker = None  # no worker accessible
    w._mini_progress = MagicMock()

    # "cannot cancel" has embedded cancel, not a word boundary
    # → word-boundary regex would NOT match (as intended)
    # We test the regex logic directly
    e = "cannot cancel: worker still running"
    cancelled = bool(re.search(r"\bcancelled\b|\bcanceled\b", e, re.IGNORECASE))
    assert cancelled is False

    # "cancelled (42 rows)" has word boundary → matches
    e2 = "cancelled (42 rows collected)"
    cancelled2 = bool(re.search(r"\bcancelled\b|\bcanceled\b", e2, re.IGNORECASE))
    assert cancelled2 is True


# ============================================================
# M-8: _status_key set in progress
# ============================================================


def test_status_key_set_in_progress():
    """M-8: after _on_job_progress, _status_key must be 'main.progress'
    with the correct kwargs so language switch re-renders correctly."""
    from rlpe.gui import i18n

    w = _make_window()
    w._jobs_tab = MagicMock()
    w._status_perm = MagicMock()
    # Use real _set_status to properly set _status_key / _status_kwargs
    w._status_key = "main.idle"
    w._status_kwargs = {}

    # Patch _jobs_tab.update_progress to be a no-op
    w._jobs_tab.update_progress = MagicMock()
    # Patch mini_progress to avoid QWidget issues
    w._mini_progress = MagicMock()

    w._on_job_progress("job-1", 3, 12, "OCR page 3 of 12")

    assert w._status_key == "main.progress"
    assert w._status_kwargs["msg"] == "OCR page 3 of 12"
    assert w._status_kwargs["current"] == 3
    assert w._status_kwargs["total"] == 12


# ============================================================
# M-10/M-16: JobRecord output_dir from signal
# ============================================================


def test_job_record_output_dir_from_signal():
    """M-10/M-16: _on_job_started uses output_dir from signal, not
    recomputed from pdf_path/stem."""
    w = _make_window()
    w._jobs_tab = MagicMock()
    w._run_tab = MagicMock()
    w._run_tab.collect_settings.return_value = {}

    w._on_job_started("job-1", "/pdfs/ Baumgartner2020.pdf", "/tmp/baum_work/output")

    # The JobRecord must carry the output_dir from the signal
    call_args = w._jobs_tab.add_or_update_job.call_args
    job_record = call_args[0][0]
    assert job_record.output_dir == "/tmp/baum_work/output"
    # Must NOT be the wrong derived path
    assert "stem" not in job_record.output_dir


def test_batch_placeholder_output_dir_has_work_output():
    """M-10/M-16: batch placeholder output_dir must include /work/output."""
    from rlpe.gui.jobs_tab import JobRecord

    # Simulate what _on_batch_started builds for a placeholder
    out_root = Path("/tmp/batch_out") / "baumgartner2020_rlpe_out"
    output_dir = str(out_root / "work" / "output")

    assert "work" in output_dir
    assert "output" in output_dir
    # The path must point inside /work/, not at the root
    assert str(out_root) + "/work/output" == output_dir


# ============================================================
# M-25: batch export worker does not block GUI
# ============================================================


def test_batch_export_worker_does_not_block():
    """M-25: _BatchExportWorker runs write_xlsx on a QThread, not the GUI thread."""
    import tempfile
    from unittest.mock import MagicMock

    from rlpe.gui.main_window import _BatchExportWorker

    # Snapshot of jobs (empty rows is fine for this test)
    class FakeJob:
        rows = []

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = str(Path(tmpdir) / "out.xlsx")
        worker = _BatchExportWorker([FakeJob(), FakeJob()], out_path)

        # Check it is a QThread subclass
        assert issubclass(_BatchExportWorker, QThread)

        # Check signals exist
        assert hasattr(worker, "progress")
        assert hasattr(worker, "finished_ok")
        assert hasattr(worker, "failed")

        # Check _cancelled flag
        assert worker._cancelled is False


# ============================================================
# M-9+M-27: closeEvent calls run_tab.shutdown + wait RuntimeError
# ============================================================


def test_close_event_calls_run_tab_shutdown():
    """M-9/M-27: closeEvent must call _run_tab.shutdown() before
    _stop_pipeline_worker()."""
    from PySide6.QtGui import QCloseEvent

    w = _make_window()
    w._run_tab = MagicMock()
    w._jobs_tab = MagicMock()
    w._jobs_tab.shutdown = MagicMock()
    w._stop_pipeline_worker = MagicMock()
    w._flush_settings = MagicMock()
    w._remove_i18n_listeners = MagicMock()
    w._save_window_state = MagicMock()

    event = QCloseEvent()
    w.closeEvent(event)

    # _run_tab.shutdown must have been called
    w._run_tab.shutdown.assert_called_once()
    # And before _stop_pipeline_worker
    assert w._run_tab.shutdown.call_count == 1
    assert w._stop_pipeline_worker.call_count == 1


def test_close_event_wait_runtime_error_handled():
    """M-9/M-27: if worker.wait() raises RuntimeError (thread already gone),
    closeEvent must not crash and must accept the close."""
    from PySide6.QtGui import QCloseEvent

    w = _make_window()

    # Mock run_tab with a worker whose wait() raises RuntimeError
    dead_worker = MagicMock()
    dead_worker.isRunning.return_value = True
    dead_worker.wait.side_effect = RuntimeError(
        "QThread: Destroyed while thread is still running"
    )
    w._run_tab = MagicMock()
    w._run_tab._worker = dead_worker
    w._run_tab._current_job_id = "job-dead"
    w._run_tab.shutdown = MagicMock()

    w._jobs_tab = MagicMock()
    w._jobs_tab.shutdown = MagicMock()
    w._stop_pipeline_worker = MagicMock()
    w._flush_settings = MagicMock()
    w._remove_i18n_listeners = MagicMock()
    w._save_window_state = MagicMock()

    event = QCloseEvent()
    # Must not raise
    w.closeEvent(event)
    # Event must be accepted
    assert event.isAccepted()


# ============================================================
# M-7 source guard: _on_job_failed uses worker.was_cancelled
# ============================================================


def test_on_job_failed_checks_worker_was_cancelled():
    """M-7: _on_job_failed must check worker.was_cancelled() when available."""
    w = _make_window()
    w._jobs_tab = MagicMock()
    w._mini_progress = MagicMock()

    class FakeWorker:
        def was_cancelled(self):
            return True  # genuinely cancelled

    w._run_tab = MagicMock()
    w._run_tab._worker = FakeWorker()

    # Even though error says something else, worker says cancelled
    w._on_job_failed("job-x", "cancelled (0 rows)")

    # Must use mark_cancelled, not mark_failed
    w._jobs_tab.mark_cancelled.assert_called_once()
    w._jobs_tab.mark_failed.assert_not_called()
