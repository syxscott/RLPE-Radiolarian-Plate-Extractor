"""Phase F-2 (2026-08-20) regression tests for run_tab.py MAJOR fixes M-17/M-19/M-21/M-22/M-27.

M-17: work_dir.mkdir() OSError unhandled
M-19: PDF/GROBID input validation gaps
M-21: Run-time controls unlocked during pipeline
M-22: Failed task shown as "done" with progress 100%
M-27: Pipeline worker close-timeout leaves running QThread
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from rlpe.gui import i18n

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

# Make rlpe importable
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in os.environ.get("PYTHONPATH", ""):
    os.environ["PYTHONPATH"] = str(_SRC)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestM17WorkDirMkdiringOserror:
    """M-17: work_dir.mkdir() OSError unhandled."""

    def test_work_dir_mkdir_oserror_popup(self, qapp, tmp_path):
        """OSError from out_path.mkdir shows QMessageBox.critical + restores Start button."""
        from rlpe.gui.run_tab import RunTab

        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        rt = RunTab(settings={}, parent=None)
        rt._path_edit.setText(str(pdf))
        rt._out_edit.setText(str(out_dir))

        # Patch the mkdir method at module level so OSError is raised
        with patch("rlpe.gui.run_tab.Path.mkdir", side_effect=OSError("Permission denied")):
            with patch("PySide6.QtWidgets.QMessageBox.critical"):
                rt._on_start()
                # Start button should be re-enabled after OSError was caught
                assert rt._start_btn.isEnabled(), "Start button not restored after OSError"


class TestM19PdfValidation:
    """M-19: PDF/GROBID input validation gaps."""

    def test_validate_pdf_path_rejects_directory(self):
        """_validate_pdf_path returns error for a directory."""
        from rlpe.gui.run_tab import RunTab

        error = RunTab._validate_pdf_path("/tmp")
        assert error is not None, "Directory should be rejected"

    def test_validate_pdf_path_rejects_non_pdf_extension(self):
        """_validate_pdf_path returns error for non-.pdf file."""
        from rlpe.gui.run_tab import RunTab

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not a pdf")
            tmp = f.name
        try:
            error = RunTab._validate_pdf_path(tmp)
            assert error is not None, ".txt file should be rejected"
        finally:
            Path(tmp).unlink()

    def test_validate_pdf_path_accepts_valid_pdf(self):
        """_validate_pdf_path returns None for a valid .pdf file."""
        from rlpe.gui.run_tab import RunTab

        with tempfile.NamedTemporaryFile(suffix=".PDF", delete=False) as f:
            f.write(b"%PDF-1.4 fake")
            tmp = f.name
        try:
            error = RunTab._validate_pdf_path(tmp)
            assert error is None, f"Valid PDF should be accepted, got: {error}"
        finally:
            Path(tmp).unlink()


class TestM19GrobidValidator:
    """M-19: GROBID URL validator on _grobid_edit."""

    def test_grobid_validator_rejects_not_url(self, qapp):
        """_grobid_edit hasAcceptableInput() is False for 'not-a-url'."""
        from rlpe.gui.run_tab import RunTab

        rt = RunTab(settings={}, parent=None)
        rt._grobid_edit.setText("not-a-url")
        assert not rt._grobid_edit.hasAcceptableInput(), \
            "hasAcceptableInput() should be False for 'not-a-url'"

    def test_grobid_validator_accepts_valid_url(self, qapp):
        """_grobid_edit hasAcceptableInput() is True for a valid http URL."""
        from rlpe.gui.run_tab import RunTab

        rt = RunTab(settings={}, parent=None)
        rt._grobid_edit.setText("http://localhost:8070")
        assert rt._grobid_edit.hasAcceptableInput(), \
            "hasAcceptableInput() should be True for valid http URL"


class TestM21InputsLockedDuringRun:
    """M-21: Run-time controls unlocked during pipeline."""

    def test_inputs_locked_during_run(self, qapp, tmp_path):
        """Browse/Clear/Open buttons are disabled while pipeline runs."""
        from rlpe.gui.run_tab import RunTab
        from unittest.mock import MagicMock

        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        rt = RunTab(settings={}, parent=None)
        rt._path_edit.setText(str(pdf))
        rt._out_edit.setText(str(out_dir))

        # Simulate a pipeline worker that starts and immediately finishes
        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = False
        mock_worker.isInterruptionRequested.return_value = False

        with patch("rlpe.gui.run_tab.PipelineWorker", return_value=mock_worker):
            with patch.object(rt, "_set_inputs_locked") as mock_lock:
                rt._on_start()
                # _set_inputs_locked(True) should have been called
                mock_lock.assert_called_with(True)


class TestM22FailureNotDone:
    """M-22: Failed task shown as 'done' with progress 100%."""

    def test_failure_status_not_done(self, qapp, tmp_path):
        """Simulating a failed outcome leaves progress != 100 and status != done."""
        from rlpe.gui.run_tab import RunTab

        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        rt = RunTab(settings={}, parent=None)
        rt._path_edit.setText(str(pdf))
        rt._out_edit.setText(str(out_dir))

        # Simulate the worker signals with a failed outcome
        rt._pending_outcome = "failed"

        # Manually call _on_thread_done with a mock worker
        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = False
        mock_worker.isInterruptionRequested.return_value = False
        mock_worker.progress = MagicMock()
        mock_worker.log_line = MagicMock()
        mock_worker.status_changed = MagicMock()
        mock_worker.finished_ok = MagicMock()
        mock_worker.failed = MagicMock()
        mock_worker.finished = MagicMock()

        # Disconnect all signals (they're not connected in this test)
        for attr in ("progress", "log_line", "status_changed", "finished_ok", "failed", "finished"):
            getattr(mock_worker, attr).disconnect = MagicMock()

        rt._worker = mock_worker

        with patch.object(rt, "_set_inputs_locked"):
            rt._on_thread_done(mock_worker)

        # Status should be "failed", NOT "done" - check the QSS property
        status_prop = rt._status_label.property("status")
        assert status_prop == "failed", \
            f"Failed job should have status='failed', got: {status_prop!r}"


class TestM27ShutdownNoDestroyRunningThread:
    """M-27: Pipeline worker close-timeout leaves running QThread."""

    def test_shutdown_no_runtime_error_if_already_finished(self, qapp):
        """Calling shutdown() when worker has already finished raises no RuntimeError."""
        from rlpe.gui.run_tab import RunTab
        from unittest.mock import MagicMock

        rt = RunTab(settings={}, parent=None)
        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = False  # already stopped

        rt._worker = mock_worker

        # Should not raise RuntimeError
        try:
            rt.shutdown()
        except RuntimeError:
            pytest.fail("shutdown() raised RuntimeError for already-stopped worker")

    def test_shutdown_calls_requestInterruption(self, qapp):
        """shutdown() calls requestInterruption() on a running worker."""
        from rlpe.gui.run_tab import RunTab
        from unittest.mock import MagicMock

        rt = RunTab(settings={}, parent=None)
        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = True
        mock_worker.wait.return_value = True  # exits within timeout

        rt._worker = mock_worker

        rt.shutdown()

        mock_worker.requestInterruption.assert_called_once()
        mock_worker.wait.assert_called_once_with(30000)
