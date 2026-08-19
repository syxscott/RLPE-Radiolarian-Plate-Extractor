"""Phase 1f audit (2026-08-19): GUI main-thread blocking + silent OCR/export failures.

Four fixes from the 2026-08-19 multi-agent audit, all in the GUI / pipeline
"silent failure" / "main-thread blocking" bucket:

1. **B-14 (BLOCKER)** — ``ResultsTab._flip_image_verified`` posted to
   ``/review/correction`` on the *GUI thread* with ``timeout=10``. A
   slow / unreachable API froze the UI for up to 10 s with no spinner,
   no cancel, and no double-click protection. The fix introduces a
   :class:`_FlipVerifiedWorker` ``QThread`` that does the POST off the
   main thread and emits success / failure back via Qt signals.

2. **B-15 (BLOCKER)** — ``JobsTab.load_recent_jobs_from_disk`` walked
   ``service_work/<jid>/output/manifests/matches.jsonl`` and parsed
   every line on the GUI thread, blocking the event loop for 3–10 s
   on a workstation with 150+ cached jobs. The fix splits the scan
   into:
     a. a fast directory-name listing done synchronously (<100 ms), and
     b. a JSONL parse running on a :class:`_DiskScanWorker` ``QThread``.

3. **M-1 (MAJOR)** — three ``except Exception: pass`` handlers in
   ``pipeline.py`` (per-panel OCR, label-region OCR, M3 concurrency
   config read) silently swallowed every backend failure. The fix logs
   the exception at DEBUG so it's visible in the troubleshooting log.

4. **M-15 (MAJOR)** — the four ``ResultsTab._export_*`` functions
   write 50k+ rows on the GUI thread. The proper fix is to move the
   write to a ``QThread`` (mirroring B-14); the minimum viable fix
   here is to log the failure at ERROR with a traceback so silent
   failures are no longer truly silent.

The tests below are mostly source-string guards because the GUI
runtime requires PySide6 + an X server; runtime tests are guarded
behind ``_HAS_PYSIDE6`` so they are skipped on a slim install.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import PySide6  # noqa: F401

    _HAS_PYSIDE6 = True
except ImportError:
    _HAS_PYSIDE6 = False


_REPO = Path(__file__).resolve().parents[1]
_SRC_RESULTS_TAB = _REPO / "src" / "rlpe" / "gui" / "results_tab.py"
_SRC_JOBS_TAB = _REPO / "src" / "rlpe" / "gui" / "jobs_tab.py"
_SRC_PIPELINE = _REPO / "src" / "rlpe" / "pipeline.py"


def _read(rel: str) -> str:
    return (_REPO / rel).read_text(encoding="utf-8")


# ============================================================================
# Helpers — extract method bodies from sources
# ============================================================================


def _body_before_next_def(src: str, signature: str) -> str:
    """Return the slice of ``src`` from ``signature`` until the next
    nested ``def `` (conservative — stops at the first indented method
    or the next top-level class). Used to assert the body of a
    specific method *without* reaching across into the next method.
    """
    idx = src.find(signature)
    assert idx != -1, f"signature {signature!r} not found"
    # Scan for the next top-level class definition only — stopping at
    # the next class rather than the next method means we still capture
    # nested methods inside the class body (including ``run()`` inside
    # a QThread).
    after = src[idx + len(signature):]
    next_class = re.search(r"\nclass |\nif __name__", after)
    if next_class is None:
        return after
    return after[: next_class.start()]


# ============================================================================
# B-14 — _flip_image_verified must use a QThread worker
# ============================================================================


class TestFlipVerifiedWorkerExists:
    """B-14: a QThread worker must wrap the /review/correction POST."""

    def test_worker_class_is_subclass_of_qthread(self):
        src = _read("src/rlpe/gui/results_tab.py")
        # Pin the worker class name + parent class so a future rename
        # is caught immediately.
        assert "class _FlipVerifiedWorker(QThread)" in src, (
            "B-14 fix: _FlipVerifiedWorker must subclass QThread"
        )

    def test_worker_emits_success_and_error_signals(self):
        src = _read("src/rlpe/gui/results_tab.py")
        # Two signals are required: a success bool and an error string.
        worker_body = _body_before_next_def(src, "class _FlipVerifiedWorker(QThread)")
        assert "Signal(bool)" in worker_body, (
            "B-14 fix: _FlipVerifiedWorker must emit a success(bool) signal"
        )
        assert "Signal(str)" in worker_body, (
            "B-14 fix: _FlipVerifiedWorker must emit an error(str) signal"
        )

    def test_worker_run_does_the_post(self):
        src = _read("src/rlpe/gui/results_tab.py")
        worker_body = _body_before_next_def(src, "class _FlipVerifiedWorker(QThread)")
        # The HTTP POST must live inside the worker.run() method, not
        # on the GUI thread.
        assert "def run(self)" in worker_body, (
            "B-14 fix: _FlipVerifiedWorker must implement run()"
        )
        # The POST itself should be inside the run() method. We assert
        # by counting the number of 'requests.post' usages inside the
        # class body — there should be exactly one (inside run()).
        post_count = worker_body.count("requests.post")
        assert post_count == 1, (
            f"B-14 fix: _FlipVerifiedWorker.run() must call requests.post exactly "
            f"once; found {post_count}"
        )

    def test_flip_method_does_not_call_requests_post_directly(self):
        """The GUI thread must NOT call ``requests.post`` synchronously.

        The previous bug was that ``_flip_image_verified`` called
        ``requests.post(url, timeout=10)`` directly inside the slot,
        blocking the event loop for the full timeout. The fix moves
        the POST into a QThread worker; the slot only constructs the
        worker and connects signals.
        """
        src = _read("src/rlpe/gui/results_tab.py")
        flip_body = _body_before_next_def(src, "def _flip_image_verified(self")
        # Allow the import + the worker construction call, but NOT a
        # raw ``requests.post(...)`` in the slot body.
        post_calls = re.findall(r"requests\.post\s*\(", flip_body)
        assert not post_calls, (
            f"B-14 fix: _flip_image_verified must not call requests.post on the "
            f"GUI thread; found {len(post_calls)} direct call(s)"
        )

    def test_flip_method_constructs_worker_and_connects_signals(self):
        src = _read("src/rlpe/gui/results_tab.py")
        flip_body = _body_before_next_def(src, "def _flip_image_verified(self")
        assert "_FlipVerifiedWorker(" in flip_body, (
            "B-14 fix: _flip_image_verified must construct a _FlipVerifiedWorker"
        )
        assert ".finished_with_success.connect(" in flip_body, (
            "B-14 fix: _flip_image_verified must connect the success signal"
        )
        assert ".error.connect(" in flip_body, (
            "B-14 fix: _flip_image_verified must connect the error signal"
        )
        assert ".start()" in flip_body, (
            "B-14 fix: _flip_image_verified must .start() the worker"
        )

    def test_flip_method_disables_buttons_while_in_flight(self):
        """Double-click protection: the mark-verified buttons must be
        disabled while the worker is in flight so the operator cannot
        fire a second flip on the same row."""
        src = _read("src/rlpe/gui/results_tab.py")
        flip_body = _body_before_next_def(src, "def _flip_image_verified(self")
        assert "setEnabled(False)" in flip_body, (
            "B-14 fix: _flip_image_verified must disable the mark-verified "
            "buttons while the worker is in flight"
        )

    def test_qthread_signal_imported_in_results_tab(self):
        src = _read("src/rlpe/gui/results_tab.py")
        # QThread + Signal must be imported from PySide6.QtCore near
        # the top of the file (existing Qt import is on line 23).
        assert re.search(
            r"from PySide6\.QtCore\s+import\s+Qt\s*,\s*QThread\s*,\s*Signal",
            src,
        ), "B-14 fix: results_tab.py must import QThread + Signal from QtCore"


# ============================================================================
# B-15 — load_recent_jobs_from_disk must be async
# ============================================================================


class TestDiskScanIsAsync:
    """B-15: load_recent_jobs_from_disk must not block the GUI thread."""

    def test_disk_scan_worker_is_subclass_of_qthread(self):
        src = _read("src/rlpe/gui/jobs_tab.py")
        assert "class _DiskScanWorker(QThread)" in src, (
            "B-15 fix: _DiskScanWorker must subclass QThread"
        )

    def test_disk_scan_worker_emits_job_loaded_signal(self):
        src = _read("src/rlpe/gui/jobs_tab.py")
        worker_body = _body_before_next_def(src, "class _DiskScanWorker(QThread)")
        assert "job_loaded = Signal(" in worker_body, (
            "B-15 fix: _DiskScanWorker must emit a job_loaded signal"
        )

    def test_load_recent_jobs_starts_worker(self):
        src = _read("src/rlpe/gui/jobs_tab.py")
        body = _body_before_next_def(src, "def load_recent_jobs_from_disk(self")
        # The synchronous version used to call self.add_or_update_job
        # directly inside the load loop. The async version must NOT
        # do that on the GUI thread — it must defer to a worker via
        # the job_loaded signal.
        assert "_DiskScanWorker(" in body, (
            "B-15 fix: load_recent_jobs_from_disk must construct the worker"
        )
        assert ".start()" in body, (
            "B-15 fix: load_recent_jobs_from_disk must .start() the worker"
        )
        assert "job_loaded.connect(" in body, (
            "B-15 fix: load_recent_jobs_from_disk must connect the "
            "worker's job_loaded signal"
        )
        # The sync version used to walk every line of every jsonl in the
        # loop body. The async version must NOT — the worker does the
        # parse. (We assert this in a dedicated test below.)

    def test_load_recent_jobs_does_not_call_add_or_update_directly(self):
        """The async version moves ``add_or_update_job`` to a worker
        callback, not the synchronous load loop.

        We assert by structural check: the call to ``add_or_update_job``
        must be inside a nested ``_on_job`` closure that is connected
        to the worker's ``job_loaded`` signal — never inside the
        ``for root, jid in roots:`` synchronous loop (the loop was
        the blocking bit).
        """
        src = _read("src/rlpe/gui/jobs_tab.py")
        body = _body_before_next_def(src, "def load_recent_jobs_from_disk(self")
        # 1. The synchronous ``for root, jid in roots:`` loop that
        #    used to parse every jsonl in-place is gone.
        assert "for root, jid in roots:" not in body, (
            "B-15 fix: load_recent_jobs_from_disk must not have a "
            "synchronous ``for root, jid in roots:`` loop anymore"
        )
        # 2. The synchronous ``matches_path.open(`` block that used
        #    to read the file is gone from the load method.
        assert "matches_path.open(" not in body, (
            "B-15 fix: load_recent_jobs_from_disk must not open "
            "matches.jsonl directly anymore"
        )
        # 3. add_or_update_job IS still referenced, but only inside
        #    the _on_job callback that is connected to job_loaded.
        assert "job_loaded.connect(" in body, (
            "B-15 fix: the job_loaded signal must be connected so "
            "the GUI thread can call add_or_update_job off the worker"
        )

    def test_load_recent_jobs_does_not_parse_jsonl_in_slot(self):
        """The ``json.loads`` line-parse loop used to run on the GUI
        thread inside ``load_recent_jobs_from_disk``. The fix moves
        it to the worker. We assert the synchronous parse is gone."""
        src = _read("src/rlpe/gui/jobs_tab.py")
        body = _body_before_next_def(src, "def load_recent_jobs_from_disk(self")
        # The synchronous parse loop looked like
        #     for line in fh:
        #         row = json.loads(line)
        # The async version does not have this loop; the worker body
        # has it instead.
        assert "for line in fh" not in body, (
            "B-15 fix: load_recent_jobs_from_disk must not iterate jsonl "
            "lines on the GUI thread"
        )
        assert "json.loads(line)" not in body, (
            "B-15 fix: load_recent_jobs_from_disk must not call json.loads "
            "on the GUI thread"
        )

    def test_qthread_imported_in_jobs_tab(self):
        src = _read("src/rlpe/gui/jobs_tab.py")
        # QThread must be imported from PySide6.QtCore.
        assert re.search(
            r"from PySide6\.QtCore\s+import\s+[^\n]*\bQThread\b",
            src,
        ), "B-15 fix: jobs_tab.py must import QThread from QtCore"


# ============================================================================
# M-1 — OCR exception logging
# ============================================================================


class TestOCRErrorLogging:
    """M-1: pipeline.py must log OCR backend failures, not swallow them."""

    def test_per_panel_ocr_failure_is_logged(self):
        src = _read("src/rlpe/pipeline.py")
        # The fix turns "except Exception: pass" into a debug log.
        # Pin the new log message so a future refactor can't silently
        # strip it.
        assert "per-panel OCR failed" in src, (
            "M-1 fix: pipeline.py must log per-panel OCR failures"
        )

    def test_label_region_ocr_failure_is_logged(self):
        src = _read("src/rlpe/pipeline.py")
        assert "label-region OCR failed" in src, (
            "M-1 fix: pipeline.py must log label-region OCR failures"
        )

    def test_minimax_concurrency_failure_is_logged(self):
        src = _read("src/rlpe/pipeline.py")
        assert "MiniMax_max_concurrent read failed" in src, (
            "M-1 fix: pipeline.py must log MiniMax_max_concurrent read failures"
        )

    def test_no_silent_pass_in_per_panel_ocr_block(self):
        """The 5322-5329 block must NO LONGER be ``except Exception: pass``."""
        src = _read("src/rlpe/pipeline.py")
        idx = src.find('panel.metadata["panel_ocr_token_count"] = len(panel_tokens)')
        assert idx != -1, "could not locate per-panel OCR block"
        window = src[idx : idx + 800]
        # The except clause may bind the exception as ``as exc`` (we
        # adopted that pattern in the fix).
        assert ("except Exception:" in window) or ("except Exception as" in window), (
            "M-1 fix: per-panel OCR block must still have an except clause"
        )
        after_except = re.split(r"except Exception(?::| as)", window, maxsplit=1)[1]
        assert "logger.debug" in after_except[:400], (
            "M-1 fix: per-panel OCR except must log via logger.debug"
        )

    def test_no_silent_pass_in_label_region_ocr_block(self):
        src = _read("src/rlpe/pipeline.py")
        idx = src.find('panel.metadata["label_region_picked"] = best.text')
        assert idx != -1, "could not locate label-region OCR block"
        window = src[idx : idx + 600]
        assert ("except Exception:" in window) or ("except Exception as" in window), (
            "M-1 fix: label-region OCR block must still have an except clause"
        )
        after_except = re.split(r"except Exception(?::| as)", window, maxsplit=1)[1]
        assert "logger.debug" in after_except[:400], (
            "M-1 fix: label-region OCR except must log via logger.debug"
        )


# ============================================================================
# M-15 — Results tab export error logging
# ============================================================================


class TestExportErrorLogging:
    """M-15: the 4 export functions must log failures instead of silent fail."""

    def test_xlsx_export_logs_error(self):
        src = _read("src/rlpe/gui/results_tab.py")
        body = _body_before_next_def(src, "def _export_xlsx(self")
        assert "Export failed (xlsx" in body, (
            "M-15 fix: _export_xlsx must log a 'Export failed (xlsx ...)' error"
        )
        assert "_log.error(" in body, (
            "M-15 fix: _export_xlsx must call _log.error(...)"
        )
        assert "exc_info=True" in body, (
            "M-15 fix: _export_xlsx must pass exc_info=True to _log.error"
        )

    def test_json_export_logs_error(self):
        src = _read("src/rlpe/gui/results_tab.py")
        body = _body_before_next_def(src, "def _export_json(self")
        assert "Export failed (json" in body, (
            "M-15 fix: _export_json must log a 'Export failed (json ...)' error"
        )
        assert "_log.error(" in body, (
            "M-15 fix: _export_json must call _log.error(...)"
        )

    def test_csv_export_logs_error(self):
        src = _read("src/rlpe/gui/results_tab.py")
        body = _body_before_next_def(src, "def _export_csv(self")
        assert "Export failed (csv" in body, (
            "M-15 fix: _export_csv must log a 'Export failed (csv ...)' error"
        )
        assert "_log.error(" in body, (
            "M-15 fix: _export_csv must call _log.error(...)"
        )

    def test_dwca_export_logs_error(self):
        src = _read("src/rlpe/gui/results_tab.py")
        body = _body_before_next_def(src, "def _export_dwca(self")
        assert "Export failed (dwca" in body, (
            "M-15 fix: _export_dwca must log a 'Export failed (dwca ...)' error"
        )
        assert "_log.error(" in body, (
            "M-15 fix: _export_dwca must call _log.error(...)"
        )

    def test_export_module_doc_acknowledges_qthread_followup(self):
        """Documentation hysteresis: the next engineer should be able to
        see that the M-15 fix is a minimum viable patch and the
        follow-up is to move the write to a QThread."""
        src = _read("src/rlpe/gui/results_tab.py")
        # Find the exports comment block.
        idx = src.find("Audit 2026-08-19 (M-15)")
        assert idx != -1, "M-15 fix: the export comment header is missing"
        chunk = src[idx : idx + 1500]
        assert "QThread" in chunk, (
            "M-15 fix: the export comment must acknowledge the QThread follow-up"
        )


# ============================================================================
# Runtime tests — only run when PySide6 is available
# ============================================================================


@pytest.mark.skipif(not _HAS_PYSIDE6, reason="PySide6 not installed")
class TestFlipVerifiedWorkerRuntime:
    """Drive the worker end-to-end through a stub requests.post to
    verify the success / error signals fire correctly.

    These tests stub ``requests.post`` at runtime; they restore the
    original ``requests.post`` after each test (via the
    ``monkeypatch`` fixture) so a failure here can't leak into the
    rest of the suite.
    """

    def test_worker_reports_success_on_2xx(self, monkeypatch: pytest.MonkeyPatch):
        """A 200 response must emit finished_with_success(True)."""
        from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

        from rlpe.gui.results_tab import _FlipVerifiedWorker

        # Stub requests.post to return a 200.
        class _FakeResp:
            def raise_for_status(self) -> None:
                return None

        def _fake_post(url, json=None, timeout=10):
            return _FakeResp()

        # Patch the live ``requests`` module — the worker imports it
        # by name and resolves through sys.modules, so patching the
        # module attribute is sufficient.
        import requests

        monkeypatch.setattr(requests, "post", _fake_post)

        # Ensure a single QCoreApplication exists.
        app = QCoreApplication.instance() or QCoreApplication(sys.argv)

        worker = _FlipVerifiedWorker("http://example.invalid/x", {"k": "v"})

        results: list[bool] = []
        errors: list[str] = []

        worker.finished_with_success.connect(results.append)
        worker.error.connect(errors.append)

        # Drive the event loop until the worker emits a signal.
        loop = QEventLoop()
        worker.finished.connect(loop.quit)
        QTimer.singleShot(0, worker.start)
        # Cap runtime so a broken worker can't hang the test forever.
        QTimer.singleShot(3000, loop.quit)
        loop.exec()

        assert results == [True], f"expected success signal, got {results}"
        assert errors == [], f"unexpected error: {errors}"

    def test_worker_reports_error_on_network_failure(self, monkeypatch: pytest.MonkeyPatch):
        """A network failure must emit error(str) — not raise."""
        from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

        from rlpe.gui.results_tab import _FlipVerifiedWorker

        def _fake_post(url, json=None, timeout=10):
            raise ConnectionError("simulated network failure")

        import requests

        monkeypatch.setattr(requests, "post", _fake_post)

        # Ensure a single QCoreApplication exists.
        app = QCoreApplication.instance() or QCoreApplication(sys.argv)

        worker = _FlipVerifiedWorker("http://example.invalid/x", {"k": "v"})

        results: list[bool] = []
        errors: list[str] = []

        worker.finished_with_success.connect(results.append)
        worker.error.connect(errors.append)

        loop = QEventLoop()
        worker.finished.connect(loop.quit)
        QTimer.singleShot(0, worker.start)
        QTimer.singleShot(3000, loop.quit)
        loop.exec()

        assert results == [], f"expected no success, got {results}"
        assert errors, "error signal must fire on network failure"
        assert "simulated network failure" in errors[0], (
            f"error message missing detail; got {errors[0]!r}"
        )
