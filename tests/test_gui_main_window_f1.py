"""Phase F-1 (2026-08-20) — MainWindow B-1 (close-QThread) + B-3 (async auto-open).

Two BLOCKER-level fixes from the 2026-08-20 frontend audit:

* **B-1** — closing the GUI while ``_DiskScanWorker`` (or the
  pipeline worker) was still running crashed with exit code 134
  (``QThread: Destroyed while thread is still running``). The fix
  adds ``JobsTab.shutdown()`` (interrupts + waits 30s) and wires
  it into ``MainWindow.closeEvent`` BEFORE the heavier pipeline
  worker shutdown.

* **B-3** — Phase 51's auto-open-to-Results-tab logic stopped
  firing once B-15 made the disk scan async. The previous code
  read ``self._jobs_tab._jobs`` synchronously, but the worker
  hadn't emitted anything yet, so the auto-open never fired.
  The fix adds ``JobsTab.scan_finished(records)`` and a new
  ``_on_disk_scan_done`` slot that runs AFTER the worker
  genuinely finishes.

These tests pin the new behaviour in ``main_window.py`` without
touching the other GUI files (which parallel agents are working
on). Source-string guards are used where the runtime path is
fragile (PySide6 QThread teardown can crash the interpreter if
the wait races the GC).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QThread, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def tmp_work_dirs(tmp_path, monkeypatch):
    """Synthetic project root with service_work/ + work/ subdirs.

    Three jobs are materialised on disk (``job-a`` with 2 rows,
    ``job-b`` with 1 row) plus a ``cli_<hash>`` job from the work/
    tree. Each job's ``output/manifests/complete.flag`` is created
    so the worker marks them as ``STATUS_DONE`` (without the flag
    the audit 2026-08-17 C1 honesty rule marks them as
    ``STATUS_FAILED`` to flag incomplete runs). Mirrors what the
    real pipeline writes at the end of a successful run.
    """
    import rlpe.gui.constants as consts

    monkeypatch.setattr(consts, "PROJECT_ROOT", tmp_path)

    job_a = tmp_path / "service_work" / "job-a" / "output" / "manifests"
    job_a.mkdir(parents=True)
    rows_a = [
        {"species": "Species A1", "panel_id": "job-a/fig1/p1", "page_index": 5},
        {"species": "Species A2", "panel_id": "job-a/fig1/p2", "page_index": 5},
    ]
    (job_a / "matches.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows_a) + "\n",
        encoding="utf-8",
    )
    (job_a / "complete.flag").write_text("", encoding="utf-8")

    job_b = tmp_path / "service_work" / "job-b" / "output" / "manifests"
    job_b.mkdir(parents=True)
    (job_b / "matches.jsonl").write_text(
        json.dumps({"species": "Species B1", "panel_id": "job-b/fig1/p1"}) + "\n",
        encoding="utf-8",
    )
    (job_b / "complete.flag").write_text("", encoding="utf-8")

    cli_dir = tmp_path / "work" / "output" / "manifests"
    cli_dir.mkdir(parents=True)
    (cli_dir / "matches.jsonl").write_text(
        json.dumps({"species": "Species CLI", "panel_id": "cli/p1"}) + "\n",
        encoding="utf-8",
    )
    (cli_dir / "complete.flag").write_text("", encoding="utf-8")
    return tmp_path


@pytest.fixture
def empty_work_dirs(tmp_path, monkeypatch):
    """Project root with NO service_work/ or work/ — fresh-install path."""
    import rlpe.gui.constants as consts

    monkeypatch.setattr(consts, "PROJECT_ROOT", tmp_path)
    return tmp_path


# ============================================================
# B-1: closeEvent shuts down the disk-scan worker gracefully
# ============================================================
class _SleepyThread(QThread):
    """A QThread that sleeps for a long time so we can race a close."""

    def __init__(self) -> None:
        super().__init__()
        self._stop = False

    def run(self) -> None:  # noqa: D401 - QThread contract
        # Sleep in small chunks so requestInterruption() can be observed
        # between iterations (a single long sleep would race the wait()).
        for _ in range(600):
            if self.isInterruptionRequested():
                return
            time.sleep(0.05)


class TestCloseEventStopsDiskScanWorker:
    """B-1: closing the GUI mid-scan must NOT leave a QThread running."""

    def _drain_real_worker(self, mw) -> None:
        """Drain the real ``_DiskScanWorker`` the MainWindow started.

        The MainWindow kicks off the async disk scan in ``__init__``;
        for the unit test we want to install a long-running sleeper
        *instead*, so we must wait for the real worker to finish
        first (otherwise the replacement would orphan the still-running
        QThread and crash on GC).
        """
        real = mw._jobs_tab._disk_scan_worker
        if real is None:
            return
        if real.isRunning():
            try:
                real.requestInterruption()
            except RuntimeError:
                pass
            try:
                real.wait(5000)
            except RuntimeError:
                pass
        # The real worker is finished; drop the reference so the
        # attribute can be safely reassigned.
        mw._jobs_tab._disk_scan_worker = None

    def test_close_event_stops_disk_scan_worker(self, tmp_work_dirs):
        """Build a MainWindow with a known long-running worker attached
        to ``_jobs_tab._disk_scan_worker``, then call ``closeEvent``
        directly. The worker must:

        1. Receive ``requestInterruption()`` (the loop exits at the
           next iteration).
        2. ``wait()`` for the thread to finish (within 30 s).
        3. Have its strong reference cleared so the worker can be
           GC'd normally — no SIGABRT / exit code 134.
        """
        from rlpe.gui.main_window import MainWindow

        mw = MainWindow()
        # Hold a local reference to the sleeper so it doesn't get
        # GC'd before the test exits. The test is asserting that
        # closeEvent's shutdown() drains the worker; if the local
        # variable goes out of scope before mw.close(), the QThread
        # destructor would race with the test fixture teardown.
        sleeper = _SleepyThread()
        try:
            # Drain the real worker first so we don't orphan the
            # still-running QThread when we replace it below.
            self._drain_real_worker(mw)
            # Install a long-running sleeper so closeEvent is forced
            # to interrupt + wait.
            mw._jobs_tab._disk_scan_worker = sleeper
            sleeper.start()
            # Let the thread actually enter run() so requestInterruption
            # has an effect.
            time.sleep(0.1)
            assert mw._jobs_tab._disk_scan_worker.isRunning(), (
                "sleepy thread should be running before closeEvent"
            )

            # closeEvent must NOT raise; in particular it must not
            # crash with RuntimeError / SIGABRT about a still-running
            # QThread.
            from PySide6.QtGui import QCloseEvent

            event = QCloseEvent()
            mw.closeEvent(event)

            # After closeEvent:
            # * the worker must have finished (or wait() timed out)
            # * the strong reference must be cleared so Qt can free it
            assert mw._jobs_tab._disk_scan_worker is None, (
                "Phase F-1 (B-1): shutdown() must clear "
                "_disk_scan_worker after the thread finishes"
            )
            # The sleeper itself must have finished (interrupted + wait).
            assert not sleeper.isRunning(), (
                "Phase F-1 (B-1): worker must have finished "
                "before closeEvent returns"
            )
        finally:
            # Belt-and-suspenders: belt belt. If closeEvent didn't
            # drain the worker (e.g. test failure), do it here so the
            # test fixture teardown doesn't crash with SIGABRT.
            if sleeper.isRunning():
                try:
                    sleeper.requestInterruption()
                except RuntimeError:
                    pass
                try:
                    sleeper.wait(2000)
                except RuntimeError:
                    pass
            mw.close()

    def test_close_event_handles_no_worker(self, empty_work_dirs):
        """If the worker is None (early load or already cleared),
        closeEvent must NOT raise — the shutdown path is a no-op."""
        from rlpe.gui.main_window import MainWindow

        mw = MainWindow()
        try:
            # Drain the real worker so the attribute can be safely
            # set back to None without orphaning the QThread.
            self._drain_real_worker(mw)
            mw._jobs_tab._disk_scan_worker = None
            from PySide6.QtGui import QCloseEvent

            event = QCloseEvent()
            mw.closeEvent(event)  # must not raise
        finally:
            mw.close()

    def test_close_event_handles_already_finished_worker(self, tmp_work_dirs):
        """A worker that already finished but hasn't been GC'd yet
        must be released cleanly (no wait() needed)."""
        from rlpe.gui.main_window import MainWindow

        mw = MainWindow()
        t = _SleepyThread()
        try:
            # Drain the real worker first.
            self._drain_real_worker(mw)
            t.start()
            t.requestInterruption()
            t.wait(2000)
            assert not t.isRunning()
            mw._jobs_tab._disk_scan_worker = t
            from PySide6.QtGui import QCloseEvent

            event = QCloseEvent()
            mw.closeEvent(event)
            assert mw._jobs_tab._disk_scan_worker is None
        finally:
            mw.close()


# ============================================================
# B-3: async auto-open via the scan_finished signal
# ============================================================
class TestAutoOpenResultsAfterAsyncScan:
    """B-3: the auto-open must fire AFTER the worker truly finishes."""

    def test_auto_open_after_async_scan_completes(self, tmp_work_dirs):
        """Simulate the B-15 race: construct MainWindow, then wait
        for the async scan to finish, then verify the Results tab
        was auto-populated with the latest done job.

        The previous sync code checked ``_jobs_tab._jobs`` immediately
        after kicking off the scan; the worker hadn't emitted anything
        yet so the auto-open never fired. The fix routes the auto-open
        through ``_on_disk_scan_done`` which only runs after the worker
        truly finishes.
        """
        from PySide6.QtCore import QEventLoop, QTimer

        from rlpe.gui.constants import STATUS_DONE, TAB_RESULTS
        from rlpe.gui.main_window import MainWindow

        mw = MainWindow()
        try:
            # The worker was kicked off in __init__ via
            # load_recent_jobs_from_disk; we need to drain the event
            # loop until the scan_finished signal fires.
            loop = QEventLoop()
            mw._jobs_tab.scan_finished.connect(lambda _recs: loop.quit())
            # Safety timeout so a crashed worker can't hang the test.
            QTimer.singleShot(5000, loop.quit)
            loop.exec()

            # The async scan must have populated _jobs.
            assert len(mw._jobs_tab._jobs) >= 2, (
                f"async scan should have loaded jobs, got "
                f"{list(mw._jobs_tab._jobs)}"
            )

            # The auto-open must have fired and switched to Results.
            assert mw._tabs.currentIndex() == TAB_RESULTS, (
                f"Phase F-1 (B-3): tab should switch to Results after "
                f"async scan completes, got {mw._tabs.currentIndex()} "
                f"(TAB_RUN=0, TAB_JOBS=1, TAB_RESULTS=2)"
            )
            assert mw._results_tab._current_job_id is not None, (
                "Phase F-1 (B-3): _results_tab._current_job_id should "
                "be set after _on_disk_scan_done ran"
            )

            # Pick the latest STATUS_DONE job and check the auto-open
            # landed on it.
            latest = max(
                (j for j in mw._jobs_tab._jobs.values()
                 if j.status == STATUS_DONE and j.rows),
                key=lambda j: j.finished_at,
            )
            assert mw._results_tab._current_job_id == latest.job_id, (
                f"Phase F-1 (B-3): auto-opened {mw._results_tab._current_job_id!r} "
                f"but most recent is {latest.job_id!r}"
            )
        finally:
            mw.close()

    def test_auto_open_skipped_when_no_jobs(self, empty_work_dirs):
        """With no candidates on disk, ``scan_finished`` fires with an
        empty list and the GUI must stay on the default Run tab."""
        from PySide6.QtCore import QEventLoop, QTimer

        from rlpe.gui.constants import TAB_RUN
        from rlpe.gui.main_window import MainWindow

        mw = MainWindow()
        try:
            # Drain the event loop so the QTimer.singleShot(0, ...)
            # that emits scan_finished([]) actually fires.
            loop = QEventLoop()
            mw._jobs_tab.scan_finished.connect(lambda _recs: loop.quit())
            QTimer.singleShot(5000, loop.quit)
            loop.exec()

            assert mw._jobs_tab._jobs == {}
            assert mw._tabs.currentIndex() == TAB_RUN, (
                f"Phase F-1 (B-3): GUI should stay on Run tab when "
                f"no jobs are loaded, got {mw._tabs.currentIndex()}"
            )
        finally:
            mw.close()

    def test_on_disk_scan_done_picks_latest_done(self, tmp_work_dirs):
        """Direct unit test on ``_on_disk_scan_done``: call it with
        a synthetic list of records and assert it picks the latest
        ``STATUS_DONE`` job."""
        from rlpe.gui.constants import STATUS_DONE, STATUS_RUNNING
        from rlpe.gui.jobs_tab import JobRecord
        from rlpe.gui.main_window import MainWindow

        mw = MainWindow()
        try:
            now = time.time()
            records = [
                JobRecord(
                    job_id="a",
                    pdf_path="",
                    output_dir="/tmp/a",
                    status=STATUS_DONE,
                    finished_at=now - 10,
                    rows=[{"species": "X"}],
                ),
                JobRecord(
                    job_id="b",
                    pdf_path="",
                    output_dir="/tmp/b",
                    status=STATUS_DONE,
                    finished_at=now,  # latest
                    rows=[{"species": "Y"}],
                ),
                JobRecord(
                    job_id="c",
                    pdf_path="",
                    output_dir="/tmp/c",
                    status=STATUS_RUNNING,
                    finished_at=now,
                    rows=[],
                ),
            ]
            # Directly invoke the slot.
            mw._on_disk_scan_done(records)
            assert mw._results_tab._current_job_id == "b", (
                f"_on_disk_scan_done should pick latest done job 'b', "
                f"got {mw._results_tab._current_job_id!r}"
            )
        finally:
            mw.close()

    def test_on_disk_scan_done_empty_records_is_noop(self, tmp_work_dirs):
        """Calling ``_on_disk_scan_done([])`` must NOT switch tabs or
        throw — fresh install means no jobs, no auto-open."""
        from rlpe.gui.constants import TAB_RUN
        from rlpe.gui.main_window import MainWindow

        mw = MainWindow()
        try:
            # Don't call load_recent_jobs_from_disk() again — the
            # MainWindow already kicked off the scan in __init__ and
            # the worker is still running. Just call _on_disk_scan_done
            # directly with an empty list to verify the no-op branch.
            mw._on_disk_scan_done([])
            assert mw._tabs.currentIndex() == TAB_RUN, (
                f"empty records → must stay on Run, got {mw._tabs.currentIndex()}"
            )
        finally:
            mw.close()


# ============================================================
# Source-string guards: pin the API contract
# ============================================================
def test_main_window_wires_scan_finished_signal():
    """The signal wiring must be present in ``_wire_signals`` so a
    future refactor that drops the connect doesn't silently break the
    auto-open again."""
    src = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "rlpe"
        / "gui"
        / "main_window.py"
    ).read_text(encoding="utf-8")
    assert "scan_finished.connect(self._on_disk_scan_done)" in src, (
        "Phase F-1 (B-3): main_window must wire "
        "scan_finished → _on_disk_scan_done"
    )
    assert "scan_failed.connect(self._on_disk_scan_failed)" in src, (
        "Phase F-1 (B-3): main_window must wire "
        "scan_failed → _on_disk_scan_failed"
    )


def test_main_window_close_event_calls_shutdown():
    """B-1 source-guard: ``closeEvent`` must call ``self._jobs_tab.shutdown()``
    before the pipeline worker shutdown."""
    src = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "rlpe"
        / "gui"
        / "main_window.py"
    ).read_text(encoding="utf-8")
    idx = src.find("def closeEvent")
    assert idx != -1, "closeEvent must exist"
    close_body = src[idx:src.find("def _remove_i18n_listeners", idx)]
    assert "self._jobs_tab.shutdown()" in close_body, (
        "Phase F-1 (B-1): closeEvent must call self._jobs_tab.shutdown()"
    )
    # It must come BEFORE _stop_pipeline_worker so the heavier
    # pipeline worker doesn't need to wait on the disk scan.
    assert close_body.index("self._jobs_tab.shutdown()") < close_body.index(
        "self._stop_pipeline_worker()"
    ), (
        "Phase F-1 (B-1): _jobs_tab.shutdown() must be called "
        "BEFORE _stop_pipeline_worker()"
    )


def test_load_recent_jobs_no_longer_does_sync_auto_open():
    """B-3 source-guard: ``_load_recent_jobs`` must NOT contain the
    synchronous auto-open logic anymore (it would always read
    ``_jobs`` before the worker finished emitting)."""
    src = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "rlpe"
        / "gui"
        / "main_window.py"
    ).read_text(encoding="utf-8")
    idx = src.find("def _load_recent_jobs")
    assert idx != -1
    end = src.find("def _on_disk_scan_done", idx)
    body = src[idx:end]
    assert "self._tabs.setCurrentIndex(TAB_RESULTS)" not in body, (
        "Phase F-1 (B-3): _load_recent_jobs must not switch tabs "
        "synchronously any more — that broke once B-15 made the "
        "scan async"
    )
    assert "self._results_tab.load_job(" not in body, (
        "Phase F-1 (B-3): _load_recent_jobs must not load the "
        "Results tab directly — move to _on_disk_scan_done"
    )
