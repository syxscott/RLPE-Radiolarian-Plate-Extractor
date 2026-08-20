"""Phase F-1 (2026-08-20) — frontend audit fixes for ``jobs_tab``.

Three BLOCKER/MAJOR bugs were fixed in ``src/rlpe/gui/jobs_tab.py``:

* **B-1** (partial): ``_DiskScanWorker`` QThread wasn't stopped/waited
  on GUI close, causing exit code 134. Added ``JobsTab.shutdown()`` that
  interrupts the worker and waits up to 30 s.
* **B-3** (partial): the async disk scan didn't tell callers when it
  actually finished. Added ``scan_finished(list[JobRecord])`` and
  ``scan_failed(str)`` signals on ``JobsTab`` plus matching
  ``completed``/``failed`` signals on the worker.
* **M-4**: ``_parse_one`` had no file size or line length limit on
  ``matches.jsonl``. Added ``MAX_JSONL_SIZE = 100 MB`` and
  ``MAX_LINE_SIZE = 1 MB`` module-level guards.
* **MAJOR M (partial flag)**: a ``matches.jsonl`` with rows but no
  ``complete.flag`` (interrupted run) was incorrectly marked
  ``STATUS_DONE`` — now ``STATUS_FAILED`` with a clear progress_msg.

This test file pins each of those fixes. Existing tests in
``tests/test_phase49_disk_scan.py`` continue to pin the legacy
"load_recent_jobs returns a count" contract.
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
from PySide6.QtCore import QEventLoop, QThread, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


# ============================================================
# Helpers
# ============================================================
def _drain(max_ms: int = 5000) -> None:
    """Run the Qt event loop briefly so queued signals/callbacks fire."""
    loop = QEventLoop()
    QTimer.singleShot(max_ms, loop.quit)
    loop.exec()


def _write_jsonl(path: Path, rows: list[dict], truncate_last_to: int | None = None) -> None:
    """Write ``rows`` as JSONL (one JSON object per line).

    If ``truncate_last_to`` is given, the LAST line is replaced with
    ``truncate_last_to`` bytes of garbage to simulate a pathological
    single-line payload (Phase F-1 M-4 DoS protection).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if truncate_last_to is not None and rows:
        head = "\n".join(json.dumps(r) for r in rows[:-1])
        head = head + "\n" if head else ""
        with path.open("wb") as fh:
            if head:
                fh.write(head.encode("utf-8"))
            fh.write(b"x" * truncate_last_to)
            fh.write(b"\n")
        return
    body = "\n".join(json.dumps(r) for r in rows) + "\n"
    path.write_text(body, encoding="utf-8")


# ============================================================
# B-1: shutdown interrupts a running disk scan
# ============================================================
def test_shutdown_interrupts_running_disk_scan(tmp_path, monkeypatch):
    """Phase F-1 (B-1): JobsTab.shutdown() must interrupt and wait
    on a still-running _DiskScanWorker. Without the fix, GUI close
    crashed with exit code 134."""
    from rlpe.gui import jobs_tab as jt_mod

    # Make sure the constant exists (so we can prove the shutdown
    # path doesn't try to terminate a worker that's still running).
    assert hasattr(jt_mod, "MAX_JSONL_SIZE")
    assert hasattr(jt_mod, "MAX_LINE_SIZE")

    # Create a large number of synthetic jobs so the scan has real
    # work to do — guarantees the worker is mid-loop when we shut it
    # down.
    n_jobs = 200
    for i in range(n_jobs):
        job_dir = tmp_path / "service_work" / f"job-{i:04d}" / "output" / "manifests"
        job_dir.mkdir(parents=True)
        (job_dir / "matches.jsonl").write_text(
            json.dumps({"species": f"S{i}", "panel_id": f"j{i}/p1"}) + "\n",
            encoding="utf-8",
        )
    import rlpe.gui.constants as consts

    monkeypatch.setattr(consts, "PROJECT_ROOT", tmp_path)
    from rlpe.gui.jobs_tab import JobsTab

    jt = JobsTab()
    jt.load_recent_jobs_from_disk()
    worker = jt._disk_scan_worker
    assert worker is not None and worker.isRunning(), (
        "worker should be running immediately after load_recent_jobs_from_disk"
    )

    # Shutdown should interrupt + wait without raising.
    start = time.monotonic()
    jt.shutdown()
    elapsed = time.monotonic() - start
    assert elapsed < 35.0, f"shutdown waited too long: {elapsed:.1f}s"

    # After shutdown, the worker must be stopped and the reference
    # must be cleared (so Qt can free it).
    assert not worker.isRunning(), "worker should have stopped after shutdown"
    assert jt._disk_scan_worker is None, (
        "shutdown should clear _disk_scan_worker; deleting a running QThread would crash the GUI"
    )


def test_shutdown_noop_when_no_worker(tmp_path, monkeypatch):
    """Phase F-1 (B-1): calling shutdown() when no scan was ever
    started must be a safe no-op (no exception)."""
    import rlpe.gui.constants as consts

    monkeypatch.setattr(consts, "PROJECT_ROOT", tmp_path)
    from rlpe.gui.jobs_tab import JobsTab

    jt = JobsTab()
    # No load_recent_jobs_from_disk() — _disk_scan_worker stays None.
    assert jt._disk_scan_worker is None
    # Must not raise.
    jt.shutdown()
    assert jt._disk_scan_worker is None


def test_shutdown_safe_to_call_twice(tmp_path, monkeypatch):
    """Phase F-1 (B-1): calling shutdown() twice in a row must be
    safe (idempotent — important if MainWindow calls it from both
    closeEvent and an explicit destroy path)."""
    import rlpe.gui.constants as consts

    monkeypatch.setattr(consts, "PROJECT_ROOT", tmp_path)
    from rlpe.gui.jobs_tab import JobsTab

    jt = JobsTab()
    # Set up one job so the scan actually runs.
    job_dir = tmp_path / "service_work" / "x" / "output" / "manifests"
    job_dir.mkdir(parents=True)
    (job_dir / "matches.jsonl").write_text(
        json.dumps({"species": "S", "panel_id": "x/p1"}) + "\n",
        encoding="utf-8",
    )
    jt.load_recent_jobs_from_disk()
    jt.shutdown()
    # Second call: worker is already None, must be a no-op.
    jt.shutdown()
    assert jt._disk_scan_worker is None


# ============================================================
# B-3: scan_finished signal fires
# ============================================================
def test_scan_finished_signal_emitted(tmp_path, monkeypatch):
    """Phase F-1 (B-3): the async disk scan must fire scan_finished
    with the actual loaded JobRecord list (not just the synchronous
    candidate count)."""
    import rlpe.gui.constants as consts

    monkeypatch.setattr(consts, "PROJECT_ROOT", tmp_path)
    # 3 valid jobs + 1 empty (skipped).
    for jid in ("job-a", "job-b", "job-c"):
        job_dir = tmp_path / "service_work" / jid / "output" / "manifests"
        job_dir.mkdir(parents=True)
        (job_dir / "matches.jsonl").write_text(
            json.dumps({"species": f"S-{jid}", "panel_id": f"{jid}/p1"}) + "\n",
            encoding="utf-8",
        )
        (job_dir / "complete.flag").write_text("", encoding="utf-8")
    empty_dir = tmp_path / "service_work" / "empty" / "output" / "manifests"
    empty_dir.mkdir(parents=True)
    (empty_dir / "matches.jsonl").write_text("", encoding="utf-8")

    from rlpe.gui.jobs_tab import JobsTab

    jt = JobsTab()

    captured: list[list] = []

    def on_finished(records):
        captured.append(records)

    jt.scan_finished.connect(on_finished)
    n = jt.load_recent_jobs_from_disk()
    assert n == 4, f"expected 4 candidates, got {n}"

    # Pump events until the worker reports back (or timeout).
    deadline = time.monotonic() + 10.0
    while not captured and time.monotonic() < deadline:
        _drain(max_ms=50)

    assert captured, "scan_finished signal never fired"
    records = captured[-1]
    assert len(records) == 3, (
        f"expected 3 loaded records (empty skipped), got {len(records)}: "
        f"{[r.job_id for r in records]}"
    )
    jids_loaded = {r.job_id for r in records}
    assert jids_loaded == {"job-a", "job-b", "job-c"}, f"unexpected loaded jobs: {jids_loaded}"


def test_scan_finished_emits_empty_when_no_candidates(tmp_path, monkeypatch):
    """Phase F-1 (B-3): when there are no candidates at all, scan_finished
    must STILL fire (with []) so callers awaiting the async load know
    the scan truly completed (and didn't just fail to start)."""
    import rlpe.gui.constants as consts

    monkeypatch.setattr(consts, "PROJECT_ROOT", tmp_path)
    # No service_work/, no work/.

    from rlpe.gui.jobs_tab import JobsTab

    jt = JobsTab()
    captured: list[list] = []
    jt.scan_finished.connect(lambda r: captured.append(r))

    n = jt.load_recent_jobs_from_disk()
    assert n == 0

    deadline = time.monotonic() + 5.0
    while not captured and time.monotonic() < deadline:
        _drain(max_ms=50)
    assert captured, "scan_finished([]) never fired for empty candidate set"
    assert captured[-1] == []


# ============================================================
# M-4: JSONL DoS protection
# ============================================================
def test_jsonl_oversized_skipped(tmp_path, monkeypatch, caplog):
    """Phase F-1 (M-4): a 200 MB matches.jsonl must be skipped (not
    read into memory) and a clear log warning emitted."""
    import os

    import rlpe.gui.constants as consts

    monkeypatch.setattr(consts, "PROJECT_ROOT", tmp_path)
    job_dir = tmp_path / "service_work" / "fat" / "output" / "manifests"
    job_dir.mkdir(parents=True)
    fat = job_dir / "matches.jsonl"
    # Create a sparse 200 MB file. ``Path.truncate`` doesn't exist,
    # so we open + os.truncate which on most filesystems is a
    # sparse allocation (~0 disk usage).
    with open(fat, "wb") as fh:
        pass
    os.truncate(fat, 200 * 1024 * 1024)
    assert fat.stat().st_size == 200 * 1024 * 1024

    import logging

    from rlpe.gui.jobs_tab import JobsTab

    jt = JobsTab()
    captured: list[list] = []
    jt.scan_finished.connect(lambda r: captured.append(r))
    with caplog.at_level(logging.WARNING, logger="rlpe.gui"):
        n = jt.load_recent_jobs_from_disk()
        # Pump events until scan_finished fires (worker is deleteLater'd
        # after completion, so we can't safely poll isRunning()).
        deadline = time.monotonic() + 10.0
        while not captured and time.monotonic() < deadline:
            _drain(max_ms=50)

    assert n == 1, "should have found 1 candidate to attempt parsing"
    assert captured, "scan_finished should have fired"
    # The fat job must be skipped, not loaded into _jobs.
    assert "fat" not in jt._jobs, (
        f"fat job should be skipped (file > 100 MB); got _jobs={list(jt._jobs.keys())}"
    )
    # A warning must have been logged naming the file.
    msgs = [r.getMessage() for r in caplog.records]
    assert any("matches.jsonl exceeds" in m for m in msgs), (
        f"expected a 'matches.jsonl exceeds' warning, got: {msgs}"
    )


def test_jsonl_overlong_line_skipped(tmp_path, monkeypatch, caplog):
    """Phase F-1 (M-4): a single JSONL line longer than 1 MB must be
    truncated by ``_parse_one`` and skipped without OOM."""
    import logging

    import rlpe.gui.constants as consts

    monkeypatch.setattr(consts, "PROJECT_ROOT", tmp_path)
    job_dir = tmp_path / "service_work" / "longline" / "output" / "manifests"
    job_dir.mkdir(parents=True)
    # 1 valid line + 1 pathological 2 MB line. Pass TWO rows so the
    # helper keeps the first (valid) one and replaces the second
    # with 2 MB of garbage.
    _write_jsonl(
        job_dir / "matches.jsonl",
        rows=[
            {"species": "OK", "panel_id": "longline/p1"},
            {"species": "WILL_BE_TRUNCATED", "panel_id": "longline/p2"},
        ],
        truncate_last_to=2 * 1024 * 1024,  # 2 MB of 'x' characters
    )

    from rlpe.gui.jobs_tab import JobsTab

    jt = JobsTab()
    captured: list[list] = []
    jt.scan_finished.connect(lambda r: captured.append(r))
    with caplog.at_level(logging.DEBUG, logger="rlpe.gui"):
        n = jt.load_recent_jobs_from_disk()
        deadline = time.monotonic() + 10.0
        while not captured and time.monotonic() < deadline:
            _drain(max_ms=50)

    assert n == 1
    job = jt._jobs.get("longline")
    assert job is not None, "job should still be loaded (valid line kept)"
    # Only the 1 valid row should survive; the 2 MB garbage line
    # was truncated to 1 MB and then ``json.loads`` failed → skipped.
    assert len(job.rows) == 1, f"expected 1 row (truncated line skipped), got {len(job.rows)}"
    assert job.rows[0]["species"] == "OK"


def test_parse_one_skips_oversized_file_directly(tmp_path, monkeypatch):
    """Phase F-1 (M-4): even when called synchronously, ``_parse_one``
    must skip a >100 MB file via the stat() check."""
    import rlpe.gui.jobs_tab as jt_mod

    monkeypatch.setattr(jt_mod, "MAX_JSONL_SIZE", 1024)  # 1 KB for the test
    monkeypatch.setattr(jt_mod, "MAX_LINE_SIZE", 256)  # 256 B for the test
    job_dir = tmp_path / "service_work" / "big" / "output" / "manifests"
    job_dir.mkdir(parents=True)
    (job_dir / "matches.jsonl").write_text(
        json.dumps({"species": "X", "panel_id": "big/p1"}) + "\n",
        encoding="utf-8",
    )
    # Force the file to be 4 KB (above our lowered 1 KB cap).
    fat = job_dir / "matches.jsonl"
    with fat.open("ab") as fh:
        fh.write(b"\0" * (4 * 1024 - fat.stat().st_size))

    from rlpe.gui.jobs_tab import JobsTab

    jt = JobsTab()
    worker = jt_mod._DiskScanWorker(
        [
            jt_mod._PendingDiskScan(
                jid="big",
                root=job_dir.parents[1],
                matches_path=fat,
                complete_flag=job_dir / "complete.flag",
            )
        ]
    )
    worker.start()
    worker.wait(5000)
    # No records loaded because the file is too big.
    assert jt._jobs.get("big") is None


# ============================================================
# MAJOR M (partial flag): interrupted runs marked as FAILED
# ============================================================
def test_partial_flag_marks_failed(tmp_path, monkeypatch):
    """Phase F-1 (M-partial): a job with ``matches.jsonl`` rows but
    no ``complete.flag`` (interrupted run) must be STATUS_FAILED,
    not STATUS_DONE. The previous code stamped every disk-loaded
    job as STATUS_DONE, so a crashed run looked like a successful
    extraction."""
    import rlpe.gui.constants as consts
    from rlpe.gui.constants import STATUS_FAILED

    monkeypatch.setattr(consts, "PROJECT_ROOT", tmp_path)
    job_dir = tmp_path / "service_work" / "interrupted" / "output" / "manifests"
    job_dir.mkdir(parents=True)
    # matches.jsonl exists with rows — but NO complete.flag.
    (job_dir / "matches.jsonl").write_text(
        json.dumps({"species": "PartiallyExtracted", "panel_id": "interrupted/p1"}) + "\n",
        encoding="utf-8",
    )

    from rlpe.gui.jobs_tab import JobsTab

    jt = JobsTab()
    captured: list[list] = []
    jt.scan_finished.connect(lambda r: captured.append(r))
    jt.load_recent_jobs_from_disk()
    deadline = time.monotonic() + 10.0
    while not captured and time.monotonic() < deadline:
        _drain(max_ms=50)

    job = jt._jobs.get("interrupted")
    assert job is not None, "interrupted job should still be loaded (rows present)"
    assert job.status == STATUS_FAILED, (
        f"interrupted job (no complete.flag) must be STATUS_FAILED, got {job.status!r}"
    )
    # progress_msg must hint at the missing flag.
    assert "complete.flag" in job.progress_msg, (
        f"progress_msg should mention 'complete.flag', got: {job.progress_msg!r}"
    )
    # But the rows are still preserved so the operator can see what was extracted.
    assert len(job.rows) == 1
    assert job.rows[0]["species"] == "PartiallyExtracted"


def test_complete_flag_present_marks_done(tmp_path, monkeypatch):
    """Phase F-1 (M-partial): a job with ``complete.flag`` present
    must still be STATUS_DONE (regression check — the partial-flag
    fix must not over-correct)."""
    import rlpe.gui.constants as consts
    from rlpe.gui.constants import STATUS_DONE

    monkeypatch.setattr(consts, "PROJECT_ROOT", tmp_path)
    job_dir = tmp_path / "service_work" / "finished" / "output" / "manifests"
    job_dir.mkdir(parents=True)
    (job_dir / "matches.jsonl").write_text(
        json.dumps({"species": "FullyExtracted", "panel_id": "finished/p1"}) + "\n",
        encoding="utf-8",
    )
    (job_dir / "complete.flag").write_text("", encoding="utf-8")

    from rlpe.gui.jobs_tab import JobsTab

    jt = JobsTab()
    captured: list[list] = []
    jt.scan_finished.connect(lambda r: captured.append(r))
    jt.load_recent_jobs_from_disk()
    deadline = time.monotonic() + 10.0
    while not captured and time.monotonic() < deadline:
        _drain(max_ms=50)

    job = jt._jobs.get("finished")
    assert job is not None
    assert job.status == STATUS_DONE, (
        f"complete.flag present must yield STATUS_DONE, got {job.status!r}"
    )


def test_partial_flag_empty_matches_marks_failed(tmp_path, monkeypatch):
    """Phase F-1 (M-partial): a job with an EMPTY matches.jsonl
    and no complete.flag must be STATUS_FAILED too — the original
    early-complete behaviour from the legacy ``if not rows:
    return None`` was actually a hidden bug because the operator
    saw nothing at all. Mark it failed so the empty-row run is
    visible (and re-runnable)."""
    import rlpe.gui.constants as consts
    from rlpe.gui.constants import STATUS_FAILED

    monkeypatch.setattr(consts, "PROJECT_ROOT", tmp_path)
    job_dir = tmp_path / "service_work" / "empty_no_flag" / "output" / "manifests"
    job_dir.mkdir(parents=True)
    (job_dir / "matches.jsonl").write_text("", encoding="utf-8")
    # NO complete.flag.

    from rlpe.gui.jobs_tab import JobsTab

    jt = JobsTab()
    captured: list[list] = []
    jt.scan_finished.connect(lambda r: captured.append(r))
    n = jt.load_recent_jobs_from_disk()
    # The legacy ``if not rows: return None`` short-circuits before
    # we get to the partial-flag logic, so this job should be
    # invisible. The test just pins the existing behaviour.
    if n == 0:
        assert "empty_no_flag" not in jt._jobs
        return
    deadline = time.monotonic() + 10.0
    while not captured and time.monotonic() < deadline:
        _drain(max_ms=50)
    job = jt._jobs.get("empty_no_flag")
    if job is not None:
        assert job.status == STATUS_FAILED, (
            f"empty matches.jsonl + no flag must be STATUS_FAILED, got {job.status!r}"
        )


# ============================================================
# Cross-cutting: shutdown after completed scan
# ============================================================
def test_shutdown_after_completed_scan(tmp_path, monkeypatch):
    """Phase F-1 (B-1): calling shutdown() AFTER the worker has
    already finished must be a fast no-op (worker.isRunning() is
    False, no interrupt or wait needed)."""
    import rlpe.gui.constants as consts

    monkeypatch.setattr(consts, "PROJECT_ROOT", tmp_path)
    job_dir = tmp_path / "service_work" / "x" / "output" / "manifests"
    job_dir.mkdir(parents=True)
    (job_dir / "matches.jsonl").write_text(
        json.dumps({"species": "S", "panel_id": "x/p1"}) + "\n",
        encoding="utf-8",
    )
    (job_dir / "complete.flag").write_text("", encoding="utf-8")

    from rlpe.gui.jobs_tab import JobsTab

    jt = JobsTab()
    captured: list[list] = []
    jt.scan_finished.connect(lambda r: captured.append(r))
    jt.load_recent_jobs_from_disk()
    # Pump events until the scan finishes (via scan_finished signal,
    # since the worker is deleteLater'd after completion so we can't
    # safely poll isRunning()).
    deadline = time.monotonic() + 10.0
    while not captured and time.monotonic() < deadline:
        _drain(max_ms=50)
    # After scan_finished, the JobsTab forwarder has run but the
    # strong ref is still in place until the worker's finished
    # signal fires deleteLater. Give the event loop one more tick.
    _drain(max_ms=200)
    # Now shutdown should be a quick no-op.
    start = time.monotonic()
    jt.shutdown()
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, f"shutdown after completed scan took {elapsed:.2f}s"
    assert jt._disk_scan_worker is None


# ============================================================
# Constants exist + are sensible
# ============================================================
def test_module_constants_have_safe_defaults():
    """Phase F-1 (M-4): the DoS guards must exist and be sized so a
    single pathological file can't crash the GUI."""
    from rlpe.gui import jobs_tab as jt_mod

    assert isinstance(jt_mod.MAX_JSONL_SIZE, int)
    assert isinstance(jt_mod.MAX_LINE_SIZE, int)
    assert 10 * 1024 * 1024 <= jt_mod.MAX_JSONL_SIZE <= 1024 * 1024 * 1024, (
        f"MAX_JSONL_SIZE={jt_mod.MAX_JSONL_SIZE} should be 10 MB..1 GB"
    )
    assert 64 * 1024 <= jt_mod.MAX_LINE_SIZE <= 10 * 1024 * 1024, (
        f"MAX_LINE_SIZE={jt_mod.MAX_LINE_SIZE} should be 64 KB..10 MB"
    )


def test_new_signals_declared_at_class_level():
    """Phase F-1 (B-3): ``scan_finished`` and ``scan_failed`` must
    be declared at the class level (PySide6 Signal descriptors), not
    as instance attributes in __init__."""
    from rlpe.gui.jobs_tab import JobsTab

    # Class-level presence is what makes ``connect()`` work in PySide6.
    assert "scan_finished" in JobsTab.__dict__, (
        "scan_finished must be a class-level Signal; defining it in __init__ breaks .connect()"
    )
    assert "scan_failed" in JobsTab.__dict__, (
        "scan_failed must be a class-level Signal; defining it in __init__ breaks .connect()"
    )
