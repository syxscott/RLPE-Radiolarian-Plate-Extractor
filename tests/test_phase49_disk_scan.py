"""Phase 49 — GUI startup disk scan for completed jobs.

Bug: GUI kept completed jobs purely in memory (JobsTab._jobs dict).
Restarting the GUI, or running extractions via the CLI / Web UI,
made the results invisible until the user re-ran the same PDF
inside the GUI session.

Fix: Phase 49 mirrors the Web API's
``api/app.py:_load_existing_jobs_from_disk()`` — on GUI startup,
scan ``<project_root>/service_work/<job_id>/output/manifests/matches.jsonl``
and ``<project_root>/work/output/manifests/matches.jsonl`` (CLI
runs). For each ``matches.jsonl`` found, build a JobRecord with
``status=STATUS_DONE`` and the rows loaded from disk, then call
``add_or_update_job()``.

Tests pin:
  1. ``JobsTab.load_recent_jobs_from_disk()`` returns the count.
  2. The synthetic ``service_work/<jid>/output/manifests/matches.jsonl``
     produces a job in ``_jobs`` with status=done, rows populated,
     output_dir pointing at ``<root>/output``.
  3. Empty / missing manifest → 0 loaded, no exception.
  4. Corrupt JSON line → skipped, no crash.
  5. CLI ``work/`` directory gets a stable ``cli_<hash>`` job id.
  6. ``MainWindow.__init__`` calls the scan automatically.
  7. Double-clicking a loaded job opens results in ResultsTab.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


import pytest


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def tmp_work_dirs(tmp_path, monkeypatch):
    """Create a fake project root with service_work/ and work/ subdirs
    containing synthetic matches.jsonl files. The JobsTab scans
    <project_root>/service_work/<jid>/output/manifests/matches.jsonl
    and <project_root>/work/output/manifests/matches.jsonl.

    We monkeypatch ``constants.PROJECT_ROOT`` to point at the tmp_path
    so the scan picks up our synthetic dirs without touching the
    real project tree.
    """
    import rlpe.gui.constants as consts
    monkeypatch.setattr(consts, "PROJECT_ROOT", tmp_path)
    # service_work/<job-a>/output/manifests/matches.jsonl
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
    # service_work/<job-b>/output/manifests/matches.jsonl  (1 row)
    job_b = tmp_path / "service_work" / "job-b" / "output" / "manifests"
    job_b.mkdir(parents=True)
    (job_b / "matches.jsonl").write_text(
        json.dumps({"species": "Species B1", "panel_id": "job-b/fig1/p1"}) + "\n",
        encoding="utf-8",
    )
    # service_work/<job-empty>/output/manifests/matches.jsonl (empty → skip)
    job_empty = tmp_path / "service_work" / "job-empty" / "output" / "manifests"
    job_empty.mkdir(parents=True)
    (job_empty / "matches.jsonl").write_text("", encoding="utf-8")
    # work/output/manifests/matches.jsonl (CLI run, single job_id via hash)
    cli_dir = tmp_path / "work" / "output" / "manifests"
    cli_dir.mkdir(parents=True)
    cli_rows = [{"species": "Species CLI", "panel_id": "cli/p1"}]
    (cli_dir / "matches.jsonl").write_text(
        "\n".join(json.dumps(r) for r in cli_rows) + "\n",
        encoding="utf-8",
    )
    return tmp_path


# ============================================================
# 1. load_recent_jobs_from_disk returns count
# ============================================================
def test_load_recent_jobs_returns_count(tmp_work_dirs):
    """Phase 49: load_recent_jobs_from_disk() returns the number of
    jobs loaded (service_work + work CLI = 3 in this fixture)."""
    from rlpe.gui.jobs_tab import JobsTab
    jt = JobsTab()
    n = jt.load_recent_jobs_from_disk()
    assert n == 3, f"expected 3 jobs loaded, got {n}"


# ============================================================
# 2. Synthetic matches.jsonl produces a populated JobRecord
# ============================================================
def test_loaded_job_has_correct_status_and_rows(tmp_work_dirs):
    from rlpe.gui.jobs_tab import JobsTab
    from rlpe.gui.constants import STATUS_DONE
    jt = JobsTab()
    jt.load_recent_jobs_from_disk()
    # job-a: 2 rows
    job = jt._jobs.get("job-a")
    assert job is not None, "job-a should be loaded"
    assert job.status == STATUS_DONE, (
        f"loaded job should be STATUS_DONE, got {job.status!r}"
    )
    assert len(job.rows) == 2, f"job-a should have 2 rows, got {len(job.rows)}"
    assert job.rows[0]["species"] == "Species A1"
    assert job.output_dir.endswith("service_work/job-a/output"), (
        f"output_dir should end with service_work/job-a/output, got "
        f"{job.output_dir!r}"
    )


# ============================================================
# 3. Empty / missing manifest → 0 loaded, no exception
# ============================================================
def test_load_recent_jobs_handles_missing_dirs(tmp_path, monkeypatch):
    """Phase 49: if neither service_work nor work/ exists, return 0
    without raising."""
    from rlpe.gui.constants import PROJECT_ROOT as _orig  # noqa: F401
    import rlpe.gui.constants as consts
    monkeypatch.setattr(consts, "PROJECT_ROOT", tmp_path)
    # No service_work/ or work/ created
    from rlpe.gui.jobs_tab import JobsTab
    jt = JobsTab()
    n = jt.load_recent_jobs_from_disk()
    assert n == 0, f"expected 0 jobs (no dirs), got {n}"


def test_load_recent_jobs_skips_empty_manifest(tmp_path, monkeypatch):
    """Phase 49: a service_work/<jid>/output/manifests/matches.jsonl
    that exists but is empty should be skipped (no rows = no point)."""
    import rlpe.gui.constants as consts
    monkeypatch.setattr(consts, "PROJECT_ROOT", tmp_path)
    # Create empty manifest
    job_dir = tmp_path / "service_work" / "ghost" / "output" / "manifests"
    job_dir.mkdir(parents=True)
    (job_dir / "matches.jsonl").write_text("", encoding="utf-8")
    from rlpe.gui.jobs_tab import JobsTab
    jt = JobsTab()
    n = jt.load_recent_jobs_from_disk()
    assert n == 0, f"expected 0 jobs (empty manifest), got {n}"
    assert "ghost" not in jt._jobs


# ============================================================
# 4. Corrupt JSON line → skipped, no crash
# ============================================================
def test_load_recent_jobs_skips_corrupt_manifest(tmp_path, monkeypatch):
    """Phase 49: a matches.jsonl with malformed JSON should be
    skipped silently (log warning) without crashing the GUI."""
    import rlpe.gui.constants as consts
    monkeypatch.setattr(consts, "PROJECT_ROOT", tmp_path)
    job_dir = tmp_path / "service_work" / "bad" / "output" / "manifests"
    job_dir.mkdir(parents=True)
    (job_dir / "matches.jsonl").write_text(
        "not valid json\n{also: bad}\n",
        encoding="utf-8",
    )
    from rlpe.gui.jobs_tab import JobsTab
    jt = JobsTab()
    # Must not raise
    n = jt.load_recent_jobs_from_disk()
    assert n == 0


# ============================================================
# 5. CLI work/ directory gets a stable cli_<hash> job id
# ============================================================
def test_cli_work_gets_stable_hash_id(tmp_work_dirs):
    """Phase 49: work/output/manifests/matches.jsonl produces a job
    with id 'cli_<12 hex>' — same hash each run (stable across GUI
    restarts)."""
    from rlpe.gui.jobs_tab import JobsTab
    jt1 = JobsTab()
    jt1.load_recent_jobs_from_disk()
    jt2 = JobsTab()
    jt2.load_recent_jobs_from_disk()
    cli_ids_1 = [jid for jid in jt1._jobs if jid.startswith("cli_")]
    cli_ids_2 = [jid for jid in jt2._jobs if jid.startswith("cli_")]
    assert cli_ids_1, "should have a cli_ job_id"
    assert cli_ids_1 == cli_ids_2, (
        f"cli_ job_id should be stable: {cli_ids_1} vs {cli_ids_2}"
    )
    # Hash is 12 hex chars after cli_
    jid = cli_ids_1[0]
    assert len(jid) == 4 + 12, f"unexpected cli_ job_id format: {jid!r}"


def test_cli_work_loaded_with_cli_rows(tmp_work_dirs):
    from rlpe.gui.jobs_tab import JobsTab
    jt = JobsTab()
    jt.load_recent_jobs_from_disk()
    cli_jobs = [j for j in jt._jobs.values() if j.job_id.startswith("cli_")]
    assert len(cli_jobs) == 1
    assert cli_jobs[0].rows[0]["species"] == "Species CLI"


# ============================================================
# 6. MainWindow.__init__ calls the scan automatically
# ============================================================
def test_main_window_calls_load_recent_jobs_on_init(tmp_work_dirs, monkeypatch):
    """Phase 49: MainWindow constructor must invoke the disk scan so
    the JobsTab is populated before the user opens it."""
    from rlpe.gui.main_window import MainWindow
    mw = MainWindow()
    try:
        # The scan must have populated _jobs_tab._jobs with our 3 jobs
        job_ids = list(mw._jobs_tab._jobs.keys())
        assert "job-a" in job_ids, (
            f"job-a should be loaded after MainWindow init, got {job_ids}"
        )
        assert "job-b" in job_ids
        # The CLI run uses cli_<hash>
        cli_jobs = [j for j in job_ids if j.startswith("cli_")]
        assert len(cli_jobs) == 1
    finally:
        mw.close()


def test_main_window_disk_scan_survives_scan_failure(monkeypatch):
    """Phase 49: even if the scan raises, the GUI must still start."""
    from rlpe.gui.jobs_tab import JobsTab
    # Force load_recent_jobs_from_disk to raise
    orig = JobsTab.load_recent_jobs_from_disk
    def boom(self):
        raise RuntimeError("simulated disk failure")
    monkeypatch.setattr(JobsTab, "load_recent_jobs_from_disk", boom)
    # MainWindow should swallow the exception (Phase 49 contract).
    from rlpe.gui.main_window import MainWindow
    mw = MainWindow()
    try:
        # GUI still starts; _jobs_tab exists but is empty
        assert mw._jobs_tab is not None
        assert mw._jobs_tab._jobs == {}
    finally:
        mw.close()
    # Restore for other tests
    JobsTab.load_recent_jobs_from_disk = orig


# ============================================================
# 7. Double-clicking a loaded job opens results in ResultsTab
# ============================================================
def test_loaded_job_can_be_opened_in_results_tab(tmp_work_dirs):
    """Phase 49: end-to-end — after the scan, calling MainWindow's
    _open_results on a loaded job should populate ResultsTab."""
    from rlpe.gui.main_window import MainWindow
    mw = MainWindow()
    try:
        # Sanity: job-a is loaded
        assert "job-a" in mw._jobs_tab._jobs
        # Simulate double-click: open the results tab
        mw._open_results("job-a")
        # ResultsTab should now have the job's rows
        assert mw._results_tab._current_job_id == "job-a"
        assert len(mw._results_tab._all_rows) == 2
        assert mw._results_tab._all_rows[0]["species"] == "Species A1"
    finally:
        mw.close()


# ============================================================
# 8. i18n key for status bar message
# ============================================================
def test_main_recent_loaded_i18n_key_exists():
    from rlpe.gui import strings_en, strings_zh_CN
    assert "main.recent_loaded" in strings_en.STRINGS
    assert "main.recent_loaded" in strings_zh_CN.STRINGS
    # Both have the {n} placeholder
    assert "{n}" in strings_en.STRINGS["main.recent_loaded"]
    assert "{n}" in strings_zh_CN.STRINGS["main.recent_loaded"]


# ============================================================
# 9. PROJECT_ROOT points to a Path
# ============================================================
def test_project_root_is_path():
    from rlpe.gui.constants import PROJECT_ROOT
    from pathlib import Path as _Path
    assert isinstance(PROJECT_ROOT, _Path), (
        f"PROJECT_ROOT must be a Path, got {type(PROJECT_ROOT).__name__}"
    )
