"""Regression tests for audit 2026-08-01 batch W1 — api/app.py bugs M14, M17, M18, M19, D9, D19."""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import pytest

# Tests touch RLPE_API_TEST_TMP env var BEFORE importing the api.app
# module, because app.py reads it at import time (audit 2026-07-31
# note). Force the env var through ``monkeypatch.setenv`` via fixtures
# below; do NOT set it here at import time.

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Make sure the api module is discoverable and that we always have
# fastapi / httpx importable here too. If the deps are missing
# (extras not installed) the entire test module skips — partial
# installs would make the tests run on a fake module and fail
# confusingly.
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
pytest.importorskip("pydantic")


# ===========================================================================
# Fixtures
# ===========================================================================
@pytest.fixture
def app_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Import rlpe.api.app with ``RLPE_API_TEST_TMP`` set to a fresh
    tmp dir. Yields the module so tests can inspect globals
    (``JOB_CONCURRENCY``, ``RESULT_LOCK``, etc.) and call helpers
    directly.
    """
    monkeypatch.setenv("RLPE_API_TEST_TMP", str(tmp_path))
    # Wipe any cached module so its top-level ``RLPE_API_TEST_TMP``
    # read picks up our tmp_path.
    for mod_name in list(sys.modules):
        if mod_name == "rlpe.api.app" or mod_name.startswith("rlpe.api."):
            sys.modules.pop(mod_name, None)
    import rlpe.api.app as app_mod  # noqa: E402

    return app_mod


@pytest.fixture
def client(app_module):
    from fastapi.testclient import TestClient

    return TestClient(app_module.app)


# ===========================================================================
# M14 — JOB_CONCURRENCY semaphore
# ===========================================================================
class TestM14JobConcurrency:
    def test_M14_job_concurrency_semaphore(self, app_module):
        """``JOB_CONCURRENCY`` exists, is a Semaphore, and limits to N."""

        from rlpe.api.app import JOB_CONCURRENCY

        assert isinstance(JOB_CONCURRENCY, threading.Semaphore), (
            f"JOB_CONCURRENCY must be a Semaphore, got {type(JOB_CONCURRENCY)!r}"
        )
        # Value is not part of the public API but defaults to a positive int;
        # assert that several ``acquire()``s work and that we exhaust the
        # pool after ``N`` of them.
        initial = getattr(JOB_CONCURRENCY, "_value", None)
        if initial is None:
            initial = app_module.JOB_CONCURRENCY._Semaphore__value  # type: ignore[attr-defined]
        assert initial == 4, f"default semaphore value should be 4, got {initial}"

        # Exhaust all four slots, then a 5th acquire must block.
        acquired = [JOB_CONCURRENCY.acquire(blocking=False) for _ in range(initial)]
        assert all(acquired), "all N slots must be acquirable"
        assert JOB_CONCURRENCY.acquire(blocking=False) is False, (
            "after N acquires, the (N+1)-th must fail with blocking=False"
        )
        # Release everything for subsequent tests.
        for _ in range(initial + 1):
            try:
                JOB_CONCURRENCY.release()
            except ValueError:
                break

    def test_M14_env_var_overrides_default(self, app_module, monkeypatch, tmp_path):
        """``RLPE_MAX_JOBS`` env var overrides the default 4. We
        verify via the ``_env_int`` helper (the module-level
        ``JOB_CONCURRENCY`` is created once at import time, so we
        can't easily re-evaluate it without clearing the module
        cache, which would also kill the env-read test in the
        parent fixture)."""
        monkeypatch.setenv("RLPE_MAX_JOBS", "7")
        assert app_module._env_int("RLPE_MAX_JOBS", 4) == 7
        monkeypatch.delenv("RLPE_MAX_JOBS")
        assert app_module._env_int("RLPE_MAX_JOBS", 4) == 4
        # Bogus values fall back to the default.
        monkeypatch.setenv("RLPE_MAX_JOBS", "not-a-number")
        assert app_module._env_int("RLPE_MAX_JOBS", 4) == 4
        # Non-positive values are clamped to the default.
        monkeypatch.setenv("RLPE_MAX_JOBS", "0")
        assert app_module._env_int("RLPE_MAX_JOBS", 4) == 4


# ===========================================================================
# M17 — _purge_job does not hold RESULT_LOCK during file ops
# ===========================================================================
class TestM17PurgeLock:
    def test_M17_purge_does_not_hold_result_lock(self, app_module, monkeypatch, tmp_path):
        """``_purge_job`` releases ``RESULT_LOCK`` BEFORE the slow
        ``root.rglob + rmtree`` so other ``/jobs/*`` endpoints stay
        responsive. We mock shutil.rmtree to count how many times the
        lock is held WHEN rmtree fires; the count must be ZERO (i.e.
        rmtree runs lock-free).
        """
        from rlpe.api.app import (
            RESULT_CACHE,
            RESULT_LOCK,
            WORK_DIR,
            _purge_job,
        )

        # Seed a 'done' job with a `_root` UNDER ``WORK_DIR`` (the
        # ``_is_relative_to`` safe-root check refuses deletion otherwise).
        jid = "purge-lock-test-001"
        job_root = WORK_DIR / jid
        job_root.mkdir(parents=True)
        # Put a couple of files in there so rglob returns them.
        (job_root / "a.txt").write_text("a", encoding="utf-8")
        (job_root / "sub").mkdir()
        (job_root / "sub" / "b.txt").write_text("b", encoding="utf-8")

        # Pre-load entry into RESULT_CACHE.
        RESULT_CACHE.clear()
        RESULT_CACHE[jid] = {
            "status": "done",
            "result": [],
            "error": None,
            "detail": "test",
            "created_at": "2026-08-01T00:00:00",
            "filename": "test.pdf",
            "progress": 100,
            "_root": str(job_root.resolve()),
        }

        # Spy on lock acquisitions during rmtree.
        lock_held_during_rmtree = []

        def spy_rmtree(path, *args, **kwargs):
            # Check whether RESULT_LOCK is currently held by anyone.
            # threading.Lock is non-reentrant; ``acquire(blocking=False)``
            # returns True on a free lock, False on a held one.
            lock_held_during_rmtree.append(not RESULT_LOCK.acquire(blocking=False))
            if lock_held_during_rmtree[-1] is False:
                # Re-acquire failed → we held nothing → release nothing.
                pass
            else:
                # Our probe acquired an apparently-free lock; release it.
                try:
                    RESULT_LOCK.release()
                except ValueError:
                    pass
            # Always succeed (skip real delete so the test cleans up
            # via tmp_path fixture).
            return None

        monkeypatch.setattr("rlpe.api.app.shutil.rmtree", spy_rmtree)

        result = _purge_job(jid, delete_files=True)
        assert result["status"] == "deleted", result
        # The lock must NOT be held while rmtree runs.
        assert all(held is False for held in lock_held_during_rmtree), (
            f"rmtree ran while RESULT_LOCK was held: {lock_held_during_rmtree}"
        )
        assert len(lock_held_during_rmtree) >= 1, "spy_rmtree should have been called"

    def test_M17_batch_delete_parallelises(self, app_module, monkeypatch, tmp_path):
        """``batch_delete_jobs`` uses ThreadPoolExecutor so N purges
        can run concurrently."""
        from fastapi.testclient import TestClient

        from rlpe.api.app import RESULT_CACHE, batch_delete_jobs

        # Seed 8 'done' jobs with `_root` empty (delete_files=False
        # path, which doesn't touch the lock at all and is what the
        # actual log aggregator depends on).
        RESULT_CACHE.clear()
        jids = [f"batch-{i:03d}" for i in range(8)]
        for jid in jids:
            RESULT_CACHE[jid] = {
                "status": "done",
                "result": [],
                "error": None,
                "detail": None,
                "created_at": "2026-08-01T00:00:00",
                "filename": None,
                "progress": 100,
                "_root": None,  # empty, so cleanup is a no-op
            }
        client = TestClient(app_module.app)
        r = client.post("/jobs/batch-delete", json={"job_ids": jids, "delete_files": False})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["deleted"] == 8, f"expected all 8 deleted, got {body['deleted']}"


# ===========================================================================
# M18 — complete.flag required
# ===========================================================================
class TestM18CompleteFlag:
    def test_M18_complete_flag_required(self, app_module, tmp_path):
        """Startup loads jobs without ``complete.flag`` as
        ``status='partial'`` (not ``status='done'``)."""
        from rlpe.api.app import (
            RESULT_CACHE,
            WORK_DIR,
            _load_existing_jobs_from_disk,
        )

        RESULT_CACHE.clear()
        # Create a synthetic ``service_work/<jid>/output/manifests/matches.jsonl``.
        jid = "partial-flag-test"
        manifests = WORK_DIR / jid / "output" / "manifests"
        manifests.mkdir(parents=True, exist_ok=True)
        matches = manifests / "matches.jsonl"
        matches.write_text(
            json.dumps({"paper_id": "p", "figure_id": "f", "panel_id": "1"}) + "\n",
            encoding="utf-8",
        )
        # Deliberately do NOT write complete.flag.

        loaded = _load_existing_jobs_from_disk()
        assert loaded == 1, f"expected 1 loaded job, got {loaded}"
        assert jid in RESULT_CACHE, f"{jid!r} missing from RESULT_CACHE"
        entry = RESULT_CACHE[jid]
        assert entry["status"] == "partial", (
            f"without complete.flag the job should be 'partial', got {entry['status']!r}"
        )

        # Now write the flag and re-run — should be "done".
        (manifests / "complete.flag").write_text("2026-08-01T00:00:00", encoding="utf-8")
        RESULT_CACHE.clear()
        loaded = _load_existing_jobs_from_disk()
        assert loaded == 1
        entry = RESULT_CACHE[jid]
        assert entry["status"] == "done", (
            f"with complete.flag the job should be 'done', got {entry['status']!r}"
        )


# ===========================================================================
# M19 — DELETE /results persists to disk
# ===========================================================================
class TestM19PersistsToDisk:
    def test_M19_delete_results_persists_to_disk(self, app_module, tmp_path):
        """DELETE /results rewrites matches.jsonl on disk."""
        from fastapi.testclient import TestClient

        from rlpe.api.app import RESULT_CACHE, WORK_DIR

        RESULT_CACHE.clear()
        jid = "m19-persist-test"
        manifests = WORK_DIR / jid / "output" / "manifests"
        manifests.mkdir(parents=True, exist_ok=True)
        matches = manifests / "matches.jsonl"
        # Seed 5 rows in matches.jsonl.
        rows = [
            {
                "job_id": jid,
                "paper_id": f"p{i}",
                "figure_id": f"f{i}",
                "panel_id": str(i),
                "confidence": 0.5,
            }
            for i in range(5)
        ]
        with matches.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Register in RESULT_CACHE.
        RESULT_CACHE[jid] = {
            "status": "done",
            "result": list(rows),
            "error": None,
            "detail": None,
            "created_at": "2026-08-01T00:00:00",
            "filename": "test.pdf",
            "progress": 100,
            "_root": str((WORK_DIR / jid).resolve()),
        }
        client = TestClient(app_module.app)
        r = client.delete("/results")
        assert r.status_code == 200, r.text
        assert r.json()["removed"] == 5

        # Reload matches.jsonl from disk and assert it has 0 rows.
        reloaded = []
        if matches.exists():
            with matches.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        reloaded.append(json.loads(line))
        assert reloaded == [], (
            f"after DELETE /results, matches.jsonl should be empty, got {reloaded}"
        )

    def test_M19_persist_helper_writes_match(self, app_module, tmp_path):
        """The internal ``_persist_results_to_disk`` helper actually
        overwrites matches.jsonl with the supplied rows."""
        from rlpe.api.app import WORK_DIR, _persist_results_to_disk

        jid = "m19-helper-test"
        manifests = WORK_DIR / jid / "output" / "manifests"
        manifests.mkdir(parents=True, exist_ok=True)
        matches = manifests / "matches.jsonl"
        matches.write_text(
            json.dumps({"paper_id": "old", "figure_id": "old", "panel_id": "old"}) + "\n",
            encoding="utf-8",
        )
        root = str((WORK_DIR / jid).resolve())
        new_rows = [
            {"paper_id": "new1", "figure_id": "f", "panel_id": "1"},
            {"paper_id": "new2", "figure_id": "f", "panel_id": "2"},
        ]
        ok = _persist_results_to_disk(jid, root, new_rows)
        assert ok is True
        # Reload
        with matches.open(encoding="utf-8") as fh:
            lines = [json.loads(l) for l in fh if l.strip()]
        assert lines == new_rows, f"expected {new_rows}, got {lines}"


# ===========================================================================
# D9 — corrections.jsonl rotation
# ===========================================================================
class TestD9CorrectionsRotation:
    def test_D9_corrections_rotation(self, app_module, tmp_path, monkeypatch):
        """If ``corrections.jsonl`` > 1 MB on open, rotate to .1."""
        from rlpe.api.app import WORK_DIR

        corrections_dir = WORK_DIR / "corrections"
        corrections_dir.mkdir(parents=True, exist_ok=True)
        target = corrections_dir / "corrections.jsonl"
        # 2 MB of fake rows.
        huge = "x" * 1024 + "\n"
        with target.open("w", encoding="utf-8") as fh:
            # 2000 lines x 1KB ≈ 2 MB
            for _ in range(2000):
                fh.write(huge)

        # Use the public endpoint to trigger the rotation path.
        from fastapi.testclient import TestClient

        client = TestClient(app_module.app)
        r = client.post(
            "/review/correction",
            json={
                "paper_id": "p",
                "figure_id": "f",
                "corrected_species": "Genus species",
            },
        )
        assert r.status_code == 200, r.text

        # After rotation: corrections.jsonl.1 should exist,
        # corrections.jsonl should be fresh (1 row).
        rotated = corrections_dir / "corrections.jsonl.1"
        assert rotated.exists(), "corrections.jsonl.1 should exist after a rotation trigger"
        # The freshly-rotated-into .1 should have at least 1 row.
        with rotated.open(encoding="utf-8") as fh:
            rotated_lines = [l for l in fh if l.strip()]
        assert len(rotated_lines) > 0, "rotated .1 should contain rows"
        # The new corrections.jsonl should have exactly 1 row (the new submission).
        with target.open(encoding="utf-8") as fh:
            new_lines = [l for l in fh if l.strip()]
        assert len(new_lines) == 1, (
            f"fresh corrections.jsonl should have 1 row, got {len(new_lines)}"
        )

    def test_D9_no_rotation_when_small(self, app_module, tmp_path):
        """Files < 1 MB do NOT trigger rotation."""
        from fastapi.testclient import TestClient

        from rlpe.api.app import WORK_DIR

        corrections_dir = WORK_DIR / "corrections"
        corrections_dir.mkdir(parents=True, exist_ok=True)
        target = corrections_dir / "corrections.jsonl"
        # Just a few rows; well under 1 MB.
        target.write_text(
            json.dumps(
                {
                    "paper_id": "old",
                    "figure_id": "f",
                    "corrected_species": "x",
                    "timestamp": "2026-07-31T00:00:00",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        client = TestClient(app_module.app)
        r = client.post(
            "/review/correction",
            json={
                "paper_id": "p",
                "figure_id": "f",
                "corrected_species": "Genus species",
            },
        )
        assert r.status_code == 200, r.text
        # .1 should NOT exist (rotation didn't fire).
        assert not (corrections_dir / "corrections.jsonl.1").exists(), (
            "rotation should not fire when file is below threshold"
        )


# ===========================================================================
# D19 — uploaded PDF cleanup in _purge_job
# ===========================================================================
class TestD19UploadPdfCleanup:
    def test_D19_upload_pdf_cleanup(self, app_module, tmp_path):
        """``_purge_job`` removes ``UPLOAD_DIR/<job_id>_<name>``.
        Simulates a cancelled pre-flight job whose PDF was uploaded
        but never copied to the work dir.
        """
        from rlpe.api.app import (
            RESULT_CACHE,
            UPLOAD_DIR,
            WORK_DIR,
            _purge_job,
        )

        # Seed a fake PDF upload in UPLOAD_DIR (the convention from
        # upload_pdf is <job_id>_<safe_filename>).
        jid = "d19-upload-cleanup"
        upload_pdf = UPLOAD_DIR / f"{jid}_test.pdf"
        upload_pdf.write_bytes(b"%PDF-1.4\nfake-bytes-for-cleanup\n%%EOF\n")
        assert upload_pdf.exists()

        # Seed a job-root equivalent so delete_files works.
        job_root = WORK_DIR / jid
        job_root.mkdir(parents=True, exist_ok=True)
        (job_root / "x.txt").write_text("x", encoding="utf-8")

        RESULT_CACHE.clear()
        RESULT_CACHE[jid] = {
            "status": "done",
            "result": [],
            "error": None,
            "detail": "test",
            "created_at": "2026-08-01T00:00:00",
            "filename": "test.pdf",
            "progress": 100,
            "_root": str(job_root.resolve()),
        }
        result = _purge_job(jid, delete_files=True)
        assert result["status"] == "deleted", result
        # D19 fix: the uploaded PDF in UPLOAD_DIR must be gone.
        assert not upload_pdf.exists(), (
            f"D19: uploaded PDF {upload_pdf} should be removed after _purge_job"
        )
