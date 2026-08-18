"""Sweep 7 (2026-08-02 audit follow-up): API robustness cleanup.

Five small fixes landed in this sweep:

1. **N2** — ``_run_job`` finally block now drops the
   ``MiniMax_fallback_handler`` from ``RESULT_CACHE[jid]`` AND pops
   ``FALLBACK_PENDING[jid]`` so the ``_web_fallback_popup`` closure
   (which captures ``error_info``, the threading.Event, and a
   back-reference to ``_run_job``'s frame) doesn't pin the entry for
   up to 5 minutes after the worker exits.

2. **N3** — ``JOB_CONCURRENCY.acquire()`` now has a 60-second timeout
   (``_JOB_QUEUE_TIMEOUT``, env var ``RLPE_JOB_QUEUE_TIMEOUT``). The
   bare ``acquire()`` blocked forever when ``RLPE_MAX_JOBS=1`` and a
   burst of 5+ uploads arrived. On timeout the job is marked
   ``"queued_timeout"`` with an actionable detail string so the client
   knows to retry instead of polling forever.

3. **O5** — ``hb_thread.start()`` moved from BEFORE the pre-flight
   cancel check to AFTER it. A cancelled-during-queue job no longer
   spawns the heartbeat thread (saves a Thread + RESULT_CACHE[jid]
   ref for the 1s tick before ``stop_hb.set()`` would have cleaned up).

4. **C2** — The lone ``del FALLBACK_PENDING[job_id]`` in
   ``_web_fallback_popup`` was wrapped in ``try/except KeyError``;
   replaced with ``.pop(job_id, None)`` for parity with the other
   3 cleanup sites. ``.pop`` is one line and doesn't need the
   exception guard.

5. **C5** — ``_root`` storage consistency: line 596 stored
   ``str(root.resolve())`` but line 785 stored ``str(WORK_DIR / job_id)``
   (unresolved). Brought line 785 into parity with
   ``str((WORK_DIR / job_id).resolve())`` so the cached string is
   canonical (defensive ``.resolve()`` at the read site still
   normalises, but a stable write form is more correct).

These tests pin the design so a future refactor doesn't silently
regress.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_APP = _REPO_ROOT / "src" / "rlpe" / "api" / "app.py"


class TestSweep7N2FallbackClosureRelease:
    """N2 — finally block releases the MiniMax fallback closure."""

    def test_finally_drops_MiniMax_fallback_handler(self):
        """The finally block of ``_run_job`` must ``pop("MiniMax_fallback_handler")``
        from the job's RESULT_CACHE entry."""
        src = _SRC_APP.read_text(encoding="utf-8")
        assert '"MiniMax_fallback_handler"' in src
        assert 'entry.pop("MiniMax_fallback_handler", None)' in src, (
            "_run_job finally block must drop MiniMax_fallback_handler "
            "from RESULT_CACHE[jid] so the _web_fallback_popup closure "
            "isn't pinned for up to 5 minutes after the worker exits"
        )

    def test_finally_pops_fallback_pending(self):
        """The finally block must also unconditionally pop
        FALLBACK_PENDING[jid] so a worker that exited via the exception
        path doesn't leak the entry forever."""
        src = _SRC_APP.read_text(encoding="utf-8")
        # We expect at least 4 occurrences of ``FALLBACK_PENDING.pop(job_id, None)``
        # (cancel_job, delete_job, the popup cleanup, and finally).
        count = src.count("FALLBACK_PENDING.pop(job_id, None)")
        assert count >= 4, (
            f"Expected ≥4 FALLBACK_PENDING.pop(job_id, None) sites; "
            f"found {count}. The new finally-block cleanup is missing."
        )


class TestSweep7N3ConcurrencyAcquireTimeout:
    """N3 — bounded acquire with 503-equivalent status."""

    def test_job_queue_timeout_constant_exists(self):
        """``_JOB_QUEUE_TIMEOUT`` must be defined as a module-level
        constant in api/app.py so operators can tune via env var."""
        from rlpe.api import app

        assert hasattr(app, "_JOB_QUEUE_TIMEOUT"), (
            "_JOB_QUEUE_TIMEOUT missing from api/app.py — "
            "JOB_CONCURRENCY.acquire() still blocks forever on bursts"
        )
        # Default 60s; testable as float for math.future-proof.
        assert isinstance(app._JOB_QUEUE_TIMEOUT, float)
        assert app._JOB_QUEUE_TIMEOUT > 0

    def test_rlpe_max_jobs_constant_exists(self):
        """``_RLPE_MAX_JOBS`` module-level constant is needed for the
        503 detail message to echo the configured limit."""
        from rlpe.api import app

        assert hasattr(app, "_RLPE_MAX_JOBS"), (
            "_RLPE_MAX_JOBS missing from api/app.py — "
            "queued_timeout detail can't reference the configured limit"
        )
        assert isinstance(app._RLPE_MAX_JOBS, int)
        assert app._RLPE_MAX_JOBS >= 1

    def test_concurrency_acquire_uses_timeout(self):
        """Source guard: ``JOB_CONCURRENCY.acquire(timeout=...)``."""
        src = _SRC_APP.read_text(encoding="utf-8")
        # The bounded-acquire pattern must exist; bare ``acquire()``
        # would block forever.
        assert "JOB_CONCURRENCY.acquire(timeout=" in src, (
            "JOB_CONCURRENCY.acquire() must use a timeout kwarg; "
            "bare acquire() blocks forever on bursts (5+ uploads "
            "with RLPE_MAX_JOBS=1 leaves the 5th job stuck in "
            "'queued' indefinitely)"
        )
        # The old bare call must be gone.
        assert "JOB_CONCURRENCY.acquire()\n" not in src, (
            "Bare JOB_CONCURRENCY.acquire() still present in api/app.py"
        )

    def test_concurrency_timeout_marks_queued_timeout(self):
        """Source guard: when the semaphore can't be acquired, the
        job's RESULT_CACHE entry is marked ``queued_timeout`` (not
        left in ``queued`` forever)."""
        src = _SRC_APP.read_text(encoding="utf-8")
        assert '"queued_timeout"' in src, (
            "queued_timeout status not set on semaphore-acquire timeout — "
            "clients polling /jobs/{id}/status never get actionable feedback"
        )


class TestSweep7O5HeartbeatAfterPreflight:
    """O5 — hb_thread.start() must come AFTER the pre-flight cancel check."""

    def test_heartbeat_starts_after_preflight(self):
        """``hb_thread.start()`` must appear AFTER the pre-flight
        cancel check that bails on ``status == 'cancelled'``."""
        src = _SRC_APP.read_text(encoding="utf-8")
        # Find the pre-flight cancel check.
        preflight_idx = src.find('if cur.get("status") == "cancelled":')
        assert preflight_idx > 0, "pre-flight cancel check missing"
        # Find hb_thread.start() — must appear AFTER the pre-flight.
        hb_start_idx = src.find("hb_thread.start()")
        assert hb_start_idx > 0, "hb_thread.start() missing"
        assert hb_start_idx > preflight_idx, (
            "hb_thread.start() is BEFORE the pre-flight cancel check — "
            "cancelled-during-queue jobs spawn a heartbeat thread that "
            "leaks a Thread + RESULT_CACHE[jid] ref for the 1s tick"
        )

    def test_preflight_check_appears_only_in_one_block(self):
        """There must be exactly ONE ``if cur.get('status') == 'cancelled'``
        pre-flight check (not duplicated)."""
        src = _SRC_APP.read_text(encoding="utf-8")
        count = src.count('if cur.get("status") == "cancelled":')
        assert count == 1, (
            f"pre-flight cancel check duplicated {count}× — "
            f"the O5 refactor must MOVE the heartbeat, not duplicate the check"
        )


class TestSweep7C2FallbackPendingCleanup:
    """C2 — fallback_pending cleanup uses consistent .pop(..., None)."""

    def test_no_del_fallback_pending(self):
        """The lone ``del FALLBACK_PENDING[job_id]`` must be gone."""
        src = _SRC_APP.read_text(encoding="utf-8")
        assert "del FALLBACK_PENDING[job_id]" not in src, (
            "del FALLBACK_PENDING[job_id] still present — use "
            ".pop(job_id, None) for parity with the other 3 cleanup sites"
        )

    def test_pop_fallback_pending_consistent(self):
        """All FALLBACK_PENDING cleanups must use the same
        ``.pop(job_id, None)`` pattern."""
        src = _SRC_APP.read_text(encoding="utf-8")
        pop_count = src.count("FALLBACK_PENDING.pop(job_id, None)")
        del_count = src.count("del FALLBACK_PENDING[")
        # ≥4 sites, all using the same pattern.
        assert pop_count >= 4, (
            f"Expected ≥4 FALLBACK_PENDING.pop(job_id, None); got {pop_count}"
        )
        assert del_count == 0, (
            f"del FALLBACK_PENDING still appears {del_count}× — "
            f"replace with .pop(..., None) for consistency"
        )


class TestSweep7C5RootStorageConsistency:
    """C5 — _root storage is consistent (always .resolve()-d)."""

    def test_root_storage_calls_resolve(self):
        """Both ``_root`` storage sites must call ``.resolve()``.
        Source guard ensures future additions follow the same pattern."""
        src = _SRC_APP.read_text(encoding="utf-8")
        # Find every line that writes to ``_root`` and check resolve().
        # Both should produce absolute, resolved path strings.
        # The "str(root.resolve())" pattern at line ~596 is the
        # CLI-discovered path; the new web-upload path uses
        # "str((WORK_DIR / job_id).resolve())".
        assert 'str(root.resolve())' in src, (
            "CLI-discovered _root site no longer calls .resolve()"
        )
        assert 'str((WORK_DIR / job_id).resolve())' in src, (
            "Web-upload _root site no longer calls .resolve() — "
            "storing an unresolved path can differ from the .resolve()'d "
            "form the CLI-discovered site produces, breaking "
            "string-equality checks in the audit log + cache eviction"
        )

    def test_no_unresolved_root_storage(self):
        """Source guard: no remaining ``"_root": str(... )`` site
        stores an unresolved path."""
        src = _SRC_APP.read_text(encoding="utf-8")
        import re

        # Find every '"_root": str(...)' assignment. Use a non-greedy
        # match and a manual bracket-balance scan so we don't get tripped
        # up by nested parens like ``str((WORK_DIR / job_id).resolve())``.
        bad_sites: list[str] = []
        for m in re.finditer(r'"_root":\s*str\(', src):
            start = m.end()  # just after "str("
            depth = 1
            i = start
            while i < len(src) and depth > 0:
                ch = src[i]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                i += 1
            inner = src[start:i - 1]
            if ".resolve()" not in inner:
                bad_sites.append(inner)
        assert not bad_sites, (
            "Found _root storage without .resolve(): "
            + ", ".join(f"str({s!r})" for s in bad_sites)
            + ". All _root storage sites must .resolve() the path."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])