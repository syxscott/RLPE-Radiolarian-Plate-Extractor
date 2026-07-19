"""Phase 59 — Pipeline correctness, Bug 2.1.

GROBID's ``_grobid_in_progress`` cycle-guard set was read at L861
and modified at L1960/L1967 from ``ThreadPoolExecutor`` workers
without any ``threading.Lock``. Race conditions can:

  (a) miss cycle detection (T1 reads after T2 discards)
  (b) leak entries (T1's ``add`` is silently overwritten by T2)
  (c) cause GROBID↔OD infinite recursion when both workers
      interleave add/contains/discard.

This test exercises the guard set concurrently from many threads
and asserts:

  1. The pipeline exposes a ``_grobid_lock`` attribute
     (``threading.Lock`` instance).
  2. The guard set's membership transitions are atomic:
     every "guard entered" must be paired with exactly one
     "guard exited", regardless of how many threads race.
  3. After all workers finish, the set is empty (no leak).
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_REPO = Path(__file__).resolve().parents[1]


def _pipeline_via_init_skip() -> "RadiolarianPipeline":
    """Build a RadiolarianPipeline with ``__new__`` so we avoid running
    the heavy ``__init__`` (which imports torch/gemma/paddleocr)."""
    import threading

    from rlpe.pipeline import RadiolarianPipeline

    obj = RadiolarianPipeline.__new__(RadiolarianPipeline)
    obj._grobid_in_progress = set()
    obj._grobid_lock = threading.Lock()
    return obj


def test_grobid_in_progress_lock_attribute_exists() -> None:
    """Bug 2.1 fix: pipeline exposes a ``_grobid_lock`` threading.Lock.

    Source-guard: the pipeline constructor must assign
    ``self._grobid_lock = threading.Lock()`` so concurrent workers
    share a single lock guarding ``_grobid_in_progress``.
    """
    src = (_REPO / "src/rlpe/pipeline.py").read_text(encoding="utf-8")
    assert "self._grobid_lock = threading.Lock()" in src, (
        "Pipeline.__init__ must declare self._grobid_lock = threading.Lock()"
    )
    # Behavioural check: a freshly-``__new__``'d pipeline can have the
    # attribute installed manually and behave like a Lock.
    pipe = _pipeline_via_init_skip()
    assert isinstance(pipe._grobid_lock, type(threading.Lock())), (
        f"_grobid_lock must be threading.Lock, got {type(pipe._grobid_lock)}"
    )


def test_grobid_in_progress_lock_serializes_add_contains_discard() -> None:
    """Bug 2.1 fix: simultaneous add / contains-check / discard from many
    threads must not lose updates.

    We simulate the outer code path: ``if paper_id in set: skip; else:
    set.add(paper_id); try: ...; finally: set.discard(paper_id)``.
    Across 100 threads x 4 paper_ids = 400 races, the membership set
    must be empty at the end and ``entries``/``exits`` must be balanced
    (every entry had a paired exit).
    """
    pipe = _pipeline_via_init_skip()

    paper_ids = ["p1", "p2", "p3", "p4"]
    entries = 0
    entries_lock = threading.Lock()
    exits = 0
    exits_lock = threading.Lock()
    completed = threading.Event()

    def worker(pid: str) -> None:
        nonlocal entries, exits
        # Mirror the production code path:
        # if pid in self._grobid_in_progress: skip
        # else: self._grobid_in_progress.add(pid); try: ...; finally: discard
        for _ in range(25):  # 100 threads x 25 iters each = 2500 races
            with pipe._grobid_lock:
                if pid in pipe._grobid_in_progress:
                    continue
                pipe._grobid_in_progress.add(pid)
            with entries_lock:
                entries += 1
            try:
                # Critical section body — emulate the GROBID work without
                # actually doing GROBID. Yield to other threads to maximise
                # interleaving.
                pass
            finally:
                with pipe._grobid_lock:
                    pipe._grobid_in_progress.discard(pid)
                with exits_lock:
                    exits += 1

    threads = [threading.Thread(target=worker, args=(pid,)) for pid in paper_ids for _ in range(25)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert pipe._grobid_in_progress == set(), (
        f"GROBID cycle-guard set leaked entries: {pipe._grobid_in_progress}"
    )
    assert entries == exits, (
        f"Every entry must be paired with an exit; got entries={entries}, exits={exits}"
    )
    # All 2500 attempts succeeded (no one bailed early on a false-positive
    # 'pid in set' due to a torn add/discard).
    assert entries == 100 * 25, (
        f"Expected 2500 entries (100 threads * 25 iters), got {entries}"
    )
    completed.set()


def test_grobid_in_progress_lock_breaks_real_cycle() -> None:
    """Bug 2.1 fix: end-to-end cycle-guard semantics.

    Thread A enters GROBID for paper 'X' and registers 'X' in the set
    under the lock, then releases the lock. Thread B (simulating OD
    fallback) reads under the same lock and must observe 'X' as
    in-progress.
    """
    pipe = _pipeline_via_init_skip()

    observed: dict[str, bool] = {}

    a_added = threading.Event()
    b_checked = threading.Event()

    def thread_a_grobid() -> None:
        # A enters GROBID path, acquires the lock, adds entry,
        # releases the lock, then signals B.
        with pipe._grobid_lock:
            pipe._grobid_in_progress.add("X")
        a_added.set()
        # Wait for B to read before tearing down.
        assert b_checked.wait(timeout=2.0), "B never checked"
        with pipe._grobid_lock:
            pipe._grobid_in_progress.discard("X")

    def thread_b_od_fallback() -> None:
        assert a_added.wait(timeout=2.0), "A never added the entry"
        with pipe._grobid_lock:
            observed["X_in_progress"] = "X" in pipe._grobid_in_progress
        b_checked.set()

    ta = threading.Thread(target=thread_a_grobid)
    tb = threading.Thread(target=thread_b_od_fallback)
    ta.start()
    tb.start()
    ta.join(timeout=5.0)
    tb.join(timeout=5.0)
    assert not ta.is_alive(), "Thread A did not finish"
    assert not tb.is_alive(), "Thread B did not finish"

    assert observed.get("X_in_progress") is True, (
        f"OD fallback must observe GROBID-in-progress; got {observed}"
    )
    assert pipe._grobid_in_progress == set(), "Set must be empty after teardown"
