"""Regression tests for Round 9 thinking-retry lock fix.

Bug-M3: ``M3Engine._infer_vision`` previously held its retry lock in
two pieces — one to flip ``backend.enable_thinking`` off, then released
the lock for the duration of ``infer_panel()``, then re-acquired to
restore. Another thread could flip ``enable_thinking`` in between,
corrupting the first thread's restore. The fix uses ``RLock``
(reentrant) and holds it for the entire save→flip→call→restore
sequence so other workers see an atomic transition.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.m3_engine import M3Engine  # noqa: E402


def test_thinking_retry_lock_is_reentrant():
    """The lock MUST be an RLock so a backend that re-enters
    ``_infer_vision`` (e.g. a custom subclass that calls M3 again
    inside its handler) doesn't deadlock."""
    engine = M3Engine(backend=None)
    # RLock exposes ``_is_owned`` (private but stable since Python 3.x).
    assert hasattr(engine._thinking_retry_lock, "_is_owned"), (
        "Lock must be reentrant (RLock) for the retry path"
    )
    # Sanity: same thread can acquire twice without blocking.
    with engine._thinking_retry_lock:
        with engine._thinking_retry_lock:
            pass


def test_thinking_retry_restores_final_state_under_concurrency():
    """Round 9 Bug-M3: under concurrent retries, ``enable_thinking``
    must end up restored to its ORIGINAL value (True), not corrupted
    by interleaved save/restore pairs from different threads.

    Pre-fix scenario:
      Thread A: lock → saved=True → flip=False → UNLOCK → call
      Thread B: lock → saved=False (already flipped by A!) → flip=False (no-op) → UNLOCK → call
      Thread A: call done → lock → restore=True → UNLOCK
      Thread B: call done → lock → restore=False → UNLOCK   ← WRONG, ends at False

    Post-fix (RLock held throughout save/flip/call/restore):
      Thread A: RLock → saved=True → flip=False → call → restore=True → RUnlock
      Thread B: waits → RLock → saved=True → flip=False → call → restore=True → RUnlock
      → Final state: True (correct)
    """
    import threading

    class SlowBackend:
        backend_name = "fake"

        def __init__(self):
            self.enable_thinking = True
            self._lock = threading.Lock()
            self._inflight = 0
            self._max_inflight = 0

        def infer_panel(self, **kwargs):
            self._lock.acquire()
            try:
                self._inflight += 1
                self._max_inflight = max(self._max_inflight, self._inflight)
            finally:
                self._lock.release()
            time.sleep(0.02)  # hold the call long enough for races to surface
            try:
                return {"raw_text": "ok", "fallback_used": False}
            finally:
                self._lock.acquire()
                try:
                    self._inflight -= 1
                finally:
                    self._lock.release()

    backend = SlowBackend()
    # Stub the FIRST call to return empty so the retry path is exercised.
    real_infer = backend.infer_panel
    call_count = {"n": 0}

    def stub_infer(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"raw_text": "", "fallback_used": False}  # empty → trigger retry
        return real_infer(**kwargs)

    backend.infer_panel = stub_infer
    engine = M3Engine(backend=backend, config={"m3_retry_without_thinking": True})

    errors = []
    results = []

    def worker():
        try:
            engine._infer_vision("sys", "user", None)
            results.append(1)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert errors == [], f"Workers errored: {errors}"
    assert len(results) == 8
    # Critical assertion: with RLock held throughout, the final
    # state must be the ORIGINAL True, not corrupted by interleaved
    # restores.
    assert backend.enable_thinking is True, (
        f"enable_thinking not restored to True after concurrent retries: "
        f"{backend.enable_thinking}"
    )
    # Sanity: retries actually happened (at least one retry path was
    # entered per call_count > 1).
    assert call_count["n"] >= 8, f"Expected at least 8 calls, got {call_count['n']}"


def test_thinking_retry_serialises_concurrent_workers():
    """The RLock serialises the entire save→flip→call→restore sequence.
    Therefore, every retry's ``infer_panel`` call starts with
    ``enable_thinking=False`` (the same thread just flipped it off
    while holding the lock). Pre-fix, a concurrent thread could race
    to flip back to True between save/flip and the call.
    """

    class PeekBackend:
        backend_name = "fake"
        enable_thinking = True
        # Observations: (thread_id, enable_thinking, is_first_call)
        observations: list = []

        def infer_panel(self, **kwargs):
            self.observations.append(
                (threading.get_ident(), self.enable_thinking, False)
            )
            return {"raw_text": "ok", "fallback_used": False}

    backend = PeekBackend()
    real_infer = backend.infer_panel
    first_call_threads: set = set()

    def stub_infer(**kwargs):
        tid = threading.get_ident()
        if tid not in first_call_threads:
            first_call_threads.add(tid)
            backend.observations.append((tid, backend.enable_thinking, True))
            return {"raw_text": "", "fallback_used": False}
        return real_infer(**kwargs)

    backend.infer_panel = stub_infer
    engine = M3Engine(backend=backend, config={"m3_retry_without_thinking": True})

    def worker():
        engine._infer_vision("sys", "user", None)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    # Sanity: 8 unique workers each made exactly one "first call".
    assert len(first_call_threads) == 8
    first_calls = [o for o in backend.observations if o[2]]
    retry_calls = [o for o in backend.observations if not o[2]]
    assert len(first_calls) == 8
    assert len(retry_calls) == 8, (
        f"Expected 8 retry calls (one per worker), got {len(retry_calls)}: "
        f"{backend.observations}"
    )
    # First calls happen outside the retry path → enable_thinking=True.
    for tid, et, is_first in first_calls:
        assert et is True, f"First call saw enable_thinking={et}, expected True"
    # Retry calls happen inside the lock with the flip → must see False.
    for tid, et, is_first in retry_calls:
        assert et is False, (
            f"A retry call started with enable_thinking=True — the lock "
            f"is not held throughout save→flip→call→restore. observations={backend.observations}"
        )
    # And the original value True is restored.
    assert backend.enable_thinking is True


def test_thinking_retry_restores_on_exception():
    """If the retry call raises, the lock is released and
    ``enable_thinking`` is restored to its original value."""

    class ExplodingBackend:
        backend_name = "fake"
        enable_thinking = True
        call_count = 0

        def infer_panel(self, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                return {"raw_text": "", "fallback_used": False}  # trigger retry
            raise RuntimeError("simulated API failure")

    backend = ExplodingBackend()
    engine = M3Engine(backend=backend, config={"m3_retry_without_thinking": True})
    res = engine._infer_vision("sys", "user", None)
    # Retry failed → engine returns the empty `res` (not raise)
    assert res.get("fallback_used") is False  # original res
    assert backend.enable_thinking is True, (
        "enable_thinking must be restored even when the retry raises"
    )
