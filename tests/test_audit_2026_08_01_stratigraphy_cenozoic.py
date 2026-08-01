"""Regression tests for audit 2026-08-01 batch W2 — stratigraphy C4 cenozoic stages + D1 PBDB cache lock/neg/atomic."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rlpe.stratigraphy as strat  # noqa: E402
from rlpe.stratigraphy import (  # noqa: E402
    _ICS_ROWS,
    _PBDB_INTERVALS_CACHE,
    _PBDB_INTERVALS_LOCK,
    _PBDB_INTERVALS_NEG_CACHE,
    _PBDB_NEG_TTL_SECONDS,
    fetch_pbdb_intervals,
    find_ages_in_text,
)


def _reset_pbdb_state() -> None:
    """Reset module-level PBDB cache + negative cache between tests."""
    with strat._PBDB_INTERVALS_LOCK:
        strat._PBDB_INTERVALS_CACHE = None
        strat._PBDB_LAST_FETCH = 0.0
        strat._PBDB_INTERVALS_NEG_CACHE.clear()


# ---------------------------------------------------------------------------
# C4 — missing Cenozoic stages in _ICS_ROWS
# ---------------------------------------------------------------------------


class TestCenozoicStages:
    """Bug C4: the previous ICS table only had the period / epoch ranks
    (Paleogene, Eocene, Oligocene, …) and the bare Quaternary ages.
    Specific stages (Priabonian, Burdigalian, Calabrian, …) fell
    through to the (often unavailable) PBDB network fallback.
    """

    def _find_stage(self, name: str) -> dict:
        for r in _ICS_ROWS:
            if r["name"] == name:
                return r
        raise AssertionError(f"{name!r} missing from _ICS_ROWS")

    def test_priabonian_lookup(self):
        """Priabonian must resolve as an age of the Eocene with the
        correct ICS 2023 Ma bounds (37.71-33.9)."""
        ages = find_ages_in_text("Priabonian, Tunisia")
        priab = [a for a in ages if a.age == "Priabonian"]
        assert priab, f"Priabonian not found in {ages!r}"
        cls = priab[0]
        assert cls.period == "Paleogene"
        assert cls.confidence > 0
        assert cls.ma_top == 33.9
        assert cls.ma_base == 37.71

    def test_rupelian_lookup(self):
        """Rupelian must resolve as an age of the Oligocene."""
        ages = find_ages_in_text("Rupelian, Belgium")
        rupelian = [a for a in ages if a.age == "Rupelian"]
        assert rupelian, f"Rupelian not found in {ages!r}"
        cls = rupelian[0]
        assert cls.period == "Paleogene"
        assert cls.ma_top == 27.82
        assert cls.ma_base == 33.9

    def test_ypresian_burdigalian(self):
        """Ypresian (Eocene) and Burdigalian (Miocene) must both be
        present with their ICS 2023 Ma bounds."""
        cases = [
            ("Ypresian", "Paleogene", 47.8, 56.0),
            ("Lutetian", "Paleogene", 41.2, 47.8),
            ("Burdigalian", "Neogene", 15.97, 20.44),
            ("Messinian", "Neogene", 5.333, 7.246),
            ("Calabrian", "Quaternary", 0.774, 1.80),
        ]
        for name, period, ma_top, ma_base in cases:
            row = self._find_stage(name)
            assert row["rank"] == "age", f"{name} should be rank=age, got {row['rank']}"
            # Period parent walks through (Paleogene / Neogene / Quaternary).
            # Calabrian has parent = Quaternary directly; others walk
            # through epoch first.
            ancestors = {row["parent"]}
            for r in _ICS_ROWS:
                if r["name"] in ancestors and r["parent"]:
                    ancestors.add(r["parent"])
            assert period in ancestors, f"{name} should be in {period}, ancestors={ancestors}"
            assert abs(row["ma_top"] - ma_top) < 1e-6, f"{name}: ma_top={row['ma_top']} != {ma_top}"
            assert abs(row["ma_base"] - ma_base) < 1e-6, (
                f"{name}: ma_base={row['ma_base']} != {ma_base}"
            )
            # And the public lookup API returns it.
            ages = find_ages_in_text(f"{name}, somewhere")
            hit = [a for a in ages if a.age == name]
            assert hit, f"{name} not surfaced by find_ages_in_text"
            assert abs(hit[0].ma_top - ma_top) < 1e-6
            assert abs(hit[0].ma_base - ma_base) < 1e-6

    def test_calibration_round_trip(self):
        """Each new Cenozoic stage's ma_mid equals (ma_top+ma_base)/2."""
        new_stages = [
            "Danian",
            "Selandian",
            "Thanetian",
            "Ypresian",
            "Lutetian",
            "Bartonian",
            "Priabonian",
            "Rupelian",
            "Chattian",
            "Aquitanian",
            "Burdigalian",
            "Langhian",
            "Serravallian",
            "Tortonian",
            "Messinian",
            "Zanclean",
            "Piacenzian",
            "Gelasian",
            "Calabrian",
            "Chibanian",
            "Late Pleistocene",
        ]
        for name in new_stages:
            row = self._find_stage(name)
            ma_top, ma_base = row["ma_top"], row["ma_base"]
            ma_mid = (ma_top + ma_base) / 2.0
            ages = find_ages_in_text(f"{name}, somewhere")
            hit = [a for a in ages if a.age == name]
            assert hit, f"{name} not surfaced by find_ages_in_text"
            cls = hit[0]
            assert cls.ma_mid is not None, f"{name} ma_mid is None"
            assert abs(cls.ma_mid - ma_mid) < 1e-9, (
                f"{name}: ma_mid={cls.ma_mid} != expected {ma_mid}"
            )


# ---------------------------------------------------------------------------
# D1 — PBDB cache lock / negative cache / atomic write
# ---------------------------------------------------------------------------


class _FakeResp:
    """Drop-in mock for the ``requests.Response`` object."""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return {"records": self._payload}


def _patch_requests(payload_or_callable):
    """Return a context manager that replaces ``requests.get`` with a
    callable that returns ``_FakeResp(payload_or_callable)``.

    The patch is applied to the ``requests`` module so the lazy
    ``import requests`` inside ``fetch_pbdb_intervals`` sees the
    stubbed implementation.
    """
    fake_requests = type(sys)("requests")
    if callable(payload_or_callable):
        fake_requests.get = payload_or_callable  # type: ignore[attr-defined]
    else:
        fake_requests.get = lambda url, timeout=30: _FakeResp(payload_or_callable)  # type: ignore[attr-defined]
    return patch.dict(sys.modules, {"requests": fake_requests})


class TestPBDBIntervalCache:
    """Bug D1: the previous PBDB cache had no lock, no negative cache,
    and a non-atomic write. These tests pin those invariants so a
    future regression does not re-introduce them.
    """

    def test_pbdb_cache_thread_safe(self, tmp_path: Path) -> None:
        """10 concurrent callers must all see a consistent (non-empty
        or empty but always non-corrupt) cache. Without the lock,
        multiple threads race on the read-modify-write of
        ``_PBDB_INTERVALS_CACHE`` and could each issue their own
        network call. With the lock, only the first call hits the
        network and the rest return the cached payload."""
        _reset_pbdb_state()
        sample_data = [{"oid": 1, "nam": "Priabonian", "rnk": "age", "lag": 33.9, "eag": 37.71}]

        call_counter = {"n": 0}
        call_lock = threading.Lock()

        def fake_get(url, timeout=30):
            with call_lock:
                call_counter["n"] += 1
            time.sleep(0.02)  # force concurrent arrival at the lock
            return _FakeResp(sample_data)

        errors: list[BaseException] = []
        results: list[list[dict]] = []
        barrier = threading.Barrier(10)

        def worker() -> None:
            try:
                # Sync the threads at the barrier so they all enter the
                # fetch at (approximately) the same time.
                barrier.wait(timeout=2.0)
                # No force=True → threads 2..10 should hit the
                # in-memory cache populated by thread 1.
                results.append(fetch_pbdb_intervals(cache_dir=tmp_path))
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        # The patch must wrap BOTH start() and join() so the lazy
        # ``import requests`` inside fetch_pbdb_intervals sees the
        # stubbed module when each worker runs.
        with _patch_requests(fake_get):
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert not errors, f"Concurrent fetch_pbdb_intervals raised: {errors!r}"
        # Lock ensures no race: even with 10 threads we made exactly
        # one network call (subsequent ones hit the in-memory cache).
        assert call_counter["n"] == 1, (
            f"Concurrent fetch_pbdb_intervals hit network {call_counter['n']} times; "
            "lock should serialise to a single fetch"
        )
        # Every worker returned the same payload.
        assert all(r == sample_data for r in results), (
            f"Concurrent fetches returned inconsistent payloads: {results!r}"
        )
        # The cache file was written and is valid JSON.
        cache_path = tmp_path / "intervals.json"
        assert cache_path.exists(), "PBDB cache file was not written"
        on_disk = json.loads(cache_path.read_text(encoding="utf-8"))
        assert on_disk == sample_data

    def test_pbdb_negative_cache(self, tmp_path: Path) -> None:
        """When the underlying network call raises, subsequent calls
        within the negative-cache TTL must short-circuit and NOT
        retry the network."""
        _reset_pbdb_state()
        call_counter = {"n": 0}
        call_lock = threading.Lock()

        def fake_get(url, timeout=30):
            with call_lock:
                call_counter["n"] += 1
            raise ConnectionError("simulated outage")

        with _patch_requests(fake_get):
            # First call: hits the network, fails, populates neg cache.
            out1 = fetch_pbdb_intervals(force=True, cache_dir=tmp_path)
            # Second call within TTL (no force): must short-circuit.
            out2 = fetch_pbdb_intervals(cache_dir=tmp_path)
            # Third call within TTL (no force): also no retry.
            out3 = fetch_pbdb_intervals(cache_dir=tmp_path)

        # All three calls returned empty (graceful degradation).
        assert out1 == []
        assert out2 == []
        assert out3 == []
        # Only the first call actually hit the network; the next two
        # were served by the negative cache.
        assert call_counter["n"] == 1, (
            f"Negative cache failed: {call_counter['n']} network calls "
            f"instead of 1 (TTL={_PBDB_NEG_TTL_SECONDS}s)"
        )
        # The neg cache entry is present and monotonic-clock bounded.
        assert strat._PBDB_INTERVALS_NEG_CACHE, "Negative cache was not populated on failure"
        assert all(v > time.monotonic() for v in strat._PBDB_INTERVALS_NEG_CACHE.values()), (
            "Negative cache TTL must be in the future"
        )

    def test_pbdb_cache_file_atomic(self, tmp_path: Path) -> None:
        """Atomic write: when ``os.replace`` is monkey-patched to
        raise mid-write, no stale temp file is left in the cache
        directory and the existing cache file is never truncated to
        a half-written payload."""
        _reset_pbdb_state()
        sample_data = [{"oid": 1, "nam": "Priabonian"}]

        # Seed an existing valid cache so we can assert that the
        # failed atomic write doesn't truncate it.
        seed_path = tmp_path / "intervals.json"
        seed_payload = [{"oid": 99, "nam": "Seed"}]
        seed_path.write_text(json.dumps(seed_payload), encoding="utf-8")
        # Make sure the on-disk cache is considered fresh.
        import os as _os

        now = time.time()
        _os.utime(seed_path, (now, now))

        def crashing_replace(src, dst):
            # Simulate a crash before the rename lands. The temp file
            # IS deleted by the implementation's ``except`` arm — we
            # still raise to mirror an unrecoverable error.
            try:
                Path(src).unlink(missing_ok=True)
            except OSError:
                pass
            raise OSError("simulated crash mid-write")

        with _patch_requests(sample_data):
            with patch.object(strat.os, "replace", side_effect=crashing_replace):
                # The function catches the OSError, populates the
                # negative cache and returns [].
                out = fetch_pbdb_intervals(force=True, cache_dir=tmp_path)

        assert out == [], f"Expected [] after mid-write crash, got {out!r}"
        # No stale temp files should linger in the cache directory.
        temps = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
        assert not temps, f"Atomic write leaked temp files: {temps!r}"
        # The original seed cache must be intact — the failed write
        # must not have truncated it.
        on_disk = json.loads(seed_path.read_text(encoding="utf-8"))
        assert on_disk == seed_payload, f"Atomic write corrupted the existing cache: {on_disk!r}"


# Sanity: lock + neg cache constants exist with expected types.
def test_pbdb_lock_and_neg_cache_constants_exist():
    """Guard against accidental deletion of the new module-level
    state — the tests above depend on these symbols."""
    assert callable(_PBDB_INTERVALS_LOCK.acquire)
    assert callable(_PBDB_INTERVALS_LOCK.release)
    assert isinstance(_PBDB_INTERVALS_NEG_CACHE, dict)
    assert _PBDB_NEG_TTL_SECONDS > 0


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
