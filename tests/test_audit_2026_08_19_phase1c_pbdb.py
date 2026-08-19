"""Phase 1c audit (2026-08-19): PBDB rate-limit + User-Agent compliance.

Two related fixes from the 2026-08-19 multi-agent audit:

1. **B-4 (BLOCKER)** — ``rlpe.paleodb`` defaulted to ``min_interval=0.2``
   which allowed ~300 req/min against ``paleobiodb.org``. The PBDB
   public-API ToS documents a hard limit of 30 req/min and operators
   routinely IP-ban callers that exceed it. The default is now
   ``2.0`` (= 0.5 req/sec = 30 req/min), enforced in both
   :class:`_RateLimiter` and :class:`PaleoDB.__init__`.

2. **NIT-4** — the User-Agent string passed to PBDB is configurable via
   the ``RLPE_PBDB_UA`` environment variable; the default fallback
   remains the same placeholder so existing deployments are not
   silently broken, but operators who want to identify themselves to
   PBDB (recommended by ToS) can set the env var.

These tests pin all of the above so a future refactor doesn't silently
regress to the over-aggressive rate limit or strip the env-var hook.
"""

from __future__ import annotations

import sys
import time as time_mod
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.paleodb import PaleoDB, _RateLimiter  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (_REPO / rel).read_text(encoding="utf-8")


# ============================================================================
# Fake clock
# ============================================================================


class _FakeClock:
    """Stand-in for :mod:`time` that the rate-limiter can call.

    Each ``sleep`` advances the clock by the requested duration so that
    ``time.monotonic()`` returns the post-sleep wall-time. This lets
    a test drive the rate limiter through a multi-call sequence and
    read off the actual total elapsed time.
    """

    def __init__(self) -> None:
        self.now = 1_000_000.0  # arbitrary non-zero base
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.sleeps.append(float(duration))
        self.now += float(duration)


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    """Replace ``time.monotonic`` + ``time.sleep`` inside ``rlpe.paleodb``."""
    clock = _FakeClock()
    monkeypatch.setattr("rlpe.paleodb.time", clock)
    return clock


# ============================================================================
# BLOCKER B-4 — rate-limit defaults
# ============================================================================


class TestRateLimiterDefaultInterval:
    """``_RateLimiter`` must default to 2.0 s (30 req/min)."""

    def test_default_min_interval_is_two_seconds(self):
        limiter = _RateLimiter()
        assert limiter.min_interval == 2.0, (
            f"_RateLimiter default min_interval must be 2.0 "
            f"(= 30 req/min, PBDB ToS); got {limiter.min_interval}"
        )

    def test_default_min_interval_source_guard(self):
        """Pin the source so a future edit can't silently flip it back."""
        src = _read("src/rlpe/paleodb.py")
        # The dataclass field default
        assert (
            "min_interval: float = 2.0" in src
        ), "PaleoDB / _RateLimiter default min_interval must be 2.0 (B-4 fix)"
        # Make sure the OLD 0.2 default is gone (it is allowed in non-default
        # contexts such as tests, but the only literal ``0.2`` should be the
        # test fixture's ``min_interval=0.0``-style bypass usage. We rely on
        # the dataclass default check above as the primary pin, and just
        # additionally ensure there isn't another ``= 0.2`` literal sitting
        # on a PaleoDB / _RateLimiter signature.)
        # Inspect the _RateLimiter dataclass block
        rl_start = src.find("class _RateLimiter")
        rl_end = src.find("@dataclass", rl_start + 1)
        if rl_end == -1:
            rl_end = len(src)
        rl_block = src[rl_start:rl_end]
        assert "min_interval: float = 2.0" in rl_block


class TestWaitSpacing:
    """``wait()`` must space successive calls by ``min_interval`` seconds."""

    def test_first_wait_does_not_sleep(self, fake_clock: _FakeClock) -> None:
        limiter = _RateLimiter()
        limiter.wait()
        # No prior call → gap is negative → no sleep happens.
        assert fake_clock.sleeps == []

    def test_second_wait_sleeps_at_least_min_interval(
        self, fake_clock: _FakeClock
    ) -> None:
        limiter = _RateLimiter()  # min_interval == 2.0
        limiter.wait()
        # Advance clock by 0.1 s to force a real wait.
        fake_clock.now += 0.1
        limiter.wait()
        # The gap that was slept must be at least min_interval - 0.1 = 1.9 s
        # (allowing for the tiny 0.1 s we manually advanced).
        assert fake_clock.sleeps, "wait() must call time.sleep on the 2nd call"
        assert fake_clock.sleeps[0] >= 1.9, (
            f"wait() should sleep ~min_interval seconds; slept {fake_clock.sleeps[0]}"
        )

    def test_three_waits_total_elapsed_matches_min_interval(
        self, fake_clock: _FakeClock
    ) -> None:
        """3 successive ``wait()`` calls must collectively sleep
        at least ``2 × min_interval`` seconds (first call is free)."""
        limiter = _RateLimiter()
        for _ in range(3):
            limiter.wait()
        total = sum(fake_clock.sleeps)
        # First call: 0. Second + third: each should sleep ~2.0 s.
        # Allow a hair of float slack on either side.
        assert pytest.approx(4.0, abs=1e-6) == total, (
            f"three waits with default min_interval=2.0 should sleep 4.0 s total; "
            f"got {total} (sleeps={fake_clock.sleeps})"
        )

    def test_two_waits_separated_by_more_than_min_interval_skip_sleep(
        self, fake_clock: _FakeClock
    ) -> None:
        limiter = _RateLimiter()  # min_interval == 2.0
        limiter.wait()
        # Advance past the interval; the 2nd wait should NOT sleep.
        fake_clock.now += 5.0
        limiter.wait()
        assert fake_clock.sleeps == []


class TestPaleoDBDefaultInterval:
    """``PaleoDB`` must inherit the 30 req/min default."""

    def test_paleodb_default_min_interval(self, tmp_path: Path) -> None:
        client = PaleoDB(cache_dir=tmp_path)
        assert client.min_interval == 2.0, (
            f"PaleoDB default min_interval must be 2.0; got {client.min_interval}"
        )

    def test_paleodb_limiter_matches_default(self, tmp_path: Path) -> None:
        client = PaleoDB(cache_dir=tmp_path)
        assert client._limiter.min_interval == 2.0, (
            "PaleoDB._limiter.min_interval must inherit the 2.0s default"
        )

    def test_explicit_low_min_interval_is_still_supported(
        self, tmp_path: Path
    ) -> None:
        """Operators may legitimately want a faster throughput in offline
        tests; ``min_interval=0.0`` must not be rejected — only the
        *default* needs to follow PBDB ToS."""
        client = PaleoDB(cache_dir=tmp_path, min_interval=0.0)
        assert client.min_interval == 0.0
        assert client._limiter.min_interval == 0.0

    def test_signature_default_is_two_seconds_source_guard(self) -> None:
        src = _read("src/rlpe/paleodb.py")
        # Find the ``__init__`` signature
        init_idx = src.find("def __init__(")
        assert init_idx != -1
        # The next ~400 chars should contain the min_interval kwarg with =2.0
        chunk = src[init_idx : init_idx + 600]
        assert (
            "min_interval: float = 2.0" in chunk
        ), "PaleoDB.__init__ default min_interval must be 2.0 (B-4 fix)"


# ============================================================================
# NIT-4 — User-Agent env-var
# ============================================================================


class TestUserAgentEnvVar:
    """``RLPE_PBDB_UA`` env var must override the default User-Agent."""

    def test_default_user_agent_contains_rlpe(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("RLPE_PBDB_UA", raising=False)
        client = PaleoDB(cache_dir=tmp_path)
        assert "RLPE" in client._user_agent, (
            f"Default User-Agent should self-identify as RLPE; "
            f"got {client._user_agent!r}"
        )

    def test_env_var_overrides_user_agent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("RLPE_PBDB_UA", "RLPE-Test/9.9 (+ops@example.com)")
        client = PaleoDB(cache_dir=tmp_path)
        assert client._user_agent == "RLPE-Test/9.9 (+ops@example.com)"

    def test_env_var_read_at_construction(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The env var must be read at construction time, not deferred."""
        monkeypatch.setenv("RLPE_PBDB_UA", "first-value")
        client = PaleoDB(cache_dir=tmp_path)
        # Mutating the env var after construction must NOT affect
        # the already-built client.
        monkeypatch.setenv("RLPE_PBDB_UA", "second-value")
        assert client._user_agent == "first-value"

    def test_user_agent_read_on_every_new_client(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Two separately-constructed clients must each read the env var."""
        monkeypatch.setenv("RLPE_PBDB_UA", "value-A")
        a = PaleoDB(cache_dir=tmp_path / "a")
        monkeypatch.setenv("RLPE_PBDB_UA", "value-B")
        b = PaleoDB(cache_dir=tmp_path / "b")
        assert a._user_agent == "value-A"
        assert b._user_agent == "value-B"

    def test_env_var_constant_present_in_source(self) -> None:
        """Pin the exact env var name in the source so a rename is
        caught immediately."""
        src = _read("src/rlpe/paleodb.py")
        assert '"RLPE_PBDB_UA"' in src, (
            "PBDB User-Agent must be configurable via RLPE_PBDB_UA env var"
        )
        # And it must be read via ``os.environ.get`` (not e.g. a hard-coded value)
        assert "os.environ.get" in src, (
            "PBDB User-Agent must be sourced from os.environ.get(...)"
        )


# ============================================================================
# Rate-limit simulation — end-to-end via PaleoDB._limiter.wait
# ============================================================================


class TestRateLimitSimulation:
    """Simulate a small burst of PBDB calls and verify the wall-clock
    rate stays at or under 30 req/min."""

    def test_default_burst_throttles_to_30_per_min(
        self, fake_clock: _FakeClock, tmp_path: Path
    ) -> None:
        """4 consecutive ``wait()`` calls with no other I/O in between
        must collectively advance the clock by at least ``3 × 2.0 s =
        6.0 s`` (4 calls − 1 free first call = 3 sleeps).
        At 30 req/min, 4 calls should take ≥ 6.0 s of effective
        rate-limited wall time."""
        client = PaleoDB(cache_dir=tmp_path)
        for _ in range(4):
            client._limiter.wait()
        total_sleep = sum(fake_clock.sleeps)
        assert pytest.approx(6.0, abs=1e-6) == total_sleep, (
            f"4 rate-limited calls at default interval should sleep 6.0s; "
            f"got {total_sleep}s (sleeps={fake_clock.sleeps})"
        )

    def test_explicit_zero_min_interval_nevers_sleeps(
        self, fake_clock: _FakeClock, tmp_path: Path
    ) -> None:
        """An operator-pinned ``min_interval=0.0`` (e.g. for offline tests)
        must never sleep."""
        client = PaleoDB(cache_dir=tmp_path, min_interval=0.0)
        for _ in range(10):
            client._limiter.wait()
        assert fake_clock.sleeps == []


# ============================================================================
# Pipeline integration — make sure pipeline.py no longer hard-codes 0.2s
# ============================================================================


class TestPipelineDoesNotOverrideDefault:
    """The pipeline must not pass ``min_interval=0.2`` when constructing
    :class:`PaleoDB` — that's exactly the over-aggressive value the
    B-4 fix removed from the default."""

    def test_pipeline_does_not_pass_hard_coded_min_interval(self) -> None:
        src = _read("src/rlpe/pipeline.py")
        # The hard-coded literal ``min_interval=0.2,`` (followed by
        # ``,`` or ``\n`` — i.e. an actual kwarg, not a comment that
        # talks about the historical bug) is the B-4 bug pattern.
        # A future refactor that reintroduces it as a real kwarg is
        # caught immediately.
        import re

        # Strip block + line comments first so the comment that
        # *describes* the old value doesn't trip the guard.
        no_block_comments = re.sub(r"#.*", "", src)
        # Match ``min_interval=0.2`` as a real keyword (followed
        # by a comma / closing paren / whitespace, not by code
        # words like ``min_interval=0.2s``).
        assert not re.search(r"\bmin_interval\s*=\s*0\.2\s*[,)\n]", no_block_comments), (
            "pipeline.py must not pass min_interval=0.2 to PaleoDB() — "
            "that bypasses the PBDB-ToS-compliant 2.0s default (B-4)"
        )

    def test_pipeline_can_opt_into_custom_min_interval_via_config(self) -> None:
        """Operators who really want to exceed 30 req/min may set
        ``paleodb_min_interval`` in extra config. Verify the wiring."""
        src = _read("src/rlpe/pipeline.py")
        assert "paleodb_min_interval" in src, (
            "pipeline must read paleodb_min_interval from config.extra"
        )
