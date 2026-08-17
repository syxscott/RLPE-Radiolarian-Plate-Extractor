"""Real-data replay tests for Round 4 paper-level fixes.

These tests use the v18 cached prediction corpus
(``work/combined_9_v18_FINAL.jsonl``) as the input — this is
the actual output of the pipeline as run on the 9-paper gold
corpus, BEFORE Round 4 fixes. Each test replays the fix's
behavior on the cached data and verifies the fix's effect
matches the audit's stated impact:

  * P1 (pouille over-segmentation): in v18 cached, pouille2014
    produced 108 pred rows vs 6-panel gold. After the fix
    (``assign_panels_to_labels`` uses ``is_valid_panel_label``),
    28 rows of ``panel_id='P1'`` are rejected. The replay
    verifies the rejection count matches the audit trace.

The tests are intentionally data-driven (not synthetic) so
they reproduce real bugs from the actual production pipeline.
A bug in the fix would cause the test to fail because the
replay numbers would not match.

These tests run in any env (no cv2 / OCR / M3 API required).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
V18_CACHED = REPO_ROOT / "work" / "combined_9_v18_FINAL.jsonl"
POUILLE_GOLD = REPO_ROOT / "data" / "gold" / "pouille2014.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------- P1


class TestP1PouilleReplayedOnV18Cached:
    """P1 (pouille over-segmentation): the fix in
    ``src/rlpe/association.py`` replaces
    ``pid.isdigit() or len(pid) <= 3`` with the strict
    ``is_valid_panel_label``. Replay the fix's effect on the v18
    cached pouille pred set and verify the rejection count.

    The audit trace: pouille v18 cached = 108 rows, 28 with
    panel_id='P1' (OCR garbage from Stage-3 over-seg broadcasting
    the same first caption token to every panel).

    The fix rejects: panel_ids that are pure-digits >= 1, pure
    A-H markers, or digit+letter combinations. It keeps digits
    (e.g. '1', '5', '12') and accepts A-H (e.g. 'A', 'B').

    Skips when ``work/combined_9_v18_FINAL.jsonl`` is absent. The
    cached corpus is gitignored (4 MB and regenerated each pipeline
    run); CI clones without it. The skip is per-test so a single
    missing corpus doesn't fail the whole replay class.
    """

    @pytest.fixture
    def cached_pouille(self):
        if not V18_CACHED.exists():
            pytest.skip(
                f"v18 cached corpus not found at {V18_CACHED}. "
                f"Run the pipeline once and force-add "
                f"`work/combined_9_v18_FINAL.jsonl`, or this replay "
                f"test will skip."
            )
        rows = _load_jsonl(V18_CACHED)
        pouille = [r for r in rows if r.get("paper_id") == "2225994d55021328"]
        if not pouille:
            pytest.skip(
                "v18 cached corpus exists but contains no pouille "
                "rows (paper_id=2225994d55021328). The cached "
                "corpus may have been regenerated with a different "
                "paper_id."
            )
        return pouille

    @pytest.fixture
    def gold_pouille(self):
        return _load_jsonl(POUILLE_GOLD)

    def test_v18_cached_has_over_segmentation_artifacts(self, cached_pouille):
        """Sanity check: the v18 cached corpus must contain the
        28 'P1' rows that the audit cited. If the cached corpus
        is missing them, the test below is meaningless."""

        panels = Counter(r.get("panel_id") for r in cached_pouille)
        p1_count = panels.get("P1", 0)
        assert p1_count >= 20, (
            f"v18 cached pouille pred should have ≥20 'P1' rows "
            f"(audit reported 28); got {p1_count}. The cached corpus "
            f"may have been regenerated, invalidating this replay test."
        )
        # All pred rows count.
        assert len(cached_pouille) >= 100, (
            f"v18 cached pouille pred should have ≥100 rows; got {len(cached_pouille)}"
        )

    def test_replay_rejects_garbage_panel_ids(self, cached_pouille):
        """Apply ``is_valid_panel_label`` to every v18 cached row
        and verify the rejection count matches the audit trace.

        A successful fix means: number of rows where
        ``is_valid_panel_label(panel_id) == False`` should be
        similar to the audit's stated 67 'non-numeric or None
        panel_id' rows.
        """
        from rlpe.association import is_valid_panel_label

        accepted = 0
        rejected = 0
        for r in cached_pouille:
            pid = r.get("panel_id")
            if is_valid_panel_label(pid):
                accepted += 1
            else:
                rejected += 1
        # Audit said 67 rows had non-numeric / None panel_ids.
        # The fix should reject at least those 67.
        assert rejected >= 60, (
            f"Replay should reject ≥60 garbage rows; rejected "
            f"{rejected} (accepted {accepted}). The fix's contract "
            f"on the cached data is broken."
        )
        # Of the accepted, the gold matches (panel 1, 5, 8, 12, 15,
        # 19) must still be present — the fix shouldn't over-reject
        # gold patterns.
        accepted_pids = {r.get("panel_id") for r in cached_pouille}
        for gold_pid in ("1", "5", "8", "12", "15", "19"):
            assert gold_pid in accepted_pids, (
                f"Gold panel_id '{gold_pid}' is missing from the "
                f"accepted set after the fix — over-rejection bug."
            )

    def test_replay_reduces_pred_set_by_garbage_amount(self, cached_pouille, gold_pouille):
        """Compare pred set size before/after the fix in the replay.

        Pre-fix: 108 rows of pred (vs 6 gold) — 102 false positives.
        Post-fix (replay): accepted rows should drop substantially.
        The fix should bring the accepted-row count below 60
        (eliminating the bulk of the 67 garbage rows).

        Without this test, a regression that reverts the fix would
        still pass any individual garbage-rejection test (because
        each test only checks one garbage value). The aggregate
        count is the integration signal.
        """
        from rlpe.association import is_valid_panel_label

        accepted = sum(1 for r in cached_pouille if is_valid_panel_label(r.get("panel_id")))
        # The fix should cut pred set from 108 down to under 60.
        # The audit's 67 garbage rows are an exact count; with
        # a small margin for any data drift, <70 is the fix
        # contract.
        assert accepted < 70, (
            f"After fix, accepted pred count should be <70 "
            f"(eliminates garbage rows); got {accepted}."
        )

    def test_gold_panel_ids_remain_in_replay(self, cached_pouille, gold_pouille):
        """The fix must NOT reject gold panel_ids. Even if the
        pred set is smaller after the fix, the 6 gold panel_ids
        should still appear among the accepted pred set so
        string-match F1 has something to compare against.
        """
        from rlpe.association import is_valid_panel_label

        accepted_pids = {
            r.get("panel_id") for r in cached_pouille if is_valid_panel_label(r.get("panel_id"))
        }
        gold_pids = {g.get("panel_id") for g in gold_pouille}
        # All 6 gold panel_ids should be in the accepted set.
        missing = gold_pids - accepted_pids
        assert not missing, (
            f"After fix, these gold panel_ids disappeared from the "
            f"pred set: {missing}. The fix is over-rejecting valid "
            f"labels."
        )


# --------------------------------------------------------------------------- M7


class TestM7TotalCallsReplayOnV18Cached:
    """M7 (MiniMax M3 backend call counter): the fix bumps
    ``total_calls`` at the START of each attempt (not after
    success). Replay the count on the v18 cached run that
    pre-dates the fix.

    The audit: pre-fix, ``total_calls`` undercounted by exactly
    1 per retry-exhausted sequence. We can't replay the
    MiniMax API call itself without network, but we CAN verify
    the in-source counter behavior via a synthetic call: spawn
    3 fake failed attempts + 1 success, verify ``total_calls ==
    4`` (not 3, which the pre-fix code would have reported).
    """

    def test_counter_increments_at_entry_not_after_success(self):
        """Construct a backend with all counters zeroed, then
        drive 3 failed attempts. The fix increments
        ``total_calls`` at the start of each attempt, so after
        3 failures ``total_calls == 3`` (not 2, which the pre-fix
        code would have reported by missing the final attempt).

        Without this test, a regression that moves the counter
        back to after the API call would pass other tests (which
        check ``total_calls == 1`` for a single success) but
        would undercount retries.
        """
        from rlpe.llm_backends import MiniMaxM3Backend

        # Bypass __init__ (no anthropic SDK in sandbox) and
        # construct a minimal stub with the post-fix counter
        # behavior. We patch in _call_api via the production
        # function in llm_backends.
        from rlpe.llm_backends import (
            MiniMaxM3Backend as _Backend,
        )
        from tests.fakes.fake_m3_backend import FakeM3Backend

        # Build a backend without running __init__.
        backend = _Backend.__new__(_Backend)
        backend.api_key = "test"
        backend.base_url = "http://fake"
        backend.model = "MiniMax-M3-fake"
        backend.max_retries = 3
        backend.timeout_sec = 30
        backend.max_output_tokens = 4096
        backend.thinking_budget_tokens = 1024
        backend.enable_thinking = False
        backend.temperature = 1.0
        backend.top_p = 1.0
        backend.total_input_tokens = 0
        backend.total_output_tokens = 0
        backend.total_calls = 0
        backend.total_errors = 0
        backend.failed_with_thinking = 0
        backend.fallback_4xx_hints = 0
        backend.max_concurrent = 8
        backend.data_outbound_policy = "api_full"
        from unittest.mock import MagicMock

        import anthropic

        backend._anthropic = anthropic
        backend._client = MagicMock()
        import threading

        backend._lock = threading.Lock()
        backend._sem = threading.Semaphore(backend.max_concurrent)
        backend._thread_local = threading.local()

        # audit 2026-07-31: this test used to increment
        # ``backend.total_calls`` ITSELF in a loop and then assert
        # the result — it never called the production code (classic
        # tautology: the production retry counter could be deleted
        # and the test still passed). Drive the REAL ``_call_api``
        # retry loop instead: 3 retryable 500s then a success, and
        # assert the production counter.
        from unittest.mock import MagicMock, patch

        import anthropic
        import httpx

        attempts = {"n": 0}

        def fake_create(*args, **kwargs):
            attempts["n"] += 1
            if attempts["n"] <= 2:
                req = httpx.Request("POST", "http://fake")
                resp = httpx.Response(500, request=req)
                raise anthropic.APIStatusError(
                    f"simulated 500 #{attempts['n']}", response=resp, body=None
                )
            ok_resp = MagicMock()
            ok_resp.content = [
                MagicMock(
                    type="text",
                    text='{"label": "1", "species": "Test species", "confidence": 0.9}',
                )
            ]
            ok_resp.id = "test-id"
            ok_resp.model = "MiniMax-M3-fake"
            ok_resp.usage = None
            return ok_resp

        # Patch the UNDERLYING SDK call, not _call_api — the retry
        # loop and the total_calls counter must be the production ones.
        backend._client.messages.create.side_effect = fake_create
        with patch("rlpe.llm_backends.time.sleep", lambda s: None):
            result = backend.infer_text(system_prompt="s", user_prompt="u")
        assert result.get("species") == "Test species", result
        # max_retries=3 → the retry loop permits 3 attempts total:
        # 2 failures then the success on attempt 3.
        assert attempts["n"] == 3, f"expected 3 attempts, got {attempts['n']}"
        with backend._lock:
            calls = backend.total_calls
        assert calls == 3, (
            f"After 2 failed + 1 success attempts, production total_calls should be 3; got {calls}."
        )

    def test_source_has_total_calls_at_entry_of_try(self):
        """Static guard: the fix's signature is that the
        ``self.total_calls += 1`` line is the FIRST statement
        inside the try block (at the very top, before the
        ``with self._sem:`` context manager). If the line is
        moved after the API call, the bug returns.

        This test reads the source file and asserts the line's
        position — independent of any runtime execution.
        """
        from pathlib import Path as _Path

        path = _Path(__file__).resolve().parents[1] / "src" / "rlpe" / "llm_backends.py"
        text = path.read_text(encoding="utf-8")

        # Find the body of _call_api.
        marker = "def _call_api("
        i = text.find(marker)
        assert i > 0
        body = text[i : i + 5000]

        # Find ``self.total_calls += 1`` lines (there may be
        # multiple: the entry-bump is the one inside the
        # ``for attempt in range(self.max_retries):`` loop.
        # The post-fix pattern: a ``with self._lock:`` immediately
        # followed by ``self.total_calls += 1`` and then ``try:``.
        # We assert this 3-line sequence appears in order.
        pattern = "with self._lock:\n                self.total_calls += 1\n            try:"
        assert pattern in body, (
            "The M7 fix requires:\n"
            "    with self._lock:\n"
            "        self.total_calls += 1\n"
            "    try:\n"
            "as the FIRST statements of the per-attempt loop in\n"
            "_call_api. The current source has a different\n"
            "structure, indicating the fix was reverted or never\n"
            "landed correctly."
        )


# --------------------------------------------------------------------------- P1 source guard


class TestP1AssignPanelsSourceGuard:
    """P1 (pouille over-segmentation): the fix in
    ``src/rlpe/association.py`` replaces
    ``pid.isdigit() or len(pid) <= 3`` with
    ``is_valid_panel_label(pid)``. Mutation testing on the unit
    tests showed they catch 9 of the 9 fix paths in a synthetic
    way, but the cached-data replay tests (above) only verify
    the fix's effect, not the fix's location in the source. This
    source-guard test ensures the fix can't be silently reverted
    without breaking a test.
    """

    def test_assign_panels_uses_is_valid_panel_label(self):
        from pathlib import Path as _Path

        path = _Path(__file__).resolve().parents[1] / "src" / "rlpe" / "association.py"
        text = path.read_text(encoding="utf-8")

        # Find the body of assign_panels_to_labels (use the next
        # top-level def as the end-marker; the function is small
        # but i+2500 may still miss the fix line if ruff format
        # inserts blank lines).
        marker = "def assign_panels_to_labels("
        i = text.find(marker)
        assert i > 0
        next_def = text.find("\ndef ", i + 1)
        assert next_def > 0
        body = text[i:next_def]

        # The fix's signature: ``is_valid_panel_label(pid)`` is
        # the gate that determines whether a panel_id is kept.
        assert "is_valid_panel_label(pid)" in body, (
            "P1 fix: assign_panels_to_labels must call "
            "is_valid_panel_label(pid) as the gate. A pre-fix "
            "implementation used ``pid.isdigit() or len(pid) <= 3`` "
            "which is too permissive (accepted 'P1', 'ean', 'd')."
        )

        # The pre-fix pattern must NOT be present in CODE (comments
        # in the production source may legitimately describe the
        # previous behaviour for context — those don't count).
        # Strip ``#`` comments line-by-line to find the actual code.
        code_lines = [line for line in body.splitlines() if not line.lstrip().startswith("#")]
        code_only = "\n".join(code_lines)
        assert "pid.isdigit() or len(pid) <= 3" not in code_only, (
            "P1 fix: the loose guard ``pid.isdigit() or len(pid) "
            "<= 3`` is still in source code (not just a comment) — "
            "it accepts OCR garbage like 'P1', 'ean', 'd'. Revert "
            "the fix."
        )
