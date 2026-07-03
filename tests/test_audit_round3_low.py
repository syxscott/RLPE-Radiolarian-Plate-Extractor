"""Tests for the 2026-07-03 audit low-severity backend bugs.

M7: llm_backends.MiniMaxM3Backend._call_api() previously bumped
``total_calls`` only AFTER the API call succeeded. If a retry
sequence was exhausted (all max_retries attempts failed), the
final failed attempt was never counted in ``total_calls`` — only
in ``total_errors``. The fix bumps ``total_calls`` at the START of
each attempt so a failed attempt also increments the counter.

L1: converters.run_output_from_provenance() had no guard against
``matches=None``. A caller that passed None (e.g. an early-exit
pipeline branch with no matches yet) raised ``TypeError`` deep
inside the ``*_records_from_matches`` helpers. The fix coerces None
to an empty list.

L3: llm_backends.cli_fallback_prompt() called ``input()`` even when
stdin was not a TTY (background worker thread, API server context).
``input()`` blocks forever waiting for input that never arrives.
The fix raises an explicit RuntimeError so the FallbackHandler
falls through to ``MiniMax_fallback_default``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

# --------------------------------------------------------------------------- L1


class TestRunOutputFromProvenanceNoneMatches:
    """L1: passing ``matches=None`` to ``run_output_from_provenance``
    must produce an empty RunOutput, not raise TypeError.
    """

    def _provenance(self):
        from rlpe.schema_models import ProvenanceRecord

        return ProvenanceRecord(
            pipeline_version="test",
            schema_version="1.0.0",
            git_commit="deadbeef",
            git_dirty=False,
            config_snapshot={},
            input_sha256={},
            timestamp_utc="2026-07-03T00:00:00",
            host="test-host",
            python_version="3.10.20",
        )

    def test_none_matches_returns_empty_run_output(self):
        from rlpe.converters import run_output_from_provenance

        prov = self._provenance()
        out = run_output_from_provenance(prov, None)
        assert out["schema_version"] == "1.0.0"
        assert out["panels"] == []
        assert out["papers"] == []
        assert out["figures"] == []
        assert out["taxa"] == []
        assert out["samples"] == []
        assert out["localities"] == []

    def test_empty_list_matches_works(self):
        """Sanity: an explicit empty list must behave identically to None."""
        from rlpe.converters import run_output_from_provenance

        prov = self._provenance()
        out = run_output_from_provenance(prov, [])
        assert out["panels"] == []


# --------------------------------------------------------------------------- L3


class TestCliFallbackPromptNonTTY:
    """L3: cli_fallback_prompt must raise on a non-TTY stdin instead of
    blocking on input().
    """

    def test_non_tty_stdin_raises(self, monkeypatch):
        import sys as _sys

        class _FakeStdin:
            def isatty(self):
                return False

        monkeypatch.setattr(_sys, "stdin", _FakeStdin())
        from rlpe.llm_backends import cli_fallback_prompt

        with pytest.raises(RuntimeError, match="not a TTY"):
            cli_fallback_prompt({"error_type": "X", "error": "y", "context": "z"})

    def test_tty_stdin_does_not_raise_on_prompt_setup(self, monkeypatch, capsys):
        """When stdin IS a TTY, cli_fallback_prompt runs the prompt
        setup (writes to stderr) and then calls input(); we stub
        input() to return '2' so the function returns 'rules' without
        actually waiting on stdin."""

        import sys as _sys

        class _FakeTTYStdin:
            def isatty(self):
                return True

        monkeypatch.setattr(_sys, "stdin", _FakeTTYStdin())
        monkeypatch.setattr("builtins.input", lambda prompt="": "2")
        from rlpe.llm_backends import cli_fallback_prompt

        result = cli_fallback_prompt({"error_type": "X", "error": "y", "context": "z"})
        assert result == "rules"
        # The prompt text must have been written to stderr.
        captured = capsys.readouterr()
        assert "MiniMax API ERROR" in captured.err


# --------------------------------------------------------------------------- M7


class TestM3BackendCallCounter:
    """M7: every attempt (success or failure) must bump total_calls.

    We construct a backend stub with a mock client that always raises
    RateLimitError, then verify total_calls == max_retries after
    the retry sequence exhausts.

    The backend's __init__ requires the real ``anthropic`` SDK to
    construct an Anthropic client. In sandbox test envs without the
    SDK, we bypass __init__ by setting only the attributes that
    ``_call_api`` reads (``_client`` and the anthropic exception
    classes). This isolates the test to the call-counter logic.
    """

    def _build_backend_bypassing_init(self):
        import types

        from rlpe.llm_backends import MiniMaxM3Backend

        # Build a stub anthropic module with the error classes
        # ``_call_api`` references. We don't import the real SDK.
        fake_anthropic = types.ModuleType("anthropic_test_stub")
        fake_anthropic.RateLimitError = type("RateLimitError", (Exception,), {})
        fake_anthropic.APIConnectionError = type("APIConnectionError", (Exception,), {})
        fake_anthropic.APIStatusError = type("APIStatusError", (Exception,), {})
        fake_anthropic_module = fake_anthropic

        # Manually construct an instance without running __init__.
        backend = MiniMaxM3Backend.__new__(MiniMaxM3Backend)
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
        backend._anthropic = fake_anthropic_module
        backend._sem = type(
            "_Sem", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: None}
        )()
        backend._lock = __import__("threading").Lock()
        backend.total_input_tokens = 0
        backend.total_output_tokens = 0
        backend.total_calls = 0
        backend.total_errors = 0
        return backend, fake_anthropic_module

    def test_failed_attempts_count_total_calls(self, monkeypatch):
        backend, fake_anthropic = self._build_backend_bypassing_init()

        class _BoomMessages:
            def create(self, **kwargs):
                raise fake_anthropic.RateLimitError("synthetic boom")

        class _BoomClient:
            messages = _BoomMessages()  # class attr: instance, not method

        backend._client = _BoomClient()
        # Avoid actually sleeping during the retry backoff.
        monkeypatch.setattr("time.sleep", lambda _s: None)

        with pytest.raises(fake_anthropic.RateLimitError):
            backend._call_api(system_prompt="x", messages=[{"role": "user", "content": "y"}])

        # After max_retries=3 attempts, total_calls must be 3 (not 2).
        assert backend.total_calls == 3, (
            f"_call_api did not bump total_calls for failed attempts; "
            f"expected 3, got {backend.total_calls}"
        )
        assert backend.total_errors == 1

    def test_successful_call_increments_by_one(self, monkeypatch):
        backend, _ = self._build_backend_bypassing_init()
        backend.max_retries = 2

        class _StubUsage:
            input_tokens = 100
            output_tokens = 50

        class _StubBlock:
            type = "text"
            text = "stub"

        class _StubResponse:
            usage = _StubUsage()
            content = [_StubBlock()]

        class _StubMessages:
            def create(self, **kwargs):
                return _StubResponse()

        class _StubClient:
            messages = _StubMessages()  # class attr: instance, not method

        backend._client = _StubClient()
        resp = backend._call_api(system_prompt="x", messages=[{"role": "user", "content": "y"}])
        assert isinstance(resp, _StubResponse)
        assert backend.total_calls == 1
        assert backend.total_errors == 0
        assert backend.total_input_tokens == 100
        assert backend.total_output_tokens == 50
