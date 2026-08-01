"""Regression tests for audit 2026-08-01 batch W5 — llm_backends.py 11 bugs (M2/M3/M4/M6/M7/M8/M9/M10/M11/M12/M14)."""

from __future__ import annotations

import sys
import unittest.mock as mock
from pathlib import Path
from typing import Any

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Fake helpers
# ---------------------------------------------------------------------------
def _make_fake_anthropic_module() -> Any:
    """Return a tiny stand-in for the ``anthropic`` module so we can
    exercise ``MiniMaxM3Backend._call_api`` without the real SDK."""

    class RateLimitError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    class APIStatusError(Exception):
        def __init__(self, message: str, status_code: int = 500):
            super().__init__(message)
            self.status_code = status_code

    mod = mock.MagicMock()
    mod.RateLimitError = RateLimitError
    mod.APIConnectionError = APIConnectionError
    mod.APIStatusError = APIStatusError
    return mod


def _make_MiniMax_backend(**overrides) -> Any:
    """Return a ``MiniMaxM3Backend`` with the SDK stubbed out so
    construction never tries to talk to the network."""
    fake_anth = _make_fake_anthropic_module()
    fake_client = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"anthropic": fake_anth}):
        from rlpe.llm_backends import MiniMaxM3Backend

        defaults = dict(api_key="sk-test-1234567890123456", data_outbound_policy="api_full")
        defaults.update(overrides)
        with mock.patch.object(MiniMaxM3Backend, "__post_init__", lambda self: None):
            backend = MiniMaxM3Backend(**defaults)
            backend._anthropic = fake_anth
            backend._client = fake_client
            backend._lock = type("L", (), {"__enter__": lambda s: s, "__exit__": lambda *a: None})()
            # Rebuild the lock as a real threading.Lock
            import threading

            backend._lock = threading.Lock()
            backend._sem = threading.Semaphore(backend.max_concurrent)
            # Reset the counters
            backend.total_calls = 0
            backend.total_errors = 0
            backend.total_input_tokens = 0
            backend.total_output_tokens = 0
            backend.failed_with_thinking = 0
            backend.fallback_4xx_hints = 0
            return backend


# ---------------------------------------------------------------------------
# Bug M2: max_tokens calculation when thinking is enabled
# ---------------------------------------------------------------------------
class TestM2ThinkingMaxTokens:
    def test_thinking_adds_to_max_tokens(self):
        """When enable_thinking=True, max_tokens should be
        max_output_tokens + thinking_budget_tokens (not the old
        max(max_output_tokens, thinking_budget + 256))."""
        backend = _make_MiniMax_backend(
            max_output_tokens=2048,
            thinking_budget_tokens=1024,
            enable_thinking=True,
        )
        kwargs = backend._build_request_kwargs("sys", [{"role": "user", "content": "hi"}])
        assert kwargs["max_tokens"] == 2048 + 1024  # = 3072
        # The thinking block should be present
        assert kwargs["thinking"]["budget_tokens"] == 1024

    def test_no_thinking_uses_max_output_tokens(self):
        backend = _make_MiniMax_backend(
            max_output_tokens=2048,
            thinking_budget_tokens=1024,
            enable_thinking=False,
        )
        kwargs = backend._build_request_kwargs("sys", [{"role": "user", "content": "hi"}])
        assert kwargs["max_tokens"] == 2048
        assert "thinking" not in kwargs


# ---------------------------------------------------------------------------
# Bug M3: 4xx (non-401/403) with no fallback should fail fast
# ---------------------------------------------------------------------------
class TestM3FailFastOn4xxNoFallback:
    def test_4xx_no_fallback_raises_after_one_attempt(self, monkeypatch):
        """When a 4xx occurs and no fallback is configured, the loop
        must NOT retry — it must raise the last exception after
        exactly one attempt."""
        backend = _make_MiniMax_backend(max_retries=3)
        # No fallback configured
        backend._configured_fallback = None

        fake_anth = backend._anthropic
        fake_client = backend._client
        fake_client.messages.create.side_effect = fake_anth.APIStatusError(
            "bad request", status_code=400
        )

        # Patch time.sleep so we don't actually wait
        monkeypatch.setattr("rlpe.llm_backends.time.sleep", lambda *a, **k: None)

        with pytest.raises(fake_anth.APIStatusError):
            backend._call_api("sys", [{"role": "user", "content": "hi"}])
        # Should have tried exactly once (not max_retries=3)
        assert fake_client.messages.create.call_count == 1
        # total_errors was bumped (M4) and total_calls reflects 1 attempt
        assert backend.total_calls == 1
        assert backend.total_errors == 1


# ---------------------------------------------------------------------------
# Bug M4: 401/403 + FallbackRecommendedError raise bump total_errors
# ---------------------------------------------------------------------------
class TestM4TotalErrorsBumpedOnNonRetryable:
    def test_401_bumps_total_errors_before_raise(self, monkeypatch):
        backend = _make_MiniMax_backend()
        fake_anth = backend._anthropic
        fake_client = backend._client
        fake_client.messages.create.side_effect = fake_anth.APIStatusError(
            "auth failed", status_code=401
        )

        monkeypatch.setattr("rlpe.llm_backends.time.sleep", lambda *a, **k: None)

        with pytest.raises(fake_anth.APIStatusError):
            backend._call_api("sys", [{"role": "user", "content": "hi"}])
        # M4: counter should reflect the failed attempt
        assert backend.total_errors == 1
        assert fake_client.messages.create.call_count == 1

    def test_403_bumps_total_errors_before_raise(self, monkeypatch):
        backend = _make_MiniMax_backend()
        fake_anth = backend._anthropic
        fake_client = backend._client
        fake_client.messages.create.side_effect = fake_anth.APIStatusError(
            "forbidden", status_code=403
        )

        monkeypatch.setattr("rlpe.llm_backends.time.sleep", lambda *a, **k: None)

        with pytest.raises(fake_anth.APIStatusError):
            backend._call_api("sys", [{"role": "user", "content": "hi"}])
        assert backend.total_errors == 1
        assert fake_client.messages.create.call_count == 1

    def test_4xx_with_fallback_recommended_bumps_total_errors(self, monkeypatch):
        """4xx with a real fallback configured should raise
        FallbackRecommendedError AND bump total_errors."""
        from rlpe import llm_backends
        from rlpe.llm_backends import FallbackRecommendedError

        backend = _make_MiniMax_backend()
        backend.set_fallback_backend("gemma4")
        fake_anth = backend._anthropic
        fake_client = backend._client
        # 422 is a non-429 4xx — falls into the "else" branch that
        # calls select_backend_after_4xx and may raise
        # FallbackRecommendedError when the helper recommends the
        # configured fallback. Force the helper to always recommend
        # the fallback on the first attempt so the test deterministically
        # hits the FallbackRecommendedError path.
        monkeypatch.setattr(
            llm_backends,
            "select_backend_after_4xx",
            lambda current_backend, configured_fallback, attempts_made: "gemma4",
        )
        fake_client.messages.create.side_effect = fake_anth.APIStatusError(
            "bad request", status_code=422
        )

        monkeypatch.setattr("rlpe.llm_backends.time.sleep", lambda *a, **k: None)

        with pytest.raises(FallbackRecommendedError):
            backend._call_api("sys", [{"role": "user", "content": "hi"}])
        # M4: total_errors bumped before FallbackRecommendedError raised
        assert backend.total_errors >= 1
        assert backend.fallback_4xx_hints >= 1


# ---------------------------------------------------------------------------
# Bug M6: parse failures route through a single helper
# ---------------------------------------------------------------------------
class TestM6ParseFailureCounting:
    def test_no_thinking_bumps_total_errors_only(self):
        """A parse failure without thinking should bump total_errors
        but NOT failed_with_thinking."""
        backend = _make_MiniMax_backend()
        backend._record_parse_failure(has_thinking=False)
        assert backend.total_errors == 1
        assert backend.failed_with_thinking == 0

    def test_thinking_bumps_both_counters(self):
        backend = _make_MiniMax_backend()
        backend._record_parse_failure(has_thinking=True)
        assert backend.total_errors == 1
        assert backend.failed_with_thinking == 1

    def test_record_failed_with_thinking_does_not_double_count(self):
        """The public ``record_failed_with_thinking`` should delegate
        to the helper so total_errors is bumped only once per call."""
        backend = _make_MiniMax_backend()
        backend.record_failed_with_thinking("some reasoning")
        assert backend.total_errors == 1
        assert backend.failed_with_thinking == 1
        # Calling again bumps each counter by 1 — never double
        backend.record_failed_with_thinking("more reasoning")
        assert backend.total_errors == 2
        assert backend.failed_with_thinking == 2

    def test_make_result_routes_through_helper(self, monkeypatch):
        """A parse failure in _make_result should bump total_errors
        via the helper, even when no thinking block is present."""
        from rlpe.llm_backends import MiniMaxM3Backend

        backend = _make_MiniMax_backend()
        fake_resp = mock.MagicMock()
        # text block with non-parseable content
        text_block = mock.MagicMock()
        text_block.type = "text"
        text_block.text = "not json at all"
        fake_resp.content = [text_block]
        result = backend._make_result(fake_resp)
        assert result["fallback_used"] is True
        assert result["error_type"] == "JSONParseError"
        # M6: total_errors was bumped even though there's no thinking
        assert backend.total_errors == 1
        assert backend.failed_with_thinking == 0


# ---------------------------------------------------------------------------
# Bug M7: multimodal_degraded signal
# ---------------------------------------------------------------------------
class TestM7MultimodalDegradedSignal:
    def test_text_only_path_returns_false(self):
        """When no image is provided, the chat completion call should
        return ``(text, False)`` — multimodal was never attempted."""
        from rlpe.llm_backends import LlamaCppGemmaBackend

        backend = LlamaCppGemmaBackend(host="http://127.0.0.1:8080")
        with mock.patch("rlpe.llm_backends.requests.post") as mock_post:
            mock_resp = mock.MagicMock()
            mock_resp.json.return_value = {"content": "hello"}
            mock_post.return_value = mock_resp
            text, degraded = backend._chat_completion(None, "sys", "user")
        assert text == "hello"
        assert degraded is False

    def test_multimodal_success_returns_false(self):
        from PIL import Image

        from rlpe.llm_backends import LlamaCppGemmaBackend

        backend = LlamaCppGemmaBackend(host="http://127.0.0.1:8080")
        with mock.patch("rlpe.llm_backends.requests.post") as mock_post:
            mock_resp = mock.MagicMock()
            mock_resp.json.return_value = {"choices": [{"message": {"content": "species x"}}]}
            mock_post.return_value = mock_resp
            img = Image.new("RGB", (16, 16))
            text, degraded = backend._chat_completion(img, "sys", "user")
        assert text == "species x"
        assert degraded is False

    def test_multimodal_failure_returns_true(self):
        """If the multimodal endpoint fails and we fall back to
        /completion, the degraded flag must be True."""
        from PIL import Image

        from rlpe.llm_backends import LlamaCppGemmaBackend

        backend = LlamaCppGemmaBackend(host="http://127.0.0.1:8080")
        with mock.patch("rlpe.llm_backends.requests.post") as mock_post:
            # First call (multimodal) fails, second call (text fallback) succeeds
            multimodal_resp = mock.MagicMock()
            multimodal_resp.raise_for_status.side_effect = RuntimeError("oops")
            fallback_resp = mock.MagicMock()
            fallback_resp.json.return_value = {"content": "fallback text"}
            mock_post.side_effect = [multimodal_resp, fallback_resp]
            img = Image.new("RGB", (16, 16))
            text, degraded = backend._chat_completion(img, "sys", "user")
        assert text == "fallback text"
        assert degraded is True

    def test_infer_panel_propagates_degraded_flag(self):
        """The infer_panel result dict must include ``multimodal_degraded``."""
        from PIL import Image

        from rlpe.llm_backends import LlamaCppGemmaBackend

        backend = LlamaCppGemmaBackend(host="http://127.0.0.1:8080")
        with mock.patch("rlpe.llm_backends.requests.post") as mock_post:
            multimodal_resp = mock.MagicMock()
            multimodal_resp.raise_for_status.side_effect = RuntimeError("oops")
            fallback_resp = mock.MagicMock()
            fallback_resp.json.return_value = {"content": '{"species": "X"}'}
            mock_post.side_effect = [multimodal_resp, fallback_resp]
            img = Image.new("RGB", (16, 16))
            result = backend.infer_panel(img, "", [], "sys", "user")
        assert result["multimodal_degraded"] is True
        assert result["species"] == "X"


# ---------------------------------------------------------------------------
# Bug M8: Anthropic SDK constructed with max_retries=0
# ---------------------------------------------------------------------------
class TestM8AnthropicMaxRetries:
    def test_Anthropic_constructor_called_with_max_retries_zero(self):
        """The Anthropic client must be constructed with max_retries=0
        so its internal retry loop doesn't multiply with our outer
        3-attempt loop."""
        fake_anth = _make_fake_anthropic_module()
        fake_anth.Anthropic = mock.MagicMock()

        with mock.patch.dict(sys.modules, {"anthropic": fake_anth}):
            # Re-import to ensure the patched module is used
            from rlpe.llm_backends import MiniMaxM3Backend

            MiniMaxM3Backend(
                api_key="sk-test-1234567890123456",
                data_outbound_policy="api_full",
            )
        # The constructor should have been called with max_retries=0
        call_kwargs = fake_anth.Anthropic.call_args.kwargs
        assert call_kwargs.get("max_retries") == 0


# ---------------------------------------------------------------------------
# Bug M9: exponential backoff jitter
# ---------------------------------------------------------------------------
class TestM9BackoffJitter:
    def test_rate_limit_sleep_includes_jitter(self, monkeypatch):
        backend = _make_MiniMax_backend()
        fake_anth = backend._anthropic
        fake_client = backend._client
        # First two attempts rate-limited, third succeeds
        fake_client.messages.create.side_effect = [
            fake_anth.RateLimitError("rl1"),
            fake_anth.RateLimitError("rl2"),
            mock.MagicMock(content=[mock.MagicMock(type="text", text='{"species": "ok"}')]),
        ]

        sleeps = []
        monkeypatch.setattr("rlpe.llm_backends.time.sleep", lambda s: sleeps.append(s))
        backend._call_api("sys", [{"role": "user", "content": "hi"}])
        # M9: each sleep should be 2**attempt (0 or 1 for first 2) + jitter in [0, 1)
        assert len(sleeps) >= 2
        for i, s in enumerate(sleeps):
            assert s >= 2**i, f"sleep {s} should be >= {2**i} (base)"
            assert s < 2**i + 1.5, f"sleep {s} should be < {2**i + 1.5} (base + small jitter)"

    def test_backoff_uses_random_jitter(self, monkeypatch):
        """The jitter component should be random.uniform(0, 1), so
        the sleep value should be > base (not == base)."""
        backend = _make_MiniMax_backend()
        fake_anth = backend._anthropic
        fake_client = backend._client
        # Use a long enough backoff (attempt 3) to make jitter non-trivial
        fake_client.messages.create.side_effect = [
            fake_anth.RateLimitError("rl"),
            mock.MagicMock(content=[mock.MagicMock(type="text", text='{"species": "ok"}')]),
        ]

        sleeps = []
        monkeypatch.setattr("rlpe.llm_backends.time.sleep", lambda s: sleeps.append(s))
        backend._call_api("sys", [{"role": "user", "content": "hi"}])
        # attempt 0: base = 1, sleep should be 1 + [0, 1) jitter
        assert len(sleeps) >= 1
        assert sleeps[0] >= 1.0
        # If jitter were absent, the sleep would be exactly 1.0; with
        # jitter, it's strictly > 1.0. Allow tiny float tolerance.
        assert sleeps[0] > 1.0


# ---------------------------------------------------------------------------
# Bug M10: SSRF bypass via IPv4-mapped IPv6
# ---------------------------------------------------------------------------
class TestM10SSRFGuard:
    def test_ipv4_mapped_ipv6_metadata_blocked(self, monkeypatch):
        """``::ffff:169.254.169.254`` (IPv6 wrapping the AWS metadata
        endpoint) must be rejected."""
        from rlpe.llm_backends import _validate_llm_host

        monkeypatch.delenv("RLPE_LLM_ALLOW_ANY_HOST", raising=False)
        with pytest.raises(ValueError):
            _validate_llm_host("http://[::ffff:169.254.169.254]/latest")

    def test_direct_link_local_blocked(self, monkeypatch):
        from rlpe.llm_backends import _validate_llm_host

        monkeypatch.delenv("RLPE_LLM_ALLOW_ANY_HOST", raising=False)
        with pytest.raises(ValueError):
            _validate_llm_host("http://169.254.169.254/latest")

    def test_loopback_allowed(self, monkeypatch):
        from rlpe.llm_backends import _validate_llm_host

        monkeypatch.delenv("RLPE_LLM_ALLOW_ANY_HOST", raising=False)
        # 127.0.0.1 is the default Ollama/LlamaCpp host
        assert _validate_llm_host("http://127.0.0.1:11434") == "http://127.0.0.1:11434"

    def test_ipv6_loopback_allowed(self, monkeypatch):
        from rlpe.llm_backends import _validate_llm_host

        monkeypatch.delenv("RLPE_LLM_ALLOW_ANY_HOST", raising=False)
        # [::1] is IPv6 loopback — must be allowed like 127.0.0.1
        assert _validate_llm_host("http://[::1]:11434") == "http://[::1]:11434"

    def test_rfc1918_allowed(self, monkeypatch):
        from rlpe.llm_backends import _validate_llm_host

        monkeypatch.delenv("RLPE_LLM_ALLOW_ANY_HOST", raising=False)
        for h in ("http://10.0.0.1", "http://192.168.0.1", "http://172.16.0.1"):
            assert _validate_llm_host(h) == h

    def test_override_env_disables_guard(self, monkeypatch):
        from rlpe.llm_backends import _validate_llm_host

        monkeypatch.setenv("RLPE_LLM_ALLOW_ANY_HOST", "1")
        # Even link-local should pass when override is set
        assert _validate_llm_host("http://169.254.169.254/") == "http://169.254.169.254/"


# ---------------------------------------------------------------------------
# Bug M11: max_concurrent=0 should raise at construction
# ---------------------------------------------------------------------------
class TestM11MaxConcurrentValidation:
    def test_zero_raises_value_error(self):
        from rlpe.llm_backends import MiniMaxM3Backend

        with pytest.raises(ValueError, match="max_concurrent"):
            MiniMaxM3Backend(
                api_key="sk-test-1234567890123456",
                max_concurrent=0,
            )

    def test_negative_raises_value_error(self):
        from rlpe.llm_backends import MiniMaxM3Backend

        with pytest.raises(ValueError, match="max_concurrent"):
            MiniMaxM3Backend(
                api_key="sk-test-1234567890123456",
                max_concurrent=-1,
            )

    def test_positive_works(self):
        from rlpe.llm_backends import MiniMaxM3Backend

        backend = MiniMaxM3Backend(
            api_key="sk-test-1234567890123456",
            max_concurrent=4,
            data_outbound_policy="local_only",  # skip SDK init
        )
        assert backend.max_concurrent == 4


# ---------------------------------------------------------------------------
# Bug M12: _normalize_panel_dict handles string confidence
# ---------------------------------------------------------------------------
class TestM12ConfidenceCoercion:
    def test_string_high_defaults_to_zero(self):
        from rlpe.llm_backends import _normalize_panel_dict

        out = _normalize_panel_dict({"confidence": "high", "species": "X"})
        assert out["confidence"] == 0.0

    def test_numeric_string_parses(self):
        from rlpe.llm_backends import _normalize_panel_dict

        out = _normalize_panel_dict({"confidence": "0.7", "species": "X"})
        assert out["confidence"] == 0.7

    def test_none_defaults_to_zero(self):
        from rlpe.llm_backends import _normalize_panel_dict

        out = _normalize_panel_dict({"confidence": None, "species": "X"})
        assert out["confidence"] == 0.0

    def test_missing_defaults_to_zero(self):
        from rlpe.llm_backends import _normalize_panel_dict

        out = _normalize_panel_dict({"species": "X"})
        assert out["confidence"] == 0.0

    def test_numeric_passes_through(self):
        from rlpe.llm_backends import _normalize_panel_dict

        out = _normalize_panel_dict({"confidence": 0.5, "species": "X"})
        assert out["confidence"] == 0.5

    def test_out_of_range_clamped(self):
        from rlpe.llm_backends import _normalize_panel_dict

        # Even bad inputs that parse should be clamped
        out = _normalize_panel_dict({"confidence": 1.5, "species": "X"})
        assert out["confidence"] == 1.0
        out = _normalize_panel_dict({"confidence": -0.3, "species": "X"})
        assert out["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Bug M14: _make_result redacts API keys in parse error message
# ---------------------------------------------------------------------------
class TestM14ParseErrorRedaction:
    def test_parse_error_redacts_api_key_in_result(self):
        """When parse_json_from_text raises and the error message
        contains a fake key, the result dict's ``error`` and
        ``reasoning`` fields must not contain the raw key."""
        from rlpe.llm_backends import MiniMaxM3Backend, _redact_api_keys

        backend = _make_MiniMax_backend()
        fake_resp = mock.MagicMock()
        text_block = mock.MagicMock()
        text_block.type = "text"
        # The parse failure exception's str() will be embedded in the
        # result; we want to make sure the redaction helper is applied
        # to the message we write into ``reasoning``/``error``.
        text_block.text = "garbage that will fail to parse"
        fake_resp.content = [text_block]

        # Patch parse_json_from_text to raise with a key in the message
        from rlpe import llm_backends

        original_parse = llm_backends.parse_json_from_text
        fake_key = "sk-ant-api03-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

        def _raising_parse(text):
            raise RuntimeError(f"got {fake_key} from server")

        with mock.patch.object(llm_backends, "parse_json_from_text", _raising_parse):
            result = backend._make_result(fake_resp)
        # The fake key must NOT appear in any user-visible field
        for field in ("reasoning", "error", "raw_text", "thinking"):
            value = str(result.get(field, ""))
            assert fake_key not in value, f"key leaked in {field}: {value!r}"
        # And the [REDACTED] marker should be present instead
        assert "[REDACTED]" in result["error"] or "[REDACTED]" in result["reasoning"]
        # Restore (not strictly needed, mock.patch auto-cleans)
        _ = original_parse
