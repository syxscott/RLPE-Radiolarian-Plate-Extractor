"""Regression tests for audit 2026-08-19 Phase 4E — M3 telemetry + LLM
error classification + prompt-registry version stamp.

Phase 4E covers three closely-related fixes that landed together so the
LLM-failure surfaces in ``/system/llm-status`` are debuggable:

1. **Task 1 — LLM failure reason recorded.**  Before Phase 4E the engine
   caught every M3 exception in one bucket (``except Exception``) and
   only logged ``"infer_panel failed"``. Operators had no way to
   distinguish an auth failure (which requires a key rotation) from a
   rate-limit (which is transient) from a timeout (which is
   load-related) from a parse error (which is a model-quality issue).
   The fix adds four stub exception classes
   (``LLMAuthenticationError`` / ``LLMRateLimitError`` /
   ``LLMTimeoutError`` / ``LLMSchemaError``) and a typed ``except``
   chain in ``_infer_text`` / ``_infer_vision`` that maps each one to
   a short code stored on ``_telemetry.llm_error``.

2. **Task 2 — M3 telemetry fields.**  Every M3 result now carries a
   ``_telemetry`` sub-dict with ``model`` / ``prompt_version`` /
   ``latency_ms`` / ``timestamp`` (and the optional ``llm_error`` on
   failure paths) so downstream code can correlate cost / latency /
   drift with a known prompt revision without re-walking the raw
   response dict.

3. **Task 3 — Stage prompt version field.**  ``get_prompt_registry()``
   now returns ``(dict, version_str)``; ``PROMPT_REGISTRY_VERSION`` is
   the canonical version stamp and is bumped on every prompt change
   so audit can pin a result to a known prompt revision.

These tests are run via ``pytest tests/test_audit_2026_08_19_phase4e_
telemetry.py tests/test_m3_engine.py -v``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from PIL import Image  # noqa: E402

from rlpe.m3_engine import (  # noqa: E402
    PROMPT_REGISTRY,
    PROMPT_REGISTRY_VERSION,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMSchemaError,
    M3Engine,
    get_prompt_registry,
    get_prompt_registry_version,
)
from tests.fakes.fake_m3_backend import FakeM3Backend  # noqa: E402

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _plate(size: int = 256) -> Image.Image:
    """A plate image large enough to clear the engine's <32px short-circuit."""
    return Image.new("RGB", (size, size))


def _engine_with_backend(backend) -> M3Engine:
    """Wrap ``backend`` in an M3Engine with retry-without-thinking OFF
    so the test can isolate first-attempt behaviour."""
    return M3Engine(backend, config={"m3_retry_without_thinking": False})


# ===========================================================================
# Task 3 — Stage prompt version field
# ===========================================================================


class TestPromptRegistryVersion:
    """``get_prompt_registry()`` now returns ``(dict, version_str)`` so
    callers can pin a result to a known prompt revision; ``get_prompt_
    registry_version()`` is a convenience accessor."""

    def test_get_prompt_registry_returns_two_tuple(self):
        registry, version = get_prompt_registry()
        assert isinstance(registry, dict)
        assert isinstance(version, str)

    def test_get_prompt_registry_version_is_nonempty_string(self):
        v = get_prompt_registry_version()
        assert isinstance(v, str)
        assert len(v) > 0, "version stamp must be non-empty"

    def test_prompt_registry_version_constant_is_set(self):
        """The module-level constant must be a non-empty string so other
        modules can import the stamp directly (e.g. for diagnostic dumps)."""
        assert isinstance(PROMPT_REGISTRY_VERSION, str)
        assert PROMPT_REGISTRY_VERSION
        assert get_prompt_registry_version() == PROMPT_REGISTRY_VERSION

    def test_prompt_registry_dict_contains_known_stages(self):
        """All 5+ canonical stage keys must be present (audit M-12:
        registry completeness guard). Missing a key would silently
        disable that stage in downstream code paths."""
        registry, _version = get_prompt_registry()
        expected = {
            "parse_caption",
            "parse_caption_ja",
            "classify_plate",
            "segment_panels",
            "match_panel",
            "match_panel_visual_only",
            "critique_matches",
        }
        missing = expected - set(registry.keys())
        assert not missing, f"missing registry keys: {sorted(missing)}"

    def test_get_prompt_registry_returns_independent_dicts(self):
        """Mutating the returned dict must not poison the module cache."""
        r1, v1 = get_prompt_registry()
        r1["__injected__"] = "garbage"
        r2, v2 = get_prompt_registry()
        assert "__injected__" not in r2, "registry must return a fresh dict"
        assert v1 == v2


# ===========================================================================
# Task 2 — M3 telemetry fields on the success path
# ===========================================================================


class TestInferVisionTelemetry:
    """``_infer_vision`` must stamp ``_telemetry`` on every result dict."""

    def test_telemetry_attached_on_success(self):
        """Successful vision call: telemetry must include model, prompt
        version, latency_ms (int >= 0), and an ISO-format timestamp."""
        backend = FakeM3Backend(
            canned_responses=[
                {
                    "raw_text": json.dumps(
                        {
                            "label": "1",
                            "species": "Genus species",
                            "confidence": 0.9,
                            "reasoning": "ok",
                        }
                    ),
                    "fallback_used": False,
                }
            ]
        )
        engine = _engine_with_backend(backend)
        out = engine._infer_vision("sys", "user", _plate())
        assert "_telemetry" in out, f"missing _telemetry: {out!r}"
        tel = out["_telemetry"]
        assert tel["model"] == backend.model
        assert tel["prompt_version"] == PROMPT_REGISTRY_VERSION
        assert isinstance(tel["latency_ms"], int)
        assert tel["latency_ms"] >= 0
        assert isinstance(tel["timestamp"], str)
        # ISO-8601 timestamp must contain a 'T' separator (Python's
        # datetime.isoformat() default).
        assert "T" in tel["timestamp"], f"timestamp not ISO-8601: {tel['timestamp']!r}"
        # No llm_error on the success path.
        assert "llm_error" not in tel

    def test_telemetry_latency_is_nonnegative(self):
        """Wall-clock latency must be >= 0 — a back-dated ``start`` arg
        would crash with a negative int; the engine guards via
        ``max(0, ...)``."""
        backend = FakeM3Backend(
            canned_responses=[
                {
                    "raw_text": json.dumps({"ok": 1}),
                    "fallback_used": False,
                }
            ]
        )
        engine = _engine_with_backend(backend)
        out = engine._infer_vision("sys", "user", _plate())
        assert out["_telemetry"]["latency_ms"] >= 0

    def test_telemetry_no_backend_returns_other_error(self):
        """The ``self.backend is None`` short-circuit must also stamp
        ``_telemetry`` with ``llm_error='other'`` so callers can
        distinguish 'no backend wired' from 'backend returned nothing'."""
        engine = M3Engine(None, config={"m3_retry_without_thinking": False})
        out = engine._infer_vision("sys", "user", _plate())
        assert out.get("fallback_used") is True
        assert out["_telemetry"]["llm_error"] == "other"
        # No actual model — keep the field but accept None.
        assert "model" in out["_telemetry"]
        assert out["_telemetry"]["model"] is None


class TestInferTextTelemetry:
    """``_infer_text`` must stamp ``_telemetry`` symmetrically to ``_infer_vision``."""

    def test_telemetry_attached_on_success(self):
        backend = FakeM3Backend(
            canned_responses=[
                {
                    "raw_text": json.dumps({"ok": True}),
                    "fallback_used": False,
                }
            ]
        )
        engine = _engine_with_backend(backend)
        out = engine._infer_text("sys", "user")
        tel = out["_telemetry"]
        assert tel["model"] == backend.model
        assert tel["prompt_version"] == PROMPT_REGISTRY_VERSION
        assert isinstance(tel["latency_ms"], int)
        assert tel["latency_ms"] >= 0
        assert "llm_error" not in tel

    def test_telemetry_no_backend_returns_other_error(self):
        engine = M3Engine(None, config={"m3_retry_without_thinking": False})
        out = engine._infer_text("sys", "user")
        assert out.get("fallback_used") is True
        assert out["_telemetry"]["llm_error"] == "other"


# ===========================================================================
# Task 1 — LLM failure reason classification
# ===========================================================================


class TestInferVisionErrorClassification:
    """The typed-exception chain in ``_infer_vision`` must map each
    stub exception to a distinct ``llm_error`` code."""

    def test_authentication_error_classified_as_auth(self):
        class _AuthBackend:
            backend_name = "fake-auth"
            model = "fake"
            enable_thinking = False
            max_concurrent = 1

            def infer_panel(self, **_):
                raise LLMAuthenticationError("401 invalid api key")

        engine = M3Engine(_AuthBackend(), config={"m3_retry_without_thinking": False})
        out = engine._infer_vision("sys", "user", _plate())
        assert out.get("fallback_used") is True
        assert out["_telemetry"]["llm_error"] == "auth"
        # Error string from the exception flows into ``error``.
        assert "401" in out.get("error", "")

    def test_rate_limit_error_classified_as_rate_limit(self):
        class _RateLimitBackend:
            backend_name = "fake-rate"
            model = "fake"
            enable_thinking = False
            max_concurrent = 1

            def infer_panel(self, **_):
                raise LLMRateLimitError("429 too many requests")

        engine = M3Engine(_RateLimitBackend(), config={"m3_retry_without_thinking": False})
        out = engine._infer_vision("sys", "user", _plate())
        assert out.get("fallback_used") is True
        assert out["_telemetry"]["llm_error"] == "rate_limit"

    def test_timeout_error_classified_as_timeout(self):
        """``TimeoutError`` is the Python built-in; the engine aliases
        it as ``LLMTimeoutError`` and must catch both."""

        class _TimeoutBackend:
            backend_name = "fake-timeout"
            model = "fake"
            enable_thinking = False
            max_concurrent = 1

            def infer_panel(self, **_):
                raise TimeoutError("socket read timeout")

        engine = M3Engine(_TimeoutBackend(), config={"m3_retry_without_thinking": False})
        out = engine._infer_vision("sys", "user", _plate())
        assert out.get("fallback_used") is True
        assert out["_telemetry"]["llm_error"] == "timeout"

    def test_schema_error_classified_as_parse(self):
        """``LLMSchemaError`` covers schema violations AFTER a successful
        JSON parse — distinguishable from a raw JSONDecodeError so audit
        can split the parse-failure modes apart."""

        class _SchemaBackend:
            backend_name = "fake-schema"
            model = "fake"
            enable_thinking = False
            max_concurrent = 1

            def infer_panel(self, **_):
                raise LLMSchemaError("missing required field 'species'")

        engine = M3Engine(_SchemaBackend(), config={"m3_retry_without_thinking": False})
        out = engine._infer_vision("sys", "user", _plate())
        assert out.get("fallback_used") is True
        assert out["_telemetry"]["llm_error"] == "parse"

    def test_generic_exception_classified_as_other(self):
        """Unclassified exceptions must end up as ``llm_error='other'``
        rather than being silently dropped — that way audit still
        sees a failure happened."""

        class _BoomBackend:
            backend_name = "fake-boom"
            model = "fake"
            enable_thinking = False
            max_concurrent = 1

            def infer_panel(self, **_):
                raise RuntimeError("kaboom")

        engine = M3Engine(_BoomBackend(), config={"m3_retry_without_thinking": False})
        out = engine._infer_vision("sys", "user", _plate())
        assert out.get("fallback_used") is True
        assert out["_telemetry"]["llm_error"] == "other"

    def test_failure_telemetry_includes_latency_and_timestamp(self):
        """Even on failure the telemetry block must carry latency_ms +
        timestamp + model + prompt_version so dashboards can correlate
        the failure time with backend logs."""

        class _BoomBackend:
            backend_name = "fake-boom"
            model = "fake-model-42"
            enable_thinking = False
            max_concurrent = 1

            def infer_panel(self, **_):
                raise LLMAuthenticationError("nope")

        engine = M3Engine(_BoomBackend(), config={"m3_retry_without_thinking": False})
        out = engine._infer_vision("sys", "user", _plate())
        tel = out["_telemetry"]
        assert tel["model"] == "fake-model-42"
        assert tel["prompt_version"] == PROMPT_REGISTRY_VERSION
        assert isinstance(tel["latency_ms"], int)
        assert tel["latency_ms"] >= 0
        assert "T" in tel["timestamp"]


class TestInferTextErrorClassification:
    """Same classification chain on ``_infer_text``."""

    def test_authentication_error(self):
        class _AuthBackend:
            backend_name = "fake-auth"
            model = "fake"
            enable_thinking = False

            def infer_text(self, **kwargs):
                raise LLMAuthenticationError("401 bad key")

        engine = M3Engine(_AuthBackend(), config={"m3_retry_without_thinking": False})
        out = engine._infer_text("sys", "user")
        assert out.get("fallback_used") is True
        assert out["_telemetry"]["llm_error"] == "auth"

    def test_timeout_error(self):
        class _TimeoutBackend:
            backend_name = "fake-timeout"
            model = "fake"
            enable_thinking = False

            def infer_text(self, **kwargs):
                raise TimeoutError("read timeout")

        engine = M3Engine(_TimeoutBackend(), config={"m3_retry_without_thinking": False})
        out = engine._infer_text("sys", "user")
        assert out.get("fallback_used") is True
        assert out["_telemetry"]["llm_error"] == "timeout"

    def test_rate_limit_error(self):
        class _RateLimitBackend:
            backend_name = "fake-rate"
            model = "fake"
            enable_thinking = False

            def infer_text(self, **kwargs):
                raise LLMRateLimitError("429 thundering herd")

        engine = M3Engine(_RateLimitBackend(), config={"m3_retry_without_thinking": False})
        out = engine._infer_text("sys", "user")
        assert out["_telemetry"]["llm_error"] == "rate_limit"


# ===========================================================================
# Integration smoke: telemetry flows from _infer_vision into PanelMatch
# ===========================================================================


class TestTelemetryPropagation:
    """The match_panel stage consumes ``_infer_vision`` results. Even
    though PanelMatch.raw is the canonical carrier of MiniMax_*
    telemetry, the new ``_telemetry`` field must NOT corrupt the
    PanelMatch contract (no extra unknown key that downstream JSON
    schema would reject)."""

    def test_match_panel_preserves_existing_MiniMax_keys(self):
        from rlpe.m3_engine import CaptionPair

        class _Backend:
            backend_name = "MiniMax"
            model = "MiniMax-M3"
            enable_thinking = False
            max_concurrent = 1

            def infer_panel(self, **kwargs):
                return {
                    "label": "1",
                    "species": "Actinomma leptodermum",
                    "confidence": 0.9,
                    "reasoning": "ok",
                    "fallback_used": False,
                    "raw_text": json.dumps(
                        {
                            "label": "1",
                            "species": "Actinomma leptodermum",
                            "confidence": 0.9,
                            "reasoning": "ok",
                            "is_radiolarian": True,
                        }
                    ),
                    "request_id": "req-1",
                    "model_version": "MiniMax-M3",
                    "cost_cny": 0.045,
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                }

        engine = _engine_with_backend(_Backend())
        pairs = [
            CaptionPair(
                labels=["1"],
                species="Actinomma leptodermum",
                raw_text="Plate 1. figs 1. Actinomma leptodermum",
            )
        ]
        out = engine.match_panel(
            panel_image=_plate(),
            caption_pairs=pairs,
            caption_text="Plate 1. figs 1. Actinomma leptodermum",
        )
        # Existing MiniMax_* keys must still be on PanelMatch.raw.
        raw = out.raw or {}
        assert raw.get("MiniMax_request_id") == "req-1"
        assert abs(float(raw.get("MiniMax_cost_cny", 0)) - 0.045) < 1e-9
        assert raw.get("MiniMax_model_version") == "MiniMax-M3"
