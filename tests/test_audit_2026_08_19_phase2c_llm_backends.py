"""Regression tests for audit 2026-08-19 Phase 2c — LLM backend robustness.

Bug fixes covered:
- M-14: ``cross_figure_visual_inference`` in ``m3_engine.py`` accepts a
  ``strat_image`` parameter but the previous implementation dropped it
  silently and only sent ``plate_image`` to the backend. The fix
  forwards BOTH images to the Anthropic-backed ``MiniMaxM3Backend``
  via a new ``extra_image`` keyword on ``infer_panel``. Local backends
  (llama.cpp / Ollama) accept ``extra_image`` too and inject a prompt
  note that the second image is dropped (single-image contract).

- B-4: ``LlamaCppGemmaBackend._chat_completion`` used to fall back to
  ``/completion`` on ANY ``Exception`` — including non-transient
  4xx errors (401 unauthorized, 403 forbidden, 404 wrong model, 413
  payload too large, …). The fix inspects ``exc.response.status_code``
  (or ``exc.status_code``) and re-raises any 4xx so the caller sees
  the real failure rather than a silently degraded text-only path.

- M-4: ``MiniMaxM3Backend._call_api`` retry loop ignored the
  ``Retry-After`` header sent by the MiniMax endpoint. The fix
  parses the header (numeric form) via the new
  ``_parse_retry_after`` static method and uses it (capped at 60s)
  INSTEAD of the exponential backoff when present.

These tests are read-only against the live source so they catch
prompt / contract drift and accidental removal of the helpers.
"""

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
# Mocks / fixtures
# ---------------------------------------------------------------------------


class _DummyImage:
    """Stand-in for PIL.Image.Image — exposes .width and .height only.

    For tests that need to bypass the ``_encode_image_anthropic_block``
    real-PIL path we use this. Tests that actually exercise the
    encoder use :func:`_make_pil_image` instead.
    """

    def __init__(self, width: int = 64, height: int = 64, label: str = "img") -> None:
        self.width = width
        self.height = height
        self.label = label

    def __repr__(self) -> str:
        return f"_DummyImage({self.label!r}, {self.width}x{self.height})"


def _make_pil_image(width: int = 64, height: int = 64, color: str = "red") -> Any:
    """Build a real PIL Image so the Anthropic encoder can serialize it."""
    from PIL import Image

    return Image.new("RGB", (width, height), color=color)


def _make_fake_anthropic_module() -> Any:
    """Tiny stand-in for the ``anthropic`` SDK so ``MiniMaxM3Backend``
    can be constructed without the real package."""

    class RateLimitError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    class APIStatusError(Exception):
        def __init__(self, message: str, status_code: int = 500, response: Any = None):
            super().__init__(message)
            self.status_code = status_code
            self.response = response

    mod = mock.MagicMock()
    mod.RateLimitError = RateLimitError
    mod.APIConnectionError = APIConnectionError
    mod.APIStatusError = APIStatusError
    return mod


def _make_MiniMax_backend(**overrides) -> Any:
    """Return a ``MiniMaxM3Backend`` with the SDK stubbed out."""
    fake_anth = _make_fake_anthropic_module()
    fake_client = mock.MagicMock()
    import threading

    with mock.patch.dict(sys.modules, {"anthropic": fake_anth}):
        from rlpe.llm_backends import MiniMaxM3Backend

        defaults = dict(api_key="sk-test-1234567890123456", data_outbound_policy="api_full")
        defaults.update(overrides)
        with mock.patch.object(MiniMaxM3Backend, "__post_init__", lambda self: None):
            backend = MiniMaxM3Backend(**defaults)
            backend._anthropic = fake_anth
            backend._client = fake_client
            backend._lock = threading.Lock()
            backend._sem = threading.Semaphore(backend.max_concurrent)
            backend._thread_local = threading.local()
            backend.total_calls = 0
            backend.total_errors = 0
            backend.total_input_tokens = 0
            backend.total_output_tokens = 0
            backend.failed_with_thinking = 0
            backend.fallback_4xx_hints = 0
            return backend


class _CaptureMessagesBackend:
    """Test-only backend that records the messages passed to
    ``infer_panel`` so we can assert ``extra_image`` was forwarded.

    The contract this backend implements is the same as
    ``MiniMaxM3Backend.infer_panel`` for our purposes: it receives
    ``panel_image`` + ``extra_image`` and passes them through to
    ``_build_messages``. The engine sees the result dict and returns
    a canned ``plate_panels`` payload so the cross-figure flow exits
    cleanly without an actual network call.
    """

    def __init__(self, canned_response: dict[str, Any] | None = None) -> None:
        self.canned_response = canned_response or {
            "raw_text": '{"plate_panels": []}',
            "fallback_used": False,
            "label": None,
            "species": None,
            "confidence": 0.0,
            "reasoning": "fake",
        }
        self.calls: list[dict[str, Any]] = []

    def infer_panel(
        self,
        panel_image: Any = None,
        caption_text: str = "",
        ocr_labels: list[str] | None = None,
        system_prompt: str = "",
        user_prompt: str = "",
        extra_image: Any = None,
        **_unused: Any,
    ) -> dict[str, Any]:
        # Mirror the production wiring so ``cross_figure_visual_inference``
        # can treat this backend as a drop-in replacement.
        self.calls.append(
            {
                "panel_image": panel_image,
                "extra_image": extra_image,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "caption_text": caption_text,
                "ocr_labels": list(ocr_labels or []),
            }
        )
        return dict(self.canned_response)

    def infer_text(self, system_prompt: str = "", user_prompt: str = "", **_kw: Any) -> dict[str, Any]:
        return dict(self.canned_response)


def _make_engine_with_capture_backend(
    canned_response: dict[str, Any] | None = None,
) -> tuple[Any, _CaptureMessagesBackend]:
    """Return ``(engine, capture_backend)`` for cross-figure tests."""
    from rlpe.m3_engine import M3Engine

    capture = _CaptureMessagesBackend(canned_response)
    engine = M3Engine(backend=capture, config={})
    return engine, capture


# ---------------------------------------------------------------------------
# Task 1: M-14 — cross_figure_visual_inference forwards strat_image
# ---------------------------------------------------------------------------


class TestM14CrossFigureStratImageForwarded:
    """The Anthropic ``MiniMaxM3Backend`` (and any backend that supports
    multi-image content blocks) must receive the strat column image as
    the ``extra_image`` keyword argument, so the model can reason over
    BOTH images in a single Messages API call."""

    def test_strat_image_forwarded_as_extra_image(self):
        engine, capture = _make_engine_with_capture_backend()
        plate = _DummyImage(label="plate", width=256, height=256)
        strat = _DummyImage(label="strat", width=128, height=512)
        result = engine.cross_figure_visual_inference(
            plate_image=plate,
            strat_image=strat,
            plate_caption="Plate caption",
            strat_caption="Strat caption",
        )
        assert result == {"plate_panels": []}
        assert len(capture.calls) == 1
        call = capture.calls[0]
        # The PRIMARY image is still the plate.
        assert call["panel_image"] is plate
        # The SECONDARY image is now forwarded as extra_image (was silently
        # dropped before the fix).
        assert call["extra_image"] is strat

    def test_strat_image_none_still_works_for_backward_compat(self):
        """Backward-compat: callers that pass ``strat_image=None`` (the
        pre-M-14 default behaviour) must NOT crash. The function early-
        returns with ``{"plate_panels": []}`` when the strat image is
        absent because the visual cross-figure contract requires both
        images to be present — the title check is enforced upstream.
        """
        engine, capture = _make_engine_with_capture_backend()
        plate = _DummyImage(label="plate", width=256, height=256)
        result = engine.cross_figure_visual_inference(
            plate_image=plate,
            strat_image=None,  # type: ignore[arg-type]
            plate_caption="Plate",
            strat_caption="",
        )
        assert result == {"plate_panels": []}
        # When strat_image is None the function short-circuits before
        # invoking the backend — no calls recorded.
        assert len(capture.calls) == 0

    def test_strat_image_valid_propagates_extra_image_none_when_omitted(self):
        """The dual-image test below proves strat_image IS forwarded
        when provided. Here we verify the engine passes the
        ``extra_image`` keyword (default None) through to the backend
        even when the caller only supplies the plate image."""
        # We exercise this directly via _CaptureMessagesBackend which
        # always records calls. The cross_figure_visual_inference
        # wrapper bails out on tiny/None images, so we go through
        # _infer_vision instead — that's the layer that controls the
        # ``extra_image`` propagation.
        backend = _CaptureMessagesBackend()
        from rlpe.m3_engine import M3Engine

        engine = M3Engine(backend=backend, config={})
        plate = _make_pil_image()
        # Call _infer_vision with extra_image=None (default).
        res = engine._infer_vision(
            system_prompt="sys",
            user_prompt="user",
            image=plate,
        )
        assert "fallback_used" not in res or not res.get("fallback_used")
        assert len(backend.calls) == 1
        assert backend.calls[0]["panel_image"] is plate
        assert backend.calls[0]["extra_image"] is None

    def test_minimax_backend_build_messages_includes_two_images(self):
        """``MiniMaxM3Backend._build_messages(panel_image, user_prompt,
        extra_image=strat)`` must emit a user message with TWO image
        blocks when both images are provided (Anthropic multimodal
        contract)."""
        backend = _make_MiniMax_backend()
        plate = _make_pil_image(width=64, height=64, color="red")
        strat = _make_pil_image(width=64, height=64, color="blue")
        messages = backend._build_messages(plate, "user text", extra_image=strat)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert isinstance(content, list)
        # Two image blocks + one text block = 3 blocks.
        image_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "image"]
        text_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]
        assert len(image_blocks) == 2, f"expected 2 image blocks, got {len(image_blocks)}"
        assert len(text_blocks) == 1, f"expected 1 text block, got {len(text_blocks)}"

    def test_minimax_backend_build_messages_single_image_for_backward_compat(self):
        """When ``extra_image`` is omitted, only ONE image block is
        emitted — backward compatible with all callers from before the
        audit fix."""
        backend = _make_MiniMax_backend()
        plate = _make_pil_image(width=64, height=64, color="red")
        messages = backend._build_messages(plate, "user text")
        content = messages[0]["content"]
        image_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "image"]
        assert len(image_blocks) == 1

    def test_minimax_backend_build_messages_no_panel_image(self):
        """Edge case: ``panel_image=None`` AND ``extra_image`` set →
        the extra image becomes the sole image block (symmetric
        contract)."""
        backend = _make_MiniMax_backend()
        strat = _make_pil_image(width=64, height=64, color="blue")
        messages = backend._build_messages(None, "user text", extra_image=strat)
        content = messages[0]["content"]
        image_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "image"]
        assert len(image_blocks) == 1


# ---------------------------------------------------------------------------
# Task 1 (M-14) — llama.cpp single-image degradation note
# ---------------------------------------------------------------------------


class TestM14LlamaCppSingleImageDegradation:
    """Local backends (llama.cpp) only support single-image requests.
    The fix MUST inject a clear prompt note that the strat column
    image was dropped so downstream observability surfaces the
    missing visual signal."""

    def test_llamacpp_inject_strat_column_note_when_extra_image_given(self, monkeypatch):
        """When ``extra_image`` is supplied to a llama.cpp call, the
        user prompt sent to the backend MUST mention that the second
        image was not used."""
        from rlpe.llm_backends import LlamaCppGemmaBackend

        # Stub out requests.post so we never hit the network.
        posted_payloads: list[dict[str, Any]] = []

        class _FakeResp:
            def __init__(self, payload: dict[str, Any]):
                self._payload = payload
                self.status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return {"choices": [{"message": {"content": "{}"}}]}

        def _fake_post(url: str, json: dict[str, Any], timeout: Any = None) -> _FakeResp:
            posted_payloads.append({"url": url, "json": json})
            return _FakeResp(json)

        monkeypatch.setattr("rlpe.llm_backends.requests.post", _fake_post)

        backend = LlamaCppGemmaBackend(host="http://127.0.0.1:8080", model="m")
        plate = _make_pil_image()
        strat = _make_pil_image()

        text, multimodal_degraded = backend._chat_completion(
            panel_image=plate,
            system_prompt="system prompt",
            user_prompt="user prompt",
            extra_image=strat,
        )

        assert multimodal_degraded is False  # the multimodal endpoint succeeded
        assert len(posted_payloads) == 1
        user_msg = posted_payloads[0]["json"]["messages"][-1]
        # user_msg may be either a string (text-only) or a list of blocks
        # (multimodal). In the multimodal case the text is the LAST block.
        if isinstance(user_msg["content"], list):
            text_payload = user_msg["content"][-1]["text"]
        else:
            text_payload = user_msg["content"]
        assert "strat column image not used by this backend" in text_payload
        # Original user prompt text is still present after the note.
        assert "user prompt" in text_payload

    def test_llamacpp_no_note_when_no_extra_image(self, monkeypatch):
        """When ``extra_image`` is None, the prompt must NOT contain the
        ``strat column image not used by this backend`` note — would
        be confusing for plain panel-vision callers."""
        from rlpe.llm_backends import LlamaCppGemmaBackend

        posted_payloads: list[dict[str, Any]] = []

        class _FakeResp:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return {"choices": [{"message": {"content": "{}"}}]}

        def _fake_post(url: str, json: dict[str, Any], timeout: Any = None) -> _FakeResp:
            posted_payloads.append({"url": url, "json": json})
            return _FakeResp()

        monkeypatch.setattr("rlpe.llm_backends.requests.post", _fake_post)

        backend = LlamaCppGemmaBackend(host="http://127.0.0.1:8080", model="m")
        plate = _make_pil_image()

        backend._chat_completion(
            panel_image=plate,
            system_prompt="system",
            user_prompt="user",
        )
        assert len(posted_payloads) == 1
        user_msg = posted_payloads[0]["json"]["messages"][-1]
        if isinstance(user_msg["content"], list):
            text_payload = user_msg["content"][-1]["text"]
        else:
            text_payload = user_msg["content"]
        assert "strat column image not used by this backend" not in text_payload

    def test_llamacpp_infer_panel_marks_extra_image_unsupported(self, monkeypatch):
        """``LlamaCppGemmaBackend.infer_panel(extra_image=strat)`` must
        set ``extra_image_unsupported=True`` in the result so the caller
        knows the strat column visual signal is missing."""
        from rlpe.llm_backends import LlamaCppGemmaBackend

        class _FakeResp:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return {"choices": [{"message": {"content": '{"label":"1","species":"X","confidence":0.5,"reasoning":""}'}}]}

        def _fake_post(url: str, json: dict[str, Any], timeout: Any = None) -> _FakeResp:
            return _FakeResp()

        monkeypatch.setattr("rlpe.llm_backends.requests.post", _fake_post)

        backend = LlamaCppGemmaBackend(host="http://127.0.0.1:8080", model="m")
        plate = _make_pil_image()
        strat = _make_pil_image()

        result = backend.infer_panel(
            panel_image=plate,
            caption_text="",
            ocr_labels=[],
            system_prompt="sys",
            user_prompt="user",
            extra_image=strat,
        )
        assert result["extra_image_unsupported"] is True


# ---------------------------------------------------------------------------
# Task 2: B-4 — LlamaCpp 4xx must not degrade to /completion
# ---------------------------------------------------------------------------


class TestB4LlamaCppNoDegradeOn4xx:
    """NON-transient 4xx errors (401/403/404/413/...) from the
    OpenAI-compatible ``/v1/chat/completions`` endpoint must be
    RE-RAISED — silently swapping to ``/completion`` hides a
    misconfigured API key from the user."""

    def test_401_unauthorized_re_raises_no_fallback(self, monkeypatch):
        from rlpe.llm_backends import LlamaCppGemmaBackend

        class _Fake401Resp:
            status_code = 401
            text = "Unauthorized"

            def raise_for_status(self) -> None:
                from requests import HTTPError

                raise HTTPError("401 Unauthorized", response=self)

        def _fake_post(url: str, json: dict[str, Any], timeout: Any = None) -> Any:
            return _Fake401Resp()

        monkeypatch.setattr("rlpe.llm_backends.requests.post", _fake_post)
        backend = LlamaCppGemmaBackend(host="http://127.0.0.1:8080", model="m")
        plate = _make_pil_image()

        # Phase 4B introduced typed HTTPError subclasses
        # (``LLMAuthenticationError``) so the re-raise is no longer
        # bare ``Exception``. Assert the SPECIFIC subclass instead of
        # the blind ``Exception`` (B017).
        from rlpe.llm_backends import LLMAuthenticationError

        with pytest.raises(LLMAuthenticationError):
            backend._chat_completion(
                panel_image=plate,
                system_prompt="sys",
                user_prompt="user",
            )

    def test_403_forbidden_re_raises_no_fallback(self, monkeypatch):
        from rlpe.llm_backends import LlamaCppGemmaBackend

        class _Fake403Resp:
            status_code = 403
            text = "Forbidden"

            def raise_for_status(self) -> None:
                from requests import HTTPError

                raise HTTPError("403 Forbidden", response=self)

        def _fake_post(url: str, json: dict[str, Any], timeout: Any = None) -> Any:
            return _Fake403Resp()

        monkeypatch.setattr("rlpe.llm_backends.requests.post", _fake_post)
        backend = LlamaCppGemmaBackend(host="http://127.0.0.1:8080", model="m")
        plate = _make_pil_image()

        from rlpe.llm_backends import LLMAuthenticationError

        with pytest.raises(LLMAuthenticationError):
            backend._chat_completion(
                panel_image=plate,
                system_prompt="sys",
                user_prompt="user",
            )

    def test_500_still_falls_back_to_completion(self, monkeypatch):
        """5xx errors are transient — the fallback to ``/completion``
        should still happen, and ``multimodal_degraded`` should be True."""
        from rlpe.llm_backends import LlamaCppGemmaBackend

        posted_to: list[str] = []

        class _Fake500Resp:
            status_code = 500
            text = "Internal Server Error"

            def raise_for_status(self) -> None:
                from requests import HTTPError

                raise HTTPError("500 Server Error", response=self)

        class _FakeCompletionResp:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return {"content": "{}"}

        def _fake_post(url: str, json: dict[str, Any], timeout: Any = None) -> Any:
            posted_to.append(url)
            if "/v1/chat/completions" in url:
                return _Fake500Resp()
            return _FakeCompletionResp()

        monkeypatch.setattr("rlpe.llm_backends.requests.post", _fake_post)
        backend = LlamaCppGemmaBackend(host="http://127.0.0.1:8080", model="m")
        plate = _make_pil_image()

        text, multimodal_degraded = backend._chat_completion(
            panel_image=plate,
            system_prompt="sys",
            user_prompt="user",
        )
        assert multimodal_degraded is True
        # Both endpoints were hit (chat first, then /completion fallback)
        assert any("/v1/chat/completions" in u for u in posted_to)
        assert any(u.endswith("/completion") for u in posted_to)

    def test_connection_error_still_falls_back_to_completion(self, monkeypatch):
        """Connection errors (no HTTP status code) are transient — the
        fallback to ``/completion`` must still happen."""
        from requests import ConnectionError

        from rlpe.llm_backends import LlamaCppGemmaBackend

        class _FakeCompletionResp:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return {"content": "{}"}

        def _fake_post(url: str, json: dict[str, Any], timeout: Any = None) -> Any:
            if "/v1/chat/completions" in url:
                raise ConnectionError("refused")
            return _FakeCompletionResp()

        monkeypatch.setattr("rlpe.llm_backends.requests.post", _fake_post)
        backend = LlamaCppGemmaBackend(host="http://127.0.0.1:8080", model="m")
        plate = _make_pil_image()

        text, multimodal_degraded = backend._chat_completion(
            panel_image=plate,
            system_prompt="sys",
            user_prompt="user",
        )
        assert multimodal_degraded is True


# ---------------------------------------------------------------------------
# Task 3: M-4 — Retry-After header parsing
# ---------------------------------------------------------------------------


class TestM4RetryAfterHeader:
    """``MiniMaxM3Backend._parse_retry_after`` extracts the
    ``Retry-After`` value from an exception's response headers."""

    def test_retry_after_30_seconds_parsed(self):
        """A header value of ``'30'`` (seconds) returns 30.0."""
        from rlpe.llm_backends import MiniMaxM3Backend

        class _Resp:
            headers = {"Retry-After": "30"}

        class _Exc(Exception):
            response = _Resp()

        assert MiniMaxM3Backend._parse_retry_after(_Exc()) == 30.0

    def test_retry_after_missing_returns_none(self):
        """When no ``Retry-After`` header is present, returns ``None``
        so the caller falls back to exponential backoff."""
        from rlpe.llm_backends import MiniMaxM3Backend

        class _Resp:
            headers = {"Content-Type": "application/json"}

        class _Exc(Exception):
            response = _Resp()

        assert MiniMaxM3Backend._parse_retry_after(_Exc()) is None

    def test_retry_after_no_response_attribute_returns_none(self):
        from rlpe.llm_backends import MiniMaxM3Backend

        class _Exc(Exception):
            pass

        assert MiniMaxM3Backend._parse_retry_after(_Exc()) is None

    def test_retry_after_non_numeric_returns_none(self):
        from rlpe.llm_backends import MiniMaxM3Backend

        class _Resp:
            headers = {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}

        class _Exc(Exception):
            response = _Resp()

        # Date-form Retry-After is intentionally not parsed — see
        # _parse_retry_after docstring. Returns None so the caller
        # uses exponential backoff.
        assert MiniMaxM3Backend._parse_retry_after(_Exc()) is None

    def test_retry_after_zero_or_negative_returns_none(self):
        from rlpe.llm_backends import MiniMaxM3Backend

        class _RespZero:
            headers = {"Retry-After": "0"}

        class _RespNegative:
            headers = {"Retry-After": "-5"}

        class _Exc1(Exception):
            response = _RespZero()

        class _Exc2(Exception):
            response = _RespNegative()

        assert MiniMaxM3Backend._parse_retry_after(_Exc1()) is None
        assert MiniMaxM3Backend._parse_retry_after(_Exc2()) is None


class TestM4CallApiRespectsRetryAfter:
    """``_call_api`` retry loop must honour ``Retry-After`` when present
    AND fall back to exponential backoff when not."""

    def test_429_with_retry_after_uses_header(self, monkeypatch):
        """When the 429 response has ``Retry-After: 30``, ``time.sleep``
        must be called with a value >= 30 (plus a small jitter)."""
        backend = _make_MiniMax_backend(max_retries=2)
        fake_anth = backend._anthropic
        fake_client = backend._client

        class _Resp:
            headers = {"Retry-After": "30"}

        fake_client.messages.create.side_effect = fake_anth.APIStatusError(
            "rate limited", status_code=429, response=_Resp()
        )

        sleeps: list[float] = []
        monkeypatch.setattr(
            "rlpe.llm_backends.time.sleep",
            lambda s: sleeps.append(s),
        )

        with pytest.raises(fake_anth.APIStatusError):
            backend._call_api("sys", [{"role": "user", "content": "hi"}])
        assert len(sleeps) >= 1
        # First sleep must be >= 30 (Retry-After value). The cap is 60
        # so the wait never exceeds that.
        assert sleeps[0] >= 30.0
        # The cap is 60 — the wait must not exceed that.
        assert sleeps[0] <= 60.0 + 1.0  # +1 for jitter
        # The retry loop should have given up after max_retries (==2) attempts
        # because the same 429 is raised every time and is retriable.

    def test_429_without_retry_after_uses_exponential_backoff(self, monkeypatch):
        """When the 429 response has NO ``Retry-After`` header, the
        loop must use the existing exponential backoff (min(2**attempt, 30))."""
        backend = _make_MiniMax_backend(max_retries=3)
        fake_anth = backend._anthropic
        fake_client = backend._client

        class _Resp:
            headers = {}  # no Retry-After

        fake_client.messages.create.side_effect = fake_anth.APIStatusError(
            "rate limited", status_code=429, response=_Resp()
        )

        sleeps: list[float] = []
        monkeypatch.setattr(
            "rlpe.llm_backends.time.sleep",
            lambda s: sleeps.append(s),
        )

        with pytest.raises(fake_anth.APIStatusError):
            backend._call_api("sys", [{"role": "user", "content": "hi"}])
        assert len(sleeps) >= 1
        # First sleep should be ~2**0 = 1 + jitter < 30, NOT >= 30.
        assert sleeps[0] < 30.0

    def test_5xx_with_retry_after_uses_header(self, monkeypatch):
        """Same Retry-After handling applies to 5xx APIStatusErrors."""
        backend = _make_MiniMax_backend(max_retries=2)
        fake_anth = backend._anthropic
        fake_client = backend._client

        class _Resp:
            headers = {"Retry-After": "45"}

        fake_client.messages.create.side_effect = fake_anth.APIStatusError(
            "server error", status_code=503, response=_Resp()
        )

        sleeps: list[float] = []
        monkeypatch.setattr(
            "rlpe.llm_backends.time.sleep",
            lambda s: sleeps.append(s),
        )

        with pytest.raises(fake_anth.APIStatusError):
            backend._call_api("sys", [{"role": "user", "content": "hi"}])
        assert sleeps[0] >= 45.0
        assert sleeps[0] <= 60.0 + 1.0  # +1 jitter, 60s cap

    def test_rate_limit_error_with_retry_after_uses_header(self, monkeypatch):
        """``anthropic.RateLimitError`` exceptions also carry a response
        with Retry-After — the fix must respect it the same way."""
        backend = _make_MiniMax_backend(max_retries=2)
        fake_anth = backend._anthropic
        fake_client = backend._client

        class _Resp:
            headers = {"Retry-After": "20"}

        # RateLimitError in the real SDK carries ``response``; emulate that.
        def _raise_rate_limit(*_args: Any, **_kw: Any) -> None:
            err = fake_anth.RateLimitError("rate limited")
            err.response = _Resp()
            raise err

        fake_client.messages.create.side_effect = _raise_rate_limit

        sleeps: list[float] = []
        monkeypatch.setattr(
            "rlpe.llm_backends.time.sleep",
            lambda s: sleeps.append(s),
        )

        with pytest.raises(fake_anth.RateLimitError):
            backend._call_api("sys", [{"role": "user", "content": "hi"}])
        assert sleeps[0] >= 20.0
        assert sleeps[0] <= 60.0 + 1.0


# ---------------------------------------------------------------------------
# Source guard — keep the helper methods from being accidentally removed
# ---------------------------------------------------------------------------


class TestSourceGuard:
    """Source-guard tests: detect accidental removal of the new helpers."""

    def test_parse_retry_after_still_defined(self):
        from rlpe.llm_backends import MiniMaxM3Backend

        assert hasattr(MiniMaxM3Backend, "_parse_retry_after")
        assert callable(MiniMaxM3Backend._parse_retry_after)

    def test_extract_status_code_still_defined(self):
        from rlpe.llm_backends import LlamaCppGemmaBackend

        assert hasattr(LlamaCppGemmaBackend, "_extract_status_code")
        assert callable(LlamaCppGemmaBackend._extract_status_code)

    def test_infer_panel_accepts_extra_image(self):
        """Both backends' ``infer_panel`` must accept the ``extra_image``
        keyword (backward-compatible default ``None``)."""
        import inspect

        from rlpe.llm_backends import LlamaCppGemmaBackend, MiniMaxM3Backend

        sig_minimax = inspect.signature(MiniMaxM3Backend.infer_panel)
        sig_llamacpp = inspect.signature(LlamaCppGemmaBackend.infer_panel)
        assert "extra_image" in sig_minimax.parameters
        assert "extra_image" in sig_llamacpp.parameters
        # Both must default to None so existing callers are unaffected.
        assert sig_minimax.parameters["extra_image"].default is None
        assert sig_llamacpp.parameters["extra_image"].default is None
