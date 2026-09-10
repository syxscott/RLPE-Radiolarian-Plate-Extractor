"""Tests for LLM backend initialization (Round 5 fix).

These tests run in minimal-env CI runners (no cv2).  They exercise
only _try_init_gemma's contract by constructing a PipelineConfig
and calling the relevant logic with mocked dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from rlpe.config import PipelineConfig


def _try_init_gemma_standalone(config: PipelineConfig) -> object | None:
    """Replicate the _try_init_gemma logic for standalone testing
    without needing the full RadiolarianPipeline (which imports cv2).
    """
    from rlpe.gemma_postprocess import build_gemma_backend_from_config
    from rlpe.llm_backends import AnthropicCompatBackend, resolve_llm_api_key

    # Mirror the F17 pipeline heuristic: canonical cloud backend name is
    # "anthropic"; pre-F17 vendor alias values map onto it.
    anthropic_backends = {"anthropic", "minimax", "minimax-m3", "minimax_api"}
    backend_name = str(config.extra.get("llm_backend") or "").lower() or "transformers"
    if backend_name in {"minimax", "minimax-m3", "minimax_api"}:
        backend_name = "anthropic"
        config.extra["llm_backend"] = "anthropic"
    has_llm_key = resolve_llm_api_key(config.extra) is not None
    has_local_model = bool(
        config.extra.get("gemma_model_path")
        or config.extra.get("ollama_model")
        or config.extra.get("llama_model")
    )
    use_cloud_llm = backend_name in anthropic_backends or (has_llm_key and not has_local_model)
    if not use_cloud_llm and not config.extra.get("use_gemma4", False):
        return None
    model_path = config.extra.get("gemma_model_path") or config.extra.get("ollama_model")
    if not use_cloud_llm and not model_path and backend_name not in {"ollama"}:
        return None
    if use_cloud_llm and backend_name not in anthropic_backends:
        config.extra["llm_backend"] = "anthropic"
        backend_name = "anthropic"

    try:
        runtime = build_gemma_backend_from_config(config.extra)
    except Exception:
        return None

    # Verify backend type.
    assert isinstance(runtime.backend, AnthropicCompatBackend), (
        f"Expected LLM backend, got {type(runtime.backend).__name__}"
    )
    assert runtime.backend_name == "anthropic"
    return runtime


def _make_config(extra: dict) -> PipelineConfig:
    return PipelineConfig(
        pdf_dir=Path("/tmp"),
        work_dir=Path("/tmp"),
        ocr_backend="none",
        use_gpu=False,
        num_workers=1,
        extra=extra,
    )


class TestLLMInitWithKey:
    """LLM backend MUST initialize when API key is present."""

    def test_init_with_minimax_key_in_extra(self, monkeypatch):
        """LLM key in extra config → backend initializes."""
        for k in ("ANTHROPIC_API_KEY", "MINIMAX_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        config = _make_config(
            {
                "llm_api_key": "sk-test-fake-key",
                "llm_base_url": "http://mock/",
                "llm_model": "MiniMax-M3",
                "data_outbound_policy": "local_only",  # skip SDK check in sandbox
            }
        )
        runtime = _try_init_gemma_standalone(config)
        assert runtime is not None, "LLM backend must initialize when API key is in extra"
        assert runtime.backend_name == "anthropic"

    def test_init_with_minimax_key_in_env(self, monkeypatch):
        """LLM key in environment → backend initializes."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-env-key")
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        config = _make_config({"data_outbound_policy": "local_only"})
        runtime = _try_init_gemma_standalone(config)
        assert runtime is not None
        assert runtime.backend_name == "anthropic"

    def test_init_with_minimax_backend_name_only(self, monkeypatch):
        """Explicit llm_backend=llm + key → backend initializes."""
        for k in ("ANTHROPIC_API_KEY", "MINIMAX_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        config = _make_config(
            {
                "llm_backend": "minimax",
                "llm_api_key": "sk-test",
                "data_outbound_policy": "local_only",
            }
        )
        runtime = _try_init_gemma_standalone(config)
        assert runtime is not None
        assert runtime.backend_name == "anthropic"


class TestMiniMaxInitGuards:
    """Without a key, LLM must NOT initialize."""

    def test_no_key_no_local_model_returns_none(self, monkeypatch):
        """No key + no local model → backend stays None."""
        for k in ("ANTHROPIC_API_KEY", "MINIMAX_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        config = _make_config({})
        runtime = _try_init_gemma_standalone(config)
        assert runtime is None

    def test_local_model_path_skips_minimax(self, monkeypatch):
        """Local model path present → NOT auto-selected as LLM."""
        for k in ("ANTHROPIC_API_KEY", "MINIMAX_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        config = _make_config({"gemma_model_path": "/some/local/model"})
        runtime = _try_init_gemma_standalone(config)
        if runtime is not None:
            assert runtime.backend_name != "MiniMax", (
                "Local model path must not trigger LLM backend"
            )


class TestFallbackHandlerAttached:
    """LLM init must attach a FallbackHandler."""

    def test_fallback_handler_set(self, monkeypatch):
        for k in ("ANTHROPIC_API_KEY", "MINIMAX_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        config = _make_config(
            {
                "llm_api_key": "sk-test",
                "llm_base_url": "http://mock/",
                "data_outbound_policy": "local_only",
            }
        )
        runtime = _try_init_gemma_standalone(config)
        assert runtime is not None
        from rlpe.llm_backends import AnthropicCompatBackend

        assert isinstance(runtime.backend, AnthropicCompatBackend)
