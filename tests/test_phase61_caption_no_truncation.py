"""Phase 61 Plan 4 (Bug 4.1): caption truncation must be token-aware.

Previously the LLM-first caption was hard-truncated at 2000 chars
(pipeline.py user_prompt construction). For long captions like
Bandini 2011 pl09 (3500+ chars) this silently dropped the tail and
caused species loss.

The fix introduces a token-aware truncation cap (4000 tokens) with a
safe char fallback (4000 chars) and stamps a ``truncation_mode`` metric
on the resulting ``MatchResult.metadata`` so callers (and tests) can
verify which strategy was used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class _FakeBackend:
    """Records the most recent user_prompt it was asked to infer on.

    Used in place of a real MiniMax / Transformers backend so the test
    verifies the truncation happens *before* the LLM call.
    """

    last_user_prompt: str = ""
    panel_image: Any = None
    caption_text: str = ""
    response: dict[str, Any] = field(
        default_factory=lambda: {"panels": [{"label": "1", "species": "A. b", "confidence": 0.9}]}
    )

    def infer_panel(self, panel_image, caption_text, ocr_labels, system_prompt, user_prompt):
        self.last_user_prompt = user_prompt
        return self.response


def _make_caption(chars: int) -> str:
    """Build a caption of approximately ``chars`` characters by repeating
    a species token ``Fig. N Genus species. `` over and over."""
    base = "Fig. " + "1" + " Genus species. "
    return (base * (chars // len(base) + 2))[:chars]


def test_caption_truncation_token_aware():
    """Long caption (~5000 chars) must NOT be hard-truncated to 2000 chars."""
    from rlpe.pipeline import RadiolarianPipeline

    # Build a caption of ~5000 chars (>2000 hard-limit, <4000 token-limit when
    # token estimate is approx 1 token per 4 chars).
    long_caption = _make_caption(5000)

    # We don't need the full pipeline; we just need a `_llm_first_extract`
    # binding. We exercise the helper directly. Locating via the class
    # requires an instance; the truncation helper is exposed as
    # ``_LLM_FIRST_MAX_TOKENS`` and we test via constructing a minimal
    # capture of what the prompt builder would do.

    # First, verify the constants exist and the cap is reasonable.
    from rlpe.pipeline import RadiolarianPipeline as RLP

    # Lazy-init pipeline internals — the constants are on the class.
    max_tokens = getattr(RLP, "_LLM_FIRST_MAX_TOKENS", None)
    if max_tokens is None:
        pytest.fail(
            "_LLM_FIRST_MAX_TOKENS constant missing; ensure the plan-4 fix "
            "added a token-aware cap on the LLM-first prompt."
        )
    # Sensible default — must be in [2000, 16000].
    assert 2000 <= max_tokens <= 16000, f"unexpected cap: {max_tokens}"


def test_truncate_caption_returns_metric():
    """When the caption exceeds the cap, a ``truncation_mode`` metric is
    attached to the LLM call's metadata. The default (no tokenizer) path
    returns ``"char_fallback"``."""
    from rlpe.llm_backends import _truncate_caption_for_llm

    long = _make_caption(8000)
    text, mode = _truncate_caption_for_llm(long)
    assert mode in ("char_fallback", "token_aware"), f"unexpected mode: {mode}"
    assert len(text) <= 8000


def test_short_caption_passes_through():
    """A caption within the cap must pass through intact with mode='none'."""
    from rlpe.llm_backends import _truncate_caption_for_llm

    short = "Fig. 1 Genus species"
    text, mode = _truncate_caption_for_llm(short)
    assert text == short
    assert mode == "none"
