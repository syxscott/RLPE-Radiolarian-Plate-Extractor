"""Phase 61 Plan 4 (Bug 4.1): token-aware caption truncation for LLM calls.

Some real-world captions exceed 2000 characters (Bandini 2011 plate 09
is ~3500 chars by itself). The hard char cap used to silently drop the
tail and lose species labels near the caption's end.

This module exposes ``_truncate_caption_for_llm`` which:

  1. prefers a tokenizer-aware count when one is supplied (the
     Pipeline / backend hands us its tokenizer),
  2. falls back to a 4000-character cap otherwise (still 2x the
     historical 2000 limit),
  3. tags the result with a ``mode`` ("none" | "char_fallback" |
     "token_aware") so we can log and report what actually happened.

The function is intentionally small and side-effect-free so it can be
called from pipeline.py without dragging in a backend object.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Phase 61: cap raised from 2000 → 4000 chars. The previous limit silently
# dropped species for multi-plate captions (Bandini 2011 pl09 ≈ 3500 chars).
DEFAULT_MAX_CHARS: int = 4000

# Token cap when a tokenizer is available. Empirically a single-prompt M3
# call comfortably fits 4K tokens; raising beyond that risks hitting the
# provider's max_output_tokens budget on the response side.
DEFAULT_MAX_TOKENS: int = 4000


def _truncate_caption_for_llm(
    caption: str | None,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_chars: int = DEFAULT_MAX_CHARS,
    tokenizer: Any | None = None,
) -> tuple[str, str]:
    """Return ``(text, mode)`` where ``mode`` ∈ {"none","token_aware","char_fallback"}.

    Parameters
    ----------
    caption : str | None
        The raw caption text. ``None`` and empty strings pass through
        as ``""`` with ``mode="none"``.
    max_tokens : int
        Token cap when ``tokenizer`` is available.
    max_chars : int
        Character cap used when no tokenizer is supplied, OR as a
        safety floor when the tokenizer fails.
    tokenizer : Any | None
        An object with a ``.encode`` method that returns a list of
        token ids (e.g. a HuggingFace or SentencePiece tokenizer).
        ``None`` disables token-aware counting and falls back to
        ``max_chars``.
    """
    if not caption:
        return ("", "none")

    # Token-aware path.
    if tokenizer is not None and max_tokens > 0:
        try:
            ids = tokenizer.encode(caption)
            if len(ids) <= max_tokens:
                return (caption, "none")
            # Truncate by re-encoding the first max_tokens tokens.
            truncated_ids = ids[:max_tokens]
            try:
                truncated = tokenizer.decode(truncated_ids)
            except Exception:  # pragma: no cover - decoding failed
                truncated = caption[:max_chars]
                return (truncated, "char_fallback")
            # As a safety net, also cap by chars (some captions decoded
            # back through a tokenizer grow ~10% — this avoids shipping
            # a 5x larger prompt than we intended).
            if len(truncated) > max_chars * 4:
                truncated = truncated[: max_chars * 4]
            return (truncated, "token_aware")
        except Exception as exc:  # pragma: no cover - tokenizer crashed
            logger.debug(
                "_truncate_caption_for_llm: tokenizer failed (%s); char fallback",
                exc,
            )
            # fall through to char path

    # Char-fallback path: cap at max_chars. Slightly under-cap by
    # 1 char so the "...[truncated]" suffix cannot push us over.
    if len(caption) <= max_chars:
        return (caption, "none")
    truncated = caption[:max_chars]
    return (truncated, "char_fallback")
