"""Lightweight, dependency-free text filters.

This module is intentionally torch-/gemma-/paddleocr-free so that the
evaluation harness (``scripts/evaluate.py``) and the unit tests
covering placeholder detection can import it without paying the cost
of the heavy pipeline imports. Keep the public surface small and
side-effect-free.
"""
from __future__ import annotations

import re
from typing import Final


_PLACEHOLDER_CAPTION_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"page\s+\d+\s*(auto[- ]?generated|placeholder|header|footer)", re.IGNORECASE),
    re.compile(r"(auto[- ]?generated|placeholder)\s+(image|figure|page)", re.IGNORECASE),
    # Chinese: 自动生成 (auto-generated), 占位 (placeholder), 页眉/页脚 (header/footer)
    re.compile(r"(自动生成|占位图|占位|页眉|页脚)"),
    re.compile(r"^\s*page\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*(running\s+head|header|footer)\s*$", re.IGNORECASE),
    # Copyright/license lines (allow attribution continuation)
    re.compile(r"^\s*(©|copyright|licen[sc]e|creative\s+commons)[\s.©]", re.IGNORECASE),
    re.compile(r"\b(scientific|elsevier|springer|wiley|tandfonline)\s*$", re.IGNORECASE),
)


def matches_have_fallback_error(matches: list) -> bool:
    """Decide whether a list of panel matches has at least one "real"
    gemma fallback error.

    The metadata fields carry three distinct signals:
      - ``gemma_error``: a real API or runtime failure
      - ``gemma_fallback``: M3 returned a low-confidence verdict
      - ``m3_rejected_non_radiolarian``: a normal "this isn't a specimen"
        answer, which is *not* an error

    The original method lived on ``RadiolarianPipeline`` and was tested
    by importing the full pipeline (which drags in torch / gemma /
    paddleocr). Extracting it here keeps the test surface small and
    the import graph shallow.
    """
    return any(
        (m.metadata.get("gemma_error") or m.metadata.get("gemma_fallback"))
        and not m.metadata.get("m3_rejected_non_radiolarian")
        for m in matches
    )


def looks_like_placeholder_caption(caption_text: str) -> bool:
    """Heuristic: return True when the caption itself signals non-specimen content.

    The OpenDataLoader extractor sometimes picks up page headers, running
    titles, or auto-generated watermarks as a "figure" with a short caption.
    Sending those to M3 wastes API calls and produces confusing "not a
    specimen" responses that get surfaced as fallback errors.
    """
    if not caption_text:
        return False
    text = caption_text.strip()
    if len(text) <= 3:
        return True
    return any(p.search(text) for p in _PLACEHOLDER_CAPTION_PATTERNS)
