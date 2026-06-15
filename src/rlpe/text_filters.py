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
    # Publisher footers: a SHORT line that is essentially just the
    # publisher name (with optional copyright preamble), e.g.
    # "© 2020 Elsevier" or "Springer Nature". The previous pattern
    # ``\b(scientific|elsevier|...)\s*$`` matched any caption ending
    # in those words — which mis-flagged legitimate captions like
    # "specimens collected for Elsevier" or even "...published in
    # Wiley" as placeholders. Anchor at line start with a length
    # cap (≤60 chars) so real captions (~200 chars+) don't match.
    re.compile(
        r"^\s*(?:©\s*\d{4}\s+)?"
        r"(scientific|elsevier|springer(?:\s+nature)?|wiley|tandfonline|"
        r"taylor\s+&\s+francis|sage|de\s+gruyter)"
        r"\s*\.?\s*$",
        re.IGNORECASE,
    ),
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

    NOTE: a real ``gemma_error`` ALWAYS triggers the fallback, even when
    ``m3_rejected_non_radiolarian`` is also set. The previous
    ``... and not m3_rejected_non_radiolarian`` guard masked real API
    failures whenever the response payload also said "not a specimen",
    which silently swallowed transient network / 5xx errors that the
    operator should have been told about.
    """
    for m in matches:
        md = m.metadata or {}
        # Real API/runtime error → always trigger fallback.
        if md.get("gemma_error"):
            return True
        # Low-confidence fallback signal → trigger UNLESS the same panel
        # was also rejected by M3 as "not a specimen" (in which case the
        # fallback flag is a benign side effect of stage 4's threshold,
        # not a real failure the operator can act on).
        if md.get("gemma_fallback") and not md.get("m3_rejected_non_radiolarian"):
            return True
    return False


def looks_like_placeholder_caption(caption_text: str) -> bool:
    """Heuristic: return True when the caption itself signals non-specimen content.

    The OpenDataLoader extractor sometimes picks up page headers, running
    titles, or auto-generated watermarks as a "figure" with a short caption.
    Sending those to M3 wastes API calls and produces confusing "not a
    specimen" responses that get surfaced as fallback errors.

    Empty / whitespace-only captions are also treated as placeholders:
    there is nothing to extract from them, and the downstream geology
    linker uses this signal to fall back to fulltext context. The
    previous version returned False for empty input, which let the
    LLM-first path waste an API call on empty user prompts.
    """
    if not caption_text:
        return True
    text = caption_text.strip()
    if not text:
        return True
    if len(text) <= 3:
        return True
    return any(p.search(text) for p in _PLACEHOLDER_CAPTION_PATTERNS)
