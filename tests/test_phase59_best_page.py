"""Phase 59 — Pipeline correctness, Bug 2.7.

``choose_best_page`` returns the first matching candidate without
considering which page is actually the best plate page (lowest text
density / highest figure score). When ``find_caption_pages`` returns
multiple matches — common in figure-heavy plates where the same
"Fig. 5" caption text repeats on adjacent pages — the first page is
not always the one with the actual plate image.

The fix: rank candidates by score (higher is better, computed as
``1 - page_text_density``) and return the top-ranked candidate.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _make_page(width: int, height: int, text: str) -> "PageRecord":
    """Build a PageRecord without touching the filesystem."""
    from rlpe.types import PageRecord

    return PageRecord(
        page_index=1,
        image_path="/tmp/x.png",
        text=text,
        width=width,
        height=height,
        metadata={"dpi": 200},
    )


def test_choose_best_page_picks_highest_score() -> None:
    """Bug 2.7 fix: of 3 candidate pages, the function must return the
    one with the highest plate-region score (lowest text density),
    NOT the first one.
    """
    from rlpe.layout import choose_best_page

    # Order is important: page DENSEST first, sparsest LAST.
    # Pre-fix behaviour returns page 2 (first match). After the fix,
    # the function ranks by score and returns page 5 (sparsest, best).
    page_dense1 = _make_page(1000, 1000, "Fig. 5\n" + "lorem ipsum " * 60)
    page_dense2 = _make_page(1000, 1000, "Fig. 5\n" + "lorem ipsum " * 30)
    page_medium = _make_page(1000, 1000, "Fig. 5\n" + "lorem ipsum " * 10)
    page_sparse = _make_page(1000, 1000, "Fig. 5\nplate only")

    from dataclasses import replace

    page_dense1 = replace(page_dense1, page_index=2)
    page_dense2 = replace(page_dense2, page_index=3)
    page_medium = replace(page_medium, page_index=4)
    page_sparse = replace(page_sparse, page_index=5)

    best = choose_best_page(
        [page_dense1, page_dense2, page_medium, page_sparse],
        "5",
        "Fig. 5 caption",
    )
    assert best is not None, "Should return one of the candidates"
    assert best.page_index == 5, (
        f"Expected page 5 (lowest density / highest score) to win; "
        f"got page {best.page_index}"
    )


def test_choose_best_page_no_candidate_returns_densest() -> None:
    """Bug 2.7 backward-compat: when no candidate matches the figure
    number, the function still falls back to the page with lowest
    text density (preserved).
    """
    from rlpe.layout import choose_best_page
    from dataclasses import replace

    page0 = _make_page(1000, 1000, "no figure here " * 10)
    page1 = _make_page(1000, 1000, "different caption " * 80)
    page0 = replace(page0, page_index=1)
    page1 = replace(page1, page_index=2)

    # figure_number not present anywhere → no caption-page candidate
    best = choose_best_page([page0, page1], "999", "caption")
    assert best is not None
    assert best.page_index == 1, (
        f"Fallback should pick lowest density; got page {best.page_index}"
    )


def test_choose_best_page_empty_input() -> None:
    from rlpe.layout import choose_best_page

    assert choose_best_page([], "5", "x") is None


def test_layout_source_ranks_by_score_descending() -> None:
    """Bug 2.7 source-guard: layout.choose_best_page sorts candidates
    by score (descending) before picking the top.
    """
    src = (Path(__file__).resolve().parents[1] / "src/rlpe/layout.py").read_text(
        encoding="utf-8"
    )
    # Look for sort/reverse/max with score descending semantics.
    # The function must sort candidates by score descending and pick
    # the first one. Use a regex-tolerant check.
    assert "sort" in src.lower() or "max(" in src, (
        "layout.choose_best_page must rank candidates by score and pick top."
    )
    # The bug fix path must be inside choose_best_page.
    fn_idx = src.find("def choose_best_page")
    assert fn_idx >= 0
    fn_end = src.find("\n\n\n", fn_idx)
    if fn_end < 0:
        fn_end = len(src)
    fn_body = src[fn_idx:fn_end]
    assert "sort" in fn_body or "max(" in fn_body, (
        "choose_best_page body must sort/max candidates by score, not return candidates[0]."
    )
    # Should no longer return candidates[0] unconditionally.
    assert "return candidates[0]" not in fn_body, (
        "Bug 2.7: choose_best_page must not return candidates[0] (always first match)."
    )
