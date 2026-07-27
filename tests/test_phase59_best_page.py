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


# ---- Phase X: plate-page fallback for caption-in-body / figure-at-end layouts


def test_is_likely_plate_page() -> None:
    """is_likely_plate_page returns True for plate keywords."""
    from rlpe.layout import is_likely_plate_page
    from dataclasses import replace

    plate_pages = [
        _make_page(1000, 1000, "Plate I. Radiolarian fauna"),
        _make_page(1000, 1000, "pl. 3  Fig. 7"),
        _make_page(1000, 1000, "Figure Plate Section 2"),
        _make_page(1000, 1000, "图版说明  Plate"),
    ]
    non_plate_pages = [
        _make_page(1000, 1000, "Fig. 5  This species is shown in figure 5"),
        _make_page(1000, 1000, "a plateau formation"),  # "plateau" contains "plate" but not as a plate keyword
        _make_page(1000, 1000, ""),  # empty page
    ]
    for i, p in enumerate(plate_pages):
        plate_pages[i] = replace(p, page_index=i + 1)
    for i, p in enumerate(non_plate_pages):
        non_plate_pages[i] = replace(p, page_index=i + 10)

    for p in plate_pages:
        assert is_likely_plate_page(p), f"{p.text!r} should be a plate page"

    for p in non_plate_pages:
        assert not is_likely_plate_page(p), f"{p.text!r} should NOT be a plate page"


def test_find_plate_pages_returns_only_second_half() -> None:
    """find_plate_pages searches only pages in the back half of the document."""
    from rlpe.layout import find_plate_pages
    from dataclasses import replace

    # 10 pages total; mid = 5. Pages 6-10 are "back half".
    pages = []
    for i in range(10):
        p = _make_page(1000, 1000, f"Plate {i+1}" if i >= 5 else f"Fig. {i+1}")
        pages.append(replace(p, page_index=i + 1))

    plate_pages = find_plate_pages(pages)
    assert len(plate_pages) == 5, f"Expected 5 back-half plate pages, got {len(plate_pages)}"
    assert all(p.page_index > 5 for p in plate_pages), (
        "All returned pages must be in the back half (page_index > mid)"
    )


def test_choose_best_page_source_guard_plate_fallback() -> None:
    """Source guard: ``choose_best_page`` must contain plate-keyword fallback
    logic so that caption text mentioning a plate can reach the、集中图版页
    even when ``find_caption_pages`` returns a non-empty fallback list.

    The primary "caption in body text / figure at the end" fix lives in
    ``pipeline.py`` (candidate_pages extension), not here — but the
    ``choose_best_page`` body must still contain the plate-search branch
    as a belt-and-suspenders fallback.
    """
    src = (Path(__file__).resolve().parents[1] / "src/rlpe/layout.py").read_text(
        encoding="utf-8"
    )
    fn_idx = src.find("def choose_best_page")
    assert fn_idx >= 0
    fn_end = src.find("\n\n\n", fn_idx)
    if fn_end < 0:
        fn_end = len(src)
    fn_body = src[fn_idx:fn_end]
    # Must contain plate fallback logic.
    assert "_PLATE_KEYWORD_RE" in fn_body or "find_plate_pages" in fn_body, (
        "choose_best_page must have a plate-page fallback branch "
        "(search for _PLATE_KEYWORD_RE or find_plate_pages) "
        "to handle caption-on-body / figure-on-plate layouts."
    )


def test_choose_best_page_no_plate_fallback_when_no_plate_keyword() -> None:
    """Regression: when caption mentions no plate keyword, the plate
    fallback must NOT trigger — the function should still return the
    lowest-density page from the whole document.
    """
    from rlpe.layout import choose_best_page
    from dataclasses import replace

    pages = [
        replace(_make_page(1000, 1000, "Fig. 3 body reference " * 5),
                page_index=1),
        replace(_make_page(1000, 1000, "Plate I. Some radiolarians"),  # plate page in front half
                page_index=3),
        replace(_make_page(1000, 1000, "very sparse "),
                page_index=5),
    ]

    # No plate keyword in caption → should NOT return the plate page
    best = choose_best_page(pages, "99", "Fig. 5 species name")
    assert best is not None
    # Without plate fallback, should return lowest density = page 5
    assert best.page_index == 5, (
        f"Without plate keyword in caption, fallback should be lowest density "
        f"(page 5); got page {best.page_index}"
    )
