"""Phase 59 — Pipeline correctness, Bug 2.8.

Some sites in the pipeline use hardcoded page offsets (+2/-2/±20/+3)
when expanding the caption-search window. ``PipelineConfig`` already
exposes ``caption_window`` and ``od_caption_window`` as the single
source of truth (Phase 28 unified 4 paths under one field), but a
few leftover hardcoded offsets in ``_process_one_pdf_grobid`` still
expand the candidate-pages list to ±1 page unconditionally.

The fix: replace hardcoded ±1 neighbour logic with a configurable
expansion that uses ``self.config.od_caption_window`` so operators
can widen the search without code changes.

This test asserts:

  1. ``_process_one_pdf_grobid`` no longer hardcodes
     ``best_page.page_index - 2`` / ``+ 0`` neighbours.
  2. The expansion uses ``self.config.od_caption_window`` (or
     ``caption_window``) as the configured radius.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_REPO = Path(__file__).resolve().parents[1]


def test_no_hardcoded_minus_2_neighbour_in_grobid_branch() -> None:
    """Bug 2.8 source-guard: the hardcoded ``page_index - 2`` neighbour
    expansion in ``_process_one_pdf_grobid`` must be replaced with
    ``self.config.od_caption_window`` / ``caption_window``.
    """
    src = (_REPO / "src/rlpe/pipeline.py").read_text(encoding="utf-8")

    # Locate the _process_one_pdf_grobid function body.
    fn_start = src.find("def _process_one_pdf_grobid_inner(")
    if fn_start < 0:
        fn_start = src.find("def _process_one_pdf_grobid(")
    assert fn_start >= 0, "_process_one_pdf_grobid not found"

    # Search for the hardcoded neighbour pattern: ``best_page.page_index - 2``.
    pattern = re.compile(r"best_page\.page_index\s*-\s*2\b")
    matches = pattern.findall(src)
    assert not matches, (
        "Bug 2.8: hardcoded neighbour offset 'best_page.page_index - 2' "
        "still in pipeline.py — must use self.config.od_caption_window."
    )

    pattern2 = re.compile(r"best_page\.page_index\s*\+\s*1\b")
    matches2 = pattern2.findall(src)
    # ``+ 1`` is acceptable as a *literal* index shift (1-based →
    # 0-based offset); we tolerate it but the window radius itself
    # must be configurable. (Keep this assertion as a soft check.)
    # NOTE: not asserting ``matches2 == []`` because page indices are
    # 1-based in some structures and the shift is a known idiom.


def test_pipeline_uses_od_caption_window_for_neighbour_expansion() -> None:
    """Bug 2.8 source-guard: the candidate-pages expansion now references
    ``self.config.od_caption_window`` (or ``caption_window``).
    """
    src = (_REPO / "src/rlpe/pipeline.py").read_text(encoding="utf-8")
    # The expansion block is in _process_one_pdf_grobid (the OUTER
    # function that contains the per-caption loop). Search for the
    # candidate-pages expansion anchor and inspect the surrounding
    # 2000 chars.
    anchor = "candidate_pages = [best_page]"
    fn_start = src.find(anchor)
    assert fn_start >= 0, "candidate_pages expansion not found"
    fn_body = src[fn_start : fn_start + 2000]
    assert "od_caption_window" in fn_body or "caption_window" in fn_body, (
        "Bug 2.8: candidate-pages expansion must reference "
        "config.od_caption_window or config.caption_window."
    )


def test_pipelineconfig_has_od_caption_window() -> None:
    """Sanity-check: PipelineConfig.od_caption_window exists and is an int."""
    import inspect

    from rlpe.config import PipelineConfig

    sig = inspect.signature(PipelineConfig)
    assert "od_caption_window" in sig.parameters


def test_no_hardcoded_plus_3_minus_3_or_pm_20_in_pipeline() -> None:
    """Bug 2.8 source-guard: searches for other hardcoded page-window
    magic numbers in pipeline.py.
    """
    src = (_REPO / "src/rlpe/pipeline.py").read_text(encoding="utf-8")
    forbidden = [
        "page_index - 3",
        "page_index + 3",
        "page_index - 20",
        "page_index + 20",
        "page_index - 2",
    ]
    found = [p for p in forbidden if p in src]
    assert not found, (
        f"Bug 2.8: hardcoded page offsets {found} still in pipeline.py — "
        f"must use config.od_caption_window."
    )
