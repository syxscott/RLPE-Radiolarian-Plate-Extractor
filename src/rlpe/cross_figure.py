"""Lightweight cross-figure panel reassignment.

This module is intentionally torch-/gemma-/paddleocr-free so that
``scripts/evaluate.py`` and the unit tests can call the orphan-to-plate
reassignment without dragging in the heavy pipeline imports. The
public surface is ``_cross_figure_reassign_results`` (and the
re-exported ``text_filters.looks_like_placeholder_caption`` shim that
``pipeline.py`` forwards to).
"""
from __future__ import annotations

import logging
from typing import Any

from .text_filters import looks_like_placeholder_caption

logger = logging.getLogger(__name__)


def _cross_figure_reassign_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reassign panels from orphan figures to the nearest real plate figure.

    Background: OpenDataLoader sometimes extracts a real plate image as
    one figure and a smaller sub-image of the same plate (an index map,
    a thumbnail, a half-resolution duplicate) as a separate "figure".
    The sub-image goes through the pipeline and produces N panels with
    no species (because its caption is either empty or a placeholder).
    The actual species live on the plate figure's caption, two pages
    away. Without reassignment, those N panels are silently lost.

    Strategy: a figure is "orphan" if (a) it has 0 species matched,
    (b) its caption is missing/empty/placeholder, and (c) it sits
    between two real plate figures (or within 3 pages of one). Move
    its panels to the nearest plate figure.
    """
    if not results:
        return results

    # Group by figure
    by_figure: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_figure.setdefault(r.get("figure_id", ""), []).append(r)

    # Identify "real plate" figures vs orphans. A real plate figure
    # has a non-placeholder caption OR at least one panel with a
    # species assignment.
    figure_caption: dict[str, str] = {}
    figure_page: dict[str, int] = {}
    figure_species_count: dict[str, int] = {}
    for fid, panels in by_figure.items():
        cap = ""
        page = 0
        sp_count = 0
        for r in panels:
            meta = r.get("metadata") or {}
            if not cap:
                cap = (r.get("caption_snippet") or "")
            if not page:
                page = int(meta.get("page_index") or 0)
            if r.get("species"):
                sp_count += 1
        figure_caption[fid] = cap
        figure_page[fid] = page
        figure_species_count[fid] = sp_count

    def _is_orphan(fid: str) -> bool:
        cap = figure_caption.get(fid, "") or ""
        if not cap:
            return True
        if looks_like_placeholder_caption(cap):
            return True
        if figure_species_count.get(fid, 0) == 0:
            # No species matched AND caption is non-empty (not
            # placeholder) — could still be a real plate that the
            # caption-parser missed. We keep it as a real plate
            # in that case so we don't accidentally drain it.
            return False
        return False

    real_plates = [fid for fid in by_figure if not _is_orphan(fid)]
    orphans = [fid for fid in by_figure if _is_orphan(fid)]
    if not real_plates or not orphans:
        return results

    reassigned: list[dict[str, Any]] = []
    for r in results:
        fid = r.get("figure_id", "")
        if fid in orphans and real_plates:
            # Find the nearest real plate by absolute page diff.
            # Pages in this codebase are 1-indexed (see
            # ``render_pdf_pages`` which uses ``start=1``), so a
            # ``page_index`` of 0 means the metadata is missing. When
            # either the orphan or the nearest real plate has a missing
            # page, the page-distance is meaningless and reassignment
            # would misattribute panels to an unrelated figure. Skip
            # in that case and keep the panels in place.
            rp = figure_page.get(fid, 0)
            if rp == 0:
                reassigned.append(r)
                continue
            nearest = min(
                real_plates,
                key=lambda f: abs(figure_page.get(f, 0) - rp),
            )
            nearest_page = figure_page.get(nearest, 0)
            if nearest_page == 0:
                reassigned.append(r)
                continue
            # Reassign only if the page gap is small (<=3).
            if abs(nearest_page - rp) <= 3:
                new = dict(r)
                new["figure_id"] = nearest
                new["metadata"] = dict(r.get("metadata") or {})
                new["metadata"]["reassigned_from_figure"] = fid
                new["metadata"]["reassigned_reason"] = (
                    "orphan figure, caption empty/placeholder, "
                    f"merged into plate figure {nearest}"
                )
                reassigned.append(new)
                continue
        reassigned.append(r)
    # Length invariant: every input row is appended exactly once (either
    # the reassigned copy or the original). Assert it instead of the
    # previous "if mismatched, return original" branch — that branch was
    # unreachable because the loop above always appends, so a future
    # refactor that adds a ``continue`` without an ``append`` would
    # silently drop rows. The assertion converts a silent data-loss bug
    # into a loud test failure.
    assert len(reassigned) == len(results), (
        f"cross_figure_reassign produced {len(reassigned)} rows from "
        f"{len(results)} inputs; this should never happen"
    )
    moved = sum(
        1 for r in reassigned
        if (r.get("metadata") or {}).get("reassigned_from_figure")
    )
    if moved:
        logger.info("Cross-figure reassignment: moved %d panels from orphan figures.", moved)
    return reassigned
