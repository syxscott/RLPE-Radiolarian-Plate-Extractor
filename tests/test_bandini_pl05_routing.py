"""Tests for bandini2011 pl05 routing fix (Round 4 / 2026-07-03).

Bug report (from memory project_v20_results_2026_07_02.md):

    bandini pl05 — 42 panels dropped. _find_plate_captions correctly
    detects pl05 caption (page 20, length 2630), but
    _build_figures_from_plate_captions doesn't claim page-21 images.
    Pre-existing (also in 06-29 v18 cached).

The fix must ensure that _build_figures_from_plate_captions claims
images whose page numbers fall in [caption_page_lo, caption_page_hi]
even when the next plate caption is far enough away to leave forward
room. The audit trace shows that when the previous clamp logic
applied ``page_hi = min(page_lo + 2, next_cap_page - 1)`` and the
clamp value ended up below ``page_lo``, the trailing ``max(page_hi,
page_lo)`` re-opened the window to a SINGLE page — silently dropping
any forward-page images.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


def _img(img_id: int, page: int) -> dict:
    """Build a minimal OpenDataLoader image element."""
    return {"id": img_id, "page number": page, "page_number": page}


def _cap(plate_num: int, page: int, content: str | None = None) -> dict:
    return {
        "plate_number": plate_num,
        "page_number": page,
        "content": content if content is not None else f"Plate {plate_num}.",
    }


class TestBuildFiguresFromPlateCaptions:
    """P1-3: bandini pl05 routing.

    The pre-fix bug: when caption gaps are >= 3 pages (e.g. pl04 p17,
    pl05 p20, pl06 p24), the previous clamp logic ``page_hi = min(22,
    next_cap_page - 1) = min(22, 19) = 19`` shrunk pl04's window to
    p17..p19 AND also left pl05 with a single-page window
    (page_lo=20, page_hi=20 after the max() backstop). Any forward
    images on page 21+ would either be claimed by pl04 (steal) or
    dropped entirely. The fix is the ``next_cap_page >= page_lo + 2``
    guard: clamp only when the next caption is at least 2 pages
    ahead; otherwise keep the full page_lo..page_lo+2 window so
    forward images remain claimable.
    """

    def test_pl05_caption_page20_claims_images_on_page21(self):
        """Reproduces the pl05 bug: caption page 20, 4 images on
        page 21 (forward page), 1 image on page 19 (backward).
        Expected: pl04 claims page-19 image, pl05 claims 4 page-21 images.
        """
        from rlpe.opendataloader_extractor import _build_figures_from_plate_captions

        plate_captions = [
            _cap(4, 17, "Plate 4."),
            _cap(5, 20, "Plate 5. (very long caption)"),
            _cap(6, 24, "Plate 6."),
        ]
        images = [
            _img(101, 19),
            _img(102, 21),
            _img(103, 21),
            _img(104, 21),
            _img(105, 21),
        ]
        pairs, claimed_ids = _build_figures_from_plate_captions(
            plate_captions, images, Path("/tmp"), "paper1"
        )
        # pl04 should claim image 101 (page 19); pl05 should claim
        # images 102..105 (page 21).
        assert 101 in claimed_ids, "pl04 should claim image 101 (page 19)"
        for img_id in (102, 103, 104, 105):
            assert img_id in claimed_ids, f"pl05 should claim image {img_id} (page 21) but did not"
        # Total unique claimed = 5; each image claimed exactly once.
        assert len(claimed_ids) == 5

    def test_no_images_for_caption_yields_empty_pair(self):
        """A plate caption with no nearby images still produces a pair
        (so the downstream matcher sees the caption text); image_paths
        is empty and metadata flags 'no_images'."""
        from rlpe.opendataloader_extractor import _build_figures_from_plate_captions

        plate_captions = [_cap(5, 30, "Plate 5.")]  # no images nearby
        images = [_img(1, 1), _img(2, 5)]
        pairs, claimed_ids = _build_figures_from_plate_captions(
            plate_captions, images, Path("/tmp"), "paper1"
        )
        assert len(pairs) == 1
        assert pairs[0].image_paths == []
        assert pairs[0].metadata.get("no_images") is True

    def test_next_caption_clamps_forward_window(self):
        """If pl06 caption is on page 23 (within pl05's window of
        [20, 22+1=23]), the forward window is clamped to avoid
        stealing pl06's image. The pre-existing code already does
        this; we just lock the contract. Image 302 (page 23) belongs
        to pl06, NOT pl05 — verify it was claimed by pl06, not pl05."""
        from rlpe.opendataloader_extractor import _build_figures_from_plate_captions

        plate_captions = [
            _cap(5, 20, "Plate 5."),
            _cap(6, 23, "Plate 6."),
        ]
        images = [
            _img(300, 21),
            _img(301, 22),
            _img(302, 23),  # pl06's caption page
        ]
        pairs, claimed_ids = _build_figures_from_plate_captions(
            plate_captions, images, Path("/tmp"), "paper1"
        )
        # All three images claimed across the two plates (no over/under).
        assert claimed_ids == {300, 301, 302}
        # The figure_id of the pair that contains image 302 must be
        # pl06 (its caption is on page 23). We can't easily verify
        # the image_paths list (image_paths are absolute paths that
        # may not resolve under /tmp), so we anchor on page_number
        # + plate_number metadata.
        pl05_pair = next(p for p in pairs if "pl05" in p.figure_id)
        pl06_pair = next(p for p in pairs if "pl06" in p.figure_id)
        assert pl05_pair.metadata["plate_number"] == 5
        assert pl06_pair.metadata["plate_number"] == 6
        # pl05 should anchor on its earliest image page (21), pl06 on 23.
        assert pl05_pair.page_number == 21
        assert pl06_pair.page_number == 23

    def test_all_three_plates_claim_distinct_images(self):
        """Integration: 3 plates with images spread across the window.
        Verify no over-claim and no under-claim."""
        from rlpe.opendataloader_extractor import _build_figures_from_plate_captions

        plate_captions = [
            _cap(1, 5, "Plate 1."),
            _cap(2, 9, "Plate 2."),
            _cap(3, 13, "Plate 3."),
        ]
        images = [
            _img(10, 5),
            _img(11, 6),
            _img(20, 9),
            _img(21, 10),
            _img(30, 13),
            _img(31, 14),
        ]
        pairs, claimed_ids = _build_figures_from_plate_captions(
            plate_captions, images, Path("/tmp"), "paper1"
        )
        assert claimed_ids == {10, 11, 20, 21, 30, 31}
        for pair in pairs:
            assert pair.metadata["plate_number"] in (1, 2, 3)

    def test_single_plate_uses_default_two_page_window(self):
        """A single plate caption with no neighbours gets page_lo..page_lo+2."""
        from rlpe.opendataloader_extractor import _build_figures_from_plate_captions

        plate_captions = [_cap(5, 20, "Plate 5.")]
        images = [_img(100, 20), _img(101, 21), _img(102, 22)]
        pairs, claimed_ids = _build_figures_from_plate_captions(
            plate_captions, images, Path("/tmp"), "paper1"
        )
        assert claimed_ids == {100, 101, 102}

    def test_tightly_packed_captions_keep_full_window(self):
        """When two captions are tightly packed (only 1 page apart),
        the forward window is NOT clamped (no room for forward images
        beyond the next caption page anyway)."""
        from rlpe.opendataloader_extractor import _build_figures_from_plate_captions

        plate_captions = [
            _cap(1, 5, "Plate 1."),
            _cap(2, 6, "Plate 2."),
        ]
        # Image 100 on page 7 is between pl02 caption (p6) and the
        # typical pl03 caption (p8). With the fix, the clamp only
        # activates when next_cap_page >= page_lo + 2 (=7 here for
        # pl02); pl02 window is [6, 8] = page 7 in range.
        images = [_img(100, 7)]
        pairs, claimed_ids = _build_figures_from_plate_captions(
            plate_captions, images, Path("/tmp"), "paper1"
        )
        # pl02 should claim image 100 (page 7 within [6, 8]).
        assert 100 in claimed_ids
