"""BUG-2 of the 2026-09-04 zero-rows diagnosis (Zhang 2014 GUI run).

``_build_figures_from_plate_captions`` processed every caption (both
"Explanation of Plate N" plate-kind and "Fig. N" fig-kind) in a single
document-order pass, and each caption claimed **every** unclaimed image
in its forward window. In Zhang 2014 (JMicro) the "Explanation of Plate
1" caption sits on p6 — the same page as the Fig. 4 chart — and its
window [6, 7] grabbed both the p6 chart and the p7 real plate, leaving
Fig. 4's own caption with ``no_images=True`` and mis-anchoring Plate 1's
figure_id to the chart page.

Fix under test: two-pass assignment — fig-kind captions claim their
same-page images first (precise anchor), then plate-kind captions take
the remaining unclaimed images. Plate-only papers (the vast majority of
the corpus) are untouched: with no fig-kind captions the second pass is
byte-for-byte the old behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe.opendataloader_extractor import _build_figures_from_plate_captions


def _cap(num: int, page: int, content: str, kind: str) -> dict:
    return {
        "plate_number": num,
        "page_number": page,
        "content": content,
        "element": None,
        "kind": kind,
    }


def _img(img_id: int, page: int) -> dict:
    return {
        "id": img_id,
        "page number": page,
        "bounding box": [0, 0, 100, 100],
    }


def _pair_by_plate(pairs, plate_number: int):
    for p in pairs:
        if p.metadata.get("plate_number") == plate_number:
            return p
    pytest.fail(f"no pair with plate_number={plate_number}")


# ----------------------------------------------------------------------
# The Zhang 2014 scenario
# ----------------------------------------------------------------------
class TestZhang2014Scenario:
    def test_fig4_chart_not_stolen_by_plate1(self, tmp_path):
        """Fig. 4 caption on p6 must claim the p6 chart even though the
        Explanation of Plate 1 caption is on the same page."""
        caps = [
            _cap(1, 2, "Fig. 1 Phylogenetic consensus tree of Follicucullus.", "fig"),
            _cap(
                1,
                6,
                "Explanation of Plate 1. figures 1-26. Guadalupian radiolarians "
                "from the Daoduishan: 1, 2, Follicucullus sp. A; 3, F. sp. B.",
                "plate",
            ),
            _cap(4, 6, "Fig. 4 Chart of species occurrences through the section.", "fig"),
            _cap(
                2,
                8,
                "Explanation of Plate 2. figures 1-21. Guadalupian radiolarians "
                "from the Daoduishan: 1, Follicucullus scholasticus.",
                "plate",
            ),
        ]
        images = [
            _img(1, 6),  # Fig. 4 chart
            _img(2, 7),  # real Plate 1 grid
            _img(3, 9),  # real Plate 2 grid
        ]
        pairs, claimed = _build_figures_from_plate_captions(
            caps, images, tmp_path, "zhang2014", caption_window=5
        )

        fig4 = _pair_by_plate(pairs, 4)
        assert fig4.image_paths or fig4.metadata.get("no_images") is not True, (
            "Fig. 4 must claim its same-page chart"
        )
        assert "no_images" not in fig4.metadata
        assert fig4.page_number == 6

    def test_plate1_anchored_on_real_plate_page(self, tmp_path):
        """Plate 1's figure_id/page must anchor on the real plate image
        (p7), not on the p6 chart it used to steal."""
        caps = [
            _cap(1, 6, "Explanation of Plate 1. figures 1-26. Taxa: 1, 2, F. sp.", "plate"),
            _cap(4, 6, "Fig. 4 Chart of species occurrences.", "fig"),
            _cap(2, 8, "Explanation of Plate 2. figures 1-21. Taxa: 1, F. scholasticus.", "plate"),
        ]
        images = [_img(1, 6), _img(2, 7), _img(3, 9)]
        pairs, _ = _build_figures_from_plate_captions(
            caps, images, tmp_path, "zhang2014", caption_window=5
        )
        plate1 = _pair_by_plate(pairs, 1)
        assert plate1.page_number == 7
        assert plate1.figure_id == "od_plate_zhang2014_p007_pl01"

    def test_every_image_claimed_exactly_once(self, tmp_path):
        caps = [
            _cap(1, 6, "Explanation of Plate 1. figures 1-26. Taxa.", "plate"),
            _cap(4, 6, "Fig. 4 Chart of species occurrences.", "fig"),
            _cap(2, 8, "Explanation of Plate 2. figures 1-21. Taxa.", "plate"),
        ]
        images = [_img(1, 6), _img(2, 7), _img(3, 9)]
        _, claimed = _build_figures_from_plate_captions(
            caps, images, tmp_path, "zhang2014", caption_window=5
        )
        assert claimed == {1, 2, 3}


# ----------------------------------------------------------------------
# Precision / regression guards
# ----------------------------------------------------------------------
class TestRegressionGuards:
    def test_plate_only_unchanged(self, tmp_path):
        """Plate-only caption lists (no fig-kind) keep the old single-
        pass window behaviour exactly (round28 contract)."""
        caps = [
            _cap(1, 1, "Plate 1.", "plate"),
            _cap(2, 4, "Plate 2.", "plate"),
        ]
        images = [_img(1, 3), _img(2, 5)]
        pairs, claimed = _build_figures_from_plate_captions(
            caps, images, tmp_path, "p", caption_window=5
        )
        p1 = _pair_by_plate(pairs, 1)
        p2 = _pair_by_plate(pairs, 2)
        assert p1.page_number == 3
        assert p2.page_number == 5
        assert claimed == {1, 2}

    def test_legacy_dicts_without_kind_key_unchanged(self, tmp_path):
        """Older callers/fixtures that omit ``kind`` fall into the plate
        pass — identical to the pre-fix behaviour."""
        caps = [
            {"plate_number": 1, "page_number": 6, "content": "Plate 1.", "element": None},
            {"plate_number": 4, "page_number": 6, "content": "Fig. 4 chart.", "element": None},
        ]
        images = [_img(1, 6), _img(2, 7)]
        pairs, claimed = _build_figures_from_plate_captions(
            caps, images, tmp_path, "p", caption_window=5
        )
        p1 = _pair_by_plate(pairs, 1)
        assert p1.page_number == 6  # claims both images, as before
        assert claimed == {1, 2}

    def test_fig_same_page_claim_does_not_cross_page(self, tmp_path):
        """A fig caption with a same-page image claims ONLY that image;
        the next page's plate grid stays available for the plate."""
        caps = [
            _cap(3, 5, "Fig. 3 Location map.", "fig"),
            _cap(1, 6, "Plate 1.", "plate"),
        ]
        images = [_img(1, 5), _img(2, 6)]
        pairs, claimed = _build_figures_from_plate_captions(
            caps, images, tmp_path, "p", caption_window=5
        )
        fig3 = _pair_by_plate(pairs, 3)
        assert fig3.page_number == 5
        plate1 = _pair_by_plate(pairs, 1)
        assert plate1.page_number == 6
        assert claimed == {1, 2}

    def test_fig_falls_back_to_window_when_caption_page_empty(self, tmp_path):
        """When the fig caption's own page has no image, the standard
        forward window still applies (image on the next page)."""
        caps = [_cap(5, 10, "Fig. 5 Range chart.", "fig")]
        images = [_img(1, 11)]
        pairs, claimed = _build_figures_from_plate_captions(
            caps, images, tmp_path, "p", caption_window=5
        )
        fig5 = _pair_by_plate(pairs, 5)
        assert "no_images" not in fig5.metadata
        assert fig5.page_number == 11
        assert claimed == {1}
