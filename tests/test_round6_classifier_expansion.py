"""Tests for Round-6 figure-type classifier expansion.

The previous classifier mis-categorized micro-CT and cross-section
captions as "plate", causing the classical CV path to over-segment
rendered volumes into hundreds of useless panel rows. The
expansion adds a micro-CT/XCT/tomographic keyword set and a
cross-section keyword set, both routed to "other" so the
pipeline can skip them.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from rlpe.range_chart_extractor import classify_figure_type


class TestClassifyFigureTypeRound6:
    """Lock the new micro-CT + cross-section classification."""

    @pytest.mark.parametrize(
        "caption,expected",
        [
            # Micro-CT / XCT — was plate, now other
            ("Fig. 1. Micro-CT images of Permian radiolarians", "other"),
            ("Fig. 1. XCT scan of a radiolarian", "other"),
            ("μ-CT cross-section of a radiolarian", "other"),
            ("Synchrotron tomographic image of radiolarian", "other"),
            ("X-ray computed tomography of fossil", "other"),
            # Cross-section / thin section — was plate, now other
            ("Fig. 1. Thin section of the specimen", "other"),
            ("Plane-polarized photomicrograph of a radiolarian", "other"),
            # Field photo — should remain photo
            ("Field photograph of the outcrop", "photo"),
            ("Outcrop photo of locality 5", "photo"),
            # Regular SEM plate — should remain plate
            ("Plate 1. SEM images of radiolarians from the Permian", "plate"),
            # Range chart — should remain range_chart
            ("Stratigraphic distribution of radiolarians", "range_chart"),
            # Map — should remain map
            ("Location map of the study area", "map"),
            # Xiao Micro-XCT paper — the real round-6 motivating case
            (
                "Fig. 1. Micro-CT images of Permian radiolarians",
                "other",
            ),
            # "Locality map of" 命中 map 关键字 ("of" 是关键)
            (
                "Fig. 2. Locality map of the study area",
                "map",
            ),
        ],
    )
    def test_classify(self, caption, expected):
        assert classify_figure_type(caption) == expected, (
            f"classify_figure_type({caption!r}) should be {expected!r}, "
            f"got {classify_figure_type(caption)!r}"
        )