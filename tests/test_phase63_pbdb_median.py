"""Tests for Phase 63 Plan 6 — Bug 6.21: PBDB age aggregation uses
mean instead of median.

Before: ``_pbdb_enrich_geology`` averaged ``max_ma`` / ``min_ma``
across occurrences. The mean of a bimodal biostratigraphic range
(e.g. occurrences spanning Cambrian + Carboniferous with a long gap)
sits between the modes and has no biostratigraphic meaning — a
reviewer reading the export sees a "Carboniferous-Cambrian average"
range that is meaningless.

After: aggregation uses ``statistics.median`` which picks the
middle value of the range. If the data is bimodal the median sits
on one of the modes and the export carries the actual centre of
the cluster of occurrences.

Statistics semantics aside, the median is also robust to outliers
(a single erroneous 999 Ma value won't drag the average up).
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.converters import _pbdb_enrich_geology  # noqa: E402
from rlpe.types import MatchResult  # noqa: E402


def _match(occs: list[dict]) -> MatchResult:
    return MatchResult(
        paper_id="p1",
        figure_id="fig1",
        panel_id="1",
        species="Genus species",
        panel_path=None,
        bbox=None,
        confidence=0.5,
        label_text=None,
        caption_snippet=None,
        ocr_text=None,
        metadata={
            "geology_links": [
                {
                    "biozone": None,
                    "formation": None,
                    "locality": None,
                    "country": None,
                    "latitude": None,
                    "longitude": None,
                    "ma_top": None,
                    "ma_base": None,
                    "evidence_text": "",
                }
            ],
            "paleodb": {"occurrences": occs},
        },
    )


def test_pbdb_age_aggregation_uses_median():
    """``_pbdb_enrich_geology`` must use ``statistics.median`` for
    ``ma_top`` / ``ma_base``, not arithmetic mean."""
    # Cluster: most occurrences around 250 Ma with one outlier at 100 Ma.
    # Median of [100, 248, 250, 252, 255] = 250 (mean = 221).
    occs = [
        {"max_ma": 100.0, "min_ma": 105.0},  # outlier
        {"max_ma": 248.0, "min_ma": 252.0},
        {"max_ma": 250.0, "min_ma": 254.0},
        {"max_ma": 252.0, "min_ma": 256.0},
        {"max_ma": 255.0, "min_ma": 258.0},
    ]
    matches = [_match(occs)]
    _pbdb_enrich_geology(matches)
    geo = matches[0].metadata["geology_links"][0]
    # Median is 250 for ma_top and 254 for ma_base.
    # If the code still used mean, we'd get 221.0 / 225.0.
    assert geo["ma_top"] == round(statistics.median([o["max_ma"] for o in occs]), 2), (
        f"ma_top = {geo.get('ma_top')}; expected median {statistics.median([o['max_ma'] for o in occs])}. "
        "Bug 6.21: PBDB age aggregation should use statistics.median, not sum/len."
    )
    assert geo["ma_base"] == round(statistics.median([o["min_ma"] for o in occs]), 2), (
        f"ma_base = {geo.get('ma_base')}; expected median {statistics.median([o['min_ma'] for o in occs])}. "
        "Bug 6.21: PBDB age aggregation should use statistics.median, not sum/len."
    )


def test_pbdb_median_handles_single_occurrence():
    """Single-occurrence median equals the single value."""
    occs = [{"max_ma": 251.9, "min_ma": 254.14}]
    matches = [_match(occs)]
    _pbdb_enrich_geology(matches)
    geo = matches[0].metadata["geology_links"][0]
    assert geo["ma_top"] == 251.9
    assert geo["ma_base"] == 254.14


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
