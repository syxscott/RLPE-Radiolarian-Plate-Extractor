"""Round 20 source-guard tests: extend figure classifier for strat/map/litholog.

User audit: 4 OA papers (Bandini 2006 / Boughdiri 2007 / Danelian 2006 /
Bragin 2025) were sampled. Of the 12 strat_column / paleogeographic_map /
litholog_column figures across these papers, only 3 reached the
geo_vision M3 prompt. The classifier returned ``other`` for 5 captions
and ``map`` for 1 (the latter being correct but not wired into the
geo_vision pipeline).

These tests pin the regression: each sampled caption must classify to
its expected type. They also guard the existing ``plate`` /
``range_chart`` returns so the keyword expansion doesn't over-match.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# Real captions sampled from the 4 OA papers
SAMPLE_CAPTIONS: list[tuple[str, str]] = [
    # Bandini 2006 — Fig. 2 "Geological Map" + Fig. 3 "Stratigraphic log"
    ("Fig. 2. Geological Map of the area east of Karnezeika (modified after Vernez 1990)", "map"),
    ("Fig. 3. Stratigraphic log of the Karnezeika section.", "strat_column"),
    # Boughdiri 2007 — Fig. 2 stratigraphic overview, Fig. 3/4 litholog,
    # Fig. 5 outcrop photos (photo), Fig. 1 location map
    ("Fig. 1. Location of studied sections (Jurassic, Tunisia). a) northern Tunisia.", "map"),
    (
        "Fig. 2. Overview of Tunisian Jurassic stratigraphy, comprising Tunisian Dorsale and North Atlasic Tunisia.",
        "strat_column",
    ),
    (
        "Fig. 3. Lithological sections from Jebels Jédidi and Chaâbane (Jurassic, Tunisia)",
        "litholog_column",
    ),
    ("Fig. 4. Lithological section from Oued Tazega", "litholog_column"),
    ("Fig. 5. Exposures from Oued Tazega section. a) General view", "photo"),
    # Danelian 2006 — Fig. 2 composite lithostrat column
    ("Fig. 2. A) Composite lithostratigraphic column of the Vocontian basin", "strat_column"),
    # Bragin 2025 — Fig. 2 localities, Fig. 3 zones
    (
        "Fig. 2. The most important localities of Oxfordian-Valanginian Boreal radiolarians in Russia",
        "paleogeographic_map",
    ),
    (
        "Fig. 3. Zones and beds with radiolarians, as well as Kimmeridgian-Valanginian radiolarian assemblages of the Timan-Pechora",
        "strat_column",
    ),
    # Plate caption must still return plate
    ("Plate 1. SEM-illustrations of Upper Cretaceous radiolarians from Karnezeika", "plate"),
    (
        "Plate I. Jurassic radiolarians from the Jédidi Fm (Tunisia). Scanning Electron Microscope.",
        "plate",
    ),
]


def test_classifier_handles_all_sampled_captions():
    """Each real caption from the 4-paper sample must classify correctly."""
    from rlpe.range_chart_extractor import classify_figure_type

    misses = []
    for caption, expected in SAMPLE_CAPTIONS:
        got = classify_figure_type(caption, None)
        if got != expected:
            misses.append((caption[:60], expected, got))

    assert not misses, "Classifier missed these captions:\n" + "\n".join(
        f"  cap={c!r} expected={e} got={g}" for c, e, g in misses
    )


def test_plate_keyword_still_works():
    """Regression: a clean 'Plate' caption must classify as 'plate'.
    NOTE: any caption with both 'plate' AND a range_chart keyword
    (e.g. 'distribution') is intentionally re-classified as
    'range_chart' — that's the existing override behavior at line
    197 of range_chart_extractor.py. This test only covers the
    common case where no range_chart keyword is present."""
    from rlpe.range_chart_extractor import classify_figure_type

    caption = "Plate 1. SEM-illustrations of selected species."
    assert classify_figure_type(caption, None) == "plate"


def test_plate_with_range_keyword_is_reclassified():
    """Documented behavior: 'Plate' caption with range_chart keyword
    (e.g. 'distribution') is correctly overridden to 'range_chart'."""
    from rlpe.range_chart_extractor import classify_figure_type

    caption = "Plate 4. Stratigraphic distribution of selected radiolarian species."
    assert classify_figure_type(caption, None) == "range_chart"


def test_range_chart_keywords_preserved():
    """Regression: range_chart keywords must still match."""
    from rlpe.range_chart_extractor import classify_figure_type

    assert classify_figure_type("Fig. 4. Range chart of selected species", None) == "range_chart"
    assert classify_figure_type("Fig. 2. Biozone distribution of ammonoids", None) == "range_chart"


def test_paleogeographic_takes_priority_over_map():
    """'Paleogeographic Map of...' must NOT collapse to plain 'map'."""
    from rlpe.range_chart_extractor import classify_figure_type

    cap = "Fig. 1. Paleogeographic map of the western Tethys in the Late Jurassic"
    assert classify_figure_type(cap, None) == "paleogeographic_map"


def test_keyword_list_extensions():
    """Source guard: the keyword dicts must contain the Round 20 additions."""
    src = Path(
        Path(__file__).resolve().parents[1] / "src" / "rlpe" / "range_chart_extractor.py"
    ).read_text(encoding="utf-8")

    # The 6 new phrases that were missing before Round 20
    required_new = [
        '"composite lithostrat"',
        '"zones and beds"',
        '"lithological section"',
        '"locality map"',
        '"important localities"',
        '"overview of"',
    ]
    missing = [k for k in required_new if k not in src]
    assert not missing, (
        "Round 20 keyword additions not present in range_chart_extractor.py: " + ", ".join(missing)
    )


def test_pipeline_routes_map_to_geo_vision():
    """Source guard: pipeline.py:1110 must include 'map' in the
    geo_vision figure_type set. Without it, plain map captions route
    to plate segmentation and produce no geology records."""
    src = Path(
        Path(__file__).resolve().parents[1] / "src" / "rlpe" / "pipeline.py"
    ).read_text(encoding="utf-8")
    # Look for the geo_vision routing tuple
    assert '"map"' in src, (
        "pipeline.py geo_vision routing does not include 'map' as a "
        "recognized figure type. Without this, captions like 'Geological "
        "Map of...' fall through to plate segmentation."
    )
    # And it must appear inside the geo_vision tuple
    idx = src.find('"strat_column"')
    if idx > 0:
        # Look at the next 200 chars for "map"
        window = src[idx : idx + 300]
        assert '"map"' in window, (
            "'map' is not inside the geo_vision routing tuple. "
            "Round 20 sampling showed Geological Maps lost their data."
        )
