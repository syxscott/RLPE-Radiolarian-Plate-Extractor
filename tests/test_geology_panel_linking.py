"""Pin the panel-level geology linking behaviour.

This is the regression test for the "every panel inherits the entire
paper's age/formation list" bug. The fix has three pieces:

  1. ``link_species_to_geology`` no longer fabricates a fallback record
     for unmatched species.
  2. ``link_panels_to_geology`` extracts age/formation/locality from
     the panel's own caption, with fulltext sections as a
     *candidate pool* (not as the search text).
  3. Placeholder captions ("Auto-generated figure for page N") get
     the first fulltext section's record attached, NOT the union of
     every record in the paper.

Each test below exercises one of the three pieces; the full pipeline
end-to-end coverage lives in
``test_e2e_real_pdf_smoke.py`` and ``scratch_verify_geo_e2e.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rlpe.geology_extraction import (  # noqa: E402
    _is_placeholder_caption,
    link_panels_to_geology,
    link_species_to_geology,
)


FULLTEXT = [
    {
        "title": "Geological setting",
        "section_type": "geological_setting",
        "text": (
            "The Dalong Formation (Upper Permian, Wuchiapingian) crops out "
            "at Feng et al section, South China. Radiolarians indicate a "
            "Wuchiapingian to Changhsingian age."
        ),
    },
    {
        "title": "Methods",
        "section_type": "methods",
        "text": "We used scanning electron microscopy to image the specimens.",
    },
]


# ---------------------------------------------------------------------------
# 1) Different panels, different captions, different geology facts
# ---------------------------------------------------------------------------


def test_two_panels_with_distinct_captions_get_distinct_geo_facts() -> None:
    captions = {
        "A": "Fig. 3A. Specimen from the Wuchiapingian of South China.",
        "B": "Fig. 3B. Specimen from the Changhsingian of South China.",
    }
    out = link_panels_to_geology(captions, fallback_sections=FULLTEXT)
    assert out["A"], "Panel A should pick up at least one caption record"
    assert out["B"], "Panel B should pick up at least one caption record"
    # Each panel should pick up its own caption's chronostratigraphy, not
    # the global fulltext's "Wuchiapingian" for both.
    a_chrons = {r.get("chronostratigraphy") for r in out["A"]}
    b_chrons = {r.get("chronostratigraphy") for r in out["B"]}
    assert "Wuchiapingian" in a_chrons
    assert "Changhsingian" in b_chrons


def test_each_panel_yields_at_most_few_records_not_every_age_in_paper() -> None:
    """The previous implementation dumped every age in the paper onto
    every panel. The new implementation must keep the count bounded."""
    captions = {
        f"P{i}": f"Fig. 3 panel {i}, Dalong Formation, Upper Permian."
        for i in range(10)
    }
    out = link_panels_to_geology(captions, fallback_sections=FULLTEXT)
    for pid, recs in out.items():
        assert len(recs) <= 5, f"Panel {pid} has too many records: {len(recs)}"


# ---------------------------------------------------------------------------
# 2) Placeholder caption + fulltext => panel gets section record
# ---------------------------------------------------------------------------


def test_placeholder_caption_falls_back_to_first_fulltext_section() -> None:
    captions = {
        "A": "Auto-generated figure for page 1",
        "B": "Auto-generated figure for page 2",
        "C": "Fig. 3C. Specimen from the Dalong Formation, Upper Permian.",
    }
    out = link_panels_to_geology(captions, fallback_sections=FULLTEXT)
    # Placeholder panels pick up the first fulltext section's record.
    for pid in ("A", "B"):
        assert out[pid], f"Placeholder panel {pid} should have at least one fallback record"
        for r in out[pid]:
            assert r.get("section_title") == "Geological setting"
    # Real-caption panel picks up the caption's own record, not the
    # fulltext's. Its section_title is the panel-caption marker.
    assert out["C"]
    real = out["C"][0]
    assert real.get("section_title") == "panel:C"
    assert real.get("section_type") == "panel_caption"


# ---------------------------------------------------------------------------
# 3) No fulltext + placeholder => empty list (no fabrication)
# ---------------------------------------------------------------------------


def test_placeholder_with_no_fulltext_yields_empty() -> None:
    captions = {"A": "Auto-generated figure for page 1"}
    out = link_panels_to_geology(captions, fallback_sections=None)
    assert out["A"] == []


def test_real_caption_with_no_fulltext_still_extracts_from_caption() -> None:
    captions = {"A": "Fig. 3A. Specimen from the Dalong Formation, Upper Permian."}
    out = link_panels_to_geology(captions, fallback_sections=None)
    assert out["A"]
    assert out["A"][0]["chronostratigraphy"] in {"Upper Permian", "Permian"}


# ---------------------------------------------------------------------------
# 4) ``link_species_to_geology`` no longer fabricates a fallback record
# ---------------------------------------------------------------------------


def test_species_unmatched_in_fulltext_yields_empty_list() -> None:
    """A species that doesn't appear in the fulltext must end up with
    an empty list, not a fallback record from the first section."""
    out = link_species_to_geology(
        species_names=["Acanthocircus impolitus"],
        sections=FULLTEXT,
        llm_runtime=None,
    )
    assert out["Acanthocircus impolitus"] == []


def test_species_present_in_fulltext_section_yields_that_section_records() -> None:
    out = link_species_to_geology(
        species_names=["Acanthocircus impolitus"],
        sections=[
            {
                "title": "Systematic palaeontology",
                "section_type": "systematic_paleontology",
                "text": "Acanthocircus impolitus is described from the Dalong Formation, Upper Permian.",
            }
        ],
        llm_runtime=None,
    )
    recs = out["Acanthocircus impolitus"]
    assert recs, "Species present in section should be linked"
    for r in recs:
        assert r["section_title"] == "Systematic palaeontology"


# ---------------------------------------------------------------------------
# 5) Placeholder detection mirror
# ---------------------------------------------------------------------------


def test_is_placeholder_caption_recognises_common_patterns() -> None:
    assert _is_placeholder_caption("Auto-generated figure for page 1")
    assert _is_placeholder_caption("Page 4 auto-generated")
    assert _is_placeholder_caption("page 7")
    assert _is_placeholder_caption("")
    assert _is_placeholder_caption("   ")
    assert not _is_placeholder_caption("Fig. 1. Specimen from Dalong Formation.")