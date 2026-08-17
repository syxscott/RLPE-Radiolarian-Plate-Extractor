"""Phase 62 Plan 5 (Bug 5.7): per-species chart confidence.

Previously ``RangeChartResult.confidence`` was a single chart-axis
value, but ``build_geology_links_for_panels`` stamped that same
number onto EVERY species link it emitted. A species whose identity
the LLM was very confident about (e.g. "N. optima" — genus +
species unmistakable) but whose chart-position reading was
uncertain (e.g. it overlaps with two adjacent species in the
chart) ended up with the chart-wide confidence, polluting the
species-level link.

The fix: split the LLM-emitted confidence into two fields:

  * ``confidence`` (chart-wide) — overall certainty in the chart
    extraction (axis labels, sections, biozones). Unchanged.
  * ``species_confidence`` (per-species) — the model's certainty in
    the species-range row specifically.

The per-species field is read from the JSON response
(``species_ranges[i].confidence``) and defaults to the chart-wide
confidence when the model omits it (backward compatibility).
``build_geology_links_for_panels`` then uses the per-species value
on each link instead of the chart-wide value.

The test asserts:
  * ``SpeciesRange`` carries a ``confidence`` field (default 0.0).
  * ``extract_range_chart`` populates per-species confidence from
    the JSON when present.
  * When the JSON omits per-species confidence, the species_range
    inherits the chart-wide confidence (backward-compat).
  * ``build_geology_links_for_panels`` emits a link whose
    ``confidence`` field equals the per-species value (not the
    chart-wide value).
"""

from __future__ import annotations

from dataclasses import fields

from rlpe.range_chart_extractor import (
    RangeChartResult,
    SpeciesRange,
    _parse_extraction_response,
    build_geology_links_for_panels,
)


def _make_chart(
    *,
    species_confidence: float | None,
    chart_confidence: float = 0.5,
) -> RangeChartResult:
    """Build a RangeChartResult from a parsed-JSON-like dict using the
    internal _parse_extraction_response function."""
    parsed = {
        "sections": [
            {
                "name": "Pingdingshan",
                "age_range": "Late Permian",
                "formations": ["Talung Formation"],
            }
        ],
        "species_ranges": [
            {
                "species": "Neoalbaillella optima",
                "section": "Pingdingshan",
                "range_top": "Bed 9",
                "range_base": "Bed 7",
                "biozone": "N. optima Zone",
            },
            {
                "species": "Follicucullus charveti",
                "section": "Pingdingshan",
                "range_top": "Bed 10",
                "range_base": "Bed 8",
                "biozone": "",
            },
        ],
        "biozones": [],
        "other_fossils": [],
        "confidence": chart_confidence,
    }
    if species_confidence is not None:
        # Apply same per-species confidence to both for simplicity.
        for sp in parsed["species_ranges"]:
            sp["confidence"] = species_confidence
    return _parse_extraction_response(
        parsed=parsed,
        paper_id="test",
        figure_id="fig1",
    )


def test_species_range_has_confidence_field():
    """SpeciesRange must carry a 'confidence' field (defaults to 0.0)."""
    field_names = {f.name for f in fields(SpeciesRange)}
    assert "confidence" in field_names
    sr = SpeciesRange()
    assert sr.confidence == 0.0


def test_extract_range_chart_populates_per_species_confidence():
    """When the JSON response has per-species confidence, each
    SpeciesRange inherits it."""
    chart = _make_chart(species_confidence=0.92, chart_confidence=0.5)
    assert len(chart.species_ranges) == 2
    # Per-species confidence was 0.92 — both ranges should carry it.
    for sr in chart.species_ranges:
        assert sr.confidence == 0.92, f"expected per-species conf=0.92, got {sr.confidence}"
    # Chart-wide confidence is the separate field.
    assert chart.confidence == 0.5


def test_extract_range_chart_inherits_chart_confidence_when_missing():
    """Backward-compat: if the JSON omits per-species confidence,
    each species_range falls back to the chart-wide confidence."""
    chart = _make_chart(species_confidence=None, chart_confidence=0.7)
    assert chart.confidence == 0.7
    for sr in chart.species_ranges:
        assert sr.confidence == 0.7, f"expected fallback to chart conf=0.7, got {sr.confidence}"


def test_build_geology_links_uses_per_species_confidence():
    """The link emitted for a panel must carry the per-species
    confidence, NOT the chart-wide confidence."""
    chart = _make_chart(species_confidence=0.92, chart_confidence=0.5)
    panel_records = [
        {"species": "Neoalbaillella optima", "panel_id": "p1"},
    ]
    links = build_geology_links_for_panels(chart, panel_records)
    assert len(links) >= 1
    # The link for "Neoalbaillella optima" must carry the
    # per-species conf=0.92, not the chart conf=0.5.
    for link in links:
        if link.get("species") == "Neoalbaillella optima":
            assert link["confidence"] == 0.92, (
                f"link should carry per-species confidence 0.92, got {link['confidence']}"
            )


def test_build_geology_links_distinct_per_species_confidences():
    """Different species in the same chart can carry different
    per-species confidences; the link must reflect that."""
    # Hand-build a chart with two distinct per-species confidences.
    parsed = {
        "sections": [
            {"name": "Sec1", "age_range": "Triassic", "formations": []},
        ],
        "species_ranges": [
            {
                "species": "Species A",
                "section": "Sec1",
                "range_top": "Bed 1",
                "range_base": "Bed 2",
                "biozone": "",
                "confidence": 0.95,
            },
            {
                "species": "Species B",
                "section": "Sec1",
                "range_top": "Bed 3",
                "range_base": "Bed 4",
                "biozone": "",
                "confidence": 0.30,
            },
        ],
        "biozones": [],
        "other_fossils": [],
        "confidence": 0.6,
    }
    chart = _parse_extraction_response(parsed=parsed, paper_id="t", figure_id="f")
    panel_records = [
        {"species": "Species A", "panel_id": "p1"},
        {"species": "Species B", "panel_id": "p2"},
    ]
    links = build_geology_links_for_panels(chart, panel_records)
    a_link = [l for l in links if l.get("species") == "Species A"]
    b_link = [l for l in links if l.get("species") == "Species B"]
    assert a_link and b_link
    assert a_link[0]["confidence"] == 0.95
    assert b_link[0]["confidence"] == 0.30
