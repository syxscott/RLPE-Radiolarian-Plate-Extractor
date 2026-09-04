"""Tests for Phase 3E audit 2026-08-19 — Bug M-8 / M-10 / M-11.

Phase 3E ships three biozone-related fixes:

* M-8:  ``stratigraphy._BIOZONE_TO_MA`` now contains Cenozoic
  Sanfilippo & Nigrini 1998 RP1-RP21 zones + Riedel & Sanfilippo
  1978 RN1-RN17 zones. Before this fix, every RP/RN citation
  resolved to ``None`` and was reported as ``biozone_unknown``,
  silently dropping the biozone tag on every Cenozoic paper.
* M-10: ``BiozoneRecord.zone_type`` field now distinguishes taxon-
  range / concurrent-range / interval / assemblage zones (was a
  free-text-only name before; downstream consumers could not tell
  how the zone should be plotted on the Ma axis).
* M-11: ``SpeciesRange.range_top_ma`` and ``range_base_ma`` carry
  the numeric Ma values from the chart's Ma axis (were free-text
  bed/level labels like "Bed 9" before — unreadable for FAD/LAD
  biostratigraphy without a per-section legend).

These tests exercise all three together with the existing helpers
and the M3 prompt JSON → dataclass converter
(``_parse_extraction_response``).
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.range_chart_extractor import (  # noqa: E402
    BiozoneRecord,
    RangeChartResult,
    SpeciesRange,
    _parse_extraction_response,
)
from rlpe.stratigraphy import (  # noqa: E402
    _BIOZONE_RE,
    _BIOZONE_TO_MA,
    find_ages_in_text,
    lookup_biozone_ma,
)

# ---------------------------------------------------------------------------
# Task 1 / M-8: Cenozoic RP + RN biozones
# ---------------------------------------------------------------------------


def test_rp_zone_lookup():
    """Every RP1-RP21 entry must resolve via lookup_biozone_ma with
    the documented (ma_top, ma_base) pair."""
    cases = [
        # (name, ma_top, ma_base)
        ("RP1", 30.0, 34.0),
        ("RP2", 24.0, 30.0),
        ("RP6", 17.0, 18.5),  # Burdigalian
        ("RP11", 8.5, 9.5),  # Tortonian
        ("RP17", 2.5, 3.5),  # Piacenzian
        ("RP21", 0.0, 0.5),  # Holocene
    ]
    for name, exp_top, exp_base in cases:
        out = lookup_biozone_ma(name)
        assert out is not None, f"{name!r} not in biozone table"
        ma_top, ma_base = out.top_ma, out.base_ma
        assert ma_top == exp_top, f"{name!r}: ma_top={ma_top} != {exp_top}"
        assert ma_base == exp_base, f"{name!r}: ma_base={ma_base} != {exp_base}"


def test_rp_zone_with_biozone_suffix():
    """``RP6 Biozone`` and ``RP6`` should both resolve identically
    (papers write both interchangeably)."""
    a = lookup_biozone_ma("RP6")
    b = lookup_biozone_ma("RP6 Biozone")
    assert a is not None
    assert b is not None
    assert a == b


def test_rp_zone_with_space():
    """``RP 6`` (with space between prefix and digit) must resolve
    to the same bounds as ``RP6``."""
    a = lookup_biozone_ma("RP6")
    b = lookup_biozone_ma("RP 6")
    assert a is not None
    assert b is not None
    assert a == b, f"RP6={a} vs RP 6={b}"


def test_rp21_zone():
    """``RP 21`` (with space) must resolve to RP21."""
    out = lookup_biozone_ma("RP 21")
    assert out is not None
    ma_top, ma_base = out.top_ma, out.base_ma
    assert ma_top == 0.0
    assert ma_base == 0.5


def test_rp6_rp7_range():
    """``RP6-RP7`` (without "Biozone" suffix) must resolve to the
    union of both zones' Ma bounds (youngest top, oldest base)."""
    out = lookup_biozone_ma("RP6-RP7")
    assert out is not None
    ma_top, ma_base = out.top_ma, out.base_ma
    # RP6 = (17.0, 18.5); RP7 = (14.5, 17.0) → union (14.5, 18.5)
    assert ma_top == 14.5
    assert ma_base == 18.5


def test_rn_zone_lookup():
    """Every RN1-RN17 entry must resolve via lookup_biozone_ma."""
    cases = [
        ("RN1", 0.0, 1.8),
        ("RN4", 9.0, 15.0),  # commonly cited Tortonian zone
        ("RN6", 22.0, 30.0),  # commonly cited Oligocene zone
        ("RN9", 50.0, 56.0),  # Thanetian
        ("RN17", 118.0, 127.0),  # Aptian
    ]
    for name, exp_top, exp_base in cases:
        out = lookup_biozone_ma(name)
        assert out is not None, f"{name!r} not in biozone table"
        ma_top, ma_base = out.top_ma, out.base_ma
        assert ma_top == exp_top, f"{name!r}: ma_top={ma_top} != {exp_top}"
        assert ma_base == exp_base, f"{name!r}: ma_base={ma_base} != {exp_base}"


def test_rn_unknown_subrange_does_not_throw():
    """``RN5-5`` (a syntactically valid range expression with no
    upper end) must not throw — the underlying range loop simply
    iterates once and returns the RN5 bounds. Regression guard for
    an audit fix that previously asserted "regex should not throw"."""
    out = lookup_biozone_ma("RN5-5")
    # Acceptable: either a successful single-zone lookup
    # (RN5-5 → RN5 = (15.0, 22.0)) OR a defensive ``None`` from the
    # helper. We require NO exception; we do NOT require a value.
    # The lookup now returns a BiozoneMa NamedTuple carrying
    # (top_ma, base_ma, confidence); compare only the Ma fields.
    assert out is None or (out.top_ma, out.base_ma) == (15.0, 22.0)


def test_biozone_re_matches_rp_rn_forms():
    """_BIOZONE_RE must accept every variant in real Cenozoic papers."""
    cases = [
        ("RP6 Biozone", True),
        ("RP 6", True),
        ("RP6", True),
        ("RP6-RP7", True),
        ("RP6-RP7 Biozone", True),
        ("RN4", True),
        ("RN5-RN5", True),  # degenerate but legal
        ("UAZ 4-7", True),
        ("Buryella clinata Zone", False),  # not a numbered form
        ("Late Permian", False),
    ]
    for text, expected in cases:
        m = _BIOZONE_RE.search(text)
        assert bool(m) == expected, f"_BIOZONE_RE.search({text!r}) = {m}"


def test_find_ages_in_text_finds_rp_zone():
    """find_ages_in_text must recognise RP6 / RP 6 / RP6 Biozone
    mentions in text and attach Ma bounds."""
    text = "Sample taken from RP6 Biozone in the early Miocene."
    ags = find_ages_in_text(text)
    biozone_ags = [a for a in ags if a.rank == "biozone"]
    assert len(biozone_ags) >= 1, f"no biozone hits in {text!r}"
    rp6 = next((a for a in biozone_ags if "RP6" in a.age.upper()), None)
    assert rp6 is not None
    assert rp6.ma_top == 17.0
    assert rp6.ma_base == 18.5
    assert rp6.confidence > 0.0


def test_find_ages_in_text_finds_rn_zone():
    """find_ages_in_text must recognise RN mentions in text."""
    text = "Correlative with RN4 in the tropical Pacific."
    ags = find_ages_in_text(text)
    biozone_ags = [a for a in ags if a.rank == "biozone"]
    assert len(biozone_ags) >= 1
    rn4 = next((a for a in biozone_ags if "RN4" in a.age.upper()), None)
    assert rn4 is not None
    assert rn4.ma_top == 9.0
    assert rn4.ma_base == 15.0


def test_rp_rn_table_size():
    """The RP + RN additions must have meaningfully grown the table.
    A non-regression guard: if the table shrinks again (an audit
    sweep inadvertently drops RP/RN entries), this fails loudly."""
    rp_count = sum(1 for k in _BIOZONE_TO_MA if k.startswith("RP") and "Biozone" not in k)
    rn_count = sum(1 for k in _BIOZONE_TO_MA if k.startswith("RN") and "Biozone" not in k)
    assert rp_count >= 21, f"only {rp_count} RP entries (expected ≥21)"
    assert rn_count >= 17, f"only {rn_count} RN entries (expected ≥17)"


# ---------------------------------------------------------------------------
# Task 2 / M-10: BiozoneRecord.zone_type
# ---------------------------------------------------------------------------


def test_biozone_record_zone_type_default_none():
    """A vanilla BiozoneRecord() must default zone_type to None (no
    backward-incompatibility for callers that never set it)."""
    r = BiozoneRecord()
    assert r.zone_type is None


def test_biozone_record_zone_type_set():
    """Setting zone_type explicitly must round-trip through to_dict
    without alteration — downstream consumers (xlsx export, Web UI
    biozone-type filter) read it from the JSON dump."""
    r = BiozoneRecord(name="N. optima Range Zone", zone_type="range")
    assert r.zone_type == "range"
    d = r.to_dict()
    assert d["zone_type"] == "range"
    assert d["name"] == "N. optima Range Zone"


def test_biozone_record_zone_type_accepted_values():
    """All four ICS / Salvador zone-type labels must be accepted."""
    for zt in ("range", "concurrent range", "interval", "assemblage"):
        r = BiozoneRecord(name="X Zone", zone_type=zt)
        assert r.zone_type == zt


def test_parse_extraction_response_zone_type():
    """The M3 JSON → BiozoneRecord parser must populate zone_type
    when the JSON includes it."""
    parsed = {
        "biozones": [
            {"name": "Zone A", "age": "Miocene", "zone_type": "range"},
            {"name": "Zone B", "age": "Pliocene", "zone_type": "concurrent range"},
            {"name": "Zone C", "age": "Eocene"},  # omitted zone_type → None
        ]
    }
    result = _parse_extraction_response(parsed=parsed, paper_id="p", figure_id="f")
    assert len(result.biozones) == 3
    assert result.biozones[0].zone_type == "range"
    assert result.biozones[1].zone_type == "concurrent range"
    assert result.biozones[2].zone_type is None


# ---------------------------------------------------------------------------
# Task 3 / M-11: SpeciesRange.range_top_ma / range_base_ma
# ---------------------------------------------------------------------------


def test_species_range_ma_fields_default_none():
    """SpeciesRange() with no Ma args must default both to None."""
    sr = SpeciesRange()
    assert sr.range_top_ma is None
    assert sr.range_base_ma is None


def test_species_range_ma_fields_round_trip():
    """Setting range_top_ma / range_base_ma must round-trip via
    to_dict — the export pipeline writes the JSON dump verbatim."""
    sr = SpeciesRange(
        species="N. optima",
        range_top="Bed 9",
        range_base="Bed 7",
        range_top_ma=251.9,
        range_base_ma=252.5,
    )
    d = sr.to_dict()
    assert d["range_top_ma"] == 251.9
    assert d["range_base_ma"] == 252.5
    assert d["range_top"] == "Bed 9"  # free-text label still preserved


def test_parse_extraction_response_ma_fields():
    """The parser must populate range_top_ma / range_base_ma when
    the JSON includes numeric Ma values."""
    parsed = {
        "species_ranges": [
            {
                "species": "Neoalbaillella optima",
                "section": "Pingdingshan",
                "range_top": "Bed 9",
                "range_base": "Bed 7",
                "range_top_ma": 251.9,
                "range_base_ma": 252.5,
                "biozone": "N. optima Zone",
                "confidence": 0.9,
            },
            {
                # Omitted Ma → None (allowed when chart has no axis)
                "species": "Follicucullus charveti",
                "section": "Pingdingshan",
                "range_top": "Bed 5",
                "range_base": "Bed 3",
                "biozone": "",
                "confidence": 0.8,
            },
        ]
    }
    result = _parse_extraction_response(parsed=parsed, paper_id="p", figure_id="f")
    assert len(result.species_ranges) == 2
    a = result.species_ranges[0]
    assert a.range_top_ma == 251.9
    assert a.range_base_ma == 252.5
    b = result.species_ranges[1]
    assert b.range_top_ma is None
    assert b.range_base_ma is None


def test_parse_extraction_response_ma_fields_non_finite_become_none():
    """NaN / Inf / unparseable Ma values must NOT propagate into the
    dataclass — they would corrupt downstream plot colour-bin
    arithmetic. M21-style guard."""
    parsed = {
        "species_ranges": [
            {
                "species": "A",
                "section": "S",
                "range_top": "Bed 1",
                "range_base": "Bed 1",
                "range_top_ma": float("nan"),
                "range_base_ma": float("inf"),
                "confidence": 0.5,
            },
            {
                "species": "B",
                "section": "S",
                "range_top": "Bed 2",
                "range_base": "Bed 2",
                "range_top_ma": "not-a-number",
                "range_base_ma": None,
                "confidence": 0.5,
            },
        ]
    }
    result = _parse_extraction_response(parsed=parsed, paper_id="p", figure_id="f")
    a = result.species_ranges[0]
    assert a.range_top_ma is None
    assert a.range_base_ma is None
    b = result.species_ranges[1]
    assert b.range_top_ma is None
    assert b.range_base_ma is None


def test_range_chart_result_to_dict_with_new_fields():
    """A RangeChartResult carrying the new fields must serialise
    them in to_dict() — guarantees the xlsx exporter sees them."""
    r = RangeChartResult(
        paper_id="p",
        figure_id="f",
        biozones=[
            BiozoneRecord(name="N. optima Zone", zone_type="range"),
        ],
        species_ranges=[
            SpeciesRange(
                species="N. optima",
                range_top="Bed 9",
                range_base="Bed 7",
                range_top_ma=251.9,
                range_base_ma=252.5,
            ),
        ],
    )
    d = r.to_dict()
    assert d["biozones"][0]["zone_type"] == "range"
    assert d["species_ranges"][0]["range_top_ma"] == 251.9
    assert d["species_ranges"][0]["range_base_ma"] == 252.5


# ---------------------------------------------------------------------------
# Sanity: existing fields still work (non-regression).
# ---------------------------------------------------------------------------


def test_existing_biozone_lookup_still_works():
    """UAZ 5 + Buryella clinata Zone must still resolve — the RP/RN
    additions must not have displaced or corrupted the existing
    entries (Bug 3.11 / Phase 60)."""
    assert lookup_biozone_ma("UAZ 5") is not None
    ma = lookup_biozone_ma("Buryella clinata Zone")
    assert (ma.top_ma, ma.base_ma) == (56.0, 59.0)
    ma = lookup_biozone_ma("Cryptocephalus nigricae Zone")
    assert (ma.top_ma, ma.base_ma) == (83.6, 86.3)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
