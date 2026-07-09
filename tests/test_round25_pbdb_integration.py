"""Round 25 source-guard tests: PBDB integration + isotope proxies.

The user audit's follow-up after the 5-paper sampling asked for
"high-precision biostrat zonation (cross-dating)" and
"geochemistry & paleoenvironment proxies". Round 25 delivers:

  WS-R25-A: ``TaxonRecord.family`` / ``order`` / ``class_name``
             filled from PBDB taxonomy when ``use_paleodb=True``.
  WS-R25-B: PBDB occurrence aggregation fills missing geology
             fields (biozone, formation, locality, country,
             modern lat/lon) on each panel's first geology link.
  WS-R25-C: New ``_ISOTOPE_PATTERN`` captures δ¹³C / δ¹⁸O /
             ⁸⁷Sr/⁸⁶Sr / TOC / Hg numeric values and appends
             them to ``evidence_text`` for downstream analysis.

We don't add new schema fields for isotope values themselves —
the operator can grep ``evidence_text`` to find them. The four
proxies (paleoenvironment, redox, chemostrat, facies) added in
Round 24 already cover the categorical side; Round 25 adds the
numeric side.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _read(rel: str) -> str:
    return Path(
        "/home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor/" + rel
    ).read_text(encoding="utf-8")


# --- WS-R25-A: PBDB taxonomy propagation -------------------------------


def test_taxon_records_use_paleodb_taxonomy():
    """When ``m.metadata["paleodb"]["taxonomy"]`` is set, the
    converter must populate family / order / class_name on the
    resulting TaxonRecord. Without PBDB the fields stay None.
    """
    from rlpe.converters import taxon_records_from_matches
    from rlpe.types import MatchResult

    # No PBDB payload: family/order/class stay None.
    matches = [
        MatchResult(
            paper_id="p1", figure_id="f1", panel_id="1",
            species="Nassellaria sp.", panel_path=None, bbox=None,
            confidence=0.9, caption_snippet="caption",
            metadata={"extraction_method": "heuristic"},
        )
    ]
    records = taxon_records_from_matches(matches)
    assert records[0]["family"] is None
    assert records[0]["order"] is None
    assert records[0]["class_name"] is None

    # With PBDB payload: family/order/class are filled.
    matches[0].metadata["paleodb"] = {
        "taxonomy": {
            "name": "Nassellaria",
            "family": "Nassellariidae",
            "order": "Nassellariida",
            "class": "Polycystina",
            "rank": "order",
        }
    }
    records = taxon_records_from_matches(matches)
    assert records[0]["family"] == "Nassellariidae", (
        f"PBDB family not propagated; got {records[0]['family']!r}"
    )
    assert records[0]["order"] == "Nassellariida"
    assert records[0]["class_name"] == "Polycystina"


# --- WS-R25-B: PBDB occurrence fallback ---------------------------------


def test_pbdb_enrich_fills_missing_biozone():
    """When a panel's first geology link has no biozone but the
    species has PBDB occurrences, the most-common early_interval
    is used as a biozone proxy. The operator can verify the
    source in the evidence_text annotation."""
    from rlpe.converters import _pbdb_enrich_geology
    from rlpe.types import MatchResult

    matches = [
        MatchResult(
            paper_id="p1", figure_id="f1", panel_id="1",
            species="Nassellaria sp.", panel_path=None, bbox=None,
            confidence=0.9, caption_snippet="",
            metadata={
                "paleodb": {
                    "occurrences": [
                        {"early_interval": "Changhsingian", "max_ma": 254.0, "min_ma": 252.0},
                        {"early_interval": "Changhsingian", "max_ma": 254.0, "min_ma": 252.0},
                        {"early_interval": "Wuchiapingian", "max_ma": 256.0, "min_ma": 254.0},
                    ]
                },
                "geology_links": [
                    {"formation": None, "locality": None, "country": None,
                     "latitude": None, "longitude": None,
                     "ma_top": None, "ma_base": None}
                ],
            },
        )
    ]
    _pbdb_enrich_geology(matches)
    g = matches[0].metadata["geology_links"][0]
    # Most common early_interval is Changhsingian (2/3)
    assert g["biozone"] == "Changhsingian", (
        f"PBDB biozone fallback failed; got {g.get('biozone')!r}"
    )
    # The source annotation is added to evidence_text
    assert "PBDB" in g.get("evidence_text", ""), (
        f"Evidence text should annotate the source; got {g.get('evidence_text')!r}"
    )


def test_pbdb_enrich_does_not_overwrite_existing_data():
    """If a panel already has a regex-extracted biozone / formation
    / locality, the PBDB fallback must NOT overwrite it. The
    original extraction is always preferred."""
    from rlpe.converters import _pbdb_enrich_geology
    from rlpe.types import MatchResult

    matches = [
        MatchResult(
            paper_id="p1", figure_id="f1", panel_id="1",
            species="Nassellaria sp.", panel_path=None, bbox=None,
            confidence=0.9, caption_snippet="",
            metadata={
                "paleodb": {
                    "occurrences": [
                        {"early_interval": "Changhsingian", "max_ma": 254.0, "min_ma": 252.0},
                    ]
                },
                "geology_links": [
                    {"formation": "EXISTING Fm", "locality": "Existing Town",
                     "country": "Greece", "biozone": "Existing Bio",
                     "latitude": 1.0, "longitude": 2.0,
                     "ma_top": 100.0, "ma_base": 200.0}
                ],
            },
        )
    ]
    _pbdb_enrich_geology(matches)
    g = matches[0].metadata["geology_links"][0]
    # All existing data preserved
    assert g["formation"] == "EXISTING Fm"
    assert g["locality"] == "Existing Town"
    assert g["country"] == "Greece"
    assert g["biozone"] == "Existing Bio"
    assert g["latitude"] == 1.0
    assert g["ma_top"] == 100.0


def test_pbdb_enrich_no_op_without_paleodb():
    """Without ``paleodb.occurrences`` the enrichment is a no-op
    (matches without PBDB keep their existing fields)."""
    from rlpe.converters import _pbdb_enrich_geology
    from rlpe.types import MatchResult

    matches = [
        MatchResult(
            paper_id="p1", figure_id="f1", panel_id="1",
            species="Nassellaria sp.", panel_path=None, bbox=None,
            confidence=0.9, caption_snippet="",
            metadata={
                "geology_links": [
                    {"formation": "KEEP", "biozone": "KEEP",
                     "locality": "KEEP", "country": "KEEP",
                     "latitude": 1.0, "longitude": 2.0}
                ],
            },
        )
    ]
    _pbdb_enrich_geology(matches)
    g = matches[0].metadata["geology_links"][0]
    assert g["formation"] == "KEEP"
    assert g["biozone"] == "KEEP"


# --- WS-R25-C: isotope regex ---------------------------------------------


def test_isotope_pattern_captures_delta13c():
    """δ13C values like ``δ13C = -3.2 ‰`` must be captured by
    the isotope regex."""
    from rlpe.geology_extraction import _ISOTOPE_PATTERN

    samples = [
        "δ13C = -3.2 ‰",
        "δ13C: -3.2",
        "δ13C = +5.1 ‰",
        "δ18O = -2.1 ‰",
        "δ34S = +15.3 ‰",
        "δ13C = -2.5",
    ]
    for s in samples:
        m = _ISOTOPE_PATTERN.search(s)
        assert m, f"isotope regex should match {s!r}"


def test_isotope_pattern_captures_strontium_ratio():
    """87Sr/86Sr = 0.70712 must be captured."""
    from rlpe.geology_extraction import _ISOTOPE_PATTERN

    samples = [
        "87Sr/86Sr = 0.70712",
        "87Sr/86Sr: 0.707120",
        "87Sr/86Sr = 0.70689",
    ]
    for s in samples:
        m = _ISOTOPE_PATTERN.search(s)
        assert m, f"isotope regex should match {s!r}"


def test_isotope_pattern_captures_TOC_and_Hg():
    """TOC wt% and Hg ppb must be captured — these are the
    P/T boundary proxies the user audit specifically called out."""
    from rlpe.geology_extraction import _ISOTOPE_PATTERN

    samples = [
        "TOC = 4.5 wt%",
        "TOC: 12.0%",
        "Hg anomaly = 250 ppb",
        "Hg = 180 ppb",
    ]
    for s in samples:
        m = _ISOTOPE_PATTERN.search(s)
        assert m, f"isotope regex should match {s!r}"


def test_isotope_pattern_end_to_end():
    """End-to-end: feeding a section text with δ13C + TOC must
    capture both values in evidence_text via the extraction
    pipeline."""
    from rlpe.geology_extraction import extract_geology_from_sections

    sections = [
        {
            "title": "Geological setting",
            "section_type": "geological_setting",
            "text": (
                "The P/T boundary at Meishan shows a sharp δ13C = -3.2 ‰ "
                "excursion and TOC = 4.5 wt% in the anoxic/euxinic boundary clay. "
                "Pelagic basin facies."
            ),
        }
    ]
    records = extract_geology_from_sections(sections)
    assert records, "no records produced"
    r = records[0]
    # The evidence_text should now contain the isotope annotations
    assert "δ13C" in r.evidence_text or "TOC" in r.evidence_text, (
        f"Isotope values not captured in evidence_text: {r.evidence_text!r}"
    )
    # And the categorical proxies from Round 24 are still populated
    assert r.chemostrat, (
        f"chemostrat not populated; got {r.chemostrat!r}"
    )
    assert r.paleoenvironment, (
        f"paleoenvironment not populated; got {r.paleoenvironment!r}"
    )
    assert r.facies, (
        f"facies not populated; got {r.facies!r}"
    )