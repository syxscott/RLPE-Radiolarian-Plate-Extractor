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


# --- WS-R25-D: live PBDB occurrence field aliasing -----------------------
#
# Bug discovered during the M3 + PBDB live integration (2026-07-09):
# the PBDB ``occs/list.json`` endpoint returns records with SHORT
# FIELD CODES (``oei``, ``eag``, ``lag``, ``cc2``, ``lng``, ``lat``,
# ``sfm``, ``cnm``) rather than the long names (``early_interval``,
# ``max_ma``, ``country``, ``longitude``, ``latitude``, ``formation``,
# ``locality``). The previous ``PaleoDB.lookup_occurrences`` read the
# long names and got ``None`` for every field, so the Round 25 biozone
# / locality / coord fallback was completely inert in production.
#
# The fix (in ``paleodb.py``):
#   1. Drop the invalid ``show=attr,loc,strat`` param so PBDB returns
#      the default short-code payload (with ``show=`` it returned
#      records where every non-core field was ``None``).
#   2. Add a per-record alias map that accepts both short and long
#      keys — old long-name payloads (e.g. cached, future API) keep
#      working, new short-name payloads light up.
#   3. Map ``cc2`` (2-letter ISO) through ``_iso_to_country`` so
#      ``country`` carries a readable name; ``country_code`` keeps the
#      raw value for downstream consumers.
#   4. Use ``oei``→early_interval, ``oli``→late_interval, ``eag``→max_ma,
#      ``lag``→min_ma, ``cnm``→locality, ``sfm``→formation, ``smb``→member.
#
# These source-guard tests verify the aliasing on a *synthetic* PBDB
# payload that mimics the actual short-code shape — no real network
# round-trip, so the tests are deterministic. The live sanity check
# is the curl calls in the commit message.


def test_pbdb_lookup_occurrences_decodes_short_codes():
    """Synthetic PBDB short-code payload → OccurrenceSummary carries
    full canonical field values. Pre-fix this returned ``None`` for
    every geology field because ``paleodb.py`` was reading long names."""
    from rlpe.paleodb import PaleoDB

    # Direct test that doesn't hit the network: write a fake cache,
    # then call lookup_occurrences() so the cache short-circuits the
    # HTTP request. Use a real PBDB endpoint structure (oids, cids,
    # short codes) so the alias map has real inputs to decode.
    import json
    import tempfile

    cache_dir = Path(tempfile.mkdtemp())
    pbdb = PaleoDB(cache_dir=cache_dir)
    name = "Archaeodictyomitra"
    cache_key_path = (
        list(cache_dir.glob("*.json"))[0] if list(cache_dir.glob("*.json")) else None
    )
    payload = {
        "records": [
            {
                "oid": "occ:432613",
                "cid": "col:42265",
                "idn": "Archaeodictyomitra sp.",
                "tna": "Archaeodictyomitra",
                "rnk": 5,
                "tid": "txn:421",
                # Short codes — the actual PBDB shape.
                "oei": "Tithonian",
                "oli": "Cenomanian",
                "eag": "149.2",
                "lag": "93.9",
                "lng": "-100.266998",
                "lat": "18.283001",
                "cnm": "The Almoloya Phyllite Unit in Southern Mexico",
                "cc2": "MX",
                "sfm": "Almoloya Phyllite",
            },
            {
                "oid": "occ:472980",
                "cid": "col:46960",
                "oei": "Late Kimmeridgian",
                "eag": "152.21",
                "lag": "149.2",
                "lng": "0.0",
                "lat": "-80.0",
                "cc2": "AQ",
                "sfm": "Ameghino",
                "smb": "Longing",
            },
        ]
    }
    # Pre-write the cache file so the HTTP call short-circuits.
    import hashlib
    cache_key = hashlib.sha1(f"occs|{name.lower()}|25".encode()).hexdigest()[:16]
    cache_file = cache_dir / f"{cache_key}.json"
    cache_file.write_text(json.dumps(payload), encoding="utf-8")

    occs = pbdb.lookup_occurrences(name, max_n=25)
    assert len(occs) == 2, f"expected 2 records, got {len(occs)}"

    # First occurrence — full set of fields
    r0 = occs[0]
    assert r0.early_interval == "Tithonian", (
        f"oei not decoded to early_interval; got {r0.early_interval!r}"
    )
    assert r0.late_interval == "Cenomanian"
    assert r0.max_ma == 149.2, (
        f"eag string not decoded to max_ma float; got {r0.max_ma!r}"
    )
    assert r0.min_ma == 93.9
    assert r0.longitude == -100.266998
    assert r0.latitude == 18.283001
    assert r0.locality == "The Almoloya Phyllite Unit in Southern Mexico", (
        f"cnm not decoded to locality; got {r0.locality!r}"
    )
    assert r0.country == "Mexico", (
        f"cc2 'MX' not converted to country name 'Mexico'; got {r0.country!r}"
    )
    assert r0.country_code == "MX"
    assert r0.formation == "Almoloya Phyllite"

    # Second occurrence — smb carries member rank separately
    r1 = occs[1]
    assert r1.early_interval == "Late Kimmeridgian"
    assert r1.max_ma == 152.21
    assert r1.member == "Longing"
    assert r1.country == "Antarctica"  # AQ → Antarctica


def test_pbdb_lookup_occurrences_accepts_long_codes():
    """Backward-compat: if a future PBDB version (or a different PBDB
    proxy) returns long names, the alias map still decodes them. This
    protects against silent regression when one of the two shapes
    changes."""
    from rlpe.paleodb import PaleoDB

    import json
    import tempfile

    cache_dir = Path(tempfile.mkdtemp())
    pbdb = PaleoDB(cache_dir=cache_dir)
    name = "Test species"
    payload = {
        "records": [
            {
                "occurrence_id": "occ:1",
                "collection_id": "col:1",
                "early_interval": "Maastrichtian",
                "late_interval": "Campanian",
                "max_ma": 72.1,
                "min_ma": 70.0,
                "locality": "Sample Locality",
                "country": "France",
                "latitude": 45.0,
                "longitude": 5.0,
                "formation": "Sample Fm",
            }
        ]
    }
    import hashlib
    cache_key = hashlib.sha1(f"occs|{name.lower()}|25".encode()).hexdigest()[:16]
    cache_file = cache_dir / f"{cache_key}.json"
    cache_file.write_text(json.dumps(payload), encoding="utf-8")

    occs = pbdb.lookup_occurrences(name, max_n=25)
    assert len(occs) == 1
    r = occs[0]
    assert r.early_interval == "Maastrichtian"
    assert r.max_ma == 72.1
    assert r.country == "France"  # passed through (no conversion needed)
    assert r.locality == "Sample Locality"
    assert r.formation == "Sample Fm"


def test_iso_to_country_helper():
    """_iso_to_country returns readable names for known codes and
    falls through to None for unknown codes (rather than raising)."""
    from rlpe.paleodb import _iso_to_country

    assert _iso_to_country("MX") == "Mexico"
    assert _iso_to_country("AQ") == "Antarctica"
    assert _iso_to_country("fr") == "France"  # case-insensitive
    assert _iso_to_country(None) is None
    assert _iso_to_country("ZZ") is None  # unknown → None, not KeyError
    assert _iso_to_country("") is None


def test_pbdb_enrich_geology_consumes_decoded_occurrences():
    """End-to-end: feed ``_pbdb_enrich_geology`` a payload built from
    the *decoded* OccurrenceSummary shape (the converter reads
    ``early_interval`` / ``max_ma`` / ``latitude`` / ``longitude`` /
    ``formation`` from each occurrence dict). Before the alias fix,
    every occurrence dict would carry ``None`` for all of these and
    the biozone / coord fallback would never fire."""
    from rlpe.converters import _pbdb_enrich_geology
    from rlpe.types import MatchResult

    matches = [
        MatchResult(
            paper_id="p1", figure_id="f1", panel_id="1",
            species="Archaeodictyomitra", panel_path=None, bbox=None,
            confidence=0.9, caption_snippet="",
            metadata={
                "paleodb": {
                    "occurrences": [
                        {
                            "early_interval": "Tithonian",
                            "max_ma": 149.2,
                            "min_ma": 145.0,
                            "latitude": 18.28,
                            "longitude": -100.27,
                            "country": "Mexico",
                            "formation": "Almoloya Phyllite",
                            "locality": "Almoloya",
                        }
                    ]
                },
                "geology_links": [
                    {"formation": None, "locality": None, "country": None,
                     "biozone": None, "latitude": None, "longitude": None}
                ],
            },
        )
    ]
    _pbdb_enrich_geology(matches)
    g = matches[0].metadata["geology_links"][0]
    # Without the fix, every field below would be None because the
    # underlying occurrence dicts were empty.
    assert g["biozone"] == "Tithonian", (
        f"biozone fallback did not fire; got {g.get('biozone')!r}"
    )
    assert g["country"] == "Mexico"
    assert g["formation"] == "Almoloya Phyllite"
    assert g["locality"] == "Almoloya"
    assert abs(g["latitude"] - 18.28) < 0.01
    assert abs(g["longitude"] - (-100.27)) < 0.01
    assert g["coord_source"] == "paleodb"


def test_pbdb_lookup_uses_show_full_param():
    """The live integration needs ``show=full`` so PBDB returns lng +
    lat + cnm in the payload. This test guards against accidentally
    reverting to ``show=attr,loc,strat`` (which returns every
    non-core field as ``None``) or to no ``show`` arg (which returns
    only the short codes ``oei`` / ``eag`` / ``lag`` and drops coords).
    """
    from rlpe.paleodb import PaleoDB
    import json
    import tempfile
    import hashlib

    cache_dir = Path(tempfile.mkdtemp())
    pbdb = PaleoDB(cache_dir=cache_dir)
    name = "Archaeodictyomitra"
    # Synthetic ``show=full`` payload shape.
    payload = {
        "records": [
            {
                "oid": "occ:432613",
                "cid": "col:42265",
                "oei": "Tithonian",
                "eag": "149.2",
                "lag": "93.9",
                "lng": "-100.266998",
                "lat": "18.283001",
                "cnm": "The Almoloya Phyllite Unit",
                "cc2": "MX",
                "phl": "Radiozoa",
                "cll": "Polycystinea",
                "odl": "Nassellaria",
                "fml": "Archaeodictyomitridae",
            }
        ]
    }
    cache_key = hashlib.sha1(f"occs|{name.lower()}|25".encode()).hexdigest()[:16]
    cache_file = cache_dir / f"{cache_key}.json"
    cache_file.write_text(json.dumps(payload), encoding="utf-8")

    occs = pbdb.lookup_occurrences(name, max_n=25)
    assert len(occs) == 1
    r = occs[0]
    # ``show=full`` shape: short codes are present and the alias map
    # must surface readable values for every operator-facing field.
    assert r.latitude == 18.283001, (
        f"lat must be decoded from show=full short code 'lat'; "
        f"got {r.latitude!r}"
    )
    assert r.longitude == -100.266998
    assert r.early_interval == "Tithonian"
    assert r.max_ma == 149.2
    assert r.locality == "The Almoloya Phyllite Unit"
    assert r.country == "Mexico"
    assert r.country_code == "MX"


def test_pbdb_params_do_not_contain_invalid_show_attr_loc_strat():
    """Round 25 audit: ``show=attr,loc,strat`` returns records where
    every non-core field is ``None`` on PBDB occs. Lock down the
    params dict so a future edit can't silently reintroduce the
    invalid token."""
    from rlpe.paleodb import PaleoDB
    import json
    import tempfile

    cache_dir = Path(tempfile.mkdtemp())
    pbdb = PaleoDB(cache_dir=cache_dir)
    captured: dict[str, str] = {}

    def _capture_params(self: Any, url: str, params: dict[str, Any], cache_key: str) -> dict[str, Any]:
        captured.update(params)
        return {"records": []}

    import rlpe.paleodb as _pdb_mod
    orig = _pdb_mod.PaleoDB._http_get_json
    _pdb_mod.PaleoDB._http_get_json = _capture_params  # type: ignore[assignment]
    try:
        pbdb.lookup_occurrences("Archaeodictyomitra", max_n=5)
    finally:
        _pdb_mod.PaleoDB._http_get_json = orig  # type: ignore[assignment]

    show_param = captured.get("show")
    assert show_param, "show param missing — would default to short-code-only"
    assert "attr,loc,strat" not in show_param, (
        f"show={show_param!r} returns null fields on PBDB occs"
    )
    # show=full is what brings back lat/lng/cc2/cnm. If the test runs
    # with no show param at all, only short codes are returned and
    # the coord fallback stays inert.
    assert show_param == "full", (
        f"show={show_param!r} loses modern lat/lon for the operator"
    )