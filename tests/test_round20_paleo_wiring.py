"""Round 20 source-guard tests: GPlates paleo_reconstruction wiring.

User audit: 4 OA papers were sampled. ``run_output.paleo_coordinates``
was 0/4 — the GPlates-style backend exists in
``rlpe.paleo_reconstruction`` but was never connected to
``run_output_from_provenance``. Round 20 wires it via
``paleo_coordinates_from_localities``.

These tests pin the wiring: a synthetic locality with valid
coordinates + an associated geology context must produce a
non-empty ``PaleoCoordinateRecord`` in the run output.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _build_provenance():
    from rlpe.schema_models import ProvenanceRecord

    return ProvenanceRecord(
        pipeline_version="test",
        schema_version="1.0.0",
        git_commit="test",
        git_dirty=False,
        config_snapshot={},
        input_sha256={},
        timestamp_utc="2026-07-08T00:00:00",
        host="test",
        python_version="3.13",
    )


def _build_match(
    paper_id: str,
    caption: str,
    *,
    species: str = "Genus species",
    locality: str | None = None,
    country: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    ma_mid: float | None = None,
):
    from rlpe.types import MatchResult

    geology_links = []
    if locality:
        geology_links.append(
            {
                "locality": locality,
                "country": country,
                "latitude": lat,
                "longitude": lon,
                "ma_mid": ma_mid,
                "ma_top": ma_mid,
                "ma_base": ma_mid + 5 if ma_mid else None,
            }
        )
    return MatchResult(
        paper_id=paper_id,
        figure_id="od_plate_test_p001_pl01",
        panel_id="1",
        species=species,
        panel_path=None,
        bbox=None,
        confidence=0.5,
        caption_snippet=caption,
        metadata={"geology_links": geology_links},
    )


def test_paleo_coordinates_populated_for_locality_with_coords():
    """Locality with coords + ma_mid produces a paleo_coordinates record."""
    from rlpe.converters import run_output_from_provenance

    provenance = _build_provenance()
    matches = [
        _build_match(
            "test_paper_1",
            "Plate 1. Radiolarians from Jédidi, Tunisia",
            locality="Jédidi",
            country="Tunisia",
            lat=37.0,
            lon=11.0,
            ma_mid=160.0,  # Middle Jurassic
        )
    ]
    out = run_output_from_provenance(provenance, matches)
    assert out["paleo_coordinates"], (
        f"paleo_coordinates still empty after Round 20 wiring: {out['paleo_coordinates']}"
    )
    pc = out["paleo_coordinates"][0]
    assert pc["modern_latitude"] == 37.0
    assert pc["modern_longitude"] == 11.0
    assert pc["reconstruction_age_ma"] == 160.0
    # paleo coordinates must be non-None and distinct from modern
    assert pc["paleo_latitude"] is not None
    assert pc["paleo_longitude"] is not None
    # At 160 Ma, Tunisia was in the Tethys — coords should differ from modern
    assert (
        abs(pc["paleo_latitude"] - pc["modern_latitude"]) > 0.1
        or abs(pc["paleo_longitude"] - pc["modern_longitude"]) > 0.1
    ), (
        f"Paleo coords identical to modern — reconstruction didn't run: "
        f"modern=({pc['modern_latitude']},{pc['modern_longitude']}) "
        f"paleo=({pc['paleo_latitude']},{pc['paleo_longitude']})"
    )


def test_paleo_coordinates_empty_for_locality_without_coords():
    """Locality without lat/lon is skipped (no fake records)."""
    from rlpe.converters import run_output_from_provenance

    provenance = _build_provenance()
    matches = [
        _build_match(
            "test_paper_2",
            "Plate 1. Radiolarians from a region without coordinates.",
            locality="Unknown Region",
            country="Italy",
            lat=None,
            lon=None,
            ma_mid=160.0,
        )
    ]
    out = run_output_from_provenance(provenance, matches)
    assert out["paleo_coordinates"] == [], (
        f"Fake paleo record emitted for locality without coords: {out['paleo_coordinates']}"
    )


def test_paleo_coordinates_skips_when_no_locality():
    """No locality record at all → empty paleo_coordinates."""
    from rlpe.converters import run_output_from_provenance

    provenance = _build_provenance()
    matches = [_build_match("test_paper_3", "Plate 1. No geology info.", species="Foo bar")]
    out = run_output_from_provenance(provenance, matches)
    assert out["paleo_coordinates"] == []


def test_paleocoord_missing_warning_no_longer_emitted():
    """Source guard: the deprecated ``paleocoord_backend_missing``
    warning must not appear in run_output.warnings when the backend
    is wired."""
    from rlpe.converters import run_output_from_provenance

    provenance = _build_provenance()
    matches = [
        _build_match(
            "test_paper_4",
            "Plate 1. Radiolarians from Jédidi.",
            locality="Jédidi",
            country="Tunisia",
            lat=37.0,
            lon=11.0,
            ma_mid=160.0,
        )
    ]
    out = run_output_from_provenance(provenance, matches)
    warn_codes = {w.get("code") for w in out["warnings"]}
    assert "paleocoord_backend_missing" not in warn_codes, (
        f"Stale backend-missing warning still emitted: {warn_codes}"
    )


def test_paleo_coordinates_source_guard():
    """Source guard: converters.py must define
    ``paleo_coordinates_from_localities`` AND call it from
    ``run_output_from_provenance`` instead of returning ``[]``."""
    src = Path(
        "/home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor/src/rlpe/converters.py"
    ).read_text(encoding="utf-8")
    assert "def paleo_coordinates_from_localities" in src, (
        "converters.py is missing paleo_coordinates_from_localities."
    )
    # ``paleo_coordinates: paleo_dump`` must be in the return dict
    assert '"paleo_coordinates": paleo_dump' in src or (
        "paleo_coordinates" in src and "paleo_dump" in src
    ), "paleo_coordinates_from_localities not wired into run_output"


def test_plate_id_inferred_from_country():
    """Italy → Adria, Tunisia → Africa, etc."""
    from rlpe.paleo_reconstruction import infer_plate_id

    assert infer_plate_id(country="Italy", modern_lat=37.0, modern_lon=14.0) == "Adria"
    assert infer_plate_id(country="Tunisia", modern_lat=37.0, modern_lon=11.0) == "Africa"
