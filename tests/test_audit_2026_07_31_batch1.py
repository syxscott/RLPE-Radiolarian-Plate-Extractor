"""Regression tests for audit 2026-07-31 batch 1 (domain numerics).

Covers:
  - UAZ biozone calibration (Baumgartner et al. 1995): UAZ 7-11 must
    be Jurassic, not Albian–Maastrichtian; ranges expand "UAZ 4-7"
  - paleo rotation formula: correct Rodrigues rotation, length-preserving
  - PBDB interval field mapping (rnk / eag / lag)
  - Silurian series Ma values (ICS 2023)
  - era/eon classification no longer returns high-confidence empties
  - Permian series (Cisuralian / Guadalupian / Lopingian) local hits
  - epoch-level chronostratigraphy is not downgraded to period
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class TestUAZCalibration:
    def test_uaz7_is_jurassic(self):
        from rlpe.stratigraphy import lookup_biozone_ma

        top, base = lookup_biozone_ma("UAZ 7")
        assert top is not None
        # UAZ 7 = late Bathonian–early Callovian (~163-165.3 Ma),
        # NOT Albian (100.5-113 Ma as before the fix).
        assert 155 <= top <= 168, f"UAZ 7 top should be ~163 Ma, got {top}"
        assert 155 <= base <= 168

    def test_uaz8_is_jurassic(self):
        from rlpe.stratigraphy import lookup_biozone_ma

        top, base = lookup_biozone_ma("UAZ 8")
        # middle Callovian–early Oxfordian (~158-163 Ma)
        assert 150 <= top <= 166, f"UAZ 8 top should be ~158 Ma, got {top}"

    def test_uaz10_11_are_jurassic(self):
        from rlpe.stratigraphy import lookup_biozone_ma

        for name in ("UAZ 10", "UAZ 11"):
            top, base = lookup_biozone_ma(name)
            assert top is not None and base is not None
            # Jurassic (145-160 Ma), NOT Santonian–Maastrichtian
            assert 140 <= top <= 160, f"{name} top should be ~150 Ma, got {top}"
            assert base > 140

    def test_uaz21_barremian_era(self):
        from rlpe.stratigraphy import lookup_biozone_ma

        top, base = lookup_biozone_ma("UAZ 21")
        assert top is not None
        assert 120 <= top <= 127, f"UAZ 21 top should be ~125 Ma, got {top}"

    def test_all_uaz_1_21_present_and_ordered(self):
        from rlpe.stratigraphy import lookup_biozone_ma

        prev_top = None
        for i in range(1, 22):
            top, base = lookup_biozone_ma(f"UAZ {i}")
            assert top is not None, f"UAZ {i} missing"
            assert base is not None
            assert base > top, f"UAZ {i}: base {base} must be older than top {top}"
            if prev_top is not None:
                # adjacent zones may overlap slightly; allow a small
                # overlap but each zone must be younger than the one
                # below (higher top).
                assert top <= prev_top + 0.1, (
                    f"UAZ {i} top {top} not consistent with UAZ {i-1} top {prev_top}"
                )
            prev_top = top

    def test_uaz_range_form_expands(self):
        from rlpe.stratigraphy import lookup_biozone_ma

        top, base = lookup_biozone_ma("UAZ 4-7")
        assert top is not None
        # union of UAZ 4-7: youngest boundary = UAZ 7's top
        # (higher UAZ number = younger), oldest boundary = UAZ 4's base.
        t7, _ = lookup_biozone_ma("UAZ 7")
        _, b4 = lookup_biozone_ma("UAZ 4")
        assert abs(top - t7) < 1e-9, f"union top {top} should be UAZ 7 top {t7}"
        assert abs(base - b4) < 1e-9, f"union base {base} should be UAZ 4 base {b4}"

    def test_other_zones_untouched(self):
        from rlpe.stratigraphy import lookup_biozone_ma

        # Hollis 1997 / O'Dogherty 1994 / Riedel & Sanfilippo zones
        # were already correct — they must not have changed.
        t, b = lookup_biozone_ma("Buryella clinata Zone")
        assert (56.0, 59.0) == (t, b)
        t, b = lookup_biozone_ma("Buryella tetradica Zone")
        assert (83.6, 89.0) == (t, b)


class TestRotationFormula:
    def test_rotation_around_north_pole(self):
        from rlpe.paleo_reconstruction import _rotate_point

        lat, lon = _rotate_point(0, 0, 90, 0, 90)
        assert abs(lat) < 1e-9 and abs(lon - 90) < 1e-9, (lat, lon)

    def test_rotation_30_deg_north_pole(self):
        from rlpe.paleo_reconstruction import _rotate_point

        lat, lon = _rotate_point(0, 0, 90, 0, 30)
        assert abs(lat) < 1e-9 and abs(lon - 30) < 1e-9, (lat, lon)

    def test_point_to_pole_rotation(self):
        from rlpe.paleo_reconstruction import _rotate_point

        # (0°N, 90°E) rotated 90° around (0°N, 0°E) lands at the north pole.
        lat, lon = _rotate_point(0, 90, 0, 0, 90)
        assert abs(lat - 90) < 1e-9, (lat, lon)

    def test_length_preserved(self):
        import math

        from rlpe.paleo_reconstruction import _rotate_point

        for lat0, lon0, elat, elon, rot in (
            (23, -61, 30, 20, 37),
            (-45, 120, -10, 80, 130),
            (80, 0, 0, 0, 90),
        ):
            lat, lon = _rotate_point(lat0, lon0, elat, elon, rot)
            x = math.cos(math.radians(lat)) * math.cos(math.radians(lon))
            y = math.cos(math.radians(lat)) * math.sin(math.radians(lon))
            z = math.sin(math.radians(lat))
            assert abs(x * x + y * y + z * z - 1.0) < 1e-12

    def test_360_degrees_identity(self):
        from rlpe.paleo_reconstruction import _rotate_point

        lat, lon = _rotate_point(12, 34, 90, 0, 360)
        assert abs(lat - 12) < 1e-6 and abs(lon - 34) < 1e-6


class TestEulerInterpolation:
    def test_exact_timestep_uses_own_pole(self):
        """Adria at exactly 200 Ma must use the 200 Ma pole
        (35,18,-10), not the adjacent 250 Ma pole — the interpolation
        t direction was inverted."""
        from rlpe.paleo_reconstruction import _interpolate_euler, reconstruct_paleo_position

        pole = _interpolate_euler("Adria", 200.0)
        assert pole is not None
        assert abs(pole[0] - 35.0) < 1e-9, f"pole lat {pole[0]} should be 35"
        assert abs(pole[1] - 18.0) < 1e-9
        assert abs(pole[2] - (-10.0)) < 1e-9

        lat, lon = reconstruct_paleo_position(41.0, 14.0, 200.0, plate_id="Adria")
        assert lat is not None
        # small but real displacement from the 10° rotation
        assert abs(lat - 41.0) > 0.2

    def test_zero_ma_uses_identity_pole(self):
        from rlpe.paleo_reconstruction import _interpolate_euler

        pole = _interpolate_euler("Adria", 0.0)
        assert pole is not None
        assert abs(pole[0]) < 1e-9 and abs(pole[1]) < 1e-9 and abs(pole[2]) < 1e-9


class TestPbdbIntervalMapping:
    def test_pbdb_lookup_maps_fields(self, monkeypatch):
        """The PBDB payload uses rnk/eag/lag; the old code read the
        nonexistent 'tpb' and swapped eag/lag."""
        from rlpe import stratigraphy

        fake = [
            {
                "oid": 1,
                "nam": "Early Berriasian",
                "rnk": "subperiod",
                "par": 2,
                "eag": 143.1,  # older bound
                "lag": 141.0,  # younger bound
            },
            {"oid": 2, "nam": "Cretaceous", "rnk": "period", "par": None},
        ]
        monkeypatch.setattr(stratigraphy, "fetch_pbdb_intervals", lambda: fake)
        row = stratigraphy._pbdb_lookup("Early Berriasian")
        assert row is not None
        assert row["rank"] == "age", (  # subperiod not in whitelist → age
            "unknown ranks must degrade to 'age' (whitelist)"
        )
        assert row["ma_top"] == 141.0, "ma_top must be the younger bound (lag)"
        assert row["ma_base"] == 143.1, "ma_base must be the older bound (eag)"
        assert row["parent"] == "Cretaceous"

    def test_pbdb_lookup_known_rank_passes_through(self, monkeypatch):
        from rlpe import stratigraphy

        fake = [
            {"oid": 3, "nam": "Guadalupian", "rnk": "epoch", "par": None, "eag": 273.01, "lag": 259.51},
        ]
        monkeypatch.setattr(stratigraphy, "fetch_pbdb_intervals", lambda: fake)
        row = stratigraphy._pbdb_lookup("Guadalupian")
        assert row["rank"] == "epoch"


class TestSilurianSeries:
    def test_silurian_series_ics2023(self):
        from rlpe.stratigraphy import classify_age_string

        expected = {
            "Llandovery": (433.4, 443.8),
            "Wenlock": (427.4, 433.4),
            "Ludlow": (423.0, 427.4),
            "Pridoli": (419.2, 423.0),
        }
        for name, (top, base) in expected.items():
            cls = classify_age_string(name)
            assert cls.period == "Silurian", f"{name} period"
            assert cls.epoch == name
            assert cls.ma_top == top, f"{name} ma_top {cls.ma_top} != {top}"
            assert cls.ma_base == base, f"{name} ma_base {cls.ma_base} != {base}"


class TestEraEonClassification:
    def test_era_returns_nonempty_high_confidence(self):
        from rlpe.stratigraphy import classify_age_string

        for name in ("Mesozoic", "Paleozoic", "Cenozoic", "Phanerozoic"):
            cls = classify_age_string(name)
            assert cls.period == name, f"{name} must map period -> itself"
            assert cls.confidence > 0.9
            assert cls.rank in {"era", "eon"}

    def test_permian_series_local_hits(self):
        from rlpe.stratigraphy import classify_age_string

        for name, (top, base) in {
            "Cisuralian": (273.01, 298.9),
            "Guadalupian": (259.51, 273.01),
            "Lopingian": (251.9, 259.51),
        }.items():
            cls = classify_age_string(name)
            assert cls.epoch == name, f"{name} must resolve locally (no network)"
            assert cls.period == "Permian"
            assert cls.ma_top == top and cls.ma_base == base


class TestEpochNotDowngraded:
    def test_miocene_stays_epoch(self):
        from rlpe.geology_extraction import extract_geology_from_sections

        recs = extract_geology_from_sections(
            [{"title": "geological setting", "text": "Miocene radiolarians from Cyprus.",
              "section_type": "geological_setting"}]
        )
        assert recs
        r = recs[0]
        assert r.chronostratigraphy == "Miocene", (
            f"epoch must not be downgraded to 'Neogene', got {r.chronostratigraphy!r}"
        )
