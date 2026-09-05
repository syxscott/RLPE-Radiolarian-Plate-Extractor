"""Regression: audit 2026-09-04 geo-1 — fabricated Seton-2012 Euler poles.

The embedded ``_SETON2012_INDO_AUSTRALIAN_ROTATIONS`` table placed every
non-zero timestep's Euler pole exactly on the geographic spin axis
(lat -90, lon 0). A rotation about the spin axis cannot change latitude,
so Australia's "paleo-latitude" was frozen at its modern value for all
0-250 Ma — while the record was exported with
``reconstruction_model="Seton 2012"``. The real Australian plate moved
~25 degrees of latitude north after separating from Antarctica (~35 Ma).

Fix contract (honesty-first — the module must never again invent data
and label it with a literature citation):
1. The fabricated Indo-Australian table is REMOVED. Without a real
   rotation file, ``reconstruct_paleo_position`` returns (None, None)
   for Indo-Australian and no paleo coordinate is exported.
2. A degenerate-pole guard rejects any plate whose poles sit within 5°
   of the spin axis, except explicitly polar plates (Antarctica —
   sitting at the pole IS its real geography).
3. ``reconstruction_model`` says "Seton 2012" ONLY for plates loaded
   from an external ``Seton_etal_2012_ESR.rot`` file; embedded
   approximate tables are labelled "embedded-approximate".
4. ``_load_seton2012_from_external`` gains an auto-discovery entry
   point (``ensure_rotation_source`` honouring ``RLPE_SETON2012_ROT``)
   so operators actually have a path to supply real data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe import paleo_reconstruction as pr


@pytest.fixture
def _restore_euler_poles():
    """``_load_seton2012_from_external`` mutates the module-level
    EULER_POLES dict in place — snapshot and restore it so these tests
    cannot leak a truncated rotation table into other tests."""
    snapshot = {k: list(v) for k, v in pr.EULER_POLES.items()}
    yield
    pr.EULER_POLES.clear()
    pr.EULER_POLES.update(snapshot)


class TestFabricatedIndoAustralianRemoved:
    def test_indo_australian_not_in_euler_poles(self):
        assert "Indo-Australian" not in pr.EULER_POLES

    def test_indo_australian_reconstruct_returns_none(self):
        plat, plon = pr.reconstruct_paleo_position(-35.3, 149.1, 130.0, plate_id="Indo-Australian")
        assert plat is None and plon is None

    def test_enrich_does_not_export_fake_paleo_coords(self):
        rec = {
            "latitude": -35.3,
            "longitude": 149.1,
            "age": "Late Cretaceous",
            "country": "Australia",
        }
        pr.enrich_geology_record(rec)
        assert "paleo_latitude" not in rec
        assert "paleo_longitude" not in rec
        assert "reconstruction_model" not in rec

    def test_fabricated_table_gone_from_module(self):
        assert not hasattr(pr, "_SETON2012_INDO_AUSTRALIAN_ROTATIONS")


class TestDegeneratePoleGuard:
    def test_spin_axis_pole_rejected_for_non_polar_plate(self, monkeypatch):
        # Simulate the exact historical bug: inject a spin-axis table
        # for a non-polar plate; reconstruction must refuse.
        monkeypatch.setitem(
            pr.EULER_POLES,
            "TestPlate",
            [(0.0, 0.0, 0.0, 0.0), (100.0, -90.0, 0.0, -20.0)],
        )
        assert pr._interpolate_euler("TestPlate", 100.0) is None

    def test_polar_plate_antarctica_still_allowed(self):
        # Antarctica sits at the pole; a spin-axis rotation genuinely
        # does not move its latitude. That is real geography, not
        # fabrication — the table must keep working.
        euler = pr._interpolate_euler("Antarctica", 130.0)
        assert euler is not None

    def test_non_degenerate_poles_unaffected(self):
        # Africa's table has real (non spin-axis) poles — untouched.
        assert pr._interpolate_euler("Africa", 130.0) is not None

    def test_no_degenerate_large_rotations_in_any_shipped_table(self):
        # Large spin-axis rotations = the fabrication signature (a
        # small spin-axis row is a legitimate GPlates idiom for
        # "latitude barely moves" — e.g. Africa's 10 Ma 0.5° entry).
        polar = {"Antarctica"}
        offenders = [
            (plate, age, rot)
            for plate, rows in pr.EULER_POLES.items()
            if plate not in polar
            for (age, e_lat, _e_lon, rot) in rows
            if abs(abs(e_lat) - 90.0) < 5.0 and abs(rot) > 10.0
        ]
        assert not offenders, (
            "large spin-axis Euler rotations cannot change latitude — a "
            f"table containing them fabricates paleo-latitudes: {offenders}"
        )


class TestHonestModelLabel:
    def test_embedded_plate_labelled_approximate(self):
        rec = {"latitude": 41.9, "longitude": 12.5, "ma_mid": 130.0, "country": "Italy"}
        pr.enrich_geology_record(rec)
        assert rec.get("reconstruction_model") == "embedded-approximate"

    def test_external_file_plate_labelled_seton2012(
        self, tmp_path, monkeypatch, _restore_euler_poles
    ):
        rot = tmp_path / "Seton_etal_2012_ESR.rot"
        rot.write_text(
            "# test rotation file\n101 0.0 0.0 0.0 0.0\n101 130.0 60.0 50.0 10.2\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(pr, "_EXTERNAL_MODEL_PLATES", set())
        merged = pr._load_seton2012_from_external(rot)
        assert merged >= 1
        assert "Africa" in pr._EXTERNAL_MODEL_PLATES
        rec = {"latitude": 34.0, "longitude": 9.0, "ma_mid": 130.0, "country": "Tunisia"}
        pr.enrich_geology_record(rec)
        assert rec.get("reconstruction_model") == "Seton 2012"

    def test_converters_label_uses_honest_helper(self):
        # converters.py must route its reconstruction_model stamp
        # through the module-level helper, not hardcode "Seton2012".
        import inspect

        import rlpe.converters as conv

        src = inspect.getsource(conv)
        assert 'reconstruction_model="Seton2012"' not in src


class TestExternalRotationDiscovery:
    def test_ensure_rotation_source_env_var(self, tmp_path, monkeypatch, _restore_euler_poles):
        rot = tmp_path / "s.rot"
        rot.write_text("101 0.0 0.0 0.0 0.0\n101 50.0 77.0 68.0 3.0\n", encoding="utf-8")
        monkeypatch.setenv("RLPE_SETON2012_ROT", str(rot))
        monkeypatch.setattr(pr, "_EXTERNAL_MODEL_PLATES", set())
        n = pr.ensure_rotation_source()
        assert n >= 1
        # Second call is a no-op (already loaded).
        assert pr.ensure_rotation_source() == 0

    def test_ensure_rotation_source_no_env_no_op(self, monkeypatch):
        monkeypatch.delenv("RLPE_SETON2012_ROT", raising=False)
        assert pr.ensure_rotation_source() == 0

    def test_ensure_rotation_source_missing_file_no_op(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RLPE_SETON2012_ROT", str(tmp_path / "nope.rot"))
        assert pr.ensure_rotation_source() == 0
