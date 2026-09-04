"""Regression: audit 2026-09-04 geo-2 —
:func:`rlpe.paleo_reconstruction.enrich_geology_record` read
``record["latitude"]`` and wrote it back to ``record["modern_latitude"]``.

The producer (``geology_extraction.GeologyRecord``) carries TWO
distinct fields:

    latitude / longitude          # the first coord found (may be paleo)
    modern_latitude / longitude   # only set when coord_age == "modern"
    paleo_latitude / longitude    # only set when coord_age == "paleo"

When the first-found coord happened to be a paleo coordinate (e.g.
from a section that mixes "modern 35°N, 110°E" with "during the
Cretaceous, 12°S 60°W"), the bare ``latitude`` field held the paleo
value, and ``enrich_geology_record`` then wrote THAT value back to
``modern_latitude`` — corrupting the modern anchor. The pipeline
ended up using a paleo coordinate as the modern starting point for
Rodrigues rotation, and the resulting "paleo_latitude" was just
the paleo value rotated a second time.

Fix contract: prefer the explicitly-classified ``modern_latitude`` /
``modern_longitude`` when present; fall back to ``latitude`` /
``longitude`` only when the modern pair is unset. The written-back
``modern_latitude`` then matches what the rotation actually used.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe.paleo_reconstruction import enrich_geology_record  # noqa: E402


class TestPrefersClassifiedModernCoordinate:
    def test_modern_latitude_used_when_present(self, monkeypatch):
        # Producer correctly classified: modern=(10, 20), paleo=(30, 40).
        # bare latitude/longitude hold the (now stale) first-found values.
        # We monkeypatch the rotator + infer_plate so we don't need a real
        # rotation file: assert only the WRITE contract.
        import rlpe.paleo_reconstruction as pr

        monkeypatch.setattr(
            pr,
            "infer_plate_id",
            lambda **kwargs: "Africa",
        )
        # Skip reconstruction by making reconstruct return None
        monkeypatch.setattr(
            pr,
            "reconstruct_paleo_position",
            lambda *args, **kwargs: (None, None),
        )
        rec = {
            "latitude": 30.0,         # first-found, happens to be paleo
            "longitude": 40.0,
            "modern_latitude": 10.0,  # classified modern
            "modern_longitude": 20.0,
            "chronostratigraphy": "Cretaceous",
            "country": "Testland",
        }
        enrich_geology_record(rec)
        # The write-back MUST use the classified modern pair (10, 20),
        # NOT the first-found (30, 40) that was actually paleo.
        assert rec["modern_latitude"] == 10.0
        assert rec["modern_longitude"] == 20.0

    def test_first_found_used_when_modern_unclassified(self, monkeypatch):
        # Producer didn't classify (legacy path): latitude/longitude
        # are the only available modern anchors — must still work.
        import rlpe.paleo_reconstruction as pr

        monkeypatch.setattr(pr, "infer_plate_id", lambda **kwargs: "Africa")
        monkeypatch.setattr(
            pr,
            "reconstruct_paleo_position",
            lambda *args, **kwargs: (10.0, 20.0),
        )
        rec = {
            "latitude": 10.0,
            "longitude": 20.0,
            "chronostratigraphy": "Cretaceous",
            "country": "Testland",
        }
        enrich_geology_record(rec)
        assert rec["modern_latitude"] == 10.0
        assert rec["modern_longitude"] == 20.0

    def test_paleo_first_found_does_not_leak_into_modern(self, monkeypatch):
        # The audit failure mode: paleo coordinate ended up in
        # "latitude" because it was first-found; enrich then rotated
        # it AGAIN to compute paleo_latitude.
        import rlpe.paleo_reconstruction as pr

        monkeypatch.setattr(pr, "infer_plate_id", lambda **kwargs: "Africa")

        # Capture the (lat, lon) that reconstruct_paleo_position is
        # called with — these should be the MODERN anchor coords.
        captured = {}

        def fake_recon(lat, lon, age, plate_id):
            captured["lat"] = lat
            captured["lon"] = lon
            return (15.0, 25.0)

        monkeypatch.setattr(pr, "reconstruct_paleo_position", fake_recon)

        rec = {
            "latitude": 30.0,         # paleo in this scenario
            "longitude": 40.0,
            "modern_latitude": 10.0,  # real modern
            "modern_longitude": 20.0,
            "chronostratigraphy": "Cretaceous",
            "country": "Testland",
        }
        enrich_geology_record(rec)
        # The rotator must have been called with the MODERN pair (10, 20),
        # not the paleo (30, 40) that was sitting in bare "latitude".
        assert captured["lat"] == 10.0
        assert captured["lon"] == 20.0
        # And the resulting paleo_latitude is the rotated MODERN value
        # — not "second rotation of paleo".
        assert rec["paleo_latitude"] == 15.0
        assert rec["paleo_longitude"] == 25.0
