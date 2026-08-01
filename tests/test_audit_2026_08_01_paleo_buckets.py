"""Regression tests for audit 2026-08-01 batch W2 — paleo_reconstruction C7/C8/D16."""

from __future__ import annotations

import pytest

from rlpe.paleo_reconstruction import (
    COUNTRY_PLATE,
    PLATE_OVERRIDES,
    infer_plate_id,
    reconstruct_paleo_position,
)


class TestPaleoBucketOrder:
    """Bug C7 — the original ``infer_plate_id`` coord heuristic put
    Eurasia (lat 25..75, lon -15..60) BEFORE Africa (lat -40..40,
    lon -25..55), so Mediterranean coordinates such as Tunisia
    (35, 10) and Cairo (30, 31) were misassigned to Eurasia. The
    fix tightens Eurasia to ``lat < 40`` and inserts a dedicated
    N. Africa bucket (lat 25..40, lon -15..30) before Eurasia so
    Mediterranean sites resolve to the correct plate."""

    def test_tunisia_africa_not_eurasia(self):
        """Tunisia (34°N, 10°E) must resolve to ``"Africa"``, not Eurasia."""
        assert infer_plate_id(modern_lat=34.0, modern_lon=10.0) == "Africa"

    def test_egypt_africa_not_eurasia(self):
        """Egypt (26°N, 30°E) falls inside the new N. Africa bucket
        (lat 25..40, lon -15..30). The country lookup puts Egypt on
        the ``Arabia`` plate (see ``PLATE_OVERRIDES``), but with NO
        country hint we must rely on the coord bucket and get Africa."""
        # Use a longitude that sits inside the new N. Africa bucket
        # (25..30°E for lat 25..40) yet is outside the pre-fix Eurasia
        # lat upper bound — coords that pre-fix would have returned
        # Eurasia must now return Africa.
        assert infer_plate_id(modern_lat=26.0, modern_lon=29.0) == "Africa"

    def test_paris_eurasia(self):
        """Paris (48°N, 2°E) is well above the N. Africa / tightened
        Eurasia buckets and must still resolve to ``"Eurasia"``."""
        assert infer_plate_id(modern_lat=48.0, modern_lon=2.0) == "Eurasia"

    def test_morocco_africa(self):
        """Morocco (32°N, -7°W) sits inside both the new N. Africa
        bucket and the tightened Eurasia bucket; the N. Africa bucket
        is checked first, so this must resolve to ``"Africa"``."""
        assert infer_plate_id(modern_lat=32.0, modern_lon=-7.0) == "Africa"


class TestSiberiaGuard:
    """Bug C8 — ``reconstruct_paleo_position`` line 318 had
    ``if len(poles) <= 3 and age_max <= 100.0 and abs(poles[0][3]) <= 1.0``.
    The Siberia pole table has ``age_max=200``, so the original guard
    never triggered and ages > 50 Ma silently returned the modern
    identity labelled "paleo". The fix relaxes ``age_max <= 100.0``
    to ``age_max <= 250.0`` AND also checks ``abs(poles[-1][3]) <= 1.0``
    so the oldest pole must also be identity before we refuse."""

    def test_siberia_paleo_skipped_when_age_beyond_table(self):
        """Siberia's pole table is the identity pair
        ``[(0, 0, 0, 0), (200, 0, 0, 0)]`` — both endpoints are
        identity rotations. ``age_ma=150`` is within the table's
        age range (0..200) and ``age_ma > 50``, so the relaxed
        guard must reject the lookup and return ``(None, None)``
        rather than fabricating a "no motion" paleo position."""
        result = reconstruct_paleo_position(60.0, 100.0, age_ma=150.0, plate_id="Siberia")
        assert result == (None, None), (
            f"Siberia at 150 Ma must return (None, None); got {result} — "
            "the relaxed guard did not fire and we silently returned the "
            "modern coords labelled 'paleo'."
        )

    def test_siberia_paleo_within_table(self):
        """Siberia's pole table has BOTH identity poles, so ``age_ma``
        in (50, 200] still triggers the refusal. ``age_ma=50`` is
        NOT strictly greater than 50, so the guard does NOT fire and
        the function returns — for an identity pole, the modern
        coords. The expectation here is that the call returns
        *something* (NOT (None, None)) AND the returned coords are
        approximately the modern coords (since both endpoints are
        identity, linear interpolation also yields identity)."""
        paleo_lat, paleo_lon = reconstruct_paleo_position(
            60.0, 100.0, age_ma=50.0, plate_id="Siberia"
        )
        assert paleo_lat is not None and paleo_lon is not None, (
            "Siberia at 50 Ma must NOT be refused — at this age the "
            "guard's ``age_ma > 50.0`` clause does not trigger and the "
            "interpolation should run."
        )
        # The two endpoints are identity rotations, so any
        # interpolation between them is also identity: the returned
        # lat/lon must match the modern lat/lon to within 0.1°.
        assert abs(paleo_lat - 60.0) < 0.1
        assert abs(paleo_lon - 100.0) < 0.1


class TestPlateOverrides:
    """Bug D16 — ``PLATE_OVERRIDES`` (module-level dict, documented
    as the operator extension point for country→plate remapping)
    was never read by ``infer_plate_id``. The fix inserts an
    explicit override lookup at the very TOP of ``infer_plate_id``
    so operator additions take effect without patching the file."""

    def test_plate_overrides_takes_precedence(self, monkeypatch):
        """When ``PLATE_OVERRIDES`` contains a country, that country
        must map to the override value — even if a different value
        exists in ``COUNTRY_PLATE``. Here we add ``"Testland"`` and
        expect it to win over any other resolution."""
        monkeypatch.setitem(PLATE_OVERRIDES, "Testland", "TestPlate")
        assert infer_plate_id(country="Testland") == "TestPlate"

    def test_plate_overrides_falls_back_to_country(self):
        """When ``PLATE_OVERRIDES`` does NOT contain the country, the
        function must fall back to ``COUNTRY_PLATE`` like before.
        This regression-guards against an over-broad override path
        that swallows every country."""
        # ``COUNTRY_PLATE`` is keyed by lowercase phrases; spot-check
        # that the table still has the entries the rest of the
        # pipeline relies on.
        assert "italy" in COUNTRY_PLATE
        assert COUNTRY_PLATE["italy"] == "Adria"

        # Direct assertion using the documented behaviour: a country
        # that exists in BOTH tables (Italy is in PLATE_OVERRIDES
        # AND in COUNTRY_PLATE) must still resolve to its plate via
        # the override branch — and the value must match the
        # COUNTRY_PLATE entry (both tables say Adria for Italy).
        assert infer_plate_id(country="Italy") == COUNTRY_PLATE["italy"]

        # A country that is ONLY in COUNTRY_PLATE (not in
        # PLATE_OVERRIDES) must also resolve correctly through the
        # fallback path. ``"romania"`` is in COUNTRY_PLATE but not
        # in PLATE_OVERRIDES.
        assert "romania" in COUNTRY_PLATE
        assert "Romania" not in PLATE_OVERRIDES
        assert infer_plate_id(country="Romania") == COUNTRY_PLATE["romania"]


class TestPlateOverridesCaseInsensitive:
    """Sub-check: PLATE_OVERRIDES lookup must be tolerant of leading /
    trailing whitespace and case differences so operators writing
    ``"italy"`` or ``"  Italy  "`` get the same plate as ``"Italy"``."""

    def test_override_lookup_is_case_insensitive(self, monkeypatch):
        monkeypatch.setitem(PLATE_OVERRIDES, "Capitalized", "CapPlate")
        assert infer_plate_id(country="capitalized") == "CapPlate"
        assert infer_plate_id(country="CAPITALIZED") == "CapPlate"
        assert infer_plate_id(country="  Capitalized  ") == "CapPlate"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
