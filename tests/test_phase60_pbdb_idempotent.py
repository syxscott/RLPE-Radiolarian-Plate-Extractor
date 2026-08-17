"""Tests for Phase 60 Plan 3 — Bug 3.7: PBDB geology enrichment must be
idempotent.

``_pbdb_enrich_geology`` is invoked from multiple paths (the export
converter and any direct caller of ``run_output_from_provenance``).
The previous implementation re-aggregated PBDB occurrences on every
call and rewrote ``biozone`` / ``formation`` / ``locality`` /
``country`` / ``latitude`` / ``longitude`` / ``ma_top`` / ``ma_base``
on the geology link dicts in place — even when those fields were
already filled in from a previous enrichment pass.

For the second pass this produced:

  * an extra ``[PBDB first-occurrence: <early_interval>]`` appended to
    ``evidence_text`` on every call (each call appended its own suffix)
  * a double-aggregation of ``ma_top`` / ``ma_base`` means (each pass
    re-averaged the already-averaged value, drifting the result)
  * potential repeated work for every paper, which adds up over a
    200-species batch

Phase 60 Plan 3 fix: an early-return if ``meta.get("pbdb_enriched")``
is set on every match, and the function flips the flag at the end
of the pass. Downstream callers can also short-circuit if they see
the flag.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.converters import _pbdb_enrich_geology  # noqa: E402
from rlpe.types import MatchResult  # noqa: E402


def _match(species: str, occurrences: list[dict]) -> MatchResult:
    return MatchResult(
        paper_id="p1",
        figure_id="fig1",
        panel_id="1",
        species=species,
        panel_path=None,
        bbox=None,
        confidence=0.5,
        label_text=None,
        caption_snippet=None,
        ocr_text=None,
        metadata={
            "geology_links": [
                {
                    "biozone": None,
                    "formation": None,
                    "locality": None,
                    "country": None,
                    "latitude": None,
                    "longitude": None,
                    "ma_top": None,
                    "ma_base": None,
                    "evidence_text": "",
                }
            ],
            "paleodb": {"occurrences": occurrences},
        },
    )


def test_pbdb_enrich_idempotent():
    """Two consecutive calls to ``_pbdb_enrich_geology`` must produce
    the SAME state — biozone / formation / locality / country /
    latitude / longitude / ma_top / ma_base are not re-aggregated
    and the ``evidence_text`` suffix is not appended twice."""
    occs = [
        {
            "early_interval": "Changhsingian",
            "formation": "Ford Creek",
            "locality": "Lone Mountain",
            "country": "United States",
            "latitude": 39.5,
            "longitude": -117.0,
            "max_ma": 251.9,
            "min_ma": 254.14,
        }
    ]
    matches = [_match("Entactinia compacta", occs)]

    # First pass — populates fields and sets the flag.
    _pbdb_enrich_geology(matches)
    first_state = {
        k: matches[0].metadata["geology_links"][0].get(k)
        for k in (
            "biozone",
            "formation",
            "locality",
            "country",
            "latitude",
            "longitude",
            "ma_top",
            "ma_base",
        )
    }
    first_evidence = matches[0].metadata["geology_links"][0]["evidence_text"]

    # Second pass — must be a no-op so the values are unchanged.
    _pbdb_enrich_geology(matches)
    second_state = {
        k: matches[0].metadata["geology_links"][0].get(k)
        for k in (
            "biozone",
            "formation",
            "locality",
            "country",
            "latitude",
            "longitude",
            "ma_top",
            "ma_base",
        )
    }
    second_evidence = matches[0].metadata["geology_links"][0]["evidence_text"]

    assert first_state == second_state, (
        f"PBDB enrichment was not idempotent: first={first_state}, second={second_state}"
    )
    # Evidence_text must not gain a duplicate ``[PBDB first-occurrence: ...]``
    # suffix on the second pass.
    assert first_evidence == second_evidence, (
        f"evidence_text mutated on second pass: {first_evidence!r} -> {second_evidence!r}"
    )
    # The flag is the new contract — assert it's set so other callers
    # can also short-circuit on the marker.
    assert matches[0].metadata.get("pbdb_enriched") is True


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
