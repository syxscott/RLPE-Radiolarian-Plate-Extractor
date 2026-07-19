"""Phase 63 Plan 6 — Task 6.6 pinning regression: PBDB idempotency.

Pins the Phase 60 Plan 3.7 (commit 431f5d5) fix: ``_pbdb_enrich_geology``
is invoked from ``run_output_from_provenance`` (via
``geology_contexts_from_matches``). If the converter is called twice
in the same session (re-export from the same RunOutput, or
interleaved CLI export + GUI export), the enrichment must NOT
re-aggregate PBDB occurrences and append a second
``[PBDB first-occurrence: ...]`` suffix to ``evidence_text``.

Plan 3 fix: an early-return guard checks
``all(m.metadata.get('pbdb_enriched') for m in matches)`` and the
function flips the flag at the end of a successful pass.

This test runs through ``run_output_from_provenance`` to verify
that the export converter path also benefits from the guard.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.converters import run_output_from_provenance  # noqa: E402
from rlpe.provenance import build_provenance  # noqa: E402
from rlpe.schema_models import ProvenanceRecord  # noqa: E402
from rlpe.types import MatchResult  # noqa: E402


def _match(pbdb_occs: list[dict]) -> MatchResult:
    return MatchResult(
        paper_id="p1",
        figure_id="fig1",
        panel_id="1",
        species="Genus species",
        panel_path="/tmp/p.png",
        bbox=[0, 0, 100, 100],
        confidence=0.9,
        metadata={
            "geology_links": [
                {
                    "age": None,
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
            "paleodb": {"occurrences": pbdb_occs},
        },
    )


def _occs() -> list[dict]:
    return [
        {
            "early_interval": "Changhsingian",
            "formation": "Ford Creek",
            "locality": "Lone Mountain",
            "country": "United States",
            "latitude": 39.5,
            "longitude": -117.0,
            "max_ma": 251.9,
            "min_ma": 254.14,
        },
        {
            "early_interval": "Changhsingian",
            "formation": "Ford Creek",
            "locality": "Lone Mountain",
            "country": "United States",
            "latitude": 39.5,
            "longitude": -117.0,
            "max_ma": 252.0,
            "min_ma": 254.0,
        },
    ]


def test_run_output_from_provenance_idempotent_on_pbdb():
    """Calling run_output_from_provenance twice with the same match list
    must not mutate the geology link twice (no double-aggregation,
    no double evidence suffix)."""
    m = _match(_occs())
    prov = ProvenanceRecord(**build_provenance().to_dict())

    # First pass: populates fields and sets the flag.
    out1 = run_output_from_provenance(prov, [m])
    state_after_pass_1 = dict(m.metadata["geology_links"][0])
    evidence_after_pass_1 = state_after_pass_1.get("evidence_text", "")

    # Second pass: must be a no-op.
    out2 = run_output_from_provenance(prov, [m])
    state_after_pass_2 = dict(m.metadata["geology_links"][0])

    # The same RunOutput shape (no second suffix appended).
    assert state_after_pass_1.get("formation") == "Ford Creek"
    assert state_after_pass_1.get("biozone") == "Changhsingian"
    assert state_after_pass_1.get("evidence_text") == evidence_after_pass_1
    assert state_after_pass_1 == state_after_pass_2
    # The provenance flag is the contract — other consumers can
    # also short-circuit on the marker without re-invoking.
    assert m.metadata.get("pbdb_enriched") is True
    # Both outputs have the same panel shape.
    assert out1["panels"][0]["bbox"] == out2["panels"][0]["bbox"]


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
