"""Audit regression: _LOCALITY_BLOCKLIST is centralized, not duplicated.

Audit Agent A C1 flagged the original duplication as a drift risk.
The two copies had IDENTICAL content (so behavior was unchanged),
but the duplication was a future-drift hazard. Fix: cross_figure_linker
imports the blocklist from sample_id_extractor (the single source of
truth) and re-exports it under its local name for backward compat.

This test pins:
  1. Both modules' _LOCALITY_BLOCKLIST are the same object (identity).
  2. The blocklist still filters expected false-positives.
"""

from __future__ import annotations

from rlpe.cross_figure_linker import _LOCALITY_BLOCKLIST as CFL_BLOCKLIST
from rlpe.sample_id_extractor import _LOCALITY_BLOCKLIST as SID_BLOCKLIST


def test_blocklist_is_the_same_object() -> None:
    """cross_figure_linker must reuse the sample_id_extractor blocklist (identity)."""
    assert CFL_BLOCKLIST is SID_BLOCKLIST, (
        "_LOCALITY_BLOCKLIST drifted — cross_figure_linker has its own "
        "copy. The audit fix was to centralize this. Re-import from "
        "sample_id_extractor instead of redefining."
    )


def test_blocklist_still_filters_known_false_positives() -> None:
    """The blocklist still rejects geological-age terms."""
    for term in ("late cretaceous", "early jurassic", "middle triassic"):
        assert term in SID_BLOCKLIST, (
            f"Expected {term!r} in blocklist; it's a known false-positive "
            f"that locality extractors must skip."
        )


def test_blocklist_still_filters_paper_grammar_stop_words() -> None:
    """The blocklist still rejects paper-grammar stop words."""
    for term in ("this paper", "the study", "figure", "plate"):
        assert term in SID_BLOCKLIST
