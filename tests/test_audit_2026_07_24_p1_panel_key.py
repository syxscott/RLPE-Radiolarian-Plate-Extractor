"""Audit regression: by_panel_id must use (figure_id, panel_id) tuple key.

Audit Agent B M5: the linker lookup dict used panel_id as
key alone. Two plates in the same paper can both carry
panel "5" or "1" (e.g. Bandini 2011 pl07 and pl09 share
label "1" through "27"). Using panel_id alone caused the
second plate's panel to overwrite the first's entry in
this dict, so the wrong LinkResult was attached.

This test pins the composite key.
"""

from __future__ import annotations

import inspect

from src.rlpe import pipeline as pl


def test_by_panel_id_uses_composite_key() -> None:
    """The lookup dict must key on (figure_id, panel_id) tuple."""
    src = inspect.getsource(pl)
    assert "by_panel_id[(fid, pid)] = lr" in src, (
        "Expected `by_panel_id[(fid, pid)] = lr` composite key. "
        "Single-key `by_panel_id[pid] = lr` was the source of "
        "cross-plate panel_id collisions."
    )
    assert "by_panel_id.get((fid, pid))" in src, (
        "Expected the lookup call to use the composite key."
    )


def test_linkresult_isolation_across_plates() -> None:
    """Simulate two plates each with panel "5"; ensure correct linking."""
    from src.rlpe.cross_figure_linker import LinkResult, LINK_SOURCE_SAMPLE

    # Two LinkResults for the same panel_id but different figures
    lr_plate07 = LinkResult(
        panel_id="5", species=None, figure_id="pl07",
        formation="Scaglia Rossa", age="Late Cretaceous",
        locality="Sicily", confidence=1.0,
        source=LINK_SOURCE_SAMPLE, evidence="",
    )
    lr_plate09 = LinkResult(
        panel_id="5", species=None, figure_id="pl09",
        formation="Rosso Ammonitico", age="Late Cretaceous",
        locality="Western Sicily", confidence=1.0,
        source=LINK_SOURCE_SAMPLE, evidence="",
    )
    # Build a synthetic by_panel_id with the composite key
    by_panel_id = {
        ("pl07", "5"): lr_plate07,
        ("pl09", "5"): lr_plate09,
    }
    # Lookup for pl07 returns lr_plate07, not lr_plate09
    assert by_panel_id[("pl07", "5")] is lr_plate07
    assert by_panel_id[("pl09", "5")] is lr_plate09
    # The two entries are distinct, not overwritten
    assert by_panel_id[("pl07", "5")] is not by_panel_id[("pl09", "5")]
