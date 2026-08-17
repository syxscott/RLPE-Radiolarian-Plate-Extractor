"""Audit regression: cross_figure_linker writes figure_id into geology_links.

Audit Agent B H3: pipeline.py appended geology_links dicts with no
figure_id field, so a downstream auditor cannot trace which
figure (strat column, paleogeographic map, range chart) the
panel's age/formation/locality came from. The LinkResult dataclass
already had a figure_id field — it just wasn't propagated.

This test pins the figure_id field as required in the
geology_links dict written by pipeline.py.
"""

from __future__ import annotations

import inspect

from src.rlpe import pipeline as pl


def test_geology_links_dict_contains_figure_id_field() -> None:
    """The dict appended to geology_links must carry figure_id."""
    src = inspect.getsource(pl)
    assert '"figure_id": lr.figure_id' in src, (
        "Expected `figure_id: lr.figure_id` to be propagated into the "
        "geology_links dict written by pipeline.cross_figure_linker "
        "integration. Without it, GBIF/PBDB audits cannot trace "
        "links back to source figure."
    )


def test_geology_links_dict_contains_link_source() -> None:
    """The dict appended to geology_links must carry link_source."""
    src = inspect.getsource(pl)
    assert '"link_source": lr.source' in src, (
        "Expected `link_source: lr.source` in geology_links dict."
    )


def test_linkresult_has_figure_id_field() -> None:
    """LinkResult.figure_id is already defined; just verify it's wired through."""
    from src.rlpe.cross_figure_linker import LinkResult

    fields = {f.name for f in LinkResult.__dataclass_fields__.values()}
    assert "figure_id" in fields, f"LinkResult missing figure_id field. Has: {fields}"
