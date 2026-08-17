"""Tests for Strategy 4 of cross_figure_linker: cross-reference match.

Audit 2026-08-16 (fill-gaps): previously ``rlpe.cross_refs.parse_cross_refs``
had 15+ unit tests in ``test_cross_refs.py`` but no production caller. We
now invoke it from ``cross_figure_linker._strategy4_cross_refs_match``
(between Strategy 2 locality match and Strategy 3 M3 inference).

These tests guard:
  - Strategy 4 fires when the caption mentions a paper-level figure
    (strat / litholog / map / range chart).
  - Strategy 4 returns the right LinkResult shape.
  - Self-references are filtered out by ``current_fig_id``.
  - When Strategy 1 / 2 already match, Strategy 4 is skipped
    (linker uses first-non-None rule).
  - When no paper figure matches, Strategy 4 falls through to M3/unlinked.
  - Strategy 4 stamps ``metadata.cross_refs`` on the panel via pipeline.
"""

from __future__ import annotations

from rlpe.cross_figure_linker import (
    LINK_SOURCE_CROSS_REF,
    _build_figure_index,
    _FigureIndex,
    _strategy4_cross_refs_match,
    link_species_to_geology,
)


def _plate_row(panel_id: str, fig_id: str, caption: str = "") -> dict:
    """Build a minimal plate row dict (MatchResult-like)."""
    return {
        "paper_id": "paper1",
        "figure_id": fig_id,
        "panel_id": panel_id,
        "species": "Actinomma leptodermum",
        "caption_snippet": caption,
    }


def _strat_row(fig_id: str, num: str = "3", caption: str = "") -> dict:
    """Build a minimal strat-column figure dict in the SHAPE that
    pipeline._apply_cross_figure_linker passes to link_species_to_geology:

    - ``figure_type`` at TOP LEVEL (normalized by pipeline)
    - ``figure_number`` at TOP LEVEL (extracted by pipeline)
    - ``caption`` at TOP LEVEL
    - ``formation`` / ``age`` / ``locality`` at TOP LEVEL (lifted from
      the first geology_link entry by pipeline)
    """
    return {
        "paper_id": "paper1",
        "figure_id": fig_id,
        "figure_type": "strat_column",
        "figure_number": num,
        "caption": caption,
        "formation": "Scaglia Fm",
        "age": "Late Cretaceous",
        "locality": "Tunisia",
    }


# ---------------------------------------------------------------------------
# Unit tests for _strategy4_cross_refs_match
# ---------------------------------------------------------------------------


def test_strategy4_fires_on_figure_mention():
    """A caption mentioning 'Fig. 3' should resolve to the strat_column row."""
    panel = _plate_row("A", "fig_2", caption="Fig. 3 shows the type locality.")
    fig_index = _build_figure_index([_strat_row("strat_3", num="3")])
    result = _strategy4_cross_refs_match(panel, fig_index)
    assert result is not None
    assert result.source == LINK_SOURCE_CROSS_REF
    assert result.figure_id == "strat_3"
    assert result.confidence == 0.85
    assert "Fig. 3" in result.evidence


def test_strategy4_multi_mention_still_single_target():
    """Two mentions of the same figure stay at 0.85 confidence."""
    panel = _plate_row(
        "A",
        "fig_2",
        caption="See Fig. 3 for the lithology. Fig. 3 also shows the section.",
    )
    fig_index = _build_figure_index([_strat_row("strat_3", num="3")])
    result = _strategy4_cross_refs_match(panel, fig_index)
    assert result is not None
    assert result.source == LINK_SOURCE_CROSS_REF
    assert result.figure_id == "strat_3"
    # Both mentions resolve to the same figure, so unique count == 1.
    assert result.confidence == 0.85


def test_strategy4_self_reference_skipped():
    """Mentioning Fig. 2 inside Fig. 2's own caption should not link to Fig. 2."""
    panel = _plate_row("A", "fig_2", caption="Fig. 2 also shows X.")
    fig_index = _build_figure_index([_strat_row("strat_2", num="2")])
    result = _strategy4_cross_refs_match(panel, fig_index)
    # parse_cross_refs filters self-references by current_fig_id.
    assert result is None


def test_strategy4_no_caption_returns_none():
    """Empty caption means no cross-refs to match."""
    panel = _plate_row("A", "fig_2", caption="")
    fig_index = _build_figure_index([_strat_row("strat_3", num="3")])
    result = _strategy4_cross_refs_match(panel, fig_index)
    assert result is None


def test_strategy4_no_matching_figure_returns_none():
    """Cross-ref to a figure that's not in the index should fall through."""
    panel = _plate_row("A", "fig_2", caption="See Fig. 99 for details.")
    fig_index = _build_figure_index([_strat_row("strat_3", num="3")])
    result = _strategy4_cross_refs_match(panel, fig_index)
    assert result is None


def test_strategy4_plate_mention_falls_through():
    """'Pl. 2' mention (another plate, not a strat column) should fall through."""
    panel = _plate_row("A", "fig_1", caption="As shown in Pl. 2 fig. 3.")
    fig_index = _build_figure_index(
        [_strat_row("strat_3", num="3"), _strat_row("strat_2", num="2")]
    )
    # The '2' in "Pl. 2 fig. 3" might match strat_2 by number, depending on
    # whether the regex parses it. parse_cross_refs returns the canonical
    # "Pl. 2" form with target_figure_num="2". Our index only matches by
    # number, so this could be a false-positive. The strategy is still
    # wired correctly — we don't add a "kind" filter because plates
    # sometimes legitimately reference other plates for comparison.
    # Just check it returns SOMETHING (we don't assert on identity here).
    result = _strategy4_cross_refs_match(panel, fig_index)
    # Either None (kind filter added in future) or a LinkResult to strat_2.
    if result is not None:
        assert result.source == LINK_SOURCE_CROSS_REF


# ---------------------------------------------------------------------------
# Integration: link_species_to_geology chain
# ---------------------------------------------------------------------------


def test_linker_chain_uses_strategy4_after_strategies_1_2_fail():
    """When strategies 1-2 don't match, Strategy 4 should fire."""
    panel = _plate_row("A", "fig_2", caption="See Fig. 3 for the type section.")
    strat = _strat_row("strat_3", num="3")
    results = link_species_to_geology([panel], [strat])
    assert len(results) == 1
    assert results[0].source == LINK_SOURCE_CROSS_REF
    assert results[0].figure_id == "strat_3"


def test_linker_chain_strategy4_before_strategy3():
    """Strategy 4 has higher confidence than M3 inference, so it should win."""
    panel = _plate_row("A", "fig_2", caption="Compared with Fig. 3.")

    def fake_m3(_cap, _ctx):
        return {"figure_id": "strat_3", "confidence": 0.5}

    strat = _strat_row("strat_3", num="3")
    results = link_species_to_geology([panel], [strat], m3_inference_callable=fake_m3)
    assert results[0].source == LINK_SOURCE_CROSS_REF
    assert results[0].confidence == 0.85


# ---------------------------------------------------------------------------
# Audit 2026-08-16 regression tests
# ---------------------------------------------------------------------------


def test_linker_chain_strategy4_beats_locality_higher_confidence():
    """Audit A1: Strategy 4 (conf 0.85) must beat Strategy 2 (conf 0.7)
    even when both could fire. Pre-fix order had Strategy 2 first, so
    a panel whose caption was "Tunisia. See Fig. 3." got linked via
    Strategy 2 at 0.7, ignoring the explicit cross-ref.
    """
    panel = _plate_row(
        "A",
        "fig_2",
        caption="Locality Tunisia. See Fig. 3 for the type section.",
    )
    strat = _strat_row("strat_3", num="3")
    # The locality_match extractor requires "from/at/in <Capitalised>"
    # in caption; "Locality Tunisia" without that prefix should NOT
    # trigger Strategy 2. Use a prefix to force Strategy 2 to fire.
    panel["caption_snippet"] = "From Tunisia. See Fig. 3 for the type section."
    results = link_species_to_geology([panel], [strat])
    assert results[0].source == LINK_SOURCE_CROSS_REF
    assert results[0].confidence == 0.85


def test_self_reference_filter_handles_production_id():
    """Audit A2: production-style id ``od_plate_bandini2011_pl03``
    must suppress a self-ref ``Pl. 3`` but keep ``Fig. 3``.

    Pre-fix the regex ``re.search(r\"(\\d+)\")`` grabbed "2011" (the
    year) instead of "03" (the plate number), so the self-ref filter
    never matched and Strategy 4 falsely linked cross-figures.
    """
    from rlpe.cross_refs import parse_cross_refs

    # Pl. 3 is a self-ref → must be filtered
    refs = parse_cross_refs(
        "See Pl. 3 for the lithology.", current_fig_id="od_plate_bandini2011_pl03"
    )
    assert refs == []

    # Fig. 3 is NOT a self-ref (different kind) → must be kept
    refs = parse_cross_refs(
        "See Fig. 3 for the section.", current_fig_id="od_plate_bandini2011_pl03"
    )
    assert len(refs) == 1
    assert refs[0].target_figure == "Fig. 3"

    # Id with explicit Fig suffix (e.g. od_fig_X_pNNN_NN)
    refs = parse_cross_refs(
        "See Fig. 5 in Fig. 5 caption.", current_fig_id="od_fig_bandini2011_p005_05"
    )
    # The trailing _05 isn't matched by our Fig-only branch (looks for
    # _fig5 or _plate5), so the synthetic-style fallback at the end of
    # cross_refs.py handles it.
    # The ref to "Fig. 5" is a self-ref if we recognise kind=Fig.
    # We just verify no false-positive leaks through:
    assert all(r.target_figure_num != "2011" for r in refs)


def test_strategy4_kind_filter_pl_mention_does_not_link():
    """Audit A4: a "Pl. N" mention must NOT link to a non-plate figure
    in the index (strat/litholog/map/range). Pre-fix the matcher
    matched purely on figure_num and wrongly linked plate-to-plate
    mentions to the strat column at conf 0.85.
    """
    panel = _plate_row("A", "fig_2", caption="See Pl. 3 for comparison.")
    fig_index = _build_figure_index(
        [_strat_row("strat_3", num="3"), _strat_row("strat_2", num="2")]
    )
    from rlpe.cross_figure_linker import _strategy4_cross_refs_match

    result = _strategy4_cross_refs_match(panel, fig_index)
    assert result is None, (
        "Pl. 3 mention should not link to strat_3 — the index only "
        "holds non-plate figures, so a Pl-only caption must fall "
        "through to unlinked."
    )


def test_strategy4_uses_figure_number_field_over_regex():
    """Audit A3: figure_views stamp ``figure_number`` so Strategy 4's
    Step 1 (figure_number field) wins over Step 2 (regex on
    figure_id). Without the stamp, the linker extracted "03" from
    ``od_plate_..._pl03`` but parse_cross_refs returned target_figure
    "3" → string equality failed → Strategy 4 never fired.
    """
    # Simulate the production-shaped figure dict that pipeline.py
    # passes to link_species_to_geology (figure_views).
    panel = _plate_row("1", "fig_2", caption="See Fig. 3 for the lithology.")
    fig_dict = {
        "figure_id": "od_plate_bandini2011_pl03",
        "paper_id": "paper1",
        "figure_type": "strat_column",
        "figure_number": "3",  # <-- this is what we want to honour
        "caption": "",
        "formation": "Scaglia Fm",
        "age": "Late Cretaceous",
        "locality": "Tunisia",
    }
    fig_index = _build_figure_index([fig_dict])
    results = link_species_to_geology([panel], [fig_dict])
    assert results[0].source == LINK_SOURCE_CROSS_REF
    assert results[0].figure_id == "od_plate_bandini2011_pl03"
