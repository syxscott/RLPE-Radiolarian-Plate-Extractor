"""Phase 62 Plan 5 (Bug 5.12): TEI caption dedup by lowercased text.

``tei.get_figure_caption`` walks the three child elements
``figDesc``, ``head``, ``note`` and concatenates their text with a
single space. Some GROBID versions populate two or all three of
these with the same caption text (the GROBID schema allows it),
so the returned caption doubles or triples the same sentence.

Example failure mode: GROBID returns the figure's caption in
both ``<head>`` and ``<note>`` — the caption is emitted as
``"Figure 1. Caption text Figure 1. Caption text"``, doubling it.
The figure-type classifier then sees a caption whose word
frequency is dominated by the duplicated phrase, biasing its
classification toward "Figure"-type (because "Figure 1" appears
twice) and away from stratigraphic-column / litholog / range-
chart types.

The fix: track the lowercased text of each appended part and
skip any subsequent part whose lowercased text matches a
previously-seen one. Whitespace-only differences are normalised
so ``"Caption text"`` and ``"  caption text  "`` are treated as
the same.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from rlpe.tei import get_figure_caption


def _make_fig_with_children(*children_text_by_tag: tuple[str, str]) -> ET.Element:
    """Build a <figure> with the given (tag, text) children.

    Example::
        _make_fig_with_children(("head", "Figure 1. Caption"),
                                ("note", "Figure 1. Caption"))
    """
    fig = ET.Element("figure")
    for tag, text in children_text_by_tag:
        node = ET.SubElement(fig, tag)
        node.text = text
    return fig


def test_caption_dedup_identical_head_and_note():
    """head and note carry the same text — must be deduplicated."""
    text = "Figure 1. Late Triassic radiolarians from Sicily."
    fig = _make_fig_with_children(("head", text), ("note", text))
    out = get_figure_caption(fig, root=fig)
    # The text must appear only once.
    assert out.count("Late Triassic radiolarians from Sicily") == 1, (
        f"caption not deduplicated; got: {out!r}"
    )


def test_caption_dedup_case_insensitive():
    """'Figure 1. Caption' and 'figure 1. caption' are treated as
    the same caption."""
    fig = _make_fig_with_children(
        ("head", "Figure 1. Caption"),
        ("note", "figure 1. caption"),
    )
    out = get_figure_caption(fig, root=fig)
    # Either way the resulting string contains "caption" once or
    # twice if the case difference produced a different rendered
    # form. The DEDUP applies to lowercased text, so the second
    # occurrence is skipped — the output should be just "Figure 1.
    # Caption".
    assert out == "Figure 1. Caption", f"expected deduped, got: {out!r}"


def test_caption_dedup_whitespace_insensitive():
    """Whitespace differences are ignored — '  caption text  ' and
    'caption text' are the same."""
    fig = _make_fig_with_children(
        ("head", "  Caption text  "),
        ("note", "Caption text"),
    )
    out = get_figure_caption(fig, root=fig)
    # The dedup normalises whitespace, so the second occurrence is
    # skipped. Output is the cleaned first occurrence.
    assert out.strip() == "Caption text"
    assert out.count("Caption text") == 1


def test_caption_distinct_parts_kept():
    """Regression: head and note with DIFFERENT texts are both kept."""
    fig = _make_fig_with_children(
        ("head", "Figure 1."),
        ("note", "Scale bar = 100 µm."),
    )
    out = get_figure_caption(fig, root=fig)
    # Both parts should appear in the output.
    assert "Figure 1." in out
    assert "Scale bar = 100 µm." in out


def test_caption_only_figdesc():
    """Regression: only figDesc is present (no dedup needed)."""
    fig = _make_fig_with_children(("figDesc", "Caption from figDesc"))
    out = get_figure_caption(fig, root=fig)
    assert out == "Caption from figDesc"


def test_caption_empty():
    """Empty figure — no children — returns empty string."""
    fig = ET.Element("figure")
    out = get_figure_caption(fig, root=fig)
    assert out == ""
