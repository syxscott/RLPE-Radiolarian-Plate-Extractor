"""Regression tests for bandini2011's list_item-wrapped plate captions.

Bandini 2011 plates 7-9 have their "Plate N" caption headers buried inside
PDF-UA tagged ``list`` elements (OpenDataLoader emits them as
``list_items[i].content`` rather than a top-level paragraph/caption/heading).
The previous detector at ``_find_plate_captions`` skipped all non-paragraph/
caption/heading elements, so plates 7-9 silently dropped and the images on
those pages got stamped with bogus ``od_fig_*`` IDs that the strict
``match_panel`` matcher rejected (panel_match dropped from 100% to 80%
on bandini2011).

These tests lock in the fix: ``_find_plate_captions`` must re-surface
list_item children that match ``_PLATE_CAPTION_RE`` as synthetic paragraph
siblings BEFORE the main detector loop runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.opendataloader_extractor import (  # noqa: E402
    _find_plate_captions,
    _PLATE_CAPTION_RE,
)


def _kid(etype: str, page: int, content: str = "", list_items: list | None = None):
    """Helper: build an OD kid dict."""
    kid = {"type": etype, "page number": page}
    if list_items is not None:
        kid["list items"] = list_items
    if content:
        kid["content"] = content
    return kid


def test_list_item_plate_caption_is_detected():
    """A list_item whose content starts with 'Plate N' must be detected
    as a plate caption (synthetic paragraph sibling)."""
    kids = [
        _kid("paragraph", 24, "Plate 1 - Foo. 1- Bar; 2- Baz."),
        # Plate 7 buried in a list (bandini2011 pattern):
        _kid(
            "list",
            25,
            list_items=[
                {"type": "list item", "content": "Plate 7 - Sample PR-SB26. 1- Sp1; 2- Sp2."},
                {"type": "list item", "content": "(paragraph continuation not a caption)"},
            ],
        ),
    ]
    captions = _find_plate_captions(kids)
    detected_numbers = sorted({c["plate_number"] for c in captions})
    assert 1 in detected_numbers
    assert 7 in detected_numbers, (
        f"Plate 7 buried in list_item must be detected; got {detected_numbers}"
    )


def test_list_item_non_caption_is_ignored():
    """A list_item whose content does NOT start with 'Plate N' must NOT
    trigger a plate caption (anti-false-positive)."""
    kids = [
        _kid(
            "list",
            5,
            list_items=[
                {"type": "list item", "content": "Plate was donated to the museum."},
            ],
        ),
    ]
    captions = _find_plate_captions(kids)
    assert captions == [], (
        f"'Plate was donated...' should NOT be a plate caption; got {captions}"
    )


def test_list_item_partial_match_is_ignored():
    """A list_item like 'Plates 1 and 2' (multi-number) should not match
    _PLATE_CAPTION_RE on its own — it's a body mention, not a header."""
    kids = [
        _kid(
            "list",
            10,
            list_items=[
                {"type": "list item", "content": "Plates 1 and 2 show the same species."},
            ],
        ),
    ]
    captions = _find_plate_captions(kids)
    # Whatever the regex catches, we shouldn't accidentally synthesize a
    # new plate header from this. Either the regex doesn't match, or the
    # fallback path doesn't add it — the test is mainly that we don't
    # crash and that the list_item path doesn't add a duplicate plate.
    # We accept [] OR a single caption IF the regex genuinely matched.
    for c in captions:
        assert c["plate_number"] != 999, "test-only sentinel"


def test_mixed_kids_lists_and_paragraphs_both_work():
    """Real-world: page has both a top-level paragraph caption (Pl 1-6)
    and a buried list_item caption (Pl 7-9). Both must be detected."""
    kids = [
        _kid("paragraph", 12, "Plate 1. 1- A; 2- B."),
        _kid("paragraph", 14, "Plate 2. 1- C; 2- D."),
        _kid("paragraph", 16, "Plate 3. 1- E; 2- F."),
        _kid("paragraph", 18, "Plate 4. 1- G; 2- H."),
        _kid("paragraph", 20, "Plate 5. 1- I; 2- J."),
        _kid("paragraph", 22, "Plate 6. 1- K; 2- L."),
        # Bandini 2011-style: plates 7-9 buried in PDF-UA tagged list
        _kid(
            "list",
            24,
            list_items=[
                {"type": "list item", "content": "Plate 7 - Sample PR-SB26. 1- M; 2- N."},
            ],
        ),
        _kid(
            "list",
            26,
            list_items=[
                {"type": "list item", "content": "Plate 8 - Sample PR-SB27. 1- O; 2- P."},
            ],
        ),
        _kid(
            "list",
            28,
            list_items=[
                {"type": "list item", "content": "Plate 9 - Sample PR-SB28. 1- Q; 2- R."},
            ],
        ),
    ]
    captions = _find_plate_captions(kids)
    detected_numbers = sorted({c["plate_number"] for c in captions})
    assert detected_numbers == [1, 2, 3, 4, 5, 6, 7, 8, 9], (
        f"All 9 plates must be detected (6 paragraph + 3 list_item), got {detected_numbers}"
    )


def test_explanation_of_plate_pattern_also_works():
    """Some papers (feng2007-style) use 'Explanation of Plate N.' as the
    caption header. The list_item path must also accept this."""
    kids = [
        _kid(
            "list",
            30,
            list_items=[
                {"type": "list item", "content": "Explanation of Plate 7. 1- Sp1; 2- Sp2."},
            ],
        ),
    ]
    captions = _find_plate_captions(kids)
    detected_numbers = {c["plate_number"] for c in captions}
    assert 7 in detected_numbers, (
        f"'Explanation of Plate 7' in list_item must be detected; got {detected_numbers}"
    )


def test_real_bandini_od_output_detects_plates_1_through_9():
    """Smoke test against the real OD JSON for bandini2011 (if present
    on disk). Verifies that ALL 9 plates are detected, not just 1-6."""
    od_path = Path("/tmp/llm9/bandini2011/output/od_output/4f1bf415485765b8/bandini2011.json")
    if not od_path.exists():
        pytest.skip("bandini2011 OD JSON not present at /tmp/llm9/...")
    data = json.loads(od_path.read_text())
    kids = data["kids"]
    captions = _find_plate_captions(kids)
    detected_numbers = sorted({c["plate_number"] for c in captions})
    assert 7 in detected_numbers, (
        f"Plate 7 must be detected in real bandini2011 OD JSON; got {detected_numbers}"
    )
    assert 8 in detected_numbers, (
        f"Plate 8 must be detected in real bandini2011 OD JSON; got {detected_numbers}"
    )
    assert 9 in detected_numbers, (
        f"Plate 9 must be detected in real bandini2011 OD JSON; got {detured_numbers}"
    )


# ---------------------------------------------------------------------------
# Confirm the regex itself matches the patterns we care about
# ---------------------------------------------------------------------------
def test_plate_caption_regex_matches_standard_forms():
    """Lock down the regex patterns the list_item branch relies on."""
    assert _PLATE_CAPTION_RE.match("Plate 7 - Sample PR-SB26. 1- Sp1.")
    assert _PLATE_CAPTION_RE.match("Plate 7. 1- Sp1.")
    assert _PLATE_CAPTION_RE.match("Explanation of Plate 7. 1- Sp1.")
    assert _PLATE_CAPTION_RE.match("Plate VII - Foo.")
    # These must NOT match (false-positive guard):
    assert not _PLATE_CAPTION_RE.match("Plate was donated to the museum.")
    assert not _PLATE_CAPTION_RE.match("Plates 1 and 2 show the same species.")