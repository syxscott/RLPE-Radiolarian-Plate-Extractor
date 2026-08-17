"""Phase 61 Plan 4 (Bug 4.12): extend _LIGATURE_MAP with common Latin ligatures.

The map covers U+FB00-U+FB05 (ff/fi/fl/ffi/ffl), Dutch IJ, curly
quotes/dashes, ellipsis, and NBSP — but not the Latin oe/ae/h-with-stroke
ligatures that appear in European-language species names (e.g.
Archæocyathid references, Archæan / Cœlacanth literature).

The fix extends _LIGATURE_MAP with:
  * ``œ → oe``
  * ``æ → ae``
  * ``ĥ → h``
and verifies _normalize_caption_text translates them.
"""

from __future__ import annotations

import pytest

from rlpe.m3_engine import _LIGATURE_MAP, _normalize_caption_text


def test_ligature_oe_ae_h():
    """Common Latin ligatures must be normalised to ASCII."""
    assert _normalize_caption_text("Cœlacanth") == "Coelacanth"
    assert _normalize_caption_text("Archæan") == "Archaean"
    # ĥ is rare but seen in some Latinised species epithets (the map
    # handles lowercase; capital Ĥ normalises through Unicode title
    # folding when present in the source, but we accept either
    # uppercase or lowercase input here — the spec just requires the
    # lowercase form to map).
    out = _normalize_caption_text("ĥirnant")
    # We accept either 'hirnant' (direct map) or 'Hirnant' (with the
    # capital preserved) — both are acceptable downstreams.
    assert out.lower() == "hirnant"


def test_ligature_map_contains_new_entries():
    """The map must contain the new Latin ligatures."""
    assert _LIGATURE_MAP.get("œ") == "oe"
    assert _LIGATURE_MAP.get("æ") == "ae"
    assert _LIGATURE_MAP.get("ĥ") == "h"
