"""Round 19 source-guard tests: fix the expert-flagged errors.

The user reported that an expert reviewed the extracted geology
data and found "everything wrong". The audit identified 3 root
causes, each of which produced fabricated per-panel data:

  1. GREEDY FORMATION REGEX
     Old pattern ``[A-Z][A-Za-z\\-]+(?:\\s+[A-Z][A-Za-z\\-]+){0,3}``
     would walk across sentence boundaries and capture nonsense
     like "RAM is the Fonzaso Formation" instead of "Fonzaso
     Formation". The fix uses a non-greedy ``{0,30}?`` and an
     explicit trailing word-boundary.

  2. SHARED PER-PANEL DATA
     All 27 panels of a Beccaro paper received the SAME
     figure-level geology because the panel_captions dict was
     populated with the same figure-level caption for every
     panel. Only the first panel should anchor the figure-level
     data; the rest should get empty lists + ``geology_scope=
     'none'`` so the operator sees a data gap rather than
     fabricated content.

  3. _is_placeholder_caption vs _looks_like_placeholder_caption
     Typo in the caller-side fix — the actual function name in
     ``text_filters`` is ``looks_like_placeholder_caption``. Using
     the wrong name raised NameError and silently fell back to
     the "first panel gets everything" behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _read(path: str) -> str:
    return Path(__file__).resolve().parents[1].joinpath(path).read_text(encoding="utf-8")


# --- 1) Greedy formation regex -----------------------------------------


def test_formation_regex_is_not_greedy():
    """The formation regex must NOT cross sentence boundaries. Old
    regex matched "RAM is the Fonzaso Formation" instead of
    "Fonzaso Formation" because the inner ``[A-Z][A-Za-z\\-]+``
    allowed greedy word-spanning."""
    src = _read("src/rlpe/geology_extraction.py")
    # Locate the three rank-specific regexes
    for pat in ("_GROUP_RE", "_FORMATION_RE", "_MEMBER_RE"):
        idx = src.find(f"{pat} = re.compile")
        assert idx > 0, f"{pat} missing"
        # Take the next 200 chars to see the pattern
        window = src[idx : idx + 200]
        # The new pattern must use non-greedy quantifier ``{0,30}?``
        # OR ``{0,N}?`` to prevent the old behaviour.
        assert "{0," in window and "}?" in window, (
            f"{pat} must use non-greedy quantifier (e.g. '{{0,30}}?') "
            f"to stop at the first lowercase word or sentence end. "
            f"Found: {window!r}"
        )


def test_formation_regex_does_not_match_across_sentence():
    """End-to-end: feeding the actual Beccaro text must produce
    'Fonzaso Formation' (not 'RAM is the Fonzaso Formation')."""
    from rlpe.geology_extraction import _FORMATION_RE

    text = (
        "In Western Sicily five stratigraphic sections of the Rosso "
        "Ammonitico Medio (RAM) have been studied in the Trapanese "
        "Domain. The Fonzaso Formation is siliceous limestone."
    )
    matches = [m.group(1) for m in _FORMATION_RE.finditer(text)]
    assert matches == ["Fonzaso Formation"], (
        f"Formation regex still matches across sentence boundaries. "
        f"Got: {matches!r}"
    )


# --- 2) Per-panel shared data ------------------------------------------


def test_caller_assigns_geology_scope():
    """The pipeline caller in ``_process_region`` and the
    ``_enrich_llm_first_results`` path must stamp ``geology_scope``
    on every panel so the operator can distinguish figure-level
    data from panel-specific data from empty (no info)."""
    src = _read("src/rlpe/pipeline.py")
    # Both call sites must set the scope. The function bodies are
    # large (>5000 chars), so we look at the section between the
    # function header and the next ``def `` at the same indent.
    for call in ("_process_region", "_enrich_llm_first_results"):
        idx = src.find(f"def {call}")
        assert idx > 0, f"{call} missing"
        # Find the next "def " at column 0 (next top-level function).
        next_def = src.find("\n    def ", idx + 10)
        if next_def < 0:
            next_def = idx + 10000
        window = src[idx:next_def]
        assert '"geology_scope"' in window or "'geology_scope'" in window, (
            f"{call} does not stamp geology_scope. Without the scope "
            f"marker, the operator can't tell figure-level data from "
            f"panel-specific data from empty."
        )
        # Must distinguish "panel" / "figure_anchor" / "none"
        for scope in ('"panel"', '"figure_anchor"', '"none"'):
            assert scope in window, (
                f"{call} missing scope variant {scope}. Need all three "
                f"to differentiate the data provenance."
            )


def test_first_panel_only_inherits_figure_level():
    """Only the first panel (i==0) should inherit figure-level
    geology as 'figure_anchor'. The rest get empty + 'none'."""
    src = _read("src/rlpe/pipeline.py")
    for call in ("_process_region", "_enrich_llm_first_results"):
        idx = src.find(f"def {call}")
        next_def = src.find("\n    def ", idx + 10)
        if next_def < 0:
            next_def = idx + 10000
        window = src[idx:next_def]
        # Look for the canonical "i == 0 and panel_local_geo" guard
        assert "i == 0 and panel_local_geo" in window, (
            f"{call} missing the i==0 figure-anchor guard. Without "
            f"it, every panel gets the figure-level data and we "
            f"fabricate per-panel records."
        )


# --- 3) Placeholder function name ---------------------------------------


def test_placeholder_function_name_correct():
    """The actual helper in ``text_filters`` is
    ``looks_like_placeholder_caption`` (not
    ``_is_placeholder_caption``). The pipeline caller must import
    the correct name or the panel-scope check NameErrors and
    silently falls back to fabricating data."""
    src = _read("src/rlpe/pipeline.py")
    # Must NOT use the wrong name
    assert "_is_placeholder_caption(" not in src, (
        "pipeline.py uses _is_placeholder_caption which doesn't exist. "
        "The actual function is looks_like_placeholder_caption in "
        "text_filters. This typo made every panel get the figure-"
        "level data because the guard NameError'd out."
    )
    # Must import the correct name
    assert "looks_like_placeholder_caption" in src, (
        "pipeline.py doesn't import looks_like_placeholder_caption "
        "from text_filters. The placeholder detection is silently "
        "broken."
    )


# --- 4) Schema includes geology_scope ------------------------------------


def test_panel_metadata_has_geology_scope():
    """``PanelMetadata.geology_scope`` must be declared so the
    converter can pass it through."""
    src = _read("src/rlpe/schema_models.py")
    assert "geology_scope:" in src, (
        "PanelMetadata missing geology_scope field. The expert "
        "reviewer can't see provenance without this marker."
    )


def test_converter_passes_geology_scope():
    """The converter must thread ``geology_scope`` from
    ``metadata`` into the published PanelMetadata."""
    src = _read("src/rlpe/converters.py")
    assert "geology_scope" in src, (
        "converters.py doesn't pass geology_scope through to "
        "PanelMetadata. The expert auditor's tool would see all "
        "data as 'figure-level' with no scope marker."
    )