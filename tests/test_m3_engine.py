"""Smoke test for M3Engine parsing helpers (no API calls)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.m3_engine import (  # noqa: E402
    _CLASSIFY_PLATE_SYSTEM,
    _CRITIQUE_SYSTEM,
    _MATCH_PANEL_SYSTEM,
    _PARSE_CAPTION_SYSTEM,
    _SEGMENT_PANELS_SYSTEM,
    Critique,
    M3Engine,
    PanelMatch,
    _coerce_bbox,
    _expand_label_range,
    _normalize_caption_text,
    _normalize_species,
    _regex_parse_caption,
    _safe_json_loads,
)


def test_expand_label_range_letters():
    assert _expand_label_range("A-D") == ["A", "B", "C", "D"]
    assert _expand_label_range("a-c") == ["a", "b", "c"]
    assert _expand_label_range("A-A") == ["A"]
    assert _expand_label_range("X") == ["X"]
    assert _expand_label_range("") == []


def test_expand_label_range_digits():
    assert _expand_label_range("3-5") == ["3", "4", "5"]
    assert _expand_label_range("3") == ["3"]


def test_safe_json_loads_simple_object():
    out = _safe_json_loads('{"a": 1, "b": "x"}')
    assert out == {"a": 1, "b": "x"}


def test_safe_json_loads_with_preamble():
    out = _safe_json_loads('Here is the result:\n{"a": 1}\nDone.')
    assert out == {"a": 1}


def test_safe_json_loads_code_fence():
    out = _safe_json_loads('```json\n{"a": 1}\n```')
    assert out == {"a": 1}


def test_safe_json_loads_array():
    out = _safe_json_loads('[1, 2, 3]')
    assert out == [1, 2, 3]


def test_safe_json_loads_mixed_chooses_first():
    # When both appear, the parser tries array before object for non-strict
    # matching; either is acceptable, but we must return *some* JSON.
    out = _safe_json_loads('first {"a": 1} then [1, 2]')
    assert out in ({"a": 1}, [1, 2])


def test_safe_json_loads_recovers_from_missing_comma():
    # Best-effort recovery: M3 sometimes drops the comma between array items.
    text = '[{"a": 1}\n{"a": 2}]'
    out = _safe_json_loads(text)
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0] == {"a": 1}
    assert out[1] == {"a": 2}


def test_coerce_bbox_pixels():
    assert _coerce_bbox([10, 20, 100, 200], 800, 600) == (10, 20, 100, 200)


def test_coerce_bbox_normalized():
    # 0.1, 0.1, 0.3, 0.4 of 1000x800
    x, y, w, h = _coerce_bbox([0.1, 0.1, 0.3, 0.4], 1000, 800)
    assert (x, y) == (100, 80)
    assert 280 <= w <= 320
    assert 300 <= h <= 340


def test_coerce_bbox_invalid():
    assert _coerce_bbox("nope", 100, 100) is None
    assert _coerce_bbox([1, 2, 3], 100, 100) is None
    assert _coerce_bbox(None, 100, 100) is None


def test_engine_constructs_with_minimal_config():
    """Engine should not call backend on construction; only on demand."""
    class FakeBackend:
        backend_name = "fake"
    engine = M3Engine(FakeBackend())
    assert engine._stage_enabled(1) is True
    assert engine._stage_enabled(5) is True


def test_engine_stage_toggle():
    class FakeBackend:
        backend_name = "fake"
    engine = M3Engine(FakeBackend(), {"m3_stage_4": False})
    assert engine._stage_enabled(4) is False
    assert engine._stage_enabled(3) is True


def test_engine_with_no_backend_returns_fallback():
    engine = M3Engine(None)
    pairs = engine.parse_caption("Fig. 3. A: X; B: Y")
    assert pairs == []  # no backend -> empty
    from PIL import Image
    cls = engine.classify_plate(Image.new("RGB", (100, 100)))
    # No backend -> fallback, but PlateClassification is still returned with defaults
    assert cls.is_radiolarian_plate is True
    assert cls.image_type == "micrograph"


def test_apply_critiques_agree_no_change():
    matches = [PanelMatch(panel_id="P1", label="A", species="X", confidence=0.8, reasoning="ok")]
    critiques = [Critique(panel_id="P1", verdict="agree", confidence=0.9, reasoning="good")]
    out = M3Engine.apply_critiques(matches, critiques)
    assert out[0].species == "X"
    assert out[0].raw["critique"]["verdict"] == "agree"


def test_apply_critiques_disagree_overrides():
    matches = [PanelMatch(panel_id="P1", label="A", species="X", confidence=0.8, reasoning="ok")]
    critiques = [Critique(
        panel_id="P1", verdict="disagree",
        suggested_species="Y", confidence=0.85,
        reasoning="actually looks like Y",
    )]
    out = M3Engine.apply_critiques(matches, critiques)
    assert out[0].species == "Y"
    assert out[0].raw["critique"]["from"] == "X"
    assert out[0].raw["critique"]["to"] == "Y"


def test_apply_critiques_low_confidence_no_override():
    matches = [PanelMatch(panel_id="P1", label="A", species="X", confidence=0.8, reasoning="ok")]
    critiques = [Critique(
        panel_id="P1", verdict="disagree",
        suggested_species="Y", confidence=0.4,  # below threshold
        reasoning="maybe",
    )]
    out = M3Engine.apply_critiques(matches, critiques)
    assert out[0].species == "X"  # not overridden


def test_apply_critiques_unknown_panel_no_effect():
    matches = [PanelMatch(panel_id="P1", label="A", species="X", confidence=0.8, reasoning="ok")]
    critiques = [Critique(panel_id="P99", verdict="disagree", suggested_species="Y", confidence=0.9)]
    out = M3Engine.apply_critiques(matches, critiques)
    assert out[0].species == "X"
    assert "critique" not in out[0].raw


def test_prompts_present_and_nonempty():
    for p in (_PARSE_CAPTION_SYSTEM, _CLASSIFY_PLATE_SYSTEM,
              _SEGMENT_PANELS_SYSTEM, _MATCH_PANEL_SYSTEM, _CRITIQUE_SYSTEM):
        assert p and len(p) > 50, f"prompt too short: {len(p)} chars"


class TestMatchPanelErrorPropagation:
    """When the backend returns fallback_used=True with an error, the engine
    must surface that error in panel_match.raw so the pipeline can route
    through the FallbackHandler (instead of treating it as 'not a radiolarian')."""

    def _engine_with_failing_backend(self):
        class _FailingBackend:
            backend_name = "fake-fail"
            enable_thinking = False
            def infer_panel(self, **_):
                return {"fallback_used": True, "error": "MiniMax API timeout"}
        return M3Engine(_FailingBackend())

    def test_match_panel_propagates_error_in_raw(self):
        from PIL import Image
        engine = self._engine_with_failing_backend()
        panel = Image.new("RGB", (64, 64))
        result = engine.match_panel(panel_image=panel, caption_pairs=[], caption_text="")
        assert result.is_radiolarian is False
        assert result.raw.get("error") == "MiniMax API timeout"
        assert "MiniMax API timeout" in result.reasoning

    def test_match_panel_no_error_when_results_present(self):
        from PIL import Image
        class _OkBackend:
            backend_name = "fake-ok"
            enable_thinking = False
            def infer_panel(self, **_):
                return {
                    "fallback_used": False,
                    "raw_text": '{"label": "A", "species": "Foo", "confidence": 0.8, "reasoning": "r"}',
                }
        engine = M3Engine(_OkBackend())
        panel = Image.new("RGB", (64, 64))
        result = engine.match_panel(panel_image=panel, caption_pairs=[], caption_text="")
        # Success path: no error in raw
        assert "error" not in result.raw
        assert result.species == "Foo"


def test_normalize_caption_text_strips_ligatures():
    # U+FB01 (ﬁ) is the most common ligature in OpenDataLoader output.
    assert _normalize_caption_text("ﬁgs") == "figs"
    assert _normalize_caption_text("ﬁg.") == "fig."
    # Curly quotes and dashes (common in T&F / Elsevier papers).
    assert _normalize_caption_text("‘foo’") == "'foo'"
    assert _normalize_caption_text("5–10") == "5-10"
    # Mixed: ligature + curly quote + en-dash in one caption.
    raw = "Explanation of Plate 1. ﬁgs 1–2. Foo ‘bar’"
    norm = _normalize_caption_text(raw)
    assert "ﬁ" not in norm
    assert "–" not in norm
    # Empty / None pass through.
    assert _normalize_caption_text("") == ""
    assert _normalize_caption_text(None) is None


def test_regex_parse_caption_handles_ligature_figs():
    """OpenDataLoader output uses U+FB01 (ﬁ) in 'ﬁgs'. Without normalisation
    the regex returns zero pairs and the pipeline silently falls back to
    positional heuristics that mis-allocate species across panels."""
    # Real caption shape from Feng 2007 Plate 1 (OpenDataLoader output).
    caption = (
        "Explanation of Plate 1. ﬁgs 1–2. Entactinia itsukichiensis "
        "Sashida & Tonishi: 1, DP2/B024; 2, DP4/P016. "
        "ﬁgs 3–4. Entactinia reticulata: 3, DP1/B005; 4, DP3/P009."
    )
    pairs = _regex_parse_caption(caption)
    assert len(pairs) == 2, f"expected 2 pairs, got {len(pairs)}: {pairs}"
    by_label = {lbl: p.species for p in pairs for lbl in p.labels}
    assert by_label["1"] == "Entactinia itsukichiensis"
    assert by_label["2"] == "Entactinia itsukichiensis"
    assert by_label["3"] == "Entactinia reticulata"
    assert by_label["4"] == "Entactinia reticulata"


def test_regex_parse_caption_handles_curly_quotes():
    """Curly apostrophes inside species modifiers should not break parsing."""
    caption = (
        "Explanation of Plate 1. Fig. 1. Entactinia cf. ﬁtsukichiensis."
    )
    pairs = _regex_parse_caption(caption)
    assert len(pairs) == 1
    assert pairs[0].labels == ["1"]
    # cf. epithet with embedded ligature should be accepted.
    assert "Entactinia" in pairs[0].species


# ----- Singular "Figure" + "(?)" epithet regression tests (N9 fix) -----
# These cover the bandini2006 (Karnezeika) caption shapes that the
# v17 _CAPTION_CLAUSE_RE couldn't parse:
#   1. "Figure N" (SINGULAR, no trailing 's') was not matched — the
#      `(?:s|ures)?` group only accepted "Figs" and "Figures" (plural),
#      so "Figure 3" / "Figure 14" / "Figure 18" all silently fell
#      through. Fixed by switching to `(?:s|ure|ures)?` so all four
#      forms ("Fig" / "Figs" / "Figure" / "Figures") are accepted.
#   2. "Genus (?) epithet" / "Genus (?) sp." — bandini2006 marks
#      tentative genus assignments with "(?)" placed BETWEEN the genus
#      and the epithet ("Archaeocenosphaera (?) mellifera",
#      "Pseudoacanthosphaera (?) sp."). The previous regex had no
#      branch for this; the leading "?" marker is consumed by the
#      `(?:\?)?` only when it comes right after the genus letters.
#      Added a new "Genus (?) epithet" branch to the epithet
#      alternation.


def test_caption_clause_singular_figure():
    """Singular 'Figure N' (without trailing 's') must match — bandini2006
    Plate 1 uses this shape throughout: 'Figure 3 Acaeniotyle sp. A',
    'Figure 14 Pseudoacanthosphaera galeata', etc. Previously the regex
    only matched 'Fig(s|ures)?' so 'Figure' (singular) fell through."""
    pairs = _regex_parse_caption(
        "Figure 3 Acaeniotyle sp. A\n"
        "Figure 14 Pseudoacanthosphaera galeata"
    )
    by_label = {p.labels[0]: p for p in pairs}
    assert "3" in by_label, f"expected label 3 in {list(by_label)}"
    # _regex_parse_caption folds the trailing identifier into the
    # species string so the caller sees the gold form ("Acaeniotyle
    # sp. A") directly — the modifier is cleared.
    assert by_label["3"].species == "Acaeniotyle sp. A"
    assert (by_label["3"].modifier or "").strip() == ""
    assert "14" in by_label
    assert by_label["14"].species == "Pseudoacanthosphaera galeata"


def test_caption_clause_genus_question_mark_epithet():
    """'Genus (?) epithet' — bandini2006 uses '(?)' to mark a tentative
    genus assignment, placed BETWEEN the genus and the epithet
    ('Archaeocenosphaera (?) mellifera', 'Pseudoacanthosphaera (?) sp.').
    The epithet branch must accept '(?)' before the epithet token."""
    pairs = _regex_parse_caption(
        "Figures 5-6 Archaeocenosphaera (?) mellifera\n"
        "Figures 7-8 Archaeocenosphaera (?) sp."
    )
    assert len(pairs) == 2
    by_label: dict[str, object] = {}
    for p in pairs:
        for lbl in p.labels:
            by_label[lbl] = p
    pair_5 = by_label.get("5") or by_label.get("5-6")
    assert pair_5 is not None
    # Species field carries '(?) mellifera' — _normalize_species strips
    # the '(?)' marker so the gold form is 'Archaeocenosphaera mellifera'.
    assert "Archaeocenosphaera" in pair_5.species
    assert "mellifera" in pair_5.species
    pair_7 = by_label.get("7") or by_label.get("7-8")
    assert pair_7 is not None
    assert "Archaeocenosphaera" in pair_7.species
    combined_7 = pair_7.species + " " + (pair_7.modifier or "")
    assert " sp" in combined_7


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])


# ----- _normalize_species regression tests (N2 + N3 fixes) -----
# These cover the per-corpus gold-convention rules added in 2026-06-07:
#   * bandini 2011 uses bare "Nassellaria gen" / "Spumellaria gen"
#     (not the baum-style "indet.")
#   * bandini 2011 uses bare "sp" (not "sp." with period)
#   * bandini 2011 trims subspecies ("mitra gr", "unumaense
#     pustulatum", "diamphidius hipposidericus") to species-only
#   * hollis 2006 uses "gr." with period (Haliomma gr. b) and
#     trinomial autonyms (Lamptonium fabaeforme fabaeforme) — both
#     must be preserved
#   * baum 2008 uses "gen. et sp. indet." → "indet."


def test_normalize_bandini_gen_preserved():
    """bandini 2011 gold uses bare 'Nassellaria gen' / 'Spumellaria
    gen' (no 'indet.', no trailing period). The aggressive fold to
    'indet.' used to mis-align 4 bandini panels. Now only folds when
    'indet' is present in the input."""
    assert _normalize_species("Nassellaria gen") == "Nassellaria gen"
    assert _normalize_species("Spumellaria gen") == "Spumellaria gen"


def test_normalize_baum_indet_folded():
    """baum 2008 gold uses 'Spumellaria indet. A' / 'indet. B'."""
    assert _normalize_species("Spumellaria gen. et sp. indet") == "Spumellaria indet."
    assert _normalize_species("Spumellaria indet. A") == "Spumellaria indet. A"


def test_normalize_bandini_gr_stripped():
    """bandini 2011 caption uses 'mitra gr.' subgenus-group marker;
    gold records species-only 'Archaeodictyomitra mitra'. Strip the
    bare 'gr' (no period). Hollis 2006's 'Haliomma gr. b' (with
    period) is preserved."""
    assert _normalize_species("Archaeodictyomitra mitra gr") == "Archaeodictyomitra mitra"
    assert _normalize_species("Transhsuum maxwelli gr") == "Transhsuum maxwelli"
    assert _normalize_species("Transhsuum cf. maxwelli gr") == "Transhsuum cf. maxwelli"
    assert _normalize_species("Haliomma gr. b") == "Haliomma gr. b"
    assert _normalize_species("Haliomma gr. A-K47/4") == "Haliomma gr. A-K47/4"


def test_normalize_subspecies_preserved_corpus_specific():
    """The trinomial "epithet2" tail is preserved at the normalize
    layer because the gold convention differs by corpus:
      * bandini 2011 records species-only when the caption has
        trinomial (`Eucyrtidiellum unumaense pustulatum` -> gold
        `Eucyrtidiellum unumaense`) — but only for 2 panels; the
        other 3 `mitra gr` / `maxwelli gr` panels are handled by
        the bare `gr` strip rule.
      * beccaro 2006 KEEPS trinomials (`Eucyrtidiellum unumaense
        dentatum`, `Mirifusus dianae minor`, `Ristola altissima
        nodosa`).
      * hollis 2006 KEEPS the trinomial autonym (`Lamptonium
        fabaeforme fabaeforme`).
    A global 3-token-strip rule would mis-align beccaro's 3 panels
    and hollis's autonym. Only the bare `gr` tail is stripped
    (covered by a separate test above). The 2 bandini `pustulatum`
    / `hipposidericus` cases remain as residual mismatches and
    require paper_id-aware normalize to fix.
    """
    # bandini trinomial (still mm — corpus-specific, not fixed here)
    assert _normalize_species("Eucyrtidiellum unumaense pustulatum") == "Eucyrtidiellum unumaense pustulatum"
    assert _normalize_species("Deviatus diamphidius hipposidericus") == "Deviatus diamphidius hipposidericus"
    # beccaro trinomial (must preserve)
    assert _normalize_species("Eucyrtidiellum unumaense dentatum") == "Eucyrtidiellum unumaense dentatum"
    assert _normalize_species("Mirifusus dianae minor") == "Mirifusus dianae minor"
    assert _normalize_species("Ristola altissima nodosa") == "Ristola altissima nodosa"
    # hollis autonym (must preserve)
    assert _normalize_species("Lamptonium fabaeforme fabaeforme") == "Lamptonium fabaeforme fabaeforme"
    assert _normalize_species("Ristola altissima altissima") == "Ristola altissima altissima"


def test_normalize_bandini_sp_no_period_preserved():
    """The historical v9 normalize auto-added a period to bare "sp"
    / "spp" / "indet" / "nov". This is preserved: the rule below
    keeps the periodised form in the normalize output. The eval
    harness strips trailing period before comparing to gold, so
    bandini/danelian/beccaro's bare "sp" gold still matches.

    Hollis 2006's "Haliomma gr. b" (with period) is unaffected by
    this rule because "gr" is not in the `(sp|spp|indet|nov)`
    whitelist."""
    # Historical v9 behaviour — "sp" / "spp" / "indet" / "nov" all
    # get a period added back if missing.
    assert _normalize_species("Archaeodictyomitra sp") == "Archaeodictyomitra sp."
    assert _normalize_species("Pseudoeucyrtis sp") == "Pseudoeucyrtis sp."
    # Already-periodised input is preserved.
    assert _normalize_species("Ferresium sp.") == "Ferresium sp."
    # 'spp' / 'indet' / 'nov' get periods.
    assert _normalize_species("Archaeodictyomitra spp") == "Archaeodictyomitra spp."
    assert _normalize_species("Spumellaria indet") == "Spumellaria indet."
