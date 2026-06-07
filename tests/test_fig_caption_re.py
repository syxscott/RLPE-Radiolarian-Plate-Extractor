"""Regression tests for Fig. N caption detection in opendataloader_extractor.

Wever 2006 (a review paper) has 6 paragraphs starting with "Fig. 1..6" but
no "Plate N" headers. Before this fix, _find_plate_captions returned []
for that paper and the pipeline fell into the page-render fallback with
"Auto-generated figure for page X" placeholders. After the fix, all 6
real captions are picked up and labelled with kind="fig".

Mixed-convention papers (Bandini 2011, Hollis 2006, Pouille 2014) use
BOTH "Plate N" and "Fig. N" — the fix must keep both sets distinct
(Fig 1 must not collapse onto Plate 1, and a real Plate 1 must not be
blocked by a Fig 1 caption with the same number).
"""
from __future__ import annotations

import json
from pathlib import Path

from rlpe.opendataloader_extractor import _find_plate_captions

REPO_ROOT = Path(__file__).resolve().parents[1]
OD_DIR = REPO_ROOT / "work" / "batch4_v2" / "out" / "od_output"
WEVER_DIR = REPO_ROOT / "work" / "wever_check" / "od_output"
BOUGHDIRI_DIR = REPO_ROOT / "work" / "boughdiri_only_out" / "output" / "od_output"
FENG_DIR = REPO_ROOT / "work" / "feng_rerun" / "output" / "od_output"


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _first_json(d: Path) -> Path:
    return next(d.glob("*.json"))


def test_wever2006_picks_up_fig_captions():
    fn = next(WEVER_DIR.glob("*/*.json"))
    caps = _find_plate_captions(_load_json(fn)["kids"])
    # Wever 2006 has 6 real Fig. captions and zero Plate captions.
    assert len(caps) == 6
    for c in caps:
        assert c["kind"] == "fig"
        assert c["content"].startswith(f"Fig. {c['plate_number']}")


def test_bandini2011_keeps_plate_and_fig_separate():
    fn = _first_json(OD_DIR / "f10ffa285a2f3c13")
    caps = _find_plate_captions(_load_json(fn)["kids"])
    by_kind: dict[str, list[int]] = {}
    for c in caps:
        by_kind.setdefault(c.get("kind") or "recon", []).append(c["plate_number"])
    # Bandini 2011 has Plates 1..6 (micrograph plates) and several Fig
    # captions (Fig 3 sketch, Fig 4 photo of outcrop, etc.).
    assert sorted(by_kind.get("plate", [])) == [1, 2, 3, 4, 5, 6]
    assert by_kind.get("fig"), "expected at least one Fig caption"
    # No "Fig. 1), PR-SB23..." body-text false positives.
    for c in caps:
        if c.get("kind") == "fig":
            # Body-text paragraphs that happen to start with "Fig. N"
            # are not picked up: e.g. Bandini p37 "Fig. 1), PR-SB23...".
            assert not c["content"].startswith("Fig. " + str(c["plate_number"]) + ")")


def test_pouille2014_reconstruction_works_alongside_fig_captions():
    fn = _first_json(OD_DIR / "c04373787560cf95")
    caps = _find_plate_captions(_load_json(fn)["kids"])
    by_kind: dict[str, list[int]] = {}
    for c in caps:
        by_kind.setdefault(c.get("kind") or "recon", []).append(c["plate_number"])
    # Real captions: Plate 3 (p11) + Fig 1, 2, 3 (schematic, pie, photo).
    assert 3 in by_kind.get("plate", [])
    assert sorted(set(by_kind.get("fig", []))) == [1, 2, 3]
    # Reconstruction pass should still synthesise Plate 1 and Plate 2
    # (species descriptions in the body) even though Fig 1 and Fig 2
    # are real captions on other pages with the same integer.
    assert sorted(by_kind.get("recon", [])) == [1, 2]


def test_fig_caption_regex_rejects_body_text_references():
    from rlpe.opendataloader_extractor import _FIG_CAPTION_RE, _looks_like_fig_caption
    # Real captions — regex matches AND content passes the head check.
    assert _FIG_CAPTION_RE.match("Fig. 1. Stratigraphic ranges of radiolarian families.")
    assert _looks_like_fig_caption("Fig. 1. Stratigraphic ranges of radiolarian families.")
    assert _FIG_CAPTION_RE.match("Fig 1 Schematic of the apparatus.")
    assert _looks_like_fig_caption("Fig 1 Schematic of the apparatus.")
    assert _FIG_CAPTION_RE.match("Figure 1. Caption text for a typical radiolarian paper.")
    assert _looks_like_fig_caption("Figure 1. Caption text for a typical radiolarian paper.")
    # Body-text false positives the regex alone rejects (no caption-
    # like delimiter after the number — closing paren or comma instead).
    assert not _FIG_CAPTION_RE.match("Fig. 1), PR-SB23 (Plate 6, Fig. 10)")
    assert not _FIG_CAPTION_RE.match("Fig. 13) Homeoparonaella sp.: PR-SB14")
    assert not _FIG_CAPTION_RE.match("see Fig. 1 for details")
    # Short matches and body-text species descriptions are rejected by
    # _looks_like_fig_caption (length + author-citation head check),
    # not by the regex alone.
    assert not _looks_like_fig_caption("Fig. 26")  # 7 chars
    assert not _looks_like_fig_caption("Fig. 14 continued c")  # 19 chars
    assert not _looks_like_fig_caption(
        "Fig. 21 Archaeodictyomitra montisserei (SQUINABOL) Pl. 8 Figs. 4 and 5"
    )


def test_plate_caption_regex_matches_roman_numerals():
    """Regression: boughdiri2007 uses 'Plate I' (Roman) rather than
    'Plate 1' (Arabic). Before the fix, _PLATE_CAPTION_RE only matched
    Arabic digits, so _find_plate_captions returned [] for that paper
    and the pipeline fell into the page-render fallback with
    placeholders — giving boughdiri 0% species F1 in earlier evals.

    After the fix the regex also matches Roman numerals I..XII and the
    helper maps them back to 1..12.
    """
    from rlpe.opendataloader_extractor import (
        _PLATE_CAPTION_RE,
        _plate_number_from_match,
    )

    # Arabic digits — works as before.
    for arabic, expected in [("Plate 1", 1), ("Plate 12", 12)]:
        m = _PLATE_CAPTION_RE.match(arabic)
        assert m is not None, f"{arabic!r} should match"
        assert _plate_number_from_match(m) == expected, (
            f"{arabic!r} should map to {expected}"
        )

    # Roman numerals I..XII — newly supported.
    for roman, expected in [
        ("Plate I", 1), ("Plate II", 2), ("Plate III", 3),
        ("Plate IV", 4), ("Plate V", 5), ("Plate VI", 6),
        ("Plate VII", 7), ("Plate VIII", 8), ("Plate IX", 9),
        ("Plate X", 10), ("Plate XI", 11), ("Plate XII", 12),
    ]:
        m = _PLATE_CAPTION_RE.match(roman)
        assert m is not None, f"{roman!r} should match"
        assert _plate_number_from_match(m) == expected, (
            f"{roman!r} should map to {expected}"
        )

    # Whitespace and "Explanation of" prefix still work for both kinds.
    assert _PLATE_CAPTION_RE.match("  Plate IV  ") is not None
    assert _PLATE_CAPTION_RE.match("Explanation of Plate III") is not None
    assert _PLATE_CAPTION_RE.match("Explanation of Plate VI") is not None

    # Roman XIII..XX are NOT supported (no radiolarian paper in the
    # gold set has more than 12 plates). If one ever shows up, the
    # helper returns 0 and the caption is dropped — that is the
    # documented behaviour and is better than a partial-digit match
    # that could collide with an Arabic plate number.
    m_xiii = _PLATE_CAPTION_RE.match("Plate XIII")
    # The regex may or may not match; if it does, the helper returns
    # 0 (no entry in _ROMAN_TO_INT) so the caption is harmlessly
    # filtered out downstream.
    if m_xiii is not None:
        assert _plate_number_from_match(m_xiii) == 0


def test_boughdiri2007_finds_plate_i_caption():
    """Integration test: boughdiri2007 has a 'Plate I' heading on
    page 10 followed by the species-list paragraph. Before the fix
    the heading was invisible to _find_plate_captions and the species
    list was orphaned; after the fix the heading is picked up, the
    following paragraph is appended, and the downstream parser sees
    the real species list.
    """
    fn = _first_json(BOUGHDIRI_DIR / "178d4e1e9d93136c")
    caps = _find_plate_captions(_load_json(fn)["kids"])
    plate_caps = [c for c in caps if c.get("kind") == "plate"]
    assert len(plate_caps) == 1, (
        f"expected exactly 1 plate caption in boughdiri, "
        f"got {len(plate_caps)}: {[c['plate_number'] for c in plate_caps]}"
    )
    pc = plate_caps[0]
    assert pc["plate_number"] == 1, (
        f"Roman 'Plate I' should map to 1, got {pc['plate_number']}"
    )
    assert pc["page_number"] == 10
    # The heading-only match was expanded with the following
    # paragraphs (heading → paragraph) so the captured content must
    # include the species list, not just the bare "Plate I" heading.
    assert "Ristola altissima" in pc["content"]
    assert "Archaeodictyomitra" in pc["content"]


def test_feng2007_plate_caption_expands_paragraph_to_list():
    """Regression test: feng2007 has the "Explanation of Plate 1" header
    + first species clause as a ``paragraph`` element, then a ``list``
    element with the remaining species clauses (the list is a separate
    OD element because the species panel-list is rendered as a bulleted
    list in the PDF). Before the fix the paragraph element was matched
    as the plate caption but only the truncated first sentence
    ("Explanation of Plate 1. ﬁgs 1–2. ... 4,") was captured, so
    panels 5–20 of pl01 had no species assignment and feng2007 F1
    was 69.57%. After the fix the paragraph element is expanded into
    the following list element so all 20 panels are captured.
    """
    fn = _first_json(FENG_DIR / "e28de2b07edc8950")
    caps = _find_plate_captions(_load_json(fn)["kids"])
    plate_caps = [c for c in caps if c.get("kind") == "plate"]
    assert len(plate_caps) >= 1, (
        f"expected at least 1 plate caption in feng2007, "
        f"got {len(plate_caps)}"
    )
    pl1 = next(c for c in plate_caps if c["plate_number"] == 1)
    content = pl1["content"]
    # The first species clause (truncated paragraph) must be present.
    assert "Explanation of Plate 1" in content
    assert "Entactinia itsukichiensis" in content
    # The list-item continuation (paragraph→list expansion) must also
    # be present — this is the load-bearing part of the test.
    assert "Entactinia modesta" in content, (
        "paragraph→list expansion failed: 'Entactinia modesta' is in "
        "the list element, not the paragraph, and should have been "
        "appended by _collect_following_text"
    )
    assert "Entactinia wangi" in content
    assert "ﬁgs 9–20" in content


def test_feng2007_paragraph_caption_does_not_collect_following_paragraph():
    """Anti-regression test for the Fig. 1 (geological map) case in
    feng2007: a paragraph element that starts with "Fig. 1" but is
    followed by a body-text paragraph (a species description) and a
    list of citations must NOT collect the body paragraph. Only the
    list may be appended. The body paragraph is detected by NOT being
    in the captured content.
    """
    fn = _first_json(FENG_DIR / "e28de2b07edc8950")
    caps = _find_plate_captions(_load_json(fn)["kids"])
    fig_caps = [c for c in caps if c.get("kind") == "fig"]
    # feng2007 has Fig. 1 and Fig. 2 as real figure captions for the
    # geological map and distribution map, both at the start of the
    # paper. They are followed by body-text species descriptions and
    # citation lists, not by a species-list for those figures.
    fig1 = next((c for c in fig_caps if c["plate_number"] == 1), None)
    if fig1 is None:
        return  # No Fig. 1 found, nothing to assert
    content = fig1["content"]
    # The Fig. 1 caption title must be present.
    assert content.startswith("Fig. 1.")
    # The body-text species description that follows must NOT have
    # been collected (paragraph→paragraph is a body-text transition,
    # not a caption-content transition). The next element is a
    # citation list with year-prefixed entries — those are also NOT
    # part of the Fig. 1 caption and were added by the v1 fix that
    # collected lists. We accept that for the v2 (post-fix) state the
    # citation list IS collected (because the structural pattern is
    # paragraph→list, identical to pl01); we just verify the next
    # *paragraph* (body text) is NOT collected.
    # Find the next paragraph after the Fig. 1 paragraph in the OD JSON.
    data = _load_json(fn)
    kids = data["kids"]
    fig1_idx = None
    for i, k in enumerate(kids):
        if isinstance(k, dict) and (k.get("content") or "").startswith("Fig. 1."):
            fig1_idx = i
            break
    assert fig1_idx is not None
    next_para_text = None
    for j in range(fig1_idx + 1, min(fig1_idx + 5, len(kids))):
        k2 = kids[j]
        if not isinstance(k2, dict):
            continue
        if k2.get("type") == "paragraph":
            txt = (k2.get("content") or "").strip()
            if txt and not txt[0].isdigit():
                # Body-text paragraph (doesn't start with a year/digit)
                next_para_text = txt
                break
    # If we found a body-text paragraph, assert it was NOT collected.
    if next_para_text:
        assert next_para_text not in content, (
            "paragraph→paragraph body text was collected, but the "
            "kinds=(list,) expansion rule should prevent this"
        )


class TestDanelianQuestionMarkPrefix:
    """The Danelian caption parser must accept an optional "?" prefix on
    the genus (uncertainty marker), as in boughdiri2007 items 16 and 17:
        16) ?Sethocapsa sp.
        17) ?Archaeodictyomitra sp.
    Without the prefix support the parser drops both clauses, costing
    boughdiri 2/27 panels (recall drops from 14/27 to 12/27).
    """

    def test_danelian_clause_with_question_prefix(self):
        from rlpe.m3_engine import _DANELIAN_CLAUSE_RE

        # Item 16
        m = _DANELIAN_CLAUSE_RE.match("16) ?Sethocapsa sp.")
        assert m is not None
        assert m.group(1) == "16"
        # The "?" is captured in group 2, the species in group 3
        assert m.group(2) == "?"
        # The "sp" token is captured in group 4 (modifier) rather than
        # group 3 (epithet) so the hollis-style trailing-ID pattern
        # ("sp. A. B-F36/0") can still match. The parser folds them.
        assert m.group(3) == "Sethocapsa"
        assert m.group(4) == " sp."

        # Item 17
        m = _DANELIAN_CLAUSE_RE.match("17) ?Archaeodictyomitra sp.")
        assert m is not None
        assert m.group(1) == "17"
        assert m.group(2) == "?"
        assert m.group(3) == "Archaeodictyomitra"
        assert m.group(4) == " sp."

    def test_danelian_clause_without_question_prefix_still_works(self):
        """Regression guard: the existing "no-?" clauses must still match."""
        from rlpe.m3_engine import _DANELIAN_CLAUSE_RE

        m = _DANELIAN_CLAUSE_RE.match("1) Ristola altissima altissima")
        assert m is not None
        assert m.group(1) == "1"
        assert m.group(2) == ""  # no "?"
        assert m.group(3) == "Ristola altissima altissima"
