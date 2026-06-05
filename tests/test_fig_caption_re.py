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


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
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
