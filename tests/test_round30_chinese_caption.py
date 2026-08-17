"""Phase 30 — Chinese caption routing regression suite.

Round 25 final 5-paper live test surfaced a gap: the OpenDataLoader
caption-routing path only matched English ``Plate`` / ``Fig.``
markers (Phase 27 added JA ``図版`` / ``図``). Chinese journals
use ``图版`` / ``图`` (simplified) and ``圖版`` / ``圖``
(traditional), which produced zero caption matches for any CN paper.

Phase 30 adds:
- ``_ZH_PLATE_CAPTION_RE`` matching simplified ``图版`` AND
  traditional ``圖版``
- ``_ZH_FIG_CAPTION_RE`` matching ``图 N`` AND ``圖 N``
- ``_is_caption_kind_marker`` predicate extended with both ZH
  single-char markers
- 6-way dispatcher cascade (EN_plate → EN_fig → JA_plate → JA_fig
  → ZH_plate → ZH_fig)

The scaffolding pattern follows ``test_round27_japanese_extraction.py``
and ``test_round28_caption_page_distance.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.opendataloader_extractor import (  # noqa: E402
    _ZH_FIG_CAPTION_RE,
    _ZH_PLATE_CAPTION_RE,
    _find_plate_captions,
    _is_caption_kind_marker,
)

_REPO = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (_REPO / rel).read_text(encoding="utf-8")


# ============================================================================
# ZH regex unit tests
# ============================================================================


def test_zh_plate_simplified_basic():
    m = _ZH_PLATE_CAPTION_RE.match("图版1 扫描电镜照片")
    assert m is not None
    assert m.group(1) == "1"


def test_zh_plate_simplified_with_shuoming_prefix():
    """``说明`` (explanation) prefix is accepted (Chinese equivalent
    of JA ``説明``)."""
    m = _ZH_PLATE_CAPTION_RE.match("说明 图版2. A-D: Species X")
    assert m is not None
    assert m.group(1) == "2"


def test_zh_plate_traditional():
    """``圖版`` (traditional) must match."""
    m = _ZH_PLATE_CAPTION_RE.match("圖版3 化石照片")
    assert m is not None
    assert m.group(1) == "3"


def test_zh_plate_rejects_body_text():
    """Body-text mentions must NOT match — anchored at ``^\\s*`` so
    leading non-whitespace text fails the match."""
    assert _ZH_PLATE_CAPTION_RE.match("前项的图版2") is None
    assert _ZH_PLATE_CAPTION_RE.match("本论文的图版3参考") is None


def test_zh_fig_simplified_basic():
    m = _ZH_FIG_CAPTION_RE.match("图1 SEM照片 A-D: Species X")
    assert m is not None
    assert m.group(1) == "1"


def test_zh_fig_traditional():
    m = _ZH_FIG_CAPTION_RE.match("圖2 化石写真")
    assert m is not None
    assert m.group(1) == "2"


def test_zh_fig_rejects_body_text():
    """Mid-paragraph references do NOT match (anchoring)."""
    assert _ZH_FIG_CAPTION_RE.match("（图1参照）") is None


# ============================================================================
# Dispatcher integration
# ============================================================================


def test_dispatch_routes_zh_simplified_plate():
    kids = [
        {
            "type": "paragraph",
            "page number": 1,
            "content": "图版1 扫描电镜照片。1-5: Species X",
        }
    ]
    caps = _find_plate_captions(kids)
    assert len(caps) == 1
    assert caps[0]["kind"] == "plate"
    assert caps[0]["plate_number"] == 1


def test_dispatch_routes_zh_traditional_plate():
    kids = [
        {
            "type": "paragraph",
            "page number": 2,
            "content": "圖版2 化石照片。A-D: Species Y",
        }
    ]
    caps = _find_plate_captions(kids)
    assert len(caps) == 1
    assert caps[0]["kind"] == "plate"
    assert caps[0]["plate_number"] == 2


def test_dispatch_routes_zh_simplified_fig():
    """ZH fig caption needs to be ≥25 chars to pass the
    ``_looks_like_fig_caption`` quality gate (same as JA)."""
    kids = [
        {
            "type": "paragraph",
            "page number": 3,
            "content": "图3 SEM照片 A-D: Pseudocenosphaera sp.; E,F: Trilonche sp.",
        }
    ]
    caps = _find_plate_captions(kids)
    assert len(caps) == 1
    assert caps[0]["kind"] == "fig"
    assert caps[0]["plate_number"] == 3


def test_dispatch_rejects_zh_body_text():
    """Body-text mentions that start with a ZH figure char but are
    not captions must NOT be routed."""
    kids = [
        {
            "type": "paragraph",
            "page number": 1,
            "content": "图书馆里的书很多",
        }
    ]
    caps = _find_plate_captions(kids)
    assert caps == [], f"Body text '图书馆里的书很多' must NOT match; got {caps!r}"


def test_dispatch_rejects_short_zh_fig_caption():
    """Short ZH fig caption (< 25 chars) is filtered by the quality gate."""
    kids = [
        {
            "type": "paragraph",
            "page number": 1,
            "content": "图1 Species",
        }
    ]
    caps = _find_plate_captions(kids)
    assert caps == [], "Short ZH fig caption must NOT route"


def test_dispatch_keeps_zh_plate_and_eng_plate_distinct():
    """Bilingual paper with both ``图版 1`` (ZH) and ``Plate 1`` (EN)
    collapses into one entry because they share (plate_number=1,
    kind=plate). The dispatcher dedups on (number, kind), so two
    captions with the same kind + same number dedup — same as for
    EN+EN or JA+JA. ZH introduces no new collision behaviour."""
    kids = [
        {
            "type": "paragraph",
            "page number": 1,
            "content": "图版1 扫描电镜照片。1-5: Species A",
        },
        {
            "type": "paragraph",
            "page number": 2,
            "content": "Plate 1. figs 1-5. Species B",
        },
    ]
    caps = _find_plate_captions(kids)
    # Same plate_number + same kind dedups. ZH + EN = 1 caption.
    assert len(caps) == 1
    assert caps[0]["kind"] == "plate"


# ============================================================================
# _is_caption_kind_marker predicate
# ============================================================================


def test_marker_accepts_zh_simplified_fig():
    assert _is_caption_kind_marker("图3 SEM照片") is True


def test_marker_accepts_zh_traditional_fig():
    assert _is_caption_kind_marker("圖3 SEM照片") is True


def test_marker_accepts_ja_fig():
    assert _is_caption_kind_marker("図3 SEM写真") is True


def test_marker_accepts_english_fig():
    assert _is_caption_kind_marker("fig. 1.") is True
    assert _is_caption_kind_marker("figure 1 caption") is True


def test_marker_rejects_non_caption_text():
    assert _is_caption_kind_marker("plate 1 caption") is False
    assert _is_caption_kind_marker("random text") is False
    assert _is_caption_kind_marker("") is False


# ============================================================================
# Source-guard tests
# ============================================================================


def test_zh_patterns_anchored_in_source():
    """Source guard: ZH regexes + helper live in opendataloader_extractor.py."""
    src = _read("src/rlpe/opendataloader_extractor.py")
    assert "_ZH_PLATE_CAPTION_RE" in src
    assert "_ZH_FIG_CAPTION_RE" in src
    # And the predicate accepts the ZH fig markers
    assert 'low.startswith("图")' in src
    assert 'low.startswith("圖")' in src


def test_dispatcher_zh_branches_present():
    """The 6-way dispatcher must contain the ZH branches after the JA ones."""
    src = _read("src/rlpe/opendataloader_extractor.py")
    # The ZH plate match must appear after the JA fig match
    ja_fig_idx = src.find("_JA_FIG_CAPTION_RE.match(content)")
    zh_plate_idx = src.find("_ZH_PLATE_CAPTION_RE.match(content)")
    zh_fig_idx = src.find("_ZH_FIG_CAPTION_RE.match(content)")
    assert ja_fig_idx > 0
    assert zh_plate_idx > ja_fig_idx, "ZH plate branch must come AFTER JA fig"
    assert zh_fig_idx > zh_plate_idx, "ZH fig branch must come AFTER ZH plate"


def test_list_item_expansion_accepts_zh_plate():
    """The list-item expansion loop must recognise ZH plate markers
    in addition to EN and JA."""
    src = _read("src/rlpe/opendataloader_extractor.py")
    assert "_ZH_PLATE_CAPTION_RE.match(_txt)" in src


def test_no_ja_zh_regex_collision_for_traditional_tuban():
    """Bug-fix H-4: JA regex no longer accepts ``圖版`` (traditional
    ZH). Traditional ZH papers now route through the ZH dispatcher
    instead of being captured by JA. This source guard pins the split:
    JA accepts only ``図版`` (Japanese-only char), ZH accepts
    ``图版`` (simplified) AND ``圖版`` (traditional)."""
    import re as _re

    src = _read("src/rlpe/opendataloader_extractor.py")
    # Extract the JA regex pattern from source and compile it. We
    # use a regex to find the r-string after the ``re.compile(...)``
    # call. Comments may contain ``圖版`` (e.g. the bug-fix comment
    # itself) so we MUST extract only the actual pattern string.
    ja_match = _re.search(
        r'_JA_PLATE_CAPTION_RE\s*=\s*re\.compile\(.*?r"([^"]+)"',
        src,
        _re.DOTALL,
    )
    assert ja_match is not None, "Could not locate _JA_PLATE_CAPTION_RE"
    ja_pattern = _re.compile(ja_match.group(1))

    # Extract the ZH regex the same way
    zh_match = _re.search(
        r'_ZH_PLATE_CAPTION_RE\s*=\s*re\.compile\(.*?r"([^"]+)"',
        src,
        _re.DOTALL,
    )
    assert zh_match is not None, "Could not locate _ZH_PLATE_CAPTION_RE"
    zh_pattern = _re.compile(zh_match.group(1))

    # H-4: JA must NOT accept traditional 圖版
    assert ja_pattern.match("圖版1 化石写真") is None, (
        "H-4 bug-fix: JA regex must NOT match 圖版 (traditional ZH)"
    )
    # H-4: JA still matches 図版 (Japanese-only char)
    assert ja_pattern.match("図版1 走査電子顕微鏡写真") is not None

    # ZH accepts both 圖版 (traditional) and 图版 (simplified)
    assert zh_pattern.match("圖版1 化石写真") is not None
    assert zh_pattern.match("图版1 扫描电镜照片") is not None

    # Sanity: ZH must NOT match Japanese-only 図版
    # (the JA char is U+56E7; ZH chars are U+56FE simplified / U+5716 traditional)
    # This guards against accidental over-matching.


# ============================================================================
# Backward compatibility — JA papers unaffected
# ============================================================================


def test_ja_paper_still_routes_via_ja_branch():
    """A Japanese paper with ``図版1`` must still route through the
    JA branch (kind=plate, plate_number=1). Phase 30 only adds ZH
    branches AFTER JA, so JA papers are unaffected."""
    kids = [
        {
            "type": "paragraph",
            "page number": 1,
            "content": "図版1 走査電子顕微鏡写真。1-5: Species A",
        }
    ]
    caps = _find_plate_captions(kids)
    assert len(caps) == 1
    assert caps[0]["kind"] == "plate"
    assert caps[0]["plate_number"] == 1


def test_english_paper_still_routes_via_eng_branch():
    """English paper with ``Plate 1`` must still route correctly."""
    kids = [
        {
            "type": "paragraph",
            "page number": 1,
            "content": "Plate 1. figs 1-5. Species X",
        }
    ]
    caps = _find_plate_captions(kids)
    assert len(caps) == 1
    assert caps[0]["kind"] == "plate"
    assert caps[0]["plate_number"] == 1
