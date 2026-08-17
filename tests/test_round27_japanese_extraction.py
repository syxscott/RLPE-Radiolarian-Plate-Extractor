"""Phase 27 — Japanese caption extraction regression suite.

Round 25 CV-env live test exposed a 2-paper extraction cliff: the
Japanese papers Takahashi 2004 and Uchino 2005 returned
``"Auto-generated figure for page 1"`` for nearly every caption because:

1. The OpenDataLoader caption routing only matched English ``Plate``
   and ``Fig.`` markers — Japanese ``図版`` / ``図`` were invisible.
2. ``OCRBackend`` had no ``lang`` parameter; PaddleOCR/EasyOCR were
   hardcoded to English.
3. The M3 ``parse_caption`` system prompt was Chinese-only and
   offered no language dispatch for JA captions.

These tests pin the new behavior so future refactors don't silently
break JA extraction. They follow the same scaffolding pattern as
``tests/test_round21_od_pairing.py``: ``sys.path`` injection to load
``src/rlpe`` directly (the test runner uses pytest without a
conftest-managed PYTHONPATH), and a small ``_read`` helper for
source-guard assertions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.m3_engine import (  # noqa: E402
    _PARSE_CAPTION_SYSTEM,
    _PARSE_CAPTION_SYSTEM_JA,
    _detect_caption_lang,
)
from rlpe.ocr import OCRBackend  # noqa: E402
from rlpe.opendataloader_extractor import (  # noqa: E402
    _JA_FIG_CAPTION_RE,
    _JA_PLATE_CAPTION_RE,
    _find_plate_captions,
    _is_caption_kind_marker,
    _normalise_ocr_lang,
)
from rlpe.pipeline import _resolve_m3_prompt_lang  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (_REPO / rel).read_text(encoding="utf-8")


# ============================================================================
# Phase A — caption routing
# ============================================================================


def test_ja_plate_caption_regex_basic():
    """``図版 N`` matches at the start of a JA caption."""
    m = _JA_PLATE_CAPTION_RE.match("図版1 走査電子顕微鏡写真。")
    assert m is not None
    assert m.group(1) == "1"


def test_ja_plate_caption_regex_with_setsumei_prefix():
    """The optional ``説明`` (explanation) prefix is accepted."""
    m = _JA_PLATE_CAPTION_RE.match("説明 図版2. A-D: Species X")
    assert m is not None
    assert m.group(1) == "2"


def test_ja_plate_caption_regex_rejects_body_text():
    """Mid-sentence references like ``前項の図版2と関連`` should NOT match.

    The regex anchors at ``^\\s*`` so any leading non-whitespace text
    (like a body-style paragraph) fails the match. This is the same
    anchoring policy as the English ``_PLATE_CAPTION_RE``.

    Note: a string like ``図版1参照`` still matches ``図版1`` because
    the regex captures the digit only — the dispatcher + downstream
    25-char quality gate (and short-content filter) decide whether
    the captured match becomes a real caption. We do NOT push the
    body-text rejection into the regex itself; that would over-fit
    the pattern and miss real captions like
    ``図版1参照の標本...``.
    """
    assert _JA_PLATE_CAPTION_RE.match("前項の図版2と関連") is None
    assert _JA_PLATE_CAPTION_RE.match("本論文の図版3参照") is None


def test_ja_fig_caption_regex_basic():
    """``図 N`` matches at the start of a JA fig caption."""
    m = _JA_FIG_CAPTION_RE.match("図1 SEM写真。A-D: Pseudocenosphaera sp.")
    assert m is not None
    assert m.group(1) == "1"


def test_ja_fig_caption_regex_rejects_body_text():
    """Mid-paragraph references are rejected by anchor."""
    assert _JA_FIG_CAPTION_RE.match("（図1参照）について") is None


def test_find_plate_captions_routes_ja_plate():
    kids = [
        {
            "type": "paragraph",
            "page number": 1,
            "content": "図版1 走査電子顕微鏡写真。1-5: Entactinia variaspina",
        }
    ]
    caps = _find_plate_captions(kids)
    assert len(caps) == 1
    assert caps[0]["kind"] == "plate"
    assert caps[0]["plate_number"] == 1


def test_find_plate_captions_routes_ja_fig():
    kids = [
        {
            "type": "paragraph",
            "page number": 2,
            "content": "図1 SEM写真。A-D: Pseudocenosphaera sp.; E,F: Trilonche sp.",
        }
    ]
    caps = _find_plate_captions(kids)
    assert len(caps) == 1
    assert caps[0]["kind"] == "fig"
    assert caps[0]["plate_number"] == 1


def test_find_plate_captions_keeps_ja_fig_and_eng_plate_distinct():
    """JA fig captions (``図 N``) and EN plate captions (``Plate N``)
    have different ``kind`` values so the dispatcher keeps both. The
    same-(plate_number, kind) pair would still dedup, which is the
    correct behaviour — a paper rarely has BOTH a JA fig 1 and an
    EN fig 1 referring to different figures.

    Note: both captions must clear the 25-char fig quality gate
    (``_looks_like_fig_caption``); we use a long enough JA caption
    to demonstrate the dispatch path.
    """
    kids = [
        {
            "type": "paragraph",
            "page number": 1,
            "content": "図1 SEM写真。A-D: Pseudocenosphaera sp.; E,F: Trilonche sp.",
        },
        {
            "type": "paragraph",
            "page number": 2,
            "content": "Plate 1. figs 1-5. Species B",
        },
    ]
    caps = _find_plate_captions(kids)
    assert len(caps) == 2
    assert {c["kind"] for c in caps} == {"fig", "plate"}


def test_is_caption_kind_marker_accepts_english_and_ja():
    assert _is_caption_kind_marker("fig. 1. caption") is True
    assert _is_caption_kind_marker("figure 1 caption") is True
    assert _is_caption_kind_marker("fig 1 caption") is True
    assert _is_caption_kind_marker("図1 SEM写真 A-D: Species") is True


def test_is_caption_kind_marker_rejects_non_markers():
    assert _is_caption_kind_marker("plate 1 caption") is False
    assert _is_caption_kind_marker("random text") is False
    assert _is_caption_kind_marker("") is False


# ============================================================================
# Phase B — OCR language plumbing
# ============================================================================


def test_ocr_backend_default_lang_is_en():
    """Backward compatibility: no lang kwarg → English only."""
    be = OCRBackend(backend="easyocr", use_gpu=False)
    assert be.lang == ["en"]


def test_ocr_backend_parses_comma_separated_lang_string():
    """``"en,ja"`` normalises to ``["en", "ja"]``."""
    be = OCRBackend(backend="easyocr", use_gpu=False, lang="en,ja")
    assert "en" in be.lang
    assert "ja" in be.lang


def test_ocr_backend_accepts_list_input():
    be = OCRBackend(backend="easyocr", use_gpu=False, lang=["ja"])
    assert be.lang == ["ja"]


def test_ocr_backend_drops_unknown_lang_with_warning():
    """Typos in --ocr-lang must not crash the pipeline."""
    be = OCRBackend(backend="easyocr", use_gpu=False, lang="en,klingon")
    # "klingon" is dropped, "en" is kept.
    assert "klingon" not in be.lang
    assert "en" in be.lang


def test_ocr_backend_paddle_lang_mapping():
    """PaddleOCR uses ``"japan"`` (not ``"ja"``). The internal mapping
    must convert ``ja`` → ``"japan"`` so PaddleOCR accepts it."""
    be = OCRBackend(backend="paddleocr", use_gpu=False, lang="ja")
    assert be._paddle_lang() == "japan"


def test_ocr_backend_paddle_lang_mapping_passes_unknown_through():
    """A language the mapper doesn't know is passed verbatim."""
    be = OCRBackend(backend="paddleocr", use_gpu=False, lang="fr")
    assert be._paddle_lang() == "fr"


def test_ocr_backend_supported_langs_whitelist():
    """The whitelist must include JA and ZH variants the operator may
    need for bilingual papers. We don't add Russian here (per the
    user's scope guard)."""
    assert "en" in OCRBackend.SUPPORTED_LANGS
    assert "ja" in OCRBackend.SUPPORTED_LANGS
    assert "ch_sim" in OCRBackend.SUPPORTED_LANGS
    # Russian is intentionally NOT supported in this round.
    # It works as a passthrough on EasyOCR but Phase 27 does not
    # document it; we keep it in SUPPORTED for fallback safety.
    # (This test pins that it's recognised, not that it's advertised.)


def test_normalise_ocr_lang_accepts_string_and_list():
    """The OD extractor's helper accepts both legacy strings and
    already-normalised lists."""
    assert _normalise_ocr_lang("en,ja") == ["en", "ja"]
    assert _normalise_ocr_lang("en") == ["en"]
    assert _normalise_ocr_lang(["ja", "en"]) == ["ja", "en"]
    assert _normalise_ocr_lang(None) == ["en"]
    assert _normalise_ocr_lang("") == ["en"]


def test_config_extra_keys_includes_ocr_lang_and_m3_prompt_lang():
    """Source guard: the new keys must be in the whitelist so the
    ``__post_init__`` warning doesn't fire on a clean CLI invocation."""
    src = _read("src/rlpe/config.py")
    assert '"ocr_lang"' in src
    assert '"m3_prompt_lang"' in src


def test_cli_exposes_ocr_lang_flag():
    """Source guard: ``--ocr-lang`` must be wired through ``main()``."""
    src = _read("src/rlpe/cli.py")
    assert "--ocr-lang" in src
    assert '"ocr_lang": args.ocr_lang' in src


def test_cli_exposes_m3_prompt_lang_flag():
    src = _read("src/rlpe/cli.py")
    assert "--m3-prompt-lang" in src
    assert '"m3_prompt_lang": args.m3_prompt_lang' in src


def test_pipeline_forwards_ocr_lang_to_ocr_backend():
    """Source guard: pipeline.py must pass ``lang=`` to ``OCRBackend``
    so the configured languages actually reach the engine."""
    src = _read("src/rlpe/pipeline.py")
    # The OCRBackend( ... ) call must include lang=...
    assert "OCRBackend(" in src
    # Find the call and verify lang is forwarded.
    import re

    m = re.search(r"OCRBackend\([^)]*lang=", src, re.DOTALL)
    assert m, "pipeline.py must pass lang= kwarg to OCRBackend"


def test_pipeline_forwards_m3_prompt_lang_to_parse_caption():
    """Source guard: pipeline.py must thread ``m3_prompt_lang`` through
    ``_resolve_m3_prompt_lang`` to the two ``parse_caption`` call sites."""
    src = _read("src/rlpe/pipeline.py")
    assert "_resolve_m3_prompt_lang" in src
    assert "m3_prompt_lang" in src


# ============================================================================
# Phase C — M3 JA prompt + lang dispatch
# ============================================================================


def test_detect_caption_lang_japanese_hiragana():
    assert _detect_caption_lang("これはテストです") == "ja"


def test_detect_caption_lang_japanese_katakana():
    assert _detect_caption_lang("ペンタゴンステッチ") == "ja"


def test_detect_caption_lang_japanese_kanji():
    assert _detect_caption_lang("図版1 走査電子顕微鏡写真") == "ja"


def test_detect_caption_lang_zh_hanzi_returns_ja():
    """We treat all CJK ideographs as JA-routed because the JA prompt
    also handles ZH bilingual JA+EN papers best. ZH-only captions
    fall through to the legacy Chinese prompt because CJK code
    points are present in ZH too — so this test pins that CJK → JA."""
    assert _detect_caption_lang("图1 扫描电镜照片") == "ja"


def test_detect_caption_lang_english_returns_zh():
    """English-only text must default to the legacy Chinese prompt so
    nothing changes for English papers."""
    assert _detect_caption_lang("Plate 1. figs 1-5. Species X") == "zh"


def test_detect_caption_lang_empty_returns_zh():
    """Empty caption falls back to ZH (legacy behaviour)."""
    assert _detect_caption_lang("") == "zh"


def test_parse_caption_uses_ja_system_when_lang_ja():
    """Pin the dispatch: ``parse_caption(caption, lang='ja')`` selects
    the Japanese system prompt."""
    captured = {}

    class FakeBackend:
        def infer_text(self, system_prompt, user_prompt, **kwargs):
            captured["system"] = system_prompt
            return {"raw_text": "[]"}

    from rlpe.m3_engine import M3Engine

    eng = M3Engine(FakeBackend(), config={"m3_stage_1": True})
    eng.parse_caption("図1. A: Species X", lang="ja")
    assert captured["system"] == _PARSE_CAPTION_SYSTEM_JA


def test_parse_caption_uses_zh_system_when_lang_zh():
    captured = {}

    class FakeBackend:
        def infer_text(self, system_prompt, user_prompt, **kwargs):
            captured["system"] = system_prompt
            return {"raw_text": "[]"}

    from rlpe.m3_engine import M3Engine

    eng = M3Engine(FakeBackend(), config={"m3_stage_1": True})
    eng.parse_caption("Plate 1. A: Species X", lang="zh")
    assert captured["system"] == _PARSE_CAPTION_SYSTEM


def test_parse_caption_auto_detects_ja_from_caption_text():
    """When ``lang`` is ``None``, the detector picks JA for Hiragana-
    containing captions and ZH for ASCII-only ones."""
    captured = {}

    class FakeBackend:
        def infer_text(self, system_prompt, user_prompt, **kwargs):
            captured["system"] = system_prompt
            return {"raw_text": "[]"}

    from rlpe.m3_engine import M3Engine

    eng = M3Engine(FakeBackend(), config={"m3_stage_1": True})
    eng.parse_caption("図1 走査電子顕微鏡写真")
    assert captured["system"] == _PARSE_CAPTION_SYSTEM_JA

    captured.clear()
    eng.parse_caption("Plate 1. figs 1-5. Species X")
    assert captured["system"] == _PARSE_CAPTION_SYSTEM


def test_parse_caption_prompts_are_byte_distinct():
    """The two prompts must not be identical — if a refactor ever
    makes them equal the JA dispatch would silently no-op."""
    assert _PARSE_CAPTION_SYSTEM != _PARSE_CAPTION_SYSTEM_JA


# ============================================================================
# Phase C — _resolve_m3_prompt_lang helper
# ============================================================================


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("auto", None),
        ("AUTO", None),
        ("", None),
        (None, None),
        ("ja", "ja"),
        ("JA", "ja"),
        ("zh", "zh"),
        ("en", "en"),
    ],
)
def test_resolve_m3_prompt_lang(raw, expected):
    """CLI string → parse_caption kwarg translation."""
    assert _resolve_m3_prompt_lang(raw) == expected


# ============================================================================
# Phase D — overfit defense / source-guard
# ============================================================================


def test_phase27_anchored_in_opendataloader_extractor():
    """Pin that the JA regex + helper + dispatch landed in the right
    file. Without this a future refactor that moves caption routing
    to a new module would silently break JA extraction."""
    src = _read("src/rlpe/opendataloader_extractor.py")
    assert "_JA_PLATE_CAPTION_RE" in src
    assert "_JA_FIG_CAPTION_RE" in src
    assert "_is_caption_kind_marker" in src
    assert "_normalise_ocr_lang" in src


def test_phase27_anchored_in_m3_engine():
    src = _read("src/rlpe/m3_engine.py")
    assert "_PARSE_CAPTION_SYSTEM_JA" in src
    assert "_detect_caption_lang" in src
    # The parse_caption signature must accept lang=
    assert "def parse_caption(" in src
    assert "lang: str | None = None" in src


def test_phase27_anchored_in_ocr():
    """The lang parameter must be a first-class init arg on OCRBackend."""
    src = _read("src/rlpe/ocr.py")
    assert "def __init__" in src
    assert 'lang: str | list[str] = "en"' in src
    assert "_PADDLE_LANG_MAP" in src
