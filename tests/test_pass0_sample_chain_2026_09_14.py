"""P0 / P1 / P2 fixes (2026-09-14): printed-number pairing gate, the
sample→unit→age chain, and honest positional stamps.

Pins:
  * P0 — the EasyOCR digit-fallback retry must fire when the PRIMARY
    backend is the known-broken ``paddleocr`` build (onednn silently
    returns 0 tokens) and the fallback differs from it. The previous
    condition compared the primary against the FALLBACK name, which
    made the retry dead code under the default config.
  * P2a — pass-0 records whether the plate showed ANY digit tokens and
    whether the OCR read crashed; plates with no printed numbers stamp
    ``caption_order_positional`` (no review alarm) instead of the
    generic ``positional_fallback``; the converter exempts that stamp
    from ``missing_printed_panel_id``.
  * P1 — sample→unit→age chain: caption cites "CH4", prose says
    "Sample CHA4 … attributed to UAZ 5-7 of latest Bajocian–early
    Callovian age" (Boughdiri 2007). Regex shapes, alias matching
    (CH4 ↔ CHA4), LLM schema (``sample_codes``), and the finalize
    wiring that prepends the resolved link.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np


# ============================================================
# stubs
# ============================================================
class _Tok:
    def __init__(self, text: str, conf: float, bbox: tuple[int, int, int, int]):
        self.text = text
        self.confidence = conf
        self.bbox = bbox


class _StubOCR:
    def __init__(self, backend: str, tokens=None, label_tokens=None):
        self.backend = backend
        self.tokens = tokens or []
        self.label_tokens = label_tokens or []
        self.panel_calls = 0
        self.label_calls = 0

    def recognize_panel(self, img, box):
        self.panel_calls += 1
        return self.tokens

    def recognize_panel_label(self, img, box, label_corner="fixed"):
        self.label_calls += 1
        return self.label_tokens


class _Cfg:
    def __init__(self, fb: str = "easyocr"):
        self.extra = {"ocr_digit_fallback_backend": fb}


def _pipeline_with(primary: _StubOCR, fallback=None, fb_name: str = "easyocr"):
    from rlpe.pipeline import RadiolarianPipeline as Pipeline

    p = object.__new__(Pipeline)
    p.config = _Cfg(fb_name)
    p.ocr = primary
    if fallback is not None:
        p._ocr_digit_fallback_backend = fallback
    return p


def _region() -> np.ndarray:
    return np.zeros((120, 160, 3), dtype=np.uint8)


def _segs(n: int = 1):
    return [SimpleNamespace(bbox=(0, 0, 60, 60)) for _ in range(n)]


# ============================================================
# P0 — the EasyOCR digit-fallback gate
# ============================================================
def test_p0_fallback_fires_for_broken_paddleocr_primary():
    primary = _StubOCR("paddleocr", tokens=[])
    fb = _StubOCR("easyocr", tokens=[_Tok("1", 0.9, (5, 5, 10, 10))])
    p = _pipeline_with(primary, fb)
    out = p._ocr_printed_panel_assignments(_region(), _segs(), {1})
    assert primary.panel_calls == 1
    assert fb.panel_calls == 1, "broken paddle primary must trigger the EasyOCR retry"
    assert 1 in out, out


def test_p0_no_fallback_when_primary_is_easyocr():
    primary = _StubOCR("easyocr", tokens=[])
    fb = _StubOCR("easyocr", tokens=[_Tok("1", 0.9, (5, 5, 10, 10))])
    p = _pipeline_with(primary, fb)
    out = p._ocr_printed_panel_assignments(_region(), _segs(), {1})
    assert fb.panel_calls == 0, "same-backend retry is pointless"
    assert out == {}


def test_p0_no_fallback_when_primary_reads_digits():
    primary = _StubOCR("paddleocr", tokens=[_Tok("2", 0.95, (8, 8, 10, 10))])
    fb = _StubOCR("easyocr", tokens=[])
    p = _pipeline_with(primary, fb)
    out = p._ocr_printed_panel_assignments(_region(), _segs(), {2})
    assert fb.panel_calls == 0
    assert 2 in out


def test_p0_fallback_disabled_by_config():
    primary = _StubOCR("paddleocr", tokens=[])
    fb = _StubOCR("easyocr", tokens=[_Tok("1", 0.9, (5, 5, 10, 10))])
    p = _pipeline_with(primary, fb, fb_name="")
    out = p._ocr_printed_panel_assignments(_region(), _segs(), {1})
    assert fb.panel_calls == 0
    assert out == {}


# ============================================================
# P2a — pass-0 diagnostics + honest stamps
# ============================================================
def test_p2a_diag_no_digits_on_plate():
    primary = _StubOCR("paddleocr", tokens=[])
    fb = _StubOCR("easyocr", tokens=[])
    p = _pipeline_with(primary, fb)
    out = p._ocr_printed_panel_assignments(_region(), _segs(), {1})
    assert out == {}
    assert p._pass0_diag["digit_tokens"] is False
    assert p._pass0_diag["read_failed"] is False


def test_p2a_diag_digits_seen_via_fallback():
    primary = _StubOCR("paddleocr", tokens=[])
    fb = _StubOCR("easyocr", tokens=[_Tok("3", 0.9, (5, 5, 10, 10))])
    p = _pipeline_with(primary, fb)
    out = p._ocr_printed_panel_assignments(_region(), _segs(), {3})
    assert 3 in out
    assert p._pass0_diag["digit_tokens"] is True


def test_p2a_diag_read_failed():
    class _Boom(_StubOCR):
        def recognize_panel(self, img, box):
            self.panel_calls += 1
            raise RuntimeError("onednn boom")

    primary = _Boom("paddleocr")
    p = _pipeline_with(primary, None)
    out = p._ocr_printed_panel_assignments(_region(), _segs(), {1})
    assert out == {}
    assert p._pass0_diag["read_failed"] is True


def test_converter_exempts_caption_order_positional():
    from rlpe.converters import _panel_review_reasons

    m = SimpleNamespace(
        species="Sp",
        panel_path="x.png",
        bbox=[0, 0, 10, 10],
        metadata={
            "association_method": "caption_order_positional",
            "panel_id_source": "phase67_segmentation_recovery",
        },
    )
    reasons = _panel_review_reasons(m)
    assert "missing_printed_panel_id" not in reasons


def test_converter_flags_plain_positional_without_printed_id():
    from rlpe.converters import _panel_review_reasons

    m = SimpleNamespace(
        species="Sp",
        panel_path="x.png",
        bbox=[0, 0, 10, 10],
        metadata={
            "association_method": "positional_fallback",
            "panel_id_source": "phase67_segmentation_recovery",
        },
    )
    reasons = _panel_review_reasons(m)
    assert "missing_printed_panel_id" in reasons


# ============================================================
# P1 — sample→unit→age chain
# ============================================================
_SECTIONS = [
    {
        "title": "Biostratigraphy",
        "text": (
            "Sample MB4 (Fig. 3) contained an association corresponding to "
            "UAZ 6; we have extended this correlation to UAZ 6-7 (middle "
            "Bathonian–early Callovian age). "
            "The O. Tazega section provided three diagnostic samples. In the "
            "lower part of the section, sample OTA1 (Fig. 4: sequence 1) has "
            "an assemblage correlative with UAZ 7 of late Bathonian or early "
            "Callovian age; sample OTA5 (Fig. 4: sequence 5) released an "
            "association attributed to UAZ 8 of middle Callovian–early "
            "Oxfordian age. "
            "Sample CHA2 corresponds to UAZ 5-6 of latest Bajocian– middle "
            "Bathonian age, a chronological correlation identical to JD4b at "
            "J. Jédidi. One meter above, sample CHA4 released an association "
            "attributed to UAZ 5-7 of latest Bajocian–early Callovian age."
        ),
    }
]


def test_sample_codes_match_aliases():
    from rlpe.geology_extraction import sample_codes_match

    assert sample_codes_match("CH4", "CHA4")
    assert sample_codes_match("cha-4", "CH 4")
    assert not sample_codes_match("MB4", "CHA4")
    assert not sample_codes_match("CH4", None)


def test_sample_unit_regex_alias_and_age():
    from rlpe.geology_extraction import extract_sample_unit_map_regex

    out = extract_sample_unit_map_regex(_SECTIONS, {"CH4"})
    assert "CH4" in out, out
    assert out["CH4"]["unit"] == "UAZ 5-7"
    assert "latest Bajocian" in (out["CH4"].get("age_text") or "")


def test_sample_unit_regex_multi_samples():
    from rlpe.geology_extraction import extract_sample_unit_map_regex

    out = extract_sample_unit_map_regex(_SECTIONS, {"MB4", "OTA1", "OTA5"})
    assert out["MB4"]["unit"] == "UAZ 6"
    assert out["OTA1"]["unit"] == "UAZ 7"
    assert "late Bathonian" in (out["OTA1"].get("age_text") or "")
    assert out["OTA5"]["unit"] == "UAZ 8"
    assert "early Oxfordian" in (out["OTA5"].get("age_text") or "")


def test_sample_unit_regex_unwanted_filtered():
    from rlpe.geology_extraction import extract_sample_unit_map_regex

    out = extract_sample_unit_map_regex(_SECTIONS, {"QQ1"})
    assert out == {}


def test_sample_unit_regex_tolerates_pdf_line_breaks():
    """OD section text wraps mid-sentence — the gaps must cross the
    newlines ("sample CHA4 released\\nan association attributed to\\nUAZ
    5-7 of latest Bajocian–early Callovian age")."""
    from rlpe.geology_extraction import extract_sample_unit_map_regex

    wrapped = [
        {
            "title": "Biostratigraphy",
            "text": (
                "One meter above, sample CHA4 released an association\n"
                "attributed to UAZ 5-7 of latest\n"
                "Bajocian–early Callovian age. No radiolarian has yet been\n"
                "recovered from the underlying beds."
            ),
        }
    ]
    out = extract_sample_unit_map_regex(wrapped, {"CH4"})
    assert "CH4" in out, out
    assert out["CH4"]["unit"] == "UAZ 5-7"
    assert "latest" in (out["CH4"].get("age_text") or "")


def test_parse_sample_resolution_response():
    from rlpe.geology_extraction import parse_sample_resolution_response

    got = parse_sample_resolution_response(
        {
            "sample_codes": {
                "CHA4": {"unit": "uaz 5 – 7", "section": "Chaâbane"},
                "ZZ9": {"unit": "UAZ 1"},
            }
        },
        {"CH4"},
    )
    assert got == {"CH4": {"unit": "UAZ 5-7", "section": "Chaâbane"}}


def test_prompt_carries_sample_schema():
    from rlpe.geology_extraction import build_unit_resolution_prompt

    sys_p, user_p = build_unit_resolution_prompt(_SECTIONS, {"UAZ A"}, {"CH4"})
    assert "sample_codes" in sys_p
    assert "CH4" in user_p


def test_item_split_boughdiri_paren_form():
    from rlpe.geology_extraction import extract_caption_item_context

    cap = (
        "Plate I\n\nJurassic radiolarians from the Jédidi Fm (Tunisia). "
        "Figures: taxon, sample number, specimen, scale. "
        "1) Ristola altissima altissima (RÜST), CH4, specimen 7, 550 µm; "
        "2) Palinandromeda podbielensis (OZVOLDOVA), MB4, specimen 15, 200 µm"
    )
    ctx1 = extract_caption_item_context(cap, "1")
    assert ctx1["sample_code"] is not None and "CH" in ctx1["sample_code"]
    ctx2 = extract_caption_item_context(cap, "2")
    assert ctx2["sample_code"] is not None and "MB" in ctx2["sample_code"]


def test_attach_prepends_resolved_sample_chain():
    from rlpe.pipeline import RadiolarianPipeline as Pipeline

    p = object.__new__(Pipeline)
    p._paper_unit_geology = {
        "p1": {
            "UAZ 5-7": {
                "age_text": "latest Bajocian–early Callovian age",
                "chronostratigraphy": "Middle Jurassic",
            }
        }
    }
    p._paper_sample_unit_geology = {"p1": {"CH 4": {"unit": "UAZ 5-7"}}}
    p._paper_sections_cache = {}
    p._paper_unit_geology_sections = {"p1": {}}
    cap = (
        "Plate I\n\nJurassic radiolarians from the Jédidi Fm (Tunisia). "
        "1) Ristola altissima altissima (RÜST), CH4, specimen 7, 550 µm"
    )
    rows = [
        {
            "figure_id": "fig1",
            "panel_id": "1",
            "caption_snippet": cap,
            "metadata": {},
        }
    ]
    out = p._attach_caption_unit_geology(rows, "p1", {"fig1": cap}, sections=[])
    md = out[0]["metadata"]
    links = md["geology_links"]
    assert links[0]["label"] == "sample_unit_geology", links
    assert links[0]["age"] == "latest Bajocian–early Callovian age"
    assert links[0]["biozone"] == "UAZ 5-7"
    assert md.get("needs_review") is not True


def test_attach_keeps_review_when_chain_unresolved():
    from rlpe.pipeline import RadiolarianPipeline as Pipeline

    p = object.__new__(Pipeline)
    p._paper_unit_geology = {"p1": {}}
    p._paper_sample_unit_geology = {"p1": {}}
    p._paper_sections_cache = {}
    p._paper_unit_geology_sections = {"p1": {}}
    cap = "Plate I\n\nJurassic radiolarians. 1) Somegenus somespecies, ZZ9, specimen 3, 100 µm"
    rows = [
        {
            "figure_id": "fig1",
            "panel_id": "1",
            "caption_snippet": cap,
            "metadata": {},
        }
    ]
    out = p._attach_caption_unit_geology(rows, "p1", {"fig1": cap}, sections=[])
    md = out[0]["metadata"]
    assert md.get("needs_review") is True
    assert "unresolved_unit_age" in md.get("review_reasons", [])
