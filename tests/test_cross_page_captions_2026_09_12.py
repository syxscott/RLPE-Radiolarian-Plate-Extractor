"""2026-09-12: journal-style cross-page caption binding (strategy 4).

Paleontological Journal house layout (afanasieva2020c): full-bleed
plate pages carry only a bare "Plate N" title + running header; the
species caption is a body paragraph "Figs. X–Y. Genus species
(Author): ..." on the NEXT page, whose text cites "Plate N". The
multi-evidence strategy must bind these captions WITHOUT touching
other layouts (same-page captions, singular Fig. paragraphs,
Explanation-style, reconstruction).
"""

from __future__ import annotations

from rlpe.opendataloader_extractor import _find_plate_captions


def _img(page: int, content_id: int = 0) -> dict:
    return {
        "type": "image",
        "id": f"img_{page}_{content_id}",
        "page number": page,
        "bounding box": [50, 50, 550, 780],
    }


PJ_CAPTION_P3 = (
    "Figs. 1\u20133. Albaillella tetraspinosa (Kozur, 1981): "
    "(1) specimen no. Ko-1979/II-2, scale bar 50 \u00b5m."
)
PJ_CAPTION_P4 = (
    "Figs. 1\u201311. Holdsworthella permica Kozur, 1981: "
    "(1) specimen PIN, no. 5481/16483, scale bar 100 \u00b5m."
)


def _pj_tree_page6_7() -> list[dict]:
    """Plate 3 on page 6 (images + bare title); caption on page 7."""
    return [
        {"type": "image", "id": "p6img1", "page number": 6, "bounding box": [40, 40, 560, 800]},
        {"type": "paragraph", "page number": 6, "content": "Plate 3"},
        {
            "type": "paragraph",
            "page number": 7,
            "content": (
                "REVISION OF THE GENERA SPINODEFLANDRELLA KOZUR, 1981. "
                "The genus Pseudoalbaillella was found in siliceous "
                "Permian sediments (Plate 3) of southwestern Japan."
            ),
        },
        {"type": "paragraph", "page number": 7, "content": PJ_CAPTION_P3},
    ]


def test_binds_caption_to_preceding_plate_page():
    caps = _find_plate_captions(_pj_tree_page6_7(), caption_window=5)
    p3 = [c for c in caps if c["plate_number"] == 3]
    assert p3, "Plate 3 caption must exist"
    merged = p3[0]
    # Bare title merged with the species caption.
    assert "tetraspinosa" in merged["content"]
    assert merged["recovered_via"] == "journal_cross_page"
    # page_number stays on the IMAGE page (forward-window compatible).
    assert merged["page_number"] == 6


def test_appends_when_bare_title_was_header_dropped():
    """Plate 4 variant: the plate page's only text was a running header
    (dropped by the header filter), so NO caption exists — the strategy
    appends a new caption anchored on the image page."""
    kids = [
        {"type": "image", "id": "p10img1", "page number": 10, "bounding box": [40, 40, 560, 800]},
        {
            "type": "paragraph",
            "page number": 10,
            "content": "1452 AFANASIEVA Plate 4 10 PALEONTOLOGICAL JOURNAL Vol. 54 No. 12 2020",
        },
        {
            "type": "paragraph",
            "page number": 11,
            "content": (
                "Body text discussing the specimens (Fig. 3; Plate 4) "
                "using a new discriminant analytical method."
            ),
        },
        {"type": "paragraph", "page number": 11, "content": PJ_CAPTION_P4},
    ]
    caps = _find_plate_captions(kids, caption_window=5)
    p4 = [c for c in caps if c["plate_number"] == 4]
    assert p4, "Plate 4 caption must be appended"
    assert "Holdsworthella permica" in p4[0]["content"]
    assert p4[0]["page_number"] == 10
    assert p4[0]["recovered_via"] == "journal_cross_page"


def test_no_plate_ref_on_caption_page_no_bind():
    """G2: without a "Plate N" reference on the caption page, no bind."""
    kids = [
        {"type": "image", "id": "p6img1", "page number": 6, "bounding box": [40, 40, 560, 800]},
        {"type": "paragraph", "page number": 6, "content": "Plate 3"},
        {
            "type": "paragraph",
            "page number": 7,
            "content": "Generic body text about ammonoids and conodonts.",
        },
        {"type": "paragraph", "page number": 7, "content": PJ_CAPTION_P3},
    ]
    caps = _find_plate_captions(kids, caption_window=5)
    merged = [c for c in caps if c.get("recovered_via") == "journal_cross_page"]
    assert not merged


def test_species_rich_caption_on_image_page_no_bind():
    """G3: a real species-bearing caption already on the image page
    means the figure is handled — the strategy must not re-bind."""
    kids = [
        {"type": "image", "id": "p6img1", "page number": 6, "bounding box": [40, 40, 560, 800]},
        {
            "type": "paragraph",
            "page number": 6,
            "content": "Plate 3. Spinodeflandrella tetraspinosa (pl. 3, figs. 1)",
        },
        {
            "type": "paragraph",
            "page number": 7,
            "content": "Discussion text citing (Plate 3) with more details.",
        },
        {"type": "paragraph", "page number": 7, "content": PJ_CAPTION_P3},
    ]
    caps = _find_plate_captions(kids, caption_window=5)
    p3 = [c for c in caps if c["plate_number"] == 3]
    # The ORIGINAL caption survives untouched; no cross-page merge.
    assert "Figs. 1\u20133" not in p3[0]["content"]
    assert not any(c.get("recovered_via") == "journal_cross_page" for c in caps)


def test_cross_page_false_disables_strategy():
    caps = _find_plate_captions(_pj_tree_page6_7(), caption_window=5, cross_page=False)
    assert not any(c.get("recovered_via") == "journal_cross_page" for c in caps)


def test_singular_fig_paragraph_untouched():
    """Singular "Fig. N." paragraphs belong to existing paths — the
    strategy must not treat them as cross-page plate captions."""
    kids = [
        {"type": "image", "id": "p6img1", "page number": 6, "bounding box": [40, 40, 560, 800]},
        {"type": "paragraph", "page number": 6, "content": "Plate 3"},
        {
            "type": "paragraph",
            "page number": 7,
            "content": "Location discussion citing (Plate 3).",
        },
        {
            "type": "paragraph",
            "page number": 7,
            "content": "Fig. 2. Some single figure caption with Haeckel 1862 text.",
        },
    ]
    caps = _find_plate_captions(kids, caption_window=5)
    # Plate 3 stays bare ("Plate 3" title only) — no cross-page merge.
    p3 = [c for c in caps if c["plate_number"] == 3]
    assert p3 and p3[0]["content"].strip() == "Plate 3"
