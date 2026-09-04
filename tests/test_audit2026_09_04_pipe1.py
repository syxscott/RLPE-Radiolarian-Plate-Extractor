"""audit 2026-09-04 pipe-1: _extract_unpaired_captions image ownership.

The rescue attached the same physical plate image to several different
figure_ids: once to the plate pair that legitimately owns it and once
per rescued ``Fig. N`` caption (no ``claimed`` set, no per-call ``used``
set). Downstream the pipeline iterates figures with no image-path dedup,
so the identical PNG was segmented once per caption and each pass
applied a *different* caption's species list to the same panels —
duplicate occurrence rows plus species credited to a caption that never
printed them.

The sibling rescue ``_rescue_missing_images`` already implements exactly
this protection (``claimed_basenames`` + ``rescued_used_keys``); this
pins the same contract on ``_extract_unpaired_captions``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.opendataloader_extractor import FigureCaptionPair, OpenDataLoaderExtractor


def _make_extractor() -> OpenDataLoaderExtractor:
    ext = OpenDataLoaderExtractor.__new__(OpenDataLoaderExtractor)
    # Phase 28: ``_extract_unpaired_captions`` reads ``self.caption_window``.
    ext.caption_window = 5
    return ext


def _write_od(tmp: Path, kids: list[dict]) -> None:
    paper_dir = tmp / "od_output" / "paperX"
    paper_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / "paperX.json").write_text(json.dumps({"kids": kids}), encoding="utf-8")


def _dupes(pairs: list[FigureCaptionPair]) -> dict[str, int]:
    return {
        name: n
        for name, n in Counter(
            Path(p).name for f in pairs for p in (f.image_paths or [])
        ).items()
        if n > 1
    }


def test_rescued_captions_do_not_steal_an_existing_figures_image():
    """The plate pair owns imageFile1.png; two rescued Fig. captions must
    not both receive it (pre-fix: both did -> 3x processing of one PNG)."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        imgdir = tmp / "od_output" / "paperX" / "paperX_images"
        imgdir.mkdir(parents=True)
        (imgdir / "imageFile1.png").write_bytes(b"\x89PNG")
        kids = [
            {
                "type": "caption",
                "page number": 3,
                "content": "Fig. 1. Location map of the studied area",
            },
            {
                "type": "caption",
                "page number": 4,
                "content": "Fig. 2. Paleogeographic reconstruction",
            },
            {
                "type": "image",
                "page number": 5,
                "id": 1,
                "bounding box": [0, 0, 100, 200],
                "source": "paperX_images/imageFile1.png",
            },
        ]
        _write_od(tmp, kids)
        existing = [
            FigureCaptionPair(
                figure_id="od_plate_paperX_p005_pl01",
                page_number=5,
                image_paths=[str(imgdir / "imageFile1.png")],
                caption_text="Plate 1. Radiolaria",
                merged_bbox=None,
            )
        ]
        rescued = _make_extractor()._extract_unpaired_captions(
            {"kids": kids}, existing, tmp, "paperX"
        )
        owned = {Path(p).name for f in existing for p in (f.image_paths or [])}
        for r in rescued:
            overlap = owned & {Path(p).name for p in (r.image_paths or [])}
            assert not overlap, (
                f"{r.figure_id} re-attached an image already owned by an "
                f"existing figure: {overlap}. pipe-1 duplicate-processing "
                f"regression."
            )
        assert not _dupes(existing + rescued), (
            f"Same image attached to several figures: {_dupes(existing + rescued)}"
        )


def test_two_rescued_captions_cannot_share_one_image():
    """Two rescued captions and one unclaimed image: exactly one caption
    may claim it (mirrors ``_rescue_missing_images.rescued_used_keys``)."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        imgdir = tmp / "od_output" / "paperX" / "paperX_images"
        imgdir.mkdir(parents=True)
        (imgdir / "imageFile1.png").write_bytes(b"\x89PNG")
        kids = [
            {
                "type": "caption",
                "page number": 3,
                "content": "Fig. 1. Location map of the studied area",
            },
            {
                "type": "caption",
                "page number": 4,
                "content": "Fig. 2. Paleogeographic reconstruction",
            },
            {
                "type": "image",
                "page number": 4,
                "id": 1,
                "bounding box": [0, 0, 100, 200],
                "source": "paperX_images/imageFile1.png",
            },
        ]
        _write_od(tmp, kids)
        rescued = _make_extractor()._extract_unpaired_captions(
            {"kids": kids}, [], tmp, "paperX"
        )
        assert len(rescued) == 2, f"expected both captions rescued, got {rescued!r}"
        dupes = _dupes(rescued)
        assert not dupes, (
            f"Two rescued captions share one image: {dupes}. pipe-1 regression."
        )


def test_rescued_captions_still_claim_a_stub_owned_image():
    """Round 21 contract must survive: a stub pair (empty caption_text)
    that OD did pair with an image is not 'represented', so the real
    Fig. caption must still be able to claim that image."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        imgdir = tmp / "od_output" / "paperX" / "paperX_images"
        imgdir.mkdir(parents=True)
        (imgdir / "imageFile1.png").write_bytes(b"\x89PNG")
        kids = [
            {
                "type": "caption",
                "page number": 3,
                "content": "Fig. 2. Overview of Tunisian Jurassic stratigraphy",
            },
            {
                "type": "image",
                "page number": 3,
                "id": 1,
                "bounding box": [0, 0, 100, 200],
                "source": "paperX_images/imageFile1.png",
            },
        ]
        _write_od(tmp, kids)
        stub = FigureCaptionPair(
            figure_id="od_fig_paperX_p002_01",
            page_number=2,
            image_paths=[str(imgdir / "imageFile1.png")],
            caption_text="",  # FALLBACK-branch stub
            merged_bbox=None,
        )
        rescued = _make_extractor()._extract_unpaired_captions(
            {"kids": kids}, [stub], tmp, "paperX"
        )
        assert rescued, "Round 21 regression: real Fig. caption was not rescued"
        assert any(
            Path(p).name == "imageFile1.png"
            for r in rescued
            for p in (r.image_paths or [])
        ), (
            "Rescue stopped claiming a stub-owned image — the Round 21 "
            "real-caption-wins contract regressed."
        )
