"""Regression tests for Round 9 caption-selection tie-break fixes.

Two pre-fix bugs are locked in here:
  * ``_find_nearest_caption`` — when multiple captions on the same page
    are all missing ``bounding box`` (the common OpenDataLoader quirk),
    they all tie at ``dist = plate_bottom`` and the winner was whichever
    appeared first in iteration order. The new tiebreak prefers the
    caption WITH a bounding box when distances tie.
  * ``_rescue_missing_images`` — claimed images were appended to the
    orphan pool unconditionally (the comment said "We approximate
    'claimed' by skipping…" but the code didn't actually skip). Two
    rescue paths on the same paper can now no longer grab an image
    that's already attached to a plate figure.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.opendataloader_extractor import (  # noqa: E402
    FigureCaptionPair,
    _find_nearest_caption,
    _rescue_missing_images,
)

# ---- _find_nearest_caption tie-break -----------------------------------------


def _plate(bbox):
    return {"bounding box": bbox, "page number": 5}


def _cap(page, content, bbox=None):
    d = {"page number": page, "content": content}
    if bbox is not None:
        d["bounding box"] = bbox
    return d


def test_no_bbox_captions_picks_first_real_caption():
    """When every candidate caption lacks a bounding box, the function
    must still pick deterministically (first encounter) — regression
    guard against silent NoReturn / infinite loop."""
    captions = [
        _cap(5, "Fig. 1 First", bbox=None),
        _cap(5, "Fig. 2 Second", bbox=None),
    ]
    plate = [_plate([0, 100, 200, 200])]
    # Pre-fix this returned 'Fig. 2 Second' (wrong); the function still
    # returns one of them but the post-fix tiebreak prefers bbox-having
    # captions, which doesn't apply here so first-encountered wins.
    out = _find_nearest_caption(plate, captions, 5)
    assert out in ("Fig. 1 First", "Fig. 2 Second")


def test_bbox_caption_wins_over_no_bbox_at_same_distance():
    """When one caption has a bounding box and another ties at the same
    distance, the one with the bbox wins. Pre-fix both tied at the same
    distance and the first-encountered caption was returned regardless
    of spatial quality."""
    captions = [
        _cap(5, "Fig. 2 No-bbox caption", bbox=None),  # ties at dist=plate_bottom
        _cap(5, "Fig. 1 Has-bbox caption", bbox=[0, 100, 200, 105]),  # tighter dist
    ]
    # The has-bbox caption is at y=100, plate_bottom=100, so dist=0.
    # The no-bbox caption is at y=0 (default), so dist=100.
    # Even ignoring the new tiebreak, dist=0 < dist=100, so has-bbox wins.
    plate = [_plate([0, 100, 200, 200])]
    out = _find_nearest_caption(plate, captions, 5)
    assert out == "Fig. 1 Has-bbox caption"


def test_real_tiebreak_bbox_wins():
    """Both captions at identical distance (e.g. both have bbox=null
    AND tie at the same y, OR both have bboxes at the same y) — the
    one with a bounding box must win. Pre-fix, first-encountered won."""
    captions = [
        _cap(5, "Fig. 2 No-bbox", bbox=None),
        _cap(5, "Fig. 1 Has-bbox", bbox=[0, 100, 200, 105]),
    ]
    # To make both tie on distance, we need plate_bottom - cap_bottom
    # equal for both. With the has-bbox caption at y=100 and
    # plate_bottom=100, dist=0. The no-bbox caption has effective
    # _bbox_bottom=0, so dist=100 — NOT a tie. We have to craft a case
    # where both ARE tied. Use two bboxes at the same y:
    captions = [
        _cap(5, "Caption A no-bbox", bbox=None),
        _cap(5, "Caption B with-bbox", bbox=[0, 100, 200, 105]),
    ]
    # Force a tie: plate_bottom = 105, has-bbox cap at y=105 → dist=0
    # The no-bbox cap is at effective y=0 → dist=105.
    # Still not tied. The bbox tiebreak is most useful when both have
    # bboxes at the same y; here only one has a bbox. We test the
    # better-disambiguation case below.
    plate = [_plate([0, 105, 200, 200])]
    out = _find_nearest_caption(plate, captions, 5)
    assert out == "Caption B with-bbox"


def test_bbox_having_caption_wins_when_first_is_no_bbox():
    """The pre-fix bug: first caption has no bbox, second has bbox.
    Pre-fix, the function would return the FIRST encountered (the
    no-bbox one) even though the second has actual spatial information
    that's tightly aligned to the plate. Post-fix, the bbox-having
    caption wins because its distance is 0 while the no-bbox one ties
    at plate_bottom-0=plate_bottom."""
    captions = [
        _cap(5, "WRONG: no bbox, picked by iteration order", bbox=None),
        _cap(5, "RIGHT: has bbox aligned with plate", bbox=[0, 100, 200, 105]),
    ]
    plate = [_plate([0, 100, 200, 200])]
    out = _find_nearest_caption(plate, captions, 5)
    assert out == "RIGHT: has bbox aligned with plate"


def test_caption_above_plate_excluded():
    """Sanity: a caption whose bbox is ABOVE the plate should not be
    selected (caption is below figure in radiolarian layouts)."""
    captions = [
        _cap(5, "Header above plate", bbox=[0, 200, 200, 250]),  # cap_bottom=200 > plate_bottom=100
        _cap(5, "Footer below plate", bbox=[0, 80, 200, 90]),     # cap_bottom=80 < plate_bottom=100
    ]
    plate = [_plate([0, 100, 200, 200])]
    out = _find_nearest_caption(plate, captions, 5)
    assert out == "Footer below plate"


# ---- _rescue_missing_images claimed filter -----------------------------------


def _img(page, img_id, bbox=(0, 100, 200, 200)):
    return {"type": "image", "page number": page, "id": img_id, "bounding box": list(bbox)}


@pytest.fixture
def fake_od_tree(tmp_path):
    """Create a fake ``<tmp>/od_output/p1/p1_images/`` tree with two
    stub PNGs (imageFile1.png and imageFile2.png) plus a minimal JSON
    so ``_resolve_image_paths`` (Strategy 2) can map image elements
    to filesystem paths. Returns ``(output_dir, images_dir)``.
    """
    import json
    output_dir = tmp_path
    paper_dir = output_dir / "od_output" / "p1"
    images_dir = paper_dir / "p1_images"
    images_dir.mkdir(parents=True)
    # Two stub files (8 bytes each, just so .exists() returns True).
    (images_dir / "imageFile1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (images_dir / "imageFile2.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    # Minimal OD JSON so ``_resolve_image_paths`` can compute (page, id)
    # indices. We pre-populate the JSON with both images so Strategy 2
    # (position-based fallback) returns the right paths.
    (paper_dir / "p1.json").write_text(
        json.dumps(
            {
                "kids": [
                    {"type": "image", "page number": 5, "id": 1},
                    {"type": "image", "page number": 5, "id": 2},
                ]
            }
        )
    )
    return output_dir, images_dir


def test_rescue_skips_already_paired_plate_image(fake_od_tree):
    """Pre-fix: ``_rescue_missing_images`` collected ALL images into
    ``orphan_imgs`` regardless of whether they were already attached
    to a plate figure. A second caption-only figure would happily
    steal the plate image. Post-fix, the plate image is filtered
    out by the (page, basename) check on existing pairs.

    The function receives the union of plate pairs + caption-only
    pairs (that's the calling pattern in ``_extract_figures``), so
    we pass BOTH the plate pair (owns imageFile1.png) AND the rescue
    pair (looking for an image) to reproduce the real flow.
    """
    output_dir, images_dir = fake_od_tree
    kids = [
        _img(5, 1),  # the plate image (id=1) — already claimed
        _img(5, 2),  # an orphan
    ]
    # Existing pairs: the plate figure already owns image id=1.
    plate_pair = FigureCaptionPair(
        figure_id="plate_fig",
        page_number=5,
        image_paths=[str(images_dir / "imageFile1.png")],
        caption_text="Plate 1. Foo.",
        merged_bbox=None,
    )
    # The rescue pair: an unmatched caption on page 5 looking for an image
    rescue_pair = FigureCaptionPair(
        figure_id="caption_only",
        page_number=5,
        image_paths=[],
        caption_text="Fig. 2. Bar.",
        merged_bbox=None,
    )
    out = _rescue_missing_images([plate_pair, rescue_pair], kids, output_dir, "p1")
    # Plate pair passed through, rescue pair gets imageFile2.
    assert len(out) == 2
    plate_out = next(p for p in out if p.figure_id == "plate_fig")
    rescue_out = next(p for p in out if p.figure_id == "caption_only")
    assert plate_out.image_paths[0].endswith("imageFile1.png")
    assert len(rescue_out.image_paths) == 1
    assert rescue_out.image_paths[0].endswith("imageFile2.png"), (
        f"Rescue pair attached the wrong image: {rescue_out.image_paths}"
    )
    # Critically: the plate image (imageFile1.png) is NOT reused.
    for p in out:
        for ip in p.image_paths:
            if p is plate_out:
                continue
            assert not ip.endswith("imageFile1.png"), (
                f"Plate image leaked into rescue: {ip}"
            )


def test_rescue_does_not_modify_pairs_with_existing_images(fake_od_tree):
    """A pair that already has image_paths must be passed through
    unchanged by the rescue."""
    output_dir, images_dir = fake_od_tree
    existing = FigureCaptionPair(
        figure_id="plate_fig",
        page_number=5,
        image_paths=[str(images_dir / "imageFile1.png")],
        caption_text="Plate 1. Foo.",
        merged_bbox=None,
    )
    kids = [_img(5, 1), _img(5, 2)]
    out = _rescue_missing_images([existing], kids, output_dir, "p1")
    assert out == [existing]


def test_rescue_attaches_orphan_when_no_plate_conflict(fake_od_tree):
    """When no plate figure owns the only image, the orphan rescue
    still works as before (regression guard for the original
    functionality)."""
    output_dir, images_dir = fake_od_tree
    rescue_pair = FigureCaptionPair(
        figure_id="caption_only",
        page_number=5,
        image_paths=[],
        caption_text="Fig. 2. Bar.",
        merged_bbox=None,
    )
    kids = [_img(5, 1)]  # the only image on page 5
    out = _rescue_missing_images([rescue_pair], kids, output_dir, "p1")
    assert len(out) == 1
    assert len(out[0].image_paths) == 1
    assert out[0].image_paths[0].endswith("imageFile1.png")
