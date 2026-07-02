"""Regression tests for the v20 multi-paper routing fixes.

Three related bugs surfaced after v19:

1. **Multi-paper ``_collect_images_from_output_dir`` collision.** In a
   work_dir with N papers, ``rglob("*.json")`` returned only the
   FIRST JSON (alphabetically first by path) regardless of which
   paper the caller was processing. This caused ``_resolve_image_paths``
   to look up image indices in the wrong paper's image list, missing
   most images and silently producing empty ``image_paths`` for
   every plate after the first. Bandini pl05 (page 21, image id=1123)
   was the canonical victim: its image index 2383 in the global
   walk was computed against the boughdiri JSON, but its actual
   file was ``imageFile15.png`` in bandini's own images directory.

2. **Multi-paper ``_resolve_image_paths`` images_dir collision.**
   Strategy 2's fallback ``output_dir.rglob("*_images")`` picked the
   first matching images directory (often a different paper's), then
   used it to compute imageFileN indices for the current paper.

3. **Body-text ``Fig. N Photograph ...`` false-positive.** bandini's
   body text "Fig. 7 Photograph of the Early Cretaceous radiolarite..."
   was matched as a plate figure caption, hijacking page-27 images
   from plate 8 (which needs those images for itself).

These tests lock down the fixes: paper_id-scoped JSON lookup,
paper_id-scoped images_dir, and body-text rejection.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.opendataloader_extractor import (  # noqa: E402
    _collect_images_from_output_dir,
    _looks_like_fig_caption,
    _resolve_image_paths,
)

# ---------------------------------------------------------------------------
# _looks_like_fig_caption: body-text rejection
# ---------------------------------------------------------------------------


def test_rejects_fig_photograph_body_text():
    """The bandini 2011 body text "Fig. 7 Photograph of the Early
    Cretaceous radiolarite..." must NOT be classified as a plate
    figure caption — it would hijack page-27 images from plate 8."""
    body = (
        "Fig. 7 Photograph of the Early Cretaceous radiolarite, which "
        "crops out in 2-15 cm boudinaged layers of dark green to black "
        "radiolarian cherts rich in organic matter that weather orange "
        "(samples PR-SB26, PR-SB27, and PR-SB28)"
    )
    assert _looks_like_fig_caption(body) is False, (
        "Body-text 'Fig. N Photograph ...' must be rejected"
    )


def test_rejects_fig_photographs_plural():
    """Plural form 'Photographs' should also be rejected."""
    body = "Fig. 2 Photographs of the apparatus used in the field study"
    assert _looks_like_fig_caption(body) is False


def test_accepts_real_caption_starting_with_schematic():
    """Schematic / Diagram / Map are LEGITIMATE caption titles —
    only 'Photograph(s)' is body-text signal. (Per test_fig_caption_re
    the regex match on 'Fig 1 Schematic of the apparatus.' must
    remain valid.)"""
    assert _looks_like_fig_caption("Fig. 1 Schematic of the apparatus.")


def test_accepts_real_caption_with_stratigraphic():
    """Real captions like 'Fig. 1. Stratigraphic ranges...' must still
    pass."""
    assert _looks_like_fig_caption("Fig. 1. Stratigraphic ranges of radiolarian families.")


def test_rejects_short_body_text():
    """Defensive: very short 'Fig. 26' references are body text."""
    assert _looks_like_fig_caption("Fig. 26") is False


# ---------------------------------------------------------------------------
# _collect_images_from_output_dir: paper_id scoping
# ---------------------------------------------------------------------------


def _write_paper_json(paper_dir: Path, paper_id: str, num_images: int = 3) -> None:
    """Helper: write a synthetic OD JSON + images dir for one paper."""
    paper_dir.mkdir(parents=True, exist_ok=True)
    images_dir = paper_dir / f"{paper_id}_images"
    images_dir.mkdir(exist_ok=True)
    # Write minimal imageFile1.png etc. so _resolve_image_paths can find them.
    for i in range(1, num_images + 1):
        (images_dir / f"imageFile{i}.png").write_bytes(b"x")
    # Write the OD JSON with N image elements per page.
    kids = []
    for i in range(1, num_images + 1):
        kids.append(
            {
                "type": "image",
                "id": i,
                "page number": i,
            }
        )
    (paper_dir / f"{paper_id}.json").write_text(json.dumps({"kids": kids}))


def test_multi_paper_collect_scoped_by_paper_id():
    """In a work_dir with two papers A and B, calling
    _collect_images_from_output_dir with paper_id='A' must return
    A's images only — not B's."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        _write_paper_json(out / "od_output" / "aaaa", "aaaa", num_images=2)
        _write_paper_json(out / "od_output" / "bbbb", "bbbb", num_images=5)

        a_imgs = _collect_images_from_output_dir(out, paper_id="aaaa")
        b_imgs = _collect_images_from_output_dir(out, paper_id="bbbb")

        # A has 2 images (id 1, 2), B has 5 (id 1..5).
        assert len(a_imgs) == 2, f"AAAA: expected 2, got {len(a_imgs)}"
        assert len(b_imgs) == 5, f"BBBB: expected 5, got {len(b_imgs)}"
        # B's id=3 must appear in b_imgs but NOT in a_imgs.
        a_ids = {img.get("id") for img in a_imgs}
        b_ids = {img.get("id") for img in b_imgs}
        assert 3 in b_ids and 3 not in a_ids


def test_multi_paper_collect_without_paper_id_falls_back():
    """Without paper_id, the fallback must accumulate all papers'
    images (with possible (page, id) collisions; callers should
    pass paper_id when available)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        _write_paper_json(out / "od_output" / "aaaa", "aaaa", num_images=2)
        _write_paper_json(out / "od_output" / "bbbb", "bbbb", num_images=3)
        all_imgs = _collect_images_from_output_dir(out)
        assert len(all_imgs) == 5


def test_unknown_paper_id_returns_empty():
    """An unknown paper_id in a work_dir with no matching JSON must
    return [], not silently fall through to a wrong paper."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        _write_paper_json(out / "od_output" / "aaaa", "aaaa", num_images=3)
        # paper_id 'zzzz' doesn't exist
        result = _collect_images_from_output_dir(out, paper_id="zzzz")
        assert result == []


# ---------------------------------------------------------------------------
# _resolve_image_paths: paper_id-scoped images_dir lookup
# ---------------------------------------------------------------------------


def test_resolve_image_paths_with_paper_id_finds_correct_dir():
    """Strategy 2 must use paper_id to pick the correct images_dir,
    not the first alphabetic match."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        # Paper A: 3 images
        _write_paper_json(out / "od_output" / "aaaa", "aaaa", num_images=3)
        # Paper B: 5 images
        _write_paper_json(out / "od_output" / "bbbb", "bbbb", num_images=5)
        # Pass paper B's id=5 image (page=5) — must resolve to
        # bbbb's imageFile5.png, NOT aaaa's imageFile5.png (which
        # doesn't exist — aaaa only has 3 images).
        candidates = [{"page number": 5, "id": 5}]
        paths = _resolve_image_paths(candidates, out, paper_id="bbbb")
        assert len(paths) == 1
        assert "bbbb_images" in paths[0]
        assert paths[0].endswith("imageFile5.png")
        assert Path(paths[0]).exists()
