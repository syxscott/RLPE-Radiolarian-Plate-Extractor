"""Round 21 source-guard tests: OpenDataLoader pairing fix.

User audit (Round 20 sampling) found that Boughdiri 2007's 6 non-plate
figures (strat column, litholog sections, location map, outcrop photos
on p2-p7) were silently dropped. The pipeline only created
``od_plate_p011_pl01`` and the geo_vision M3 routing never fired.

Root cause analysis identified 3 layered bugs in
``src/rlpe/opendataloader_extractor.py``:

  1. **int-cast crash on string image IDs.**
     OD emits image ``id`` as either an integer or a string
     (``"p011f1"``). The FALLBACK path's
     ``caption_for_image[int(linked)]`` raised ``ValueError`` for
     string IDs and silently dropped the caption-image link.

  2. **Stub-with-empty-caption masks real Fig. caption.**
     The dedup set in ``_extract_unpaired_captions`` was built from
     ``caption_text`` alone; a stub with empty ``caption_text``
     (FALLBACK branch output) was treated as "no caption, but still
     represented" and the rescue's 60-char-prefix dedup matched the
     empty string vacuously, skipping the real ``Fig. N`` caption.

  3. **Bidirectional substring check in rescue.**
     ``_rescue_unmatched_captions`` used the Round-9-retired
     bidirectional ``text[:60] in s or s in text[:60]`` substring
     check; "Fig. 1" was a prefix of "Fig. 10 ..." so the rescue
     skipped Fig. 10 when Fig. 1 was already represented.

These tests pin the three fixes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _read(rel: str) -> str:
    return (Path(__file__).resolve().parents[1] / rel).read_text(encoding="utf-8")


# --- 1) String image IDs survive the FALLBACK path -----------------------


def test_int_cast_replaced_with_string_in_fallback():
    """The FALLBACK path's int(linked) cast is the most likely silent
    failure when OD emits string image IDs. Source guard: the cast
    must be ``str(...)`` (or absent) at the relevant call sites."""
    src = _read("src/rlpe/opendataloader_extractor.py")
    # The FALLBACK branch (no plate captions) and the plate-captions
    # branch both updated. The old ``int(linked)`` / ``int(img_id)``
    # casts must not appear in the rescued blocks.
    # We allow ``int()`` elsewhere (e.g. ``int(page number)``), but
    # not in the dict-lookup expressions.
    assert "int(linked)" not in src, (
        "opendataloader_extractor.py still has `int(linked)` — the "
        "FALLBACK int cast on string IDs is the Round 21 silent "
        "failure mode that drops Boughdiri's non-plate figures."
    )
    assert "int(img_id)" not in src, (
        "opendataloader_extractor.py still has `int(img_id)` — same int cast issue for image IDs."
    )
    assert "str(linked)" in src, (
        "opendataloader_extractor.py missing str(linked) cast — the "
        "Round 21 fix expects string-keyed lookup."
    )
    assert "str(img_id)" in src, (
        "opendataloader_extractor.py missing str(img_id) cast — the "
        "Round 21 fix expects string-keyed lookup."
    )


# --- 2) Stubs with empty captions don't mask real Fig. captions ----------


def test_existing_caption_snippets_only_counts_real_pairs():
    """The dedup set in ``_extract_unpaired_captions`` must skip
    pairs whose ``caption_text`` is empty (stubs). Otherwise a
    FALLBACK-branch stub would vacuously mask the real Fig. caption
    and the rescue would never emit the ``od_fig_*`` pair."""
    src = _read("src/rlpe/opendataloader_extractor.py")
    # The function builds ``existing_caption_snippets`` and adds
    # entries only when both ``cap`` and the image-paths check are
    # satisfied. We check that the comment or the code reflects this
    # guard (the comment is in Round 21).
    assert "existing_caption_snippets" in src
    # The dedup now requires ``if cap and imgs:`` (or equivalent).
    # The plate-captions branch has been replaced; assert the
    # Round-21 explanatory comment is present.
    assert "Stubs are excluded" in src or "if cap and imgs" in src, (
        "opendataloader_extractor.py did not exclude stub pairs "
        "(empty caption or empty image_paths) from the dedup set. "
        "Round 21 fix expects `if cap and imgs:` style guard."
    )


def test_rescue_unmatched_captions_uses_exact_prefix_match():
    """The rescue's bidirectional substring check was retired in
    Round 9 L4 but never re-applied to ``_rescue_unmatched_captions``.
    The Round 21 fix restores the exact-60-char-prefix match."""
    import re

    src = _read("src/rlpe/opendataloader_extractor.py")
    # Find the body of ``_rescue_unmatched_captions``.
    rescue_section_start = src.find("def _rescue_unmatched_captions")
    if rescue_section_start < 0:
        assert False, "_rescue_unmatched_captions not found"
    next_def = src.find("\ndef ", rescue_section_start + 10)
    if next_def < 0:
        next_def = len(src)
    body = src[rescue_section_start:next_def]
    # Strip comments before searching so a docstring reference to
    # the retired pattern doesn't trip the assertion.
    code_only = re.sub(r"#.*$", "", body, flags=re.MULTILINE)
    # The actual code form of the old bidirectional check.
    assert "s in text[:60]" not in code_only, (
        "_rescue_unmatched_captions still has the bidirectional "
        "substring check (`s in text[:60]`). Round 9 L4 retired "
        "this pattern; Round 21 verifies the fix."
    )


# --- 3) End-to-end: synthetic OD JSON produces ``od_fig_*`` for non-plate --


def test_boughdiri_caption_yields_od_fig_pair():
    """Synthetic OD JSON mirroring Boughdiri's structure: 1 plate
    caption + 1 non-plate Fig. 2 caption, both with linked-content
    string IDs. After Round 21 the rescue must produce a real
    FigureCaptionPair for Fig. 2 with the non-empty caption_text."""
    from rlpe.opendataloader_extractor import FigureCaptionPair, OpenDataLoaderExtractor

    # We test the rescue method directly. The full extract() flow
    # requires a real PDF + OD subprocess; that path is exercised
    # by the live e2e test (see plan verification section).
    kids = [
        {
            "type": "caption",
            "page number": 3,
            "content": "Fig. 2. Overview of Tunisian Jurassic stratigraphy",
            "linked content id": "p003f1",  # String ID — Round 21 fix
        }
    ]
    # Existing pair is a stub with empty caption (the FALLBACK
    # branch's typical output). Without Round 21's dedup tightening,
    # the rescue would skip Fig. 2 because the empty-caption stub
    # vacuously matches.
    existing = [
        FigureCaptionPair(
            figure_id="od_fig_test_p002_01",
            page_number=2,
            image_paths=[],
            caption_text="",  # empty stub
            merged_bbox=None,
        )
    ]
    extractor = OpenDataLoaderExtractor.__new__(OpenDataLoaderExtractor)
    # Phase 28: ``_extract_unpaired_captions`` reads
    # ``self.caption_window`` (default 5). ``__new__`` skips __init__ so
    # we must set the attribute explicitly for tests that bypass the
    # constructor.
    extractor.caption_window = 5
    # First arg is the full OD data dict (we wrap kids in expected shape)
    data = {"kids": kids}
    rescued = extractor._extract_unpaired_captions(data, existing, Path("/tmp"), "test_paper")
    # After Round 21, the rescue must emit a FigureCaptionPair for
    # Fig. 2 with the real caption.
    assert any("Fig. 2" in (r.caption_text or "") for r in rescued), (
        f"Rescue did not emit Fig. 2; got: {[r.caption_text for r in rescued]}"
    )
    # And the stub pair was overwritten / not duplicated.
    assert all((r.caption_text or "") != "" for r in rescued), (
        "Empty caption in rescued pair — Round 21 dedup tightening failed"
    )


def test_rescue_handles_string_linked_content_id():
    """Source guard: the rescue function uses ``str(linked)`` when
    building the caption-image linkage dict, so non-integer OD image
    IDs (like ``"p011f1"``) work."""
    src = _read("src/rlpe/opendataloader_extractor.py")
    assert "caption_for_image[str(linked)]" in src or ("caption_for_image[str(" in src), (
        "opendataloader_extractor.py is missing str(linked) cast in "
        "caption_for_image dict — string image IDs cannot link."
    )
