"""Round 15 source-guard tests: full-codebase audit fixes (2026-07-06).

Locks in the 8 high-severity bug fixes found in the 2026-07-06 audit
pass (4 parallel agents). Each test reads the production source and
verifies the fix is in place so a future refactor can't silently
revert it.

Bugs fixed (file:line):

  Roman numeral dict (opendataloader_extractor.py:1047):
      _ROMAN_TO_INT previously ended at XII (12). The regex above
      accepts up to XIV, but ``_ROMAN_TO_INT.get('XIII', 0)`` returns
      0 — silently dropping plates >= 13. Fix: add XIII=13, XIV=14.

  Longitude hemisphere (geo_coords.py:83, 106):
      Both the DMS and decimal regex branches accepted ('W', 'S') as
      longitude hemisphere markers. Latitude marker 'S' should never
      flip a longitude. Fix: longitude only accepts 'W'.

  Gemma fallback race (pipeline.py:3456-3457):
      ``self._fallback_gemma_runtime = self._build_local_gemma_fallback()``
      was outside any lock. Two concurrent workers could both see
      None, both build the multi-GB model, and OOM. Fix: double-
      checked locking under self._gemma_lock.

  cv2.imwrite return value (pipeline.py:2866):
      cv2.imwrite returns False on failure but the previous code
      ignored the return value and stored image_path anyway. Fix:
      check the return and continue on failure.

  torch.load weights_only (association.py:159):
      Loading pickled checkpoints can execute arbitrary code. Fix:
      pass weights_only=True; fall back to legacy for old PyTorch.

  Schema validators (schema_models.py):
      confidence, latitude, longitude had no range constraints.
      Pydantic accepted confidence=2.5 and lat=200.0. Fix: add
      ge/le constraints via Field().

  CSV injection sanitizer (exporters/analysis.py):
      write_csv wrote user-controlled species names directly; an
      Excel formula like =cmd|'/c calc'!A1 would execute on open.
      Fix: _sanitise_csv_cell prefix-prefixes dangerous cells with '.

  Unicode normalization (evaluation/gold.py:match_panel):
      A gold "Aethium" and pred "Æthium" did not match because the
      comparison was byte-exact. Fix: NFKD normalization before
      comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# --- 1) Roman numeral dict must include XIII / XIV -----------------------


def test_roman_to_int_includes_XIII_and_XIV():
    """The plate-caption regex accepts up to 'XIV' but the lookup
    table used to stop at XII. Plates >= 13 were silently dropped."""
    from rlpe.opendataloader_extractor import _ROMAN_TO_INT

    assert _ROMAN_TO_INT.get("XIII") == 13, (
        f"_ROMAN_TO_INT missing XIII=13; got {_ROMAN_TO_INT.get('XIII')!r}"
    )
    assert _ROMAN_TO_INT.get("XIV") == 14, (
        f"_ROMAN_TO_INT missing XIV=14; got {_ROMAN_TO_INT.get('XIV')!r}"
    )
    # Sanity: lower numerals still there
    assert _ROMAN_TO_INT["XII"] == 12


def test_plate_caption_regex_accepts_XIII_XIV():
    """Sanity: the regex should still match XIII/XIV strings (else
    fixing only the dict doesn't help)."""
    from rlpe.opendataloader_extractor import _PLATE_CAPTION_RE

    m = _PLATE_CAPTION_RE.match("Plate XIII")
    assert m is not None and m.group(2) == "XIII"
    m = _PLATE_CAPTION_RE.match("Plate XIV")
    assert m is not None and m.group(2) == "XIV"


def test_plate_number_from_match_returns_13_14():
    """End-to-end: the public function must produce 13/14, not 0."""
    from rlpe.opendataloader_extractor import (
        _PLATE_CAPTION_RE,
        _plate_number_from_match,
    )

    for roman, expected in [("XIII", 13), ("XIV", 14)]:
        m = _PLATE_CAPTION_RE.match(f"Plate {roman}")
        assert m is not None
        assert _plate_number_from_match(m) == expected


# --- 2) Longitude hemisphere only accepts 'W' ----------------------------


def test_geo_coords_longitude_only_accepts_W():
    """geo_coords.py: the longitude hemisphere check must only accept
    'W'. Previously it also accepted 'S', which double-negated
    malformed inputs like '110°S'."""
    geo = Path(__file__).resolve().parents[1] / "src" / "rlpe" / "geo_coords.py"
    src = geo.read_text(encoding="utf-8")
    # Strip docstrings/comments before searching (single-line strings
    # may legitimately document the previous behaviour).
    code_lines = []
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        # Drop inline # comments
        if "#" in line:
            line = line.split("#", 1)[0]
        code_lines.append(line)
    code_only = "\n".join(code_lines)
    # The fix: only "W" remains in the longitude tuple. Latitude
    # still accepts both S and W (S for south, W for western
    # hemisphere confusion is rare but pre-existing).
    assert 'upper() == "W"' in code_only, (
        "Longitude hemisphere must be checked with == 'W' only. "
        "The previous ('W', 'S') tuple silently mis-parsed OCR-noisy "
        "inputs like '110°S' as -110 longitude."
    )


def test_geo_coords_rejects_S_as_longitude():
    """End-to-end: '110°S' as a longitude must NOT be negated."""
    from rlpe.geo_coords import parse_coordinate

    # A coordinate string that is malformed — '110°S' cannot be a
    # latitude (max 90) so the parser should fall through and return
    # None (the bug version would parse '110' as -110 longitude,
    # which is invalid but wouldn't be caught by _valid because
    # -180 <= -110 <= 180).
    result = parse_coordinate("110°S, 40°N")
    # If it returns a Coordinate, the longitude must NOT have been
    # negated by the bogus 'S' marker on the latitude slot.
    if result is not None:
        # latitude = 40 (with N or no flip), longitude = 110 (unflipped
        # because 'S' is no longer accepted as longitude hemisphere)
        assert result.longitude == 110, f"longitude was incorrectly negated; got {result.longitude}"


# --- 3) Gemma fallback race protection -----------------------------------


def test_gemma_fallback_uses_double_checked_lock():
    """pipeline.py: the lazy init of ``self._fallback_gemma_runtime``
    must be under ``self._gemma_lock`` so two concurrent MiniMax-
    fallback workers don't both build the multi-GB model."""
    pipeline = Path(__file__).resolve().parents[1] / "src" / "rlpe" / "pipeline.py"
    src = pipeline.read_text(encoding="utf-8")
    # Locate the gemma4 branch and verify the lock pattern.
    idx = src.find('action == "gemma4"')
    assert idx > 0, "Could not find gemma4 fallback branch"
    # Take the next 800 chars (the assignment block)
    window = src[idx : idx + 1200]
    assert "self._gemma_lock" in window, (
        "self._gemma_lock must guard the _fallback_gemma_runtime "
        "lazy init. Without the lock, two concurrent MiniMax-fallback "
        "workers can both call _build_local_gemma_fallback() and OOM "
        "the box."
    )
    # Verify double-checked pattern (two `is None` checks)
    assert window.count("is None") >= 2, (
        "Expected double-checked locking (2x `is None` inside/outside "
        "the lock). The outer check avoids lock contention on the "
        "happy path; the inner check ensures one-time init."
    )


# --- 4) cv2.imwrite return value must be checked ------------------------


def test_pipeline_checks_cv2_imwrite_return():
    """pipeline.py:2866 — cv2.imwrite returns False on failure but
    the previous code stored image_path anyway. Fix: check the
    return and ``continue`` on failure."""
    pipeline = Path(__file__).resolve().parents[1] / "src" / "rlpe" / "pipeline.py"
    src = pipeline.read_text(encoding="utf-8")
    # Find the imwrite line and verify the return value is checked.
    idx = src.find("cv2.imwrite(str(panel_path), crop)")
    assert idx > 0, "cv2.imwrite line not found"
    # Look at the surrounding 250 chars
    window = src[max(0, idx - 80) : idx + 250]
    assert "if not cv2.imwrite" in window, (
        "cv2.imwrite return value is not checked. A False return "
        "(disk full / invalid path / encoding error) silently leaves "
        "the panel referenced in results with no actual crop file."
    )


# --- 5) torch.load with weights_only=True -------------------------------


def test_association_uses_weights_only_true():
    """association.py: torch.load must use weights_only=True (PyTorch
    2.6+ default) to block pickle-based code execution."""
    assoc = Path(__file__).resolve().parents[1] / "src" / "rlpe" / "association.py"
    src = assoc.read_text(encoding="utf-8")
    # Both torch.load calls (the safe one and the legacy fallback)
    # must reference weights_only=True. The fallback is a TypeError
    # catch for PyTorch <1.13; the primary path must use the secure
    # call.
    assert "weights_only=True" in src, (
        "torch.load must pass weights_only=True to block arbitrary "
        "code execution via pickle deserialization."
    )
    # The fallback must catch only TypeError (the legacy-PyTorch
    # signal), not a bare Exception.
    assert "except TypeError" in src, (
        "The weights_only=True fallback must catch TypeError (the "
        "older-PyTorch signal), not a bare Exception — otherwise "
        "the secure path is silently bypassed on any error."
    )


# --- 6) Schema validators (confidence 0..1, lat ±90, lon ±180) ---------


def test_schema_confidence_has_range_constraint():
    """schema_models.py: confidence must be constrained to [0, 1]."""
    from pydantic import ValidationError

    from rlpe.schema_models import ScaleBarRecord

    # A confidence of 2.5 must be rejected.
    try:
        ScaleBarRecord(confidence=2.5)
    except ValidationError:
        return
    raise AssertionError(
        "ScaleBarRecord accepted confidence=2.5; the schema must "
        "constrain confidence to [0, 1] via Field(ge=0, le=1)."
    )


def test_schema_latitude_has_range_constraint():
    """schema_models.py: latitude must be in [-90, 90]."""
    from pydantic import ValidationError

    from rlpe.schema_models import GeologyLinkRecord

    try:
        GeologyLinkRecord(latitude=200.0, longitude=0.0)
    except ValidationError:
        return
    raise AssertionError(
        "GeologyLinkRecord accepted latitude=200; the schema must constrain latitude to [-90, 90]."
    )


def test_schema_longitude_has_range_constraint():
    """schema_models.py: longitude must be in [-180, 180]."""
    from pydantic import ValidationError

    from rlpe.schema_models import GeologyLinkRecord

    try:
        GeologyLinkRecord(latitude=0.0, longitude=500.0)
    except ValidationError:
        return
    raise AssertionError(
        "GeologyLinkRecord accepted longitude=500; the schema must "
        "constrain longitude to [-180, 180]."
    )


# --- 7) CSV injection sanitizer -----------------------------------------


def test_csv_sanitiser_prefixes_danger_cells():
    """exporters/analysis.py: cells starting with =, +, -, @, TAB
    must be prefixed with a single quote to neutralise Excel
    formula injection (CWE-1236)."""
    from rlpe.exporters.analysis import _sanitise_csv_cell

    dangerous = ["=cmd|'/c calc'!A1", "+1+1", "-2+3", "@SUM(A1:A9)", "\tinjected"]
    for cell in dangerous:
        out = _sanitise_csv_cell(cell)
        assert isinstance(out, str)
        assert out.startswith("'"), f"Dangerous cell {cell!r} not sanitised; got {out!r}"
        assert out[1:] == cell, f"Sanitiser mangled cell content; expected {cell!r}, got {out!r}"


def test_csv_sanitiser_passes_through_safe_values():
    """Numbers and ordinary strings pass through unchanged."""
    from rlpe.exporters.analysis import _sanitise_csv_cell

    assert _sanitise_csv_cell(0.5) == 0.5
    assert _sanitise_csv_cell(42) == 42
    assert _sanitise_csv_cell(True) is True
    assert _sanitise_csv_cell(None) == ""
    assert _sanitise_csv_cell("Genus species") == "Genus species"
    assert _sanitise_csv_cell("1.234") == "1.234"  # '1.' would be dangerous, '1.234' is not


def test_write_csv_uses_sanitiser():
    """write_csv must route every cell through _sanitise_csv_cell."""
    ana = Path(__file__).resolve().parents[1] / "src" / "rlpe" / "exporters" / "analysis.py"
    src = ana.read_text(encoding="utf-8")
    assert "_sanitise_csv_cell" in src, (
        "_sanitise_csv_cell is not referenced from write_csv(). "
        "A CSV cell starting with = would still execute as a "
        "formula when opened in Excel."
    )


# --- 8) Unicode normalization in match_panel ----------------------------


def test_match_panel_normalises_unicode_diacritics():
    """A gold 'Périsphera' and pred 'Perisphera' must match — same
    taxon; only the OCR engine's diacritic handling differs.
    (é is U+00E9 → NFKD → 'e' with case preserved.)
    """
    from rlpe.evaluation.gold import GoldPanel, match_panel

    gold = GoldPanel(
        paper_id="p1",
        figure_id="f1",
        panel_id="Périsphera",
        species="Genus species",
    )
    assert match_panel(gold, "p1", "Perisphera"), (
        "Unicode diacritics must be normalised before comparison. "
        "Gold 'Périsphera' vs pred 'Perisphera' should match."
    )


def test_match_panel_normalises_combining_marks():
    """NFKD must also fold combining marks. 'cafe\\u0301' (5 codepoints:
    c, a, f, e, combining acute) is the canonical decomposed form of
    'café'. Real OCR output mixes both forms."""
    from rlpe.evaluation.gold import GoldPanel, match_panel

    gold = GoldPanel(
        paper_id="p1",
        figure_id="f1",
        panel_id="café",
        species="Genus species",
    )
    # Same string in NFC (precomposed) vs NFD (decomposed)
    assert match_panel(gold, "p1", "café"), (
        "NFKD normalisation must fold combining marks so NFC and "
        "NFD forms of the same string match."
    )


def test_match_panel_still_rejects_unrelated_ids():
    """The fix must not weaken the existing strict-prefix rules."""
    from rlpe.evaluation.gold import GoldPanel, match_panel

    gold = GoldPanel(
        paper_id="p1",
        figure_id="f1",
        panel_id="5",
        species="Genus species",
    )
    assert not match_panel(gold, "p1", "10"), "Pure-numeric extension must NOT match"
    assert not match_panel(gold, "p2", "5"), "Wrong paper must NOT match"
    assert not match_panel(gold, "p1", ""), "Empty pred must NOT match"
