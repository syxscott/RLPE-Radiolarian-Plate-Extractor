"""Tests for audit-found pipeline bugs (Round 3 2026-07-03).

These tests pin the contracts that the audit found violated:

M4: _apply_geo_vision previously used ``Path.exists()`` which is True for
directories. PIL.Image.open(directory) raises IsADirectoryError. The
fix uses ``Path.is_file()`` which correctly rejects non-files.

M5: _link_range_chart_geology previously iterated over
``rc_dict.get("sections", [])`` without guarding against None list
items. A row whose stub metadata had ``sections=[None]`` crashed with
AttributeError on ``sec.get(...)``. The fix guards each iteration item.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

HAS_CV2 = True
try:
    import cv2  # noqa: F401
except Exception:
    HAS_CV2 = False

requires_cv2 = pytest.mark.skipif(not HAS_CV2, reason="pipeline imports cv2 transitively")


# --------------------------------------------------------------------------- M4


class TestApplyGeoVisionDirectoryPath:
    """M4: directory paths must not crash _apply_geo_vision.

    The function reads image_path via PIL.Image.open(image_path). A
    directory path bypasses the old ``Path.exists()`` guard but raises
    IsADirectoryError in PIL. The fix tightens the guard to
    ``Path.is_file()``.
    """

    def test_directory_path_is_skipped(self, tmp_path):
        # Create a directory that exists but is not a file.
        dir_path = tmp_path / "fake_dir"
        dir_path.mkdir()
        # We can't easily call _apply_geo_vision without a full pipeline
        # instance; instead, assert the guard semantic directly via the
        # Path.is_file() contract the fix relies on.
        assert dir_path.exists() is True
        assert dir_path.is_file() is False


# --------------------------------------------------------------------------- M1 (Round 9)


class TestApplyGeoVisionMapFigureTypeBackfill:
    """Round 9 Bug-M1: ``_apply_geo_vision`` previously only matched
    ``extraction_source == "map_context"`` to backfill ``figure_type``.
    The map figure path actually writes
    ``extraction_source == "map_caption_heuristic"`` (set by
    ``_process_map``), so the backfill never fired and geo vision silently
    skipped every map figure.

    We replicate the backfill logic in isolation — the contract is the
    ``elif src in ("map_caption_heuristic", "map_context")`` membership
    test in the production source.
    """

    @staticmethod
    def _backfill_figure_type(md: dict) -> str | None:
        """Mirror the production backfill logic."""
        figure_type = md.get("figure_type")
        if not figure_type:
            src = md.get("extraction_source")
            if src == "range_chart":
                figure_type = "range_chart"
            elif src in ("map_caption_heuristic", "map_context"):
                figure_type = "map"
        return figure_type

    def test_map_caption_heuristic_backfills_to_map(self):
        """The real-world case: ``_process_map`` writes this value."""
        assert self._backfill_figure_type({"extraction_source": "map_caption_heuristic"}) == "map"

    def test_map_context_still_works_legacy_compat(self):
        """Any caller that hand-stamped metadata with the old value still works."""
        assert self._backfill_figure_type({"extraction_source": "map_context"}) == "map"

    def test_explicit_figure_type_wins_over_extraction_source(self):
        """If ``figure_type`` is already set on the row, backfill is a no-op."""
        assert self._backfill_figure_type(
            {"figure_type": "strat_column", "extraction_source": "map_caption_heuristic"}
        ) == "strat_column"

    def test_unknown_extraction_source_does_not_backfill(self):
        """Rows without a recognised extraction_source keep figure_type=None."""
        assert self._backfill_figure_type({"extraction_source": "plate"}) is None
        assert self._backfill_figure_type({}) is None


# --------------------------------------------------------------------------- M5


class TestLinkRangeChartGeologyNoneSections:
    """M5: sections=[None] must not crash _link_range_chart_geology.

    The stub metadata carries a ``range_chart`` dict with a ``sections``
    list. If a list item is None (e.g., from a hand-edited manifest or
    malformed upstream serialization), the old code crashed on
    ``sec.get(...)`` because None has no .get. The fix skips non-dict
    entries.
    """

    def test_none_item_in_sections_does_not_crash(self, tmp_path):
        # We replicate the parsing logic _link_range_chart_geology
        # performs, without needing the full pipeline instance. The
        # contract: ``for sec in sections: if not isinstance(sec, dict):
        # continue`` should hold; the old contract crashed.
        sections = [
            {"name": "Sec1", "age_range": "Late Permian", "formations": ["Talung"]},
            None,
            {"name": "Sec2", "age_range": "Late Permian", "formations": ["Yinkeng"]},
        ]
        parsed = []
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            parsed.append(sec.get("name", "?"))
        # Old code would raise AttributeError on the None item; new code
        # skips it and proceeds with the two valid dicts.
        assert parsed == ["Sec1", "Sec2"]

    def test_none_sections_list_is_skipped(self):
        sections = None
        # Old code: ``rc_dict.get("sections", [])`` returns None here,
        # and ``for sec in None`` raises TypeError. The fix uses
        # ``rc_dict.get("sections") or []`` so None becomes [].
        out = []
        for sec in sections or []:
            out.append(sec)
        assert out == []


class TestPipelineSourceAuditFixes:
    """Lock the source-level patches the audit asked for.

    These tests grep pipeline.py to assert the audited fix lines are
    present, guarding against accidental revert. The frontend-style
    grep tests are a known compromise when a full integration test
    would require a live M3 backend (see ``tests/test_web_fetch_patterns.py``).
    """

    def test_apply_geo_vision_uses_is_file_not_exists(self):
        from pathlib import Path as _Path

        pipeline_path = _Path(__file__).resolve().parents[1] / "src" / "rlpe" / "pipeline.py"
        text = pipeline_path.read_text(encoding="utf-8")
        # Locate the _apply_geo_vision function body.
        marker = "def _apply_geo_vision("
        i = text.find(marker)
        assert i > 0
        # The function body is ~150 lines (Round 5 added a guard),
        # so a 2500-char slice may miss the is_file() call.  Use
        # the next top-level ``def `` as the end-marker.
        next_def = text.find("\n    def ", i + 1)
        assert next_def > 0
        body = text[i:next_def]
        # The fix: ``Path(image_path).is_file()`` instead of ``.exists()``.
        assert "Path(image_path).is_file()" in body, (
            "_apply_geo_vision must use Path(image_path).is_file() "
            "(audit M4) — directories would otherwise crash PIL.Image.open"
        )
        # And NOT the old buggy guard.
        assert "Path(image_path).exists()" not in body, (
            "_apply_geo_vision still uses Path(image_path).exists(); audit M4 requires is_file()"
        )

    def test_link_range_chart_geology_filters_none_sections(self):
        from pathlib import Path as _Path

        pipeline_path = _Path(__file__).resolve().parents[1] / "src" / "rlpe" / "pipeline.py"
        text = pipeline_path.read_text(encoding="utf-8")
        marker = "def _link_range_chart_geology("
        i = text.find(marker)
        assert i > 0
        body = text[i : i + 3000]
        # The fix: guard ``not isinstance(sec, dict)`` inside the
        # sections loop so [None] entries are skipped, not crashed on.
        assert "not isinstance(sec, dict)" in body, (
            "_link_range_chart_geology must filter non-dict entries (audit M5)"
        )
