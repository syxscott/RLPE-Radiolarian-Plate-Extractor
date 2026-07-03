r"""Tests for the 2026-07-03 audit medium-severity backend bugs.

M1: panel_record_from_match() previously called ``list(match.bbox)``
unconditionally. If a row had a 2- or 3-element tuple (e.g. from a
malformed OCR result), the resulting list violated Pydantic's
``min_length=4, max_length=4`` PanelRecord.bbox constraint and
raised a confusing ValidationError. The fix rejects wrong-length
tuples by passing None for bbox (a missing bbox is already a known
+ tolerated state per audit Bug C elsewhere).

M3: _safe_json_loads() used a regex that only stripped the LAST
closing fence. A response like
``{...}\n```\nfooter text`` would fail ``json.loads`` on the first
try. The fallback ``_extract_balanced_json_object`` already extracts
the first balanced ``{...}`` regardless of trailing content, so the
real fix here is a regression test to ensure the fallback keeps
working for trailing-fence / trailing-prose responses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

# --------------------------------------------------------------------------- M1


class TestPanelRecordBboxGuard:
    """M1: wrong-length bbox tuples must produce None (not a
    Pydantic ValidationError).
    """

    def _make_match(self, bbox):
        from rlpe.types import MatchResult

        return MatchResult(
            paper_id="p1",
            figure_id="f1",
            panel_id="1",
            panel_path=None,
            species="Genus species",
            bbox=bbox,
            confidence=0.5,
        )

    def test_four_element_bbox_passes(self):
        from rlpe.converters import panel_record_from_match

        rec = panel_record_from_match(self._make_match((10, 20, 30, 40)))
        assert rec.bbox == [10, 20, 30, 40]

    def test_three_element_bbox_rejected(self):
        from rlpe.converters import panel_record_from_match

        rec = panel_record_from_match(self._make_match((10, 20, 30)))
        assert rec.bbox is None

    def test_two_element_bbox_rejected(self):
        from rlpe.converters import panel_record_from_match

        rec = panel_record_from_match(self._make_match((10, 20)))
        assert rec.bbox is None

    def test_empty_bbox_rejected(self):
        from rlpe.converters import panel_record_from_match

        rec = panel_record_from_match(self._make_match(()))
        assert rec.bbox is None

    def test_none_bbox_stays_none(self):
        from rlpe.converters import panel_record_from_match

        rec = panel_record_from_match(self._make_match(None))
        assert rec.bbox is None


# --------------------------------------------------------------------------- M3


class TestSafeJsonLoadsFallbackForTrailingJunk:
    """M3: _safe_json_loads() must recover from trailing-fence /
    trailing-prose content via the balanced-object extractor.
    """

    def test_trailing_prose_after_fenced_json(self):
        from rlpe.range_chart_extractor import _safe_json_loads

        text = (
            "```json\n"
            '{"sections": [{"name": "Sec1"}], "species_ranges": [], '
            '"biozones": [], "other_fossils": [], "confidence": 0.8}\n'
            "```\n"
            "Some trailing prose that the model appended."
        )
        result = _safe_json_loads(text)
        assert result["sections"][0]["name"] == "Sec1"
        assert result["confidence"] == 0.8

    def test_trailing_prose_no_fence(self):
        from rlpe.range_chart_extractor import _safe_json_loads

        text = (
            '{"sections": [], "species_ranges": [], "biozones": [], '
            '"other_fossils": [], "confidence": 0.5}\n'
            "Trailing prose without any fence."
        )
        result = _safe_json_loads(text)
        assert result["confidence"] == 0.5

    def test_nested_braces_with_trailing_junk(self):
        from rlpe.range_chart_extractor import _safe_json_loads

        text = (
            '{"sections": [{"name": "S1", "formations": ["A", "B"]}], '
            '"species_ranges": [{"species": "Genus species", '
            '"section": "S1", "range_top": "B9", "range_base": "B7"}], '
            '"biozones": [], "other_fossils": [], "confidence": 0.7}\n'
            "Footer content here."
        )
        result = _safe_json_loads(text)
        assert len(result["sections"]) == 1
        assert result["sections"][0]["name"] == "S1"
        assert result["species_ranges"][0]["species"] == "Genus species"

    def test_pure_clean_json_works(self):
        """Sanity check: the happy path must still work."""
        from rlpe.range_chart_extractor import _safe_json_loads

        text = '{"x": 1, "y": [1, 2, 3]}'
        assert _safe_json_loads(text) == {"x": 1, "y": [1, 2, 3]}

    def test_empty_text_raises(self):
        from rlpe.range_chart_extractor import _safe_json_loads

        with pytest.raises(ValueError, match="empty text"):
            _safe_json_loads("")

    def test_no_json_object_raises(self):
        from rlpe.range_chart_extractor import _safe_json_loads

        with pytest.raises(ValueError, match="no JSON object"):
            _safe_json_loads("just some plain text without braces")


# --------------------------------------------------------------------------- M6


class TestThinkingRetryLockScope:
    """M6: _infer_vision() must NOT hold _thinking_retry_lock for the
    duration of ``backend.infer_panel()`` — the lock should only
    bracket the enable_thinking read+write. Otherwise a custom
    backend that re-enters M3 (e.g. via callbacks) would deadlock
    against the lock the retry path is still holding.
    """

    def test_lock_released_before_infer_panel_call(self):
        """Audit M6: the ``with self._thinking_retry_lock:`` block in
        the retry path must close BEFORE backend.infer_panel is called.

        Old buggy code looked like::

            with self._thinking_retry_lock:
                saved = ...
                ...
                res2 = self.backend.infer_panel(...)   # <-- lock held here
                ...
                enable_thinking = saved               # <-- only restored after

        The fix closes the lock BEFORE the infer_panel call (and
        re-acquires it briefly in the outer finally to restore state)::
            with self._thinking_retry_lock:
                saved = ...
                enable_thinking = False            # <-- lock closed here
            try:
                res2 = self.backend.infer_panel(...)
            finally:
                with self._thinking_retry_lock:
                    enable_thinking = saved

        We assert this by checking that after the FIRST
        ``with self._thinking_retry_lock:`` block, the line
        ``try:`` appears BEFORE the next ``self.backend.infer_panel(`` —
        i.e. the lock is not held across the infer_panel call.
        """
        from pathlib import Path as _Path

        path = _Path(__file__).resolve().parents[1] / "src" / "rlpe" / "m3_engine.py"
        text = path.read_text(encoding="utf-8")
        marker = "def _infer_vision("
        i = text.find(marker)
        assert i > 0
        body = text[i : i + 4000]
        retry_marker = "retrying with thinking disabled"
        j = body.find(retry_marker)
        assert j > 0
        retry_block = body[j : j + 2000]
        # Find the first ``with self._thinking_retry_lock:``.
        lock_open = retry_block.find("with self._thinking_retry_lock:")
        assert lock_open >= 0
        # Find the next ``try:`` AFTER lock_open (the lock-stored body
        # ends, the infer_panel call lives in a new try block outside
        # the lock).
        try_pos = retry_block.find("\n            try:", lock_open)
        assert try_pos >= 0, (
            "retry block must end the lock scope and open a new 'try:' "
            "before backend.infer_panel (audit M6)"
        )
        # And ``self.backend.infer_panel(`` must be after the try_pos.
        infer_pos = retry_block.find("self.backend.infer_panel(", try_pos)
        assert infer_pos > try_pos, (
            "backend.infer_panel must be inside the outer 'try:' block (audit M6)"
        )

    def test_enable_thinking_restored_in_outer_finally(self):
        """The enable_thinking restore must happen even when
        infer_panel raises, in an outer finally block that re-acquires
        the lock briefly.
        """
        from pathlib import Path as _Path

        path = _Path(__file__).resolve().parents[1] / "src" / "rlpe" / "m3_engine.py"
        text = path.read_text(encoding="utf-8")
        marker = "def _infer_vision("
        i = text.find(marker)
        body = text[i : i + 4000]
        retry_marker = "retrying with thinking disabled"
        j = body.find(retry_marker)
        retry_block = body[j : j + 2000]
        # The outer ``finally`` block must restore enable_thinking.
        assert "finally:" in retry_block
        # And the restore inside finally must be wrapped in the lock
        # so concurrent readers see a consistent state.
        finally_pos = retry_block.find("finally:")
        post_finally = retry_block[finally_pos:]
        assert (
            "with self._thinking_retry_lock:" in post_finally
            and "self.backend.enable_thinking = saved" in post_finally
        )
