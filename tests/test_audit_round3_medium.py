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
    """M6 / Round 9 (Bug-M3): the lock scope around the thinking-retry
    path in ``_infer_vision``.

    Round 6 audit M6 originally asserted the lock was RELEASED before
    ``backend.infer_panel()`` was called, to avoid deadlock with backends
    that re-enter M3. Round 9 found that pattern introduced a race
    window: another thread could flip ``enable_thinking`` between the
    save/flip and the call, and the first thread's restore would
    overwrite the other thread's setup, corrupting the final state.

    Post-fix: the lock is held throughout the save → flip → call →
    restore sequence, and the lock is an RLock (reentrant) so a
    backend that re-enters ``_infer_vision`` doesn't deadlock. The
    tests below assert the new correct shape.
    """

    def test_lock_held_throughout_save_flip_call_restore(self):
        """Round 9 (Bug-M3): the ``with self._thinking_retry_lock:``
        block must wrap the entire save → flip → call → restore
        sequence so the whole retry is atomic from the perspective
        of other workers.

        Pre-fix shape (round 6, broken)::

            with self._thinking_retry_lock:
                saved = ...
                enable_thinking = False
            try:
                res2 = self.backend.infer_panel(...)
            finally:
                with self._thinking_retry_lock:
                    enable_thinking = saved

        Post-fix shape::

            with self._thinking_retry_lock:
                saved = ...
                enable_thinking = False
                try:
                    res2 = self.backend.infer_panel(...)
                finally:
                    enable_thinking = saved

        We assert this by checking that ``self.backend.infer_panel(``
        lives INSIDE the ``with self._thinking_retry_lock:`` block
        (i.e. the lock is not closed before the call).
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
        lock_open = retry_block.find("with self._thinking_retry_lock:")
        assert lock_open >= 0
        # The infer_panel call MUST be inside the lock block. The
        # previous "lock released before infer_panel" pattern closed
        # the with-block before the call — verify we don't have that.
        infer_pos = retry_block.find("self.backend.infer_panel(", lock_open)
        assert infer_pos > lock_open, "infer_panel must appear AFTER the lock-open line"
        # And there must NOT be a closing of the with-block before
        # infer_panel. The simplest assertion: there is no second
        # ``with self._thinking_retry_lock:`` before infer_panel.
        second_lock = retry_block.find("with self._thinking_retry_lock:", lock_open + 1)
        assert second_lock < 0 or second_lock > infer_pos, (
            "Round 9 fix: lock must be a single 'with' block wrapping "
            "save/flip/call/restore, not two separate blocks"
        )

    def test_lock_is_reentrant(self):
        """The lock MUST be an RLock so a backend that re-enters
        ``_infer_vision`` (custom subclass calling M3 inside its
        handler) doesn't deadlock."""
        from pathlib import Path as _Path

        path = _Path(__file__).resolve().parents[1] / "src" / "rlpe" / "m3_engine.py"
        text = path.read_text(encoding="utf-8")
        # The lock is constructed as either ``RLock()`` or
        # ``threading.RLock()``. We accept both spellings.
        assert "RLock" in text, (
            "Round 9 fix: _thinking_retry_lock must be RLock for "
            "reentrancy; replace ``Lock()`` with ``RLock()`` in m3_engine.py"
        )
        # And specifically the field type, not just any RLock import.
        init_marker = "self._thinking_retry_lock = "
        i = text.find(init_marker)
        assert i > 0
        snippet = text[i : i + 40]
        assert "RLock" in snippet, (
            f"_thinking_retry_lock must be assigned from RLock(), got: {snippet!r}"
        )
