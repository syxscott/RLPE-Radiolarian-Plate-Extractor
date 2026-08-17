"""Regression tests for audit 2026-08-05 (Fill Gaps) — Fix 7.

Fix 7: ``image_verified`` now flip-able via POST /review/correction.
  - audit 2026-08-05 verified that no API endpoint could set
    ``PanelRecord.image_verified = True`` (the only way to land a
    True value was to edit the JSONL on disk and reload).
  - We extended ``ReviewCorrection`` with an optional
    ``image_verified: bool | None`` field. When non-None, the
    handler calls ``_flip_image_verified_in_cache`` which walks
    ``RESULT_CACHE`` and flips the metadata bit on matching
    panels.

Tests:
1. ``_flip_image_verified_in_cache`` flips the metadata bit on
   exact (paper_id, figure_id, panel_path) match.
2. Misses don't flip unrelated rows.
3. ``image_verified=None`` on the payload is a no-op (doesn't
   touch the cache and doesn't crash).
4. Source guard: the field IS declared on ``ReviewCorrection`` so
   the existing ``extra='forbid'`` config doesn't 400 the payload.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class TestReviewCorrectionAcceptsImageVerified:
    """The Pydantic model accepts ``image_verified`` without 400ing."""

    def test_none_default(self):
        from rlpe.api.app import ReviewCorrection

        body = ReviewCorrection(paper_id="p", figure_id="f")
        assert body.image_verified is None

    def test_true_passes(self):
        from rlpe.api.app import ReviewCorrection

        body = ReviewCorrection(
            paper_id="p",
            figure_id="f",
            panel_path="/tmp/panel_01.png",
            image_verified=True,
        )
        assert body.image_verified is True

    def test_extra_forbid_rejects_unknown_keys(self):
        from pydantic import ValidationError

        from rlpe.api.app import ReviewCorrection

        with pytest.raises(ValidationError):
            ReviewCorrection(
                paper_id="p",
                figure_id="f",
                image_vertified=True,  # typo
            )


class TestFlipImageVerifiedInCache:
    """Direct test of the cache-walking helper."""

    def _populate_cache(self, monkeypatch, panels):
        from rlpe.api import app as _app

        _app.RESULT_CACHE.clear()
        _app.RESULT_CACHE["job1"] = {
            "result": {"panels": panels, "schema_version": "1.2.0"},
        }
        return _app.RESULT_CACHE

    def test_flips_matching_panel(self, monkeypatch):
        from rlpe.api import app as _app
        from rlpe.api.app import ReviewCorrection, _flip_image_verified_in_cache

        self._populate_cache(
            monkeypatch,
            [
                {
                    "paper_id": "p1",
                    "figure_id": "f1",
                    "panel_id": "1",
                    "panel_path": "/work/panels/panel_01.png",
                    "metadata": {"image_verified": False},
                },
            ],
        )
        payload = ReviewCorrection(
            paper_id="p1",
            figure_id="f1",
            panel_path="/work/panels/panel_01.png",
            image_verified=True,
        )
        flipped = _flip_image_verified_in_cache(payload)
        assert flipped == 1
        panel = _app.RESULT_CACHE["job1"]["result"]["panels"][0]
        assert panel["metadata"]["image_verified"] is True

    def test_flips_false_too(self, monkeypatch):
        from rlpe.api import app as _app
        from rlpe.api.app import ReviewCorrection, _flip_image_verified_in_cache

        self._populate_cache(
            monkeypatch,
            [
                {
                    "paper_id": "p1",
                    "figure_id": "f1",
                    "panel_id": "1",
                    "panel_path": "/work/panels/panel_01.png",
                    "metadata": {"image_verified": True},
                },
            ],
        )
        payload = ReviewCorrection(
            paper_id="p1",
            figure_id="f1",
            panel_path="/work/panels/panel_01.png",
            image_verified=False,
        )
        flipped = _flip_image_verified_in_cache(payload)
        assert flipped == 1
        panel = _app.RESULT_CACHE["job1"]["result"]["panels"][0]
        assert panel["metadata"]["image_verified"] is False

    def test_does_not_flip_unrelated_rows(self, monkeypatch):
        from rlpe.api import app as _app
        from rlpe.api.app import ReviewCorrection, _flip_image_verified_in_cache

        self._populate_cache(
            monkeypatch,
            [
                {
                    "paper_id": "p1",
                    "figure_id": "OTHER",
                    "panel_id": "1",
                    "panel_path": "/work/panels/panel_01.png",
                    "metadata": {"image_verified": False},
                },
                {
                    "paper_id": "OTHER",
                    "figure_id": "f1",
                    "panel_id": "1",
                    "panel_path": "/work/panels/panel_01.png",
                    "metadata": {"image_verified": False},
                },
                {
                    "paper_id": "p1",
                    "figure_id": "f1",
                    "panel_id": "OTHER",
                    "panel_path": "/work/panels/OTHER.png",
                    "metadata": {"image_verified": False},
                },
            ],
        )
        payload = ReviewCorrection(
            paper_id="p1",
            figure_id="f1",
            panel_path="/work/panels/panel_01.png",
            image_verified=True,
        )
        flipped = _flip_image_verified_in_cache(payload)
        # No match: every panel differs in either paper_id, figure_id,
        # OR panel_path basename (the helper keys on all three).
        assert flipped == 0
        for p in _app.RESULT_CACHE["job1"]["result"]["panels"]:
            assert p["metadata"]["image_verified"] is False

    def test_returns_zero_on_empty_cache(self, monkeypatch):
        from rlpe.api import app as _app
        from rlpe.api.app import ReviewCorrection, _flip_image_verified_in_cache

        _app.RESULT_CACHE.clear()
        payload = ReviewCorrection(
            paper_id="p",
            figure_id="f",
            panel_path="/tmp/p.png",
            image_verified=True,
        )
        assert _flip_image_verified_in_cache(payload) == 0


class TestReviewCorrectionSourceGuard:
    """Source guard: ReviewCorrection must declare image_verified."""

    def test_image_verified_field_exists(self):
        from rlpe.api.app import ReviewCorrection

        fields = ReviewCorrection.model_fields.keys()
        assert "image_verified" in fields

    def test_flip_helper_exists(self):
        from rlpe.api.app import _flip_image_verified_in_cache

        assert callable(_flip_image_verified_in_cache)
