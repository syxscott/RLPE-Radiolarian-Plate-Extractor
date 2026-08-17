"""Phase 61 Plan 4 (Bug 4.7): exercise PaddleOCR 3.x box-format variants.

``_normalize_paddle_result`` was previously only smoke-tested against
the 4-corner list-of-lists shape. PaddleOCR 3.x returns:
  * 4-corner polygons: ``[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]``
  * 4-element flat:    ``[x, y, x+w, y+h]``
  * 8-element flat:    ``[x1, y1, x2, y2, x3, y3, x4, y4]`` (xy order)
A regression test for each shape prevents future drift.
"""

from __future__ import annotations

import pytest

from rlpe.ocr import OCRBackend


def _normalize(result):
    return OCRBackend._normalize_paddle_result(result)


def test_8elem_box():
    """8-elem flat list returns the same 4-corner shape."""
    result = {
        "rec_texts": ["hello"],
        "rec_scores": [0.95],
        "dt_polys": [[10, 20, 110, 20, 110, 70, 10, 70]],
    }
    out = _normalize(result)
    assert len(out) == 1
    box, text, conf = out[0]
    assert text == "hello"
    assert conf == pytest.approx(0.95)
    # 4-corner normalised shape
    assert len(box) == 4
    assert [box[0][0], box[0][1]] == [10, 20]
    assert [box[2][0], box[2][1]] == [110, 70]


def test_4elem_box():
    """4-elem flat list [x, y, x+w, y+h] yields 4 corners."""
    result = {
        "rec_texts": ["world"],
        "rec_scores": [0.85],
        "dt_polys": [[5, 10, 25, 30]],  # x, y, x+w, y+h
    }
    out = _normalize(result)
    assert len(out) == 1
    box, text, conf = out[0]
    assert text == "world"
    assert len(box) == 4
    # All four corners must be on the box perimeter.
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    assert min(xs) == 5 and max(xs) == 25
    assert min(ys) == 10 and max(ys) == 30


def test_polygon_box():
    """Polygon (list of 4 points) passes through unchanged."""
    poly = [[0, 0], [50, 0], [50, 20], [0, 20]]
    result = {
        "rec_texts": ["polygon"],
        "rec_scores": [0.99],
        "dt_polys": [poly],
    }
    out = _normalize(result)
    assert len(out) == 1
    box, text, _ = out[0]
    assert text == "polygon"
    assert box == poly


def test_paddle_2x_legacy_tuple():
    """The 2.x legacy ``([lines], None)`` tuple still parses correctly."""
    legacy = (
        [
            [[[0, 0], [10, 0], [10, 10], [0, 10]], ("legacy1", 0.9)],
            [[[5, 5], [25, 5], [25, 15], [5, 15]], ("legacy2", 0.8)],
        ],
        None,
    )
    out = _normalize(legacy)
    assert len(out) == 2
    assert [t for _, t, _ in out] == ["legacy1", "legacy2"]
