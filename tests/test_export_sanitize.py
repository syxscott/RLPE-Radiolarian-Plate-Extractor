"""Tests for export sanitisation and CSV flattening."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from rlpe.export import (
    _sanitize,
    export_csv,
    export_json,
    export_jsonl,
    flatten_for_csv,
)


class TestSanitize:
    def test_nan_becomes_none(self):
        assert _sanitize(float("nan")) is None

    def test_inf_becomes_none(self):
        assert _sanitize(float("inf")) is None
        assert _sanitize(float("-inf")) is None

    def test_normal_floats_preserved(self):
        assert _sanitize(3.14) == 3.14
        assert _sanitize(-1.5) == -1.5

    def test_nested_dict_sanitized(self):
        d = {"a": float("nan"), "b": {"c": float("inf")}}
        out = _sanitize(d)
        assert out["a"] is None
        assert out["b"]["c"] is None

    def test_nested_list_sanitized(self):
        lst = [1.0, float("nan"), "ok", float("inf")]
        out = _sanitize(lst)
        assert out[0] == 1.0
        assert out[1] is None
        assert out[2] == "ok"
        assert out[3] is None

    def test_tuple_becomes_list(self):
        out = _sanitize((1, 2, 3))
        assert out == [1, 2, 3]

    def test_path_becomes_str(self):
        p = Path("/tmp/test.pdf")
        out = _sanitize(p)
        assert out == "/tmp/test.pdf"

    def test_bytes_becomes_str(self):
        out = _sanitize(b"hello")
        assert out == "hello"

    def test_numpy_scalars(self):
        np = pytest.importorskip("numpy")
        assert _sanitize(np.float32(1.5)) == 1.5
        assert _sanitize(np.int64(7)) == 7


class TestFlattenForCsv:
    def test_lifts_metadata_paleodb(self):
        row = {"paper_id": "p1", "metadata": {"paleodb": {"taxonomy": "x"}}}
        flat = flatten_for_csv(row)
        assert "paleodb" in flat
        # Compound value is JSON-encoded
        decoded = json.loads(flat["paleodb"])
        assert decoded["taxonomy"] == "x"

    def test_lifts_primitive(self):
        row = {"paper_id": "p1", "metadata": {"latitude": 35.7, "longitude": 110.3}}
        flat = flatten_for_csv(row)
        assert flat.get("latitude") == 35.7
        assert flat.get("longitude") == 110.3

    def test_lifts_paper_metadata(self):
        row = {"paper_id": "p1", "metadata": {"paper_metadata": {"doi": "10.1/x"}}}
        flat = flatten_for_csv(row)
        assert "paper_metadata" in flat
        decoded = json.loads(flat["paper_metadata"])
        assert decoded["doi"] == "10.1/x"

    def test_no_metadata_returns_input(self):
        row = {"paper_id": "p1"}
        flat = flatten_for_csv(row)
        assert flat == row

    def test_preserves_original_metadata(self):
        row = {"paper_id": "p1", "metadata": {"paleodb": {"x": 1}}}
        flat = flatten_for_csv(row)
        assert flat["metadata"]["paleodb"]["x"] == 1


class TestJsonlRoundtrip:
    def test_jsonl_lines_are_valid_json(self):
        rows = [
            {
                "paper_id": "p1",
                "species": "Dalongicaepa bipolaris",
                "metadata": {
                    "paleodb": {"taxonomy": {"name": "Dalongicaepa"}},
                    "latitude": float("nan"),
                    "longitude": 110.3,
                },
            },
            {
                "paper_id": "p1",
                "species": "Klaengspongus spinosus",
                "metadata": {
                    "paleodb": None,
                    "latitude": 35.7,
                    "longitude": float("inf"),
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            export_jsonl(rows, path)
            lines = path.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 2
            for line in lines:
                parsed = json.loads(line)  # Must not raise
                assert "paper_id" in parsed

    def test_nan_values_become_null_in_jsonl(self):
        rows = [{"metadata": {"latitude": float("nan")}}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            export_jsonl(rows, path)
            content = path.read_text(encoding="utf-8")
            assert "NaN" not in content
            assert "null" in content

    def test_export_json_creates_valid_json(self):
        rows = [{"paper_id": "p1", "metadata": {"x": float("nan")}}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            export_json(rows, path)
            with path.open() as f:
                loaded = json.load(f)
            assert loaded[0]["paper_id"] == "p1"
