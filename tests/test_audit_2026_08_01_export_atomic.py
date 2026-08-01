"""Regression tests for audit 2026-08-01 batch W1 — D6/M22 export.py non-atomic writes.

Covers:
  - export_jsonl / export_csv write complete file end-to-end
  - no .tmp scratch files are left behind after a successful write
  - .tmp scratch files are cleaned up if the underlying fsync fails
    (mirrors the pattern used by exporters/xlsx.py since Phase 38)
"""

from __future__ import annotations

import csv as _csv
import io
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe.export import (  # noqa: E402
    _atomic_write_text,
    export_csv,
    export_json,
    export_jsonl,
)


class TestExportAtomic:
    def test_jsonl_atomic_writes_complete_file(self, tmp_path):
        rows = [{"id": i, "name": f"row-{i}"} for i in range(100)]
        out = tmp_path / "out.jsonl"
        export_jsonl(rows, out)
        assert out.exists()
        # Read back line-by-line — every row must be present, no truncation
        loaded = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
        assert len(loaded) == 100
        assert loaded[0]["id"] == 0
        assert loaded[-1]["id"] == 99
        # File must end with a newline (one trailing newline appended)
        assert out.read_text(encoding="utf-8").endswith("\n")

    def test_csv_atomic_writes_complete_file(self, tmp_path):
        rows = [{"id": i, "name": f"row-{i}", "value": i * 1.5} for i in range(50)]
        out = tmp_path / "out.csv"
        export_csv(rows, out)
        assert out.exists()
        # UTF-8-sig BOM at the start
        raw = out.read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf")
        # Decode and parse with csv.DictReader (strip BOM via utf-8-sig)
        text = raw.decode("utf-8-sig")
        reader = _csv.DictReader(io.StringIO(text))
        loaded = list(reader)
        assert len(loaded) == 50
        assert loaded[0]["id"] == "0"
        assert loaded[-1]["id"] == "49"

    def test_json_atomic_writes_complete_file(self, tmp_path):
        rows = [{"id": i, "payload": {"nested": i}} for i in range(10)]
        out = tmp_path / "out.json"
        export_json(rows, out)
        assert out.exists()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert len(loaded) == 10
        assert loaded[3]["payload"]["nested"] == 3

    def test_no_temp_file_left_on_success(self, tmp_path):
        rows = [{"id": i} for i in range(5)]
        out = tmp_path / "out.jsonl"
        export_jsonl(rows, out)
        # No .tmp scratch file should remain after a clean write
        siblings = [p.name for p in tmp_path.iterdir()]
        assert siblings == ["out.jsonl"], f"unexpected siblings: {siblings}"

    def test_no_temp_file_left_on_failure(self, tmp_path):
        # Force fsync to raise mid-write. The helper must swallow the
        # exception, clean up the .tmp scratch file, and re-raise.
        out = tmp_path / "out.jsonl"
        with mock.patch("rlpe.export.os.fsync", side_effect=RuntimeError("disk full")):
            with pytest.raises(RuntimeError, match="disk full"):
                export_jsonl([{"id": 1}], out)
        # Final target was never installed
        assert not out.exists()
        # And no .tmp scratch file was left behind
        siblings = [p.name for p in tmp_path.iterdir()]
        assert siblings == [], f"unexpected leftovers: {siblings}"

    def test_atomic_write_text_directly(self, tmp_path):
        out = tmp_path / "plain.txt"
        _atomic_write_text(out, "hello world\n", encoding="utf-8")
        assert out.read_text(encoding="utf-8") == "hello world\n"
        assert [p.name for p in tmp_path.iterdir()] == ["plain.txt"]

    def test_atomic_write_text_overwrites_existing(self, tmp_path):
        out = tmp_path / "plain.txt"
        out.write_text("OLD", encoding="utf-8")
        _atomic_write_text(out, "NEW", encoding="utf-8")
        assert out.read_text(encoding="utf-8") == "NEW"
