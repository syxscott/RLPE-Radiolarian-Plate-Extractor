"""Tests for the 2026-07-03 audit high-severity backend bugs.

H4 (HIGH): range_chart_extractor.extract_range_chart() leaked the
HTTP connection when ``requests.post()`` raised an exception that
was NOT a ``requests.RequestException`` (e.g. MemoryError, an
urllib3 internal error, a bug in the requests library itself). The
old ``try/finally`` referenced ``resp`` in the finally block but
``resp`` was never assigned if the post call raised pre-call. The
fix uses ``with requests.post(...) as resp:`` so the connection is
always closed by the context manager.

H6 (HIGH): utils.write_json / write_jsonl were non-atomic (naive
``path.write_text``), so concurrent readers could see a partial
file. The fix uses ``tempfile.mkstemp`` + ``os.replace`` for atomic
publication. Backend ``cost_summary`` already acquires the lock
when reading counters, so the worker-thread race that the audit
flagged is mitigated by the existing lock; the file-write race
remains real and is the part we fix here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from rlpe.utils import write_json, write_jsonl

# --------------------------------------------------------------------------- H6


class TestWriteJsonIsAtomic:
    """H6: write_json must use a temp file + os.replace so concurrent
    readers never see a partial file."""

    def test_writer_produces_complete_file(self, tmp_path):
        target = tmp_path / "out.json"
        write_json(target, {"hello": "world", "n": [1, 2, 3]})
        # A reader that opens the file AFTER write_json returns must
        # see the complete payload, not a partial file or empty file.
        with target.open(encoding="utf-8") as f:
            data = json.load(f)
        assert data == {"hello": "world", "n": [1, 2, 3]}

    def test_no_temp_files_linger(self, tmp_path):
        """After a successful write, no ``*.tmp`` sibling should remain."""
        target = tmp_path / "atomic.json"
        write_json(target, {"x": 1})
        siblings = list(tmp_path.iterdir())
        # Exactly one file: the target itself.
        assert siblings == [target], f"atomic write left sibling files behind: {siblings!r}"

    def test_failure_does_not_leave_temp(self, tmp_path, monkeypatch):
        """If the JSON serializer raises inside the temp-write phase,
        the temp file must be cleaned up and the target must not exist.
        """
        import json as _json

        target = tmp_path / "fail.json"

        def _boom(*args, **kwargs):
            raise TypeError("synthetic serialization failure")

        monkeypatch.setattr(_json, "dumps", _boom)
        from rlpe import utils as _utils

        # Re-import the module-level reference (json.dumps captured at
        # import time inside write_json).
        monkeypatch.setattr(_utils.json, "dumps", _boom)

        with pytest.raises(TypeError):
            _utils.write_json(target, {"oops": 1})
        # Temp file should be gone; target should not exist.
        siblings = list(tmp_path.iterdir())
        assert siblings == [], f"failed atomic write leaked: {siblings!r}"
        assert not target.exists()

    def test_overwrites_existing_file_atomically(self, tmp_path):
        target = tmp_path / "rewrite.json"
        write_json(target, {"version": 1})
        write_json(target, {"version": 2})
        with target.open(encoding="utf-8") as f:
            data = json.load(f)
        assert data == {"version": 2}


class TestWriteJsonlIsAtomic:
    """H6 (write_jsonl variant): matches.jsonl uses the same pattern."""

    def test_jsonl_writer_produces_complete_file(self, tmp_path):
        target = tmp_path / "rows.jsonl"
        rows = [{"i": i, "species": f"Genus{i}"} for i in range(5)]
        write_jsonl(target, rows)
        with target.open(encoding="utf-8") as f:
            out = [json.loads(line) for line in f if line.strip()]
        assert out == rows

    def test_jsonl_no_temp_files_linger(self, tmp_path):
        target = tmp_path / "rows.jsonl"
        write_jsonl(target, [{"a": 1}, {"a": 2}])
        siblings = list(tmp_path.iterdir())
        assert siblings == [target]


# --------------------------------------------------------------------------- H4


class TestRangeChartConnectionCleanup:
    """H4: extract_range_chart() must not leak the HTTP connection on
    non-RequestException errors.

    We can't easily stand up a real HTTP endpoint here; instead we
    assert the source-level fix is in place: the post-call site uses
    ``with requests.post(...) as resp:`` so the context manager
    handles close even on exception.
    """

    def test_uses_with_block_for_post(self):
        from pathlib import Path as _Path

        path = _Path(__file__).resolve().parents[1] / "src" / "rlpe" / "range_chart_extractor.py"
        text = path.read_text(encoding="utf-8")
        # Locate the extract_range_chart function body.
        marker = "def extract_range_chart("
        i = text.find(marker)
        assert i > 0
        body = text[i : i + 3000]
        # The fix: ``with requests.post(...) as _resp:`` context manager.
        assert "with requests.post(" in body, (
            "extract_range_chart must use 'with requests.post(...) as resp:' "
            "(audit H4) to guarantee connection close on non-RequestException"
        )
        # And the old buggy ``try/finally: resp.close()`` pattern must be gone.
        # Specifically, the old ``finally: try: resp.close(): except: pass``
        # pattern must NOT appear (replaced by 'with' block).
        buggy_finally = "finally:\n        try:\n            resp.close()"
        assert buggy_finally not in body, (
            "extract_range_chart still uses the buggy try/finally resp.close(); "
            "audit H4 requires the 'with' context manager pattern"
        )
