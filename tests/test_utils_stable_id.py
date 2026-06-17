"""Tests for stable_id — must be content-based, not path-based.

Why: The old stable_id hashed the file path string, which meant
moving a PDF to a different directory produced a different paper_id.
This broke eval/gold matching on re-runs: gold paper_ids (recorded
under the original path) no longer matched the pipeline output under
a new path. Switching to content hashing (file size + SHA1 of bytes)
makes the id portable — the same PDF produces the same id regardless
of where it lives.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from rlpe.utils import stable_id


def test_stable_id_is_same_for_same_file_at_different_paths():
    """The same PDF must produce the same id regardless of path.

    The old behavior (path-based hash) would produce different ids for
    these two paths. Content hashing fixes that.
    """
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        content = b"%PDF-1.4\n%fake test pdf with some bytes\n%%EOF\n" * 100
        p1 = Path(d1) / "baumgartner2008.pdf"
        p2 = Path(d2) / "baumgartner2008.pdf"
        p1.write_bytes(content)
        p2.write_bytes(content)
        assert stable_id(p1) == stable_id(p2)


def test_stable_id_differs_for_different_files():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "a.pdf").write_bytes(b"a" * 1024)
        (Path(d) / "b.pdf").write_bytes(b"b" * 1024)
        assert stable_id(Path(d) / "a.pdf") != stable_id(Path(d) / "b.pdf")


def test_stable_id_differs_for_different_sizes():
    """Different content sizes produce different ids, even if bytes were
    the same — the size prefix prevents the (vanishingly rare) SHA1
    collision between two PDFs of identical length but different content.
    """
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "short.pdf").write_bytes(b"x" * 100)
        (Path(d) / "long.pdf").write_bytes(b"x" * 200)
        assert stable_id(Path(d) / "short.pdf") != stable_id(Path(d) / "long.pdf")


def test_stable_id_returns_16_char_hex():
    """The id is a 16-character hex string (truncated SHA1)."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.pdf"
        p.write_bytes(b"hello")
        sid = stable_id(p)
        assert len(sid) == 16
        assert all(c in "0123456789abcdef" for c in sid)


def test_stable_id_falls_back_to_path_hash_for_nonexistent_file():
    """If the file doesn't exist, fall back to path-based hashing so
    that unit tests passing placeholder paths still get a deterministic
    (if non-portable) id.
    """
    sid = stable_id(Path("/nonexistent/path/to/fake.pdf"))
    assert len(sid) == 16
    # Same path → same id (deterministic)
    assert stable_id(Path("/nonexistent/path/to/fake.pdf")) == sid
