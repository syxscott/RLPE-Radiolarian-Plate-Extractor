"""Regression tests for audit 2026-08-01 batch W2 — provenance D13 input_sha256 collision."""

from __future__ import annotations

from pathlib import Path

from rlpe.provenance.stamp import _input_sha256


class TestProvenanceInputSha256:
    def test_two_pdfs_same_name_different_dirs(self, tmp_path: Path):
        """Two PDFs with the same basename from different sub-directories
        must NOT collide — the second one must not silently overwrite the first.
        """
        dir_a = tmp_path / "papers_2023"
        dir_b = tmp_path / "papers_2024"
        dir_a.mkdir()
        dir_b.mkdir()
        pdf_a = dir_a / "shared.pdf"
        pdf_b = dir_b / "shared.pdf"
        pdf_a.write_bytes(b"contents-of-A")
        pdf_b.write_bytes(b"contents-of-B-different")

        result = _input_sha256([pdf_a, pdf_b])

        assert len(result) == 2, f"Expected 2 distinct keys, got {list(result)}"
        keys = list(result.keys())
        # Both keys must be unique
        assert len(set(keys)) == 2, f"Keys collide: {keys}"
        # Both hashes must be distinct and real hex digests
        values = list(result.values())
        assert len(set(values)) == 2, f"Hashes collide: {values}"
        for v in values:
            assert len(v) == 64 and all(c in "0123456789abcdef" for c in v), (
                f"Value {v!r} is not a hex SHA-256 digest"
            )

    def test_basename_collision_in_same_dir(self, tmp_path: Path):
        """Two PDFs with the same basename in the same parent directory must
        both be represented in the output — never silently dropped."""
        pdf_a = tmp_path / "collide.pdf"
        pdf_b = tmp_path / "collide.pdf"
        # Write two distinct contents. The second write will overwrite the
        # first on disk, but the function still gets two Path objects
        # passed in and must report both.
        pdf_a.write_bytes(b"first-bytes")
        pdf_b.write_bytes(b"second-bytes-different")

        result = _input_sha256([pdf_a, pdf_b])

        # We must end up with 2 entries (one for each input Path), not 1.
        assert len(result) == 2, (
            f"Expected 2 entries (one per input Path), got {len(result)}: {list(result)}"
        )
        # The collision must be disambiguated via [N] index suffix on the
        # filename portion of the parent-qualified key.
        keys = sorted(result.keys())
        assert all(k.endswith(("/collide[0].pdf", "/collide[1].pdf")) for k in keys), (
            f"Expected disambiguated [0]/[1] suffixes, got {keys!r}"
        )
        suffixes = sorted(k.rsplit("/", 1)[1] for k in keys)
        assert suffixes == ["collide[0].pdf", "collide[1].pdf"]

    def test_collision_disambiguated_with_suffix(self, tmp_path: Path):
        """Three identical-name inputs must produce three distinct keys
        using a [0]/[1]/[2] suffix scheme."""
        # We can only have one real file on disk for a given basename; the
        # function should still produce 3 entries for 3 input Paths.
        only_file = tmp_path / "a.pdf"
        only_file.write_bytes(b"the only file")
        p1 = tmp_path / "a.pdf"
        p2 = tmp_path / "a.pdf"
        p3 = tmp_path / "a.pdf"

        result = _input_sha256([p1, p2, p3])

        assert len(result) == 3, f"Expected 3 distinct keys, got {list(result)}"
        keys = list(result.keys())
        # They must all be distinct.
        assert len(set(keys)) == 3, f"Keys collide: {keys}"
        # The filename portion of each key must carry [0]/[1]/[2] markers so
        # the sequence is stable.
        filename_only = sorted(k.rsplit("/", 1)[1] for k in keys)
        assert filename_only == ["a[0].pdf", "a[1].pdf", "a[2].pdf"], (
            f"Expected a[0]/[1]/[2].pdf, got {filename_only}"
        )
        # All values should be the same hex digest (they're the same file).
        hexes = [v for v in result.values() if v not in ("missing", "unreadable")]
        assert len(hexes) == 3
        assert len(set(hexes)) == 1, f"Distinct hashes for the same file: {hexes}"
