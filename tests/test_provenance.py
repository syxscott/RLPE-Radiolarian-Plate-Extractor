"""Tests for rlpe.provenance.stamp."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from rlpe.provenance import (
    Provenance,
    build_provenance,
    write_provenance_sidecar,
)
from rlpe.provenance.stamp import _sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestBuildProvenance:
    def test_returns_provenance_instance(self):
        p = build_provenance()
        assert isinstance(p, Provenance)

    def test_versions_are_strings(self):
        p = build_provenance()
        assert isinstance(p.pipeline_version, str) and p.pipeline_version
        assert isinstance(p.schema_version, str) and p.schema_version

    def test_timestamp_is_iso_utc(self):
        p = build_provenance()
        # YYYY-MM-DDTHH:MM:SSZ format
        assert p.timestamp_utc.endswith("Z")
        assert "T" in p.timestamp_utc
        year = int(p.timestamp_utc[:4])
        assert 2020 <= year <= 2100

    def test_python_version_recorded(self):
        p = build_provenance()
        assert p.python_version.count(".") >= 1
        parts = p.python_version.split(".")
        assert len(parts) >= 2

    def test_host_string_nonempty(self):
        p = build_provenance()
        assert "/" in p.host  # format is os/release/machine/node
        parts = p.host.split("/")
        assert len(parts) == 4

    def test_config_snapshot_none(self):
        p = build_provenance(config=None)
        assert p.config_snapshot == {}

    def test_config_snapshot_with_dict(self):
        p = build_provenance(config={"a": 1, "b": "x"})
        assert p.config_snapshot == {"a": 1, "b": "x"}

    def test_config_snapshot_with_object(self):
        class Cfg:
            def to_dict(self):
                return {"k": 1}
        p = build_provenance(config=Cfg())
        assert p.config_snapshot == {"k": 1}

    def test_config_snapshot_fallback(self):
        class Cfg:
            x = 5
            y = "hi"
        p = build_provenance(config=Cfg())
        assert p.config_snapshot.get("x") == 5
        assert p.config_snapshot.get("y") == "hi"

    def test_input_sha256_missing_file(self, tmp_path):
        p = build_provenance(pdf_paths=[tmp_path / "nope.pdf"])
        assert p.input_sha256["nope.pdf"] == "missing"

    def test_input_sha256_real_file(self, tmp_path):
        f = tmp_path / "x.pdf"
        f.write_bytes(b"hello world")
        p = build_provenance(pdf_paths=[f])
        expected = _sha256_file(f)
        assert p.input_sha256["x.pdf"] == expected
        assert len(p.input_sha256["x.pdf"]) == 64  # SHA-256 hex

    def test_git_commit_present(self):
        p = build_provenance(repo_root=REPO_ROOT)
        # We are inside the repo, so git rev-parse should succeed.
        assert p.git_commit != "unknown"
        # Either clean or dirty
        assert isinstance(p.git_dirty, bool)

    def test_git_dirty_flag_matches_state(self):
        # Verify against ground truth from git directly
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        actually_dirty = bool(out.stdout.strip())
        p = build_provenance(repo_root=REPO_ROOT)
        assert p.git_dirty == actually_dirty

    def test_to_dict_is_json_safe(self):
        p = build_provenance()
        d = p.to_dict()
        # Round-trip through json.dumps
        s = json.dumps(d)
        reloaded = json.loads(s)
        assert reloaded["pipeline_version"] == p.pipeline_version
        assert reloaded["git_commit"] == p.git_commit


class TestWriteProvenanceSidecar:
    def test_creates_sidecar(self, tmp_path):
        p = build_provenance()
        out = tmp_path / "results.jsonl"
        out.write_text("")  # touch
        sidecar = write_provenance_sidecar(p, out)
        assert sidecar == out.with_suffix(out.suffix + ".provenance.json")
        assert sidecar.exists()
        # read_text needs explicit encoding on Windows (cp936/GBK default)
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert data["pipeline_version"] == p.pipeline_version

    def test_creates_parent_dirs(self, tmp_path):
        p = build_provenance()
        out = tmp_path / "deep" / "nested" / "results.json"
        sidecar = write_provenance_sidecar(p, out)
        assert sidecar.exists()
        assert sidecar.parent == out.parent
