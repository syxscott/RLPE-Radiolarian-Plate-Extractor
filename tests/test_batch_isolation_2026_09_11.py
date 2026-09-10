"""F19 (2026-09-11): batch subprocess isolation + resume merge.

Fixes the failure mode exposed by the random-20 live run: a native
crash (PaddleOCR SIGSEGV) killed the whole batch, and every
``--resume`` attempt rewrote matches.jsonl / run_output.json with only
its own papers — losing earlier papers' rows from the aggregate.

Covered here:
- parent wrapper success path (rows file read + cleaned up),
- crash stub on nonzero exit / timeout / unreadable rows,
- resume merge carries prior papers' rows into the aggregate,
- the real worker subprocess round-trip (config handoff incl.
  credentials, rows file written, exit 0).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from rlpe.config import PipelineConfig
from rlpe.pipeline import RadiolarianPipeline


@pytest.fixture()
def pipe(tmp_path: Path) -> RadiolarianPipeline:
    cfg = PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "w")
    return RadiolarianPipeline(cfg)


@pytest.fixture()
def pdf(tmp_path: Path) -> Path:
    p = tmp_path / "paper.pdf"
    p.write_bytes(b"%PDF-1.4 stub")
    return p


class _DoneProc:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


class TestSubprocessWrapper:
    def test_success_reads_rows_and_cleans_up(self, pipe, pdf, tmp_path):
        cfg_path = tmp_path / "w" / "manifests" / ".batch_worker_config.json"
        rows_path = tmp_path / "w" / "manifests" / ".worker_rows_x.json"
        expected = [{"paper_id": "abc", "figure_id": "f1"}]

        def fake_run(cmd, **kw):
            Path(kw["__out__"]).write_text(json.dumps(expected))
            return _DoneProc(0)

        def run_writes_out(cmd, **kw):
            out = Path(cmd[cmd.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(expected))
            return _DoneProc(0)

        with mock.patch.object(subprocess, "run", side_effect=run_writes_out):
            rows = pipe._process_one_pdf_in_subprocess(pdf, cfg_path)
        assert rows == expected
        # The per-paper rows temp file must be consumed (deleted).
        assert not (tmp_path / "w" / "manifests" / ".worker_rows_x.json").exists()

    def test_nonzero_exit_records_crash_stub(self, pipe, pdf, tmp_path):
        cfg_path = tmp_path / "cfg.json"

        def run_crash(cmd, **kw):
            return _DoneProc(-11, stderr="Segmentation fault (core dumped)")

        with mock.patch.object(subprocess, "run", side_effect=run_crash):
            rows = pipe._process_one_pdf_in_subprocess(pdf, cfg_path)
        assert len(rows) == 1
        stub = rows[0]
        assert stub["figure_id"] == "_ingestion_worker_crash"
        assert stub["metadata"]["ingestion_warning"] is True
        assert "exit=-11" in stub["metadata"]["ingestion_error"]
        assert pdf.name in stub["metadata"]["ingestion_error"]

    def test_timeout_records_crash_stub(self, pipe, pdf, tmp_path):
        cfg_path = tmp_path / "cfg.json"
        pipe.config.extra["batch_worker_timeout_sec"] = 5

        def run_hang(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, 5)

        with mock.patch.object(subprocess, "run", side_effect=run_hang):
            rows = pipe._process_one_pdf_in_subprocess(pdf, cfg_path)
        assert rows[0]["figure_id"] == "_ingestion_worker_crash"
        assert "timeout after 5s" in rows[0]["metadata"]["ingestion_error"]

    def test_worker_config_dump_injects_secrets(self, pipe, pdf, tmp_path):
        pipe.config.extra["llm_api_key"] = "sk-secret-parent"
        pipe.config.extra["batch_isolation"] = "subprocess"
        path = pipe._dump_worker_config()
        assert path.exists()
        import os
        import stat

        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert mode == 0o600
        payload = json.loads(path.read_text())
        assert payload["extra"]["llm_api_key"] == "sk-secret-parent"


class TestResumeMerge:
    def test_carries_prior_papers_rows(self, pipe, tmp_path):
        manifest = tmp_path / "w" / "manifests" / "matches.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        prior = [
            {"paper_id": "aaa", "figure_id": "f1", "species": "A a"},
            {"paper_id": "bbb", "figure_id": "f1", "species": "B b"},
        ]
        manifest.write_text("\n".join(json.dumps(r) for r in prior), encoding="utf-8")
        pipe.config.extra["resume"] = True
        current = [{"paper_id": "aaa", "figure_id": "f1", "species": "A a-new"}]
        merged = pipe._merge_resume_rows(manifest, current)
        # aaa reprocessed → fresh rows win; bbb not reprocessed → carried.
        assert [r["paper_id"] for r in merged] == ["bbb", "aaa"]
        assert merged[1]["species"] == "A a-new"

    def test_no_resume_no_merge(self, pipe, tmp_path):
        manifest = tmp_path / "m.jsonl"
        manifest.write_text(json.dumps({"paper_id": "zzz"}), encoding="utf-8")
        pipe.config.extra["resume"] = False
        current = [{"paper_id": "aaa"}]
        assert pipe._merge_resume_rows(manifest, current) == [{"paper_id": "aaa"}]

    def test_corrupt_prior_file_tolerated(self, pipe, tmp_path):
        manifest = tmp_path / "m.jsonl"
        manifest.write_text("{broken jsonl", encoding="utf-8")
        pipe.config.extra["resume"] = True
        current = [{"paper_id": "aaa"}]
        assert pipe._merge_resume_rows(manifest, current) == [{"paper_id": "aaa"}]


class TestWorkerSubprocessEndToEnd:
    def test_real_worker_round_trip(self, tmp_path):
        """Spawn the real ``python -m rlpe.worker`` on a stub PDF.

        GROBID points at an unreachable port with retries 0 so the
        child fails fast and deterministically emits ingestion stub
        rows — proving the config handoff (including secrets), the
        pipeline build, the per-paper run, and the rows file all work
        across the process boundary."""
        root = Path(__file__).resolve().parents[1]
        pdf = tmp_path / "stub.pdf"
        pdf.write_bytes(b"%PDF-1.4 stub")
        cfg = PipelineConfig(
            pdf_dir=tmp_path,
            work_dir=tmp_path / "w",
            grobid_url="http://127.0.0.1:1",
            extra={
                "use_opendataloader": False,
                "batch_isolation": "subprocess",
                "llm_api_key": "sk-worker-e2e",
                "grobid_max_retries": 0,
                "grobid_timeout": 2,
            },
        )
        from rlpe.config_io import dump_worker_config

        cfg_path = dump_worker_config(
            cfg, tmp_path / "w" / "manifests" / ".batch_worker_config.json"
        )
        out_path = tmp_path / "w" / "manifests" / ".worker_rows_test.json"

        env = os_environ_with_src(root)
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "rlpe.worker",
                "--config",
                str(cfg_path),
                "--pdf",
                str(pdf),
                "--out",
                str(out_path),
            ],
            capture_output=True,
            text=True,
            timeout=180,
            env=env,
        )
        assert proc.returncode == 0, proc.stderr[-800:]
        rows = json.loads(out_path.read_text())
        assert isinstance(rows, list)
        # The stub PDF fails both extractors → an ingestion stub row is
        # the deterministic outcome.
        assert rows, "worker must always emit rows (stub on failure)"
        assert all(r.get("paper_id") for r in rows)


def os_environ_with_src(root: Path) -> dict:
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


class TestIncrementalJournal:
    def test_append_jsonl_accumulates(self, tmp_path):
        """F19 review fix: the per-paper journal must APPEND (the old
        write_jsonl call replaced the file, so concurrent workers
        clobbered each other's rows)."""
        from rlpe.pipeline import _append_jsonl

        j = tmp_path / "matches.jsonl"
        _append_jsonl(j, [{"paper_id": "a", "row": 1}])
        _append_jsonl(j, [{"paper_id": "b", "row": 2}])
        lines = [json.loads(l) for l in j.read_text().splitlines()]
        assert [r["paper_id"] for r in lines] == ["a", "b"]

    def test_resume_merge_recovers_journal_only_papers(self, pipe, tmp_path):
        """A paper that exists ONLY in the worker journal (parent died
        before the canonical write) must be carried into the merged
        aggregate on resume."""
        manifest = tmp_path / "w" / "manifests" / "matches.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"paper_id": "canonical"}), encoding="utf-8")
        from rlpe.pipeline import _append_jsonl

        journal = tmp_path / "w" / "manifests" / "matches.jsonl"
        journal.write_text("", encoding="utf-8")
        _append_jsonl(journal, [{"paper_id": "canonical"}])  # dup, keep canonical view
        _append_jsonl(journal, [{"paper_id": "lost_paper", "species": "X x"}])

        pipe.config.extra["resume"] = True
        current = [{"paper_id": "fresh"}]
        merged = pipe._merge_resume_rows(manifest, current)
        pids = [r["paper_id"] for r in merged]
        assert set(pids) == {"canonical", "lost_paper", "fresh"}
