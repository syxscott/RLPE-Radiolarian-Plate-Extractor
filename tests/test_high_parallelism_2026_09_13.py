"""2026-09-13 high-parallelism batch extraction (8-16 subprocess workers).

Guards the four pieces the feature stands on:

1. ``_append_jsonl`` serialises concurrent writers through a
   cross-process lock — without it, Windows multi-process appends to
   the shared matches.jsonl journal can interleave mid-line and the
   whole resume journal becomes unparseable.
2. ``_load_prior_rows`` / ``_merge_resume_rows`` survive a torn line
   (previously one bad line dropped EVERY prior row from the merge).
3. The memory guard caps the worker pool to physical RAM
   (``_memory_capped_workers``), and the spawn stagger resolves its
   auto default (``_effective_spawn_stagger``).
4. ``_dump_worker_config`` divides the global LLM budget across
   workers (``llm_global_max_concurrent``) without mutating the
   parent's own config, and the new extra keys survive the
   worker-config round-trip.
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rlpe.config import PipelineConfig
from rlpe.pipeline import (
    RadiolarianPipeline,
    _append_jsonl,
    _effective_spawn_stagger,
    _memory_capped_workers,
)


@pytest.fixture()
def pipe(tmp_path: Path) -> RadiolarianPipeline:
    cfg = PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "w")
    return RadiolarianPipeline(cfg)


# ---------------------------------------------------------------------------
# 1. Cross-process-safe journal appends
# ---------------------------------------------------------------------------


class TestAppendJsonlConcurrency:
    def test_multithreaded_appends_no_tear(self, tmp_path):
        """8 threads × 20 appends with fat payloads: every written line
        must parse as JSON (a torn line raises here) and all 160 rows
        must be present exactly once."""
        j = tmp_path / "matches.jsonl"
        fat = "x" * 4096  # big enough that an interleaved write tears JSON

        def append_batch(worker: int) -> None:
            for i in range(20):
                _append_jsonl(j, [{"paper_id": f"p{worker}", "i": i, "fat": fat}])

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(append_batch, range(8)))

        rows = [json.loads(line) for line in j.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 160
        assert sorted((r["paper_id"], r["i"]) for r in rows) == sorted(
            (f"p{w}", i) for w in range(8) for i in range(20)
        )

    def test_multiprocess_appends_no_tear(self, tmp_path):
        """The real batch topology: 6 independent PROCESSES appending to
        one journal. This is the case the lock exists for — Windows
        gives O_APPEND no atomicity guarantee across processes."""
        j = tmp_path / "matches.jsonl"
        procs = []
        for worker in range(6):
            # Import from rlpe.utils (dependency-light) — importing the
            # full rlpe.pipeline in a cold process drags in the
            # torch/cv2 stack, which natively crashes on some Arrow
            # Lake machines and would test DLL layout, not the lock.
            src_dir = repr(str(Path(__file__).resolve().parent.parent / "src"))
            journal = repr(str(j))
            script = (
                "import sys;"
                f"sys.path.insert(0, {src_dir});"
                "from pathlib import Path;"
                "from rlpe.utils import _append_jsonl;"
                f"_append_jsonl(Path({journal}), "
                f"[{{'paper_id': 'p{worker}', 'i': i}} for i in range(10)])"
            )
            procs.append(subprocess.Popen([sys.executable, "-c", script]))
        for proc in procs:
            assert proc.wait(timeout=120) == 0

        lines = j.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 60
        rows = [json.loads(line) for line in lines]  # torn line -> ValueError
        assert sorted((r["paper_id"], r["i"]) for r in rows) == sorted(
            (f"p{w}", i) for w in range(6) for i in range(10)
        )

    def test_lock_file_sits_next_to_target(self, tmp_path):
        j = tmp_path / "matches.jsonl"
        _append_jsonl(j, [{"a": 1}])
        assert (tmp_path / "matches.jsonl.lock").exists()


# ---------------------------------------------------------------------------
# 2. Torn-line-tolerant resume merge
# ---------------------------------------------------------------------------


class TestLoadPriorRowsTolerance:
    def test_bad_line_skipped_good_lines_kept(self, pipe, tmp_path):
        manifest = tmp_path / "m.jsonl"
        manifest.write_text(
            '{"paper_id": "aaa", "row": 1}\n'
            '{"paper_id": "tor\n'  # torn mid-line
            '{"paper_id": "bbb", "row": 2}\n',
            encoding="utf-8",
        )
        rows = pipe._load_prior_rows(manifest)
        assert [r["paper_id"] for r in rows] == ["aaa", "bbb"]

    def test_all_bad_lines_returns_empty(self, pipe, tmp_path):
        manifest = tmp_path / "m.jsonl"
        manifest.write_text("{broken jsonl", encoding="utf-8")
        assert pipe._load_prior_rows(manifest) == []

    def test_blank_lines_ignored(self, pipe, tmp_path):
        manifest = tmp_path / "m.jsonl"
        manifest.write_text('\n\n{"paper_id": "aaa"}\n\n', encoding="utf-8")
        assert [r["paper_id"] for r in pipe._load_prior_rows(manifest)] == ["aaa"]

    def test_merge_survives_torn_journal(self, pipe, tmp_path):
        """The motivating regression: a torn journal line used to drop
        every journal-only paper from the resume merge. Papers whose
        lines parse must still be carried."""
        manifest = tmp_path / "m.jsonl"
        manifest.write_text("[]", encoding="utf-8")  # canonical: empty run
        journal = pipe.config.work_dir / "manifests" / "matches.jsonl"
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text(
            '{"paper_id": "old1", "species": "A"}\n'
            '{"paper_id": "torn\n'
            '{"paper_id": "old2", "species": "B"}\n',
            encoding="utf-8",
        )
        pipe.config.extra["resume"] = True
        merged = pipe._merge_resume_rows(manifest, [{"paper_id": "new"}])
        carried = {r["paper_id"] for r in merged}
        assert {"old1", "old2", "new"} <= carried


# ---------------------------------------------------------------------------
# 3. Memory guard + spawn stagger helpers
# ---------------------------------------------------------------------------


class TestMemoryCappedWorkers:
    def test_caps_to_ram_budget(self):
        # 32 GB total, 2 GB per worker -> 0.8*32768/2048 = 12.8 -> 12
        assert _memory_capped_workers(16, 32768, 2048) == 12

    def test_never_raises_above_request(self):
        assert _memory_capped_workers(8, 32768, 2048) == 8

    def test_unknown_total_memory_is_noop(self):
        assert _memory_capped_workers(16, None, 2048) == 16

    def test_zero_estimate_disables_guard(self):
        assert _memory_capped_workers(16, 32768, 0) == 16

    def test_floors_at_one_worker(self):
        assert _memory_capped_workers(4, 1024, 2048) == 1


class TestEffectiveSpawnStagger:
    def test_auto_staggers_large_subprocess_pools(self):
        assert _effective_spawn_stagger(-1, "subprocess", 8) == 15.0
        assert _effective_spawn_stagger(-1, "subprocess", 16) == 15.0

    def test_auto_no_stagger_small_pools(self):
        assert _effective_spawn_stagger(-1, "subprocess", 4) == 0.0

    def test_auto_never_staggers_inprocess(self):
        assert _effective_spawn_stagger(-1, "inprocess", 16) == 0.0

    def test_explicit_value_wins(self):
        assert _effective_spawn_stagger(5, "inprocess", 2) == 5.0

    def test_explicit_zero_disables(self):
        assert _effective_spawn_stagger(0, "subprocess", 16) == 0.0


# ---------------------------------------------------------------------------
# 4. Global LLM concurrency division + key round-trip
# ---------------------------------------------------------------------------


class TestGlobalLlmConcurrencyDivision:
    def test_dump_divides_budget_without_mutating_parent(self, pipe):
        pipe.config.extra["llm_global_max_concurrent"] = 32
        pipe.config.extra["llm_max_concurrent"] = 8
        pipe.config.num_workers = 8
        path = pipe._dump_worker_config()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["extra"]["llm_max_concurrent"] == 4
            # the parent's own config must be untouched (run() may be
            # called again on the same instance)
            assert pipe.config.extra["llm_max_concurrent"] == 8
        finally:
            path.unlink(missing_ok=True)

    def test_dump_divides_by_effective_workers_not_raw(self, pipe, tmp_path):
        """The LLM budget must be divided by the memory-guard-capped
        pool (what will actually run), not the raw num_workers: with a
        global budget of 40, 16 raw workers -> 2 each but 12 effective
        workers -> 3 each."""
        pipe.config.extra["llm_global_max_concurrent"] = 40
        pipe.config.extra["llm_max_concurrent"] = 8
        pipe.config.num_workers = 16
        path = pipe._dump_worker_config(effective_workers=12)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["extra"]["llm_max_concurrent"] == 3
        finally:
            path.unlink(missing_ok=True)

    def test_dump_without_global_cap_leaves_value(self, pipe):
        pipe.config.extra["llm_max_concurrent"] = 8
        pipe.config.num_workers = 4
        path = pipe._dump_worker_config()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["extra"]["llm_max_concurrent"] == 8
        finally:
            path.unlink(missing_ok=True)

    def test_new_guard_keys_survive_worker_round_trip(self, pipe, tmp_path):
        """The 2026-09-11 lesson: keys set after config construction
        silently vanished from the worker handoff. The three new guard
        keys must round-trip dump_worker_config -> load_worker_config."""
        from rlpe.config_io import dump_worker_config, load_worker_config

        pipe.config.extra["batch_worker_memory_mb"] = 1500
        pipe.config.extra["llm_global_max_concurrent"] = 24
        pipe.config.extra["batch_spawn_stagger_sec"] = 10.0
        path = tmp_path / "worker_cfg.json"
        dump_worker_config(pipe.config, path)
        loaded = load_worker_config(path)
        assert loaded.extra["batch_worker_memory_mb"] == 1500
        assert loaded.extra["llm_global_max_concurrent"] == 24
        assert loaded.extra["batch_spawn_stagger_sec"] == 10.0
