"""Audit 2026-09-06 — OD↔GROBID depth-guard leak regression tests.

Found during the 25-paper coverage run: ``_enter_od_grobid_guard``
incremented the thread-local depth BEFORE the cap check, and rejected
callers returned their stub without reaching the ``finally`` that calls
``_exit_od_grobid_guard`` — so every rejected paper permanently leaked
+1 into the per-thread counter. Consecutive papers then entered at
depth 3, 4, 5, … and were instantly rejected (log showed depths 3→22
across 11 papers in one second), turning the rest of a batch into
zero-row stubs. The per-paper reset in ``_process_one_pdf`` and the
increment rollback in ``_enter_od_grobid_guard`` fix both halves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def pipe(tmp_path):
    from rlpe.config import PipelineConfig
    from rlpe.pipeline import RadiolarianPipeline

    p = RadiolarianPipeline(PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "w"))
    return p


class TestDepthGuardLeak:
    def test_rejection_rolls_back_depth(self, pipe):
        """A rejected enter must NOT permanently consume depth."""
        # Reach the cap via three legit enters (OD→GROBID→OD-retry).
        assert pipe._enter_od_grobid_guard("p1", "OD") is True
        assert pipe._enter_od_grobid_guard("p1", "GROBID") is True
        assert pipe._enter_od_grobid_guard("p1", "OD") is True
        assert pipe._od_grobid_depth.depth == 3
        # Fourth enter (cap=4) is rejected...
        assert pipe._enter_od_grobid_guard("p1", "GROBID") is False
        # ...and must NOT have leaked the +1.
        assert pipe._od_grobid_depth.depth == 3

    def test_depth_does_not_leak_across_papers(self, pipe):
        """N consecutive papers that all fail through the full
        OD→GROBID→OD chain must each get a REAL attempt — paper k must
        not be rejected because papers 1..k-1 leaked depth."""
        pdf = Path("/tmp/fake.pdf")
        calls = {"od": 0, "grobid": 0}

        def fake_od_inner(paper_id, pdf_path):
            calls["od"] += 1
            # Simulate the real chain: OD found nothing → fall back to GROBID.
            # The real _process_one_pdf_grobid does its own enter/exit.
            return pipe._process_one_pdf_grobid(paper_id, pdf_path)

        def fake_grobid_impl(paper_id, pdf_path):
            calls["grobid"] += 1
            # GROBID is "down" → falls back to OD again (this hop is
            # what the depth guard exists to bound).
            return pipe._process_one_pdf_od(paper_id, pdf_path)

        pipe._process_one_pdf_od_inner = fake_od_inner
        pipe._process_one_pdf_grobid_impl = fake_grobid_impl

        for i in range(8):
            paper_id = f"paper{i}"
            # top-level per-paper reset (mirrors _process_one_pdf)
            pipe._od_grobid_depth.depth = 0
            rows = pipe._process_one_pdf_od(paper_id, pdf)
            # With the cap at 4, each paper's LEGITIMATE chain is
            # OD(1)→GROBID(2)→OD-retry(3): two real OD attempts per
            # paper, and the bounded 4th hop (GROBID again) is what
            # finally returns the cycle stub. The critical property:
            # paper k's attempts must not shrink because earlier
            # papers leaked depth.
            assert calls["od"] == 2 * (i + 1), (
                f"paper {i} got {calls['od']} OD attempts (expected "
                f"{2 * (i + 1)}) — the depth counter still leaks across "
                f"papers or the cap rejects the legitimate retry hop"
            )

    def test_cap_rejects_only_true_cycle(self, pipe):
        """Cap=4: hops 1-3 (OD→GROBID→OD-retry) allowed; hop 4 rejected."""
        assert pipe._enter_od_grobid_guard("p", "OD") is True
        assert pipe._enter_od_grobid_guard("p", "GROBID") is True
        assert pipe._enter_od_grobid_guard("p", "OD") is True  # legit retry
        assert pipe._enter_od_grobid_guard("p", "GROBID") is False  # true cycle
        # and no leak from the rejection
        assert pipe._od_grobid_depth.depth == 3

    def test_real_recursion_still_bounded(self, pipe):
        """The guard must still cap genuine runaway recursion."""
        entered = {"n": 0}

        def runaway(paper_id: str) -> list:
            if not pipe._enter_od_grobid_guard(paper_id, "OD"):
                return [{"stopped": True}]
            entered["n"] += 1
            try:
                return runaway(paper_id)
            finally:
                pipe._exit_od_grobid_guard()

        out = runaway("p1")
        # Cap=4 → three legitimate enters (OD→GROBID→OD-retry), the
        # 4th is rejected as the true cycle.
        assert entered["n"] == 3, "legit recursion depth must remain capped at 3 enters"
        assert out[0]["stopped"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
