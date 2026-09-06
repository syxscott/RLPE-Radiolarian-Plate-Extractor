"""Regression tests for audit 2026-07-31 batch 0 (P0 fixes).

Covers:
  - CHEMOSTRAT acronym word boundaries (CIE/LIP/OAE must not match
    inside ordinary words like "species" / "lip of the aperture")
  - Empty geology records are dropped; confidence is no longer
    unconditionally boosted to 0.6
  - The output-dir probe is a pure function (no Qt) and behaves
  - Source guards: GUI P0 regressions (missing ``import threading``
    in pipeline_worker, the run_tab worker-bound thread-done slot)
  - GROBID↔OD fallback depth guard terminates the recursion
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_GUI = _SRC / "rlpe" / "gui"


# ---------------------------------------------------------------------------
# 0.5 — CHEMOSTRAT acronym word boundaries
# ---------------------------------------------------------------------------


class TestChemostratWordBoundaries:
    def test_ordinary_words_do_not_trigger_acronyms(self):
        from rlpe.geology_extraction import CHEMOSTRAT_PATTERN

        for text in (
            "species",
            "several views",
            "science",
            "lip of the aperture",
            "slip",
            "eclipse",
            "Sicilian",
            "for each species, several views",
        ):
            m = CHEMOSTRAT_PATTERN.search(text)
            assert m is None, f"{text!r} must not match, got {m.group(0)!r}"

    def test_upper_case_acronyms_still_match(self):
        from rlpe.geology_extraction import CHEMOSTRAT_PATTERN

        for text in ("CIE", "the CIE occurred", "LIP (large igneous province)", "OAE 2"):
            assert CHEMOSTRAT_PATTERN.search(text), f"{text!r} should match"

    def test_long_phrases_stay_case_insensitive(self):
        from rlpe.geology_extraction import CHEMOSTRAT_PATTERN

        for text in ("carbon isotope excursion", "mass extinction", "mercury anomaly"):
            assert CHEMOSTRAT_PATTERN.search(text), f"{text!r} should match"

    def test_sem_plate_caption_yields_no_chemostrat(self):
        """The exact real-world failure: a plain SEM plate caption used to
        produce chemostrat='cie' because 'species' contains 'cie'."""
        from rlpe.geology_extraction import extract_geology_from_sections

        cap = (
            "Plate 1\nScanning electron micrographs of the most important "
            "radiolarians used for the definition of the UAZ A-F are "
            "illustrated. For each species, several views."
        )
        recs = extract_geology_from_sections(
            [{"title": "panel:6", "text": cap, "section_type": "panel_caption"}]
        )
        assert recs == [], f"plain SEM caption must yield no geology records, got {recs}"


# ---------------------------------------------------------------------------
# 0.6 — empty geology records dropped, no unconditional 0.6 confidence boost
# ---------------------------------------------------------------------------


class TestGeologyConfidenceNoBoost:
    def test_record_with_content_keeps_extractor_confidence(self):
        from rlpe.geology_extraction import extract_geology_from_sections

        cap = (
            "Fig. 1. Paleogeographic reconstruction of the Sicilian area "
            "at the Late Triassic time (Catalano et al. 1996)."
        )
        recs = extract_geology_from_sections(
            [{"title": "panel:1", "text": cap, "section_type": "panel_caption"}]
        )
        assert recs, "caption with a real age must yield a record"
        for r in recs:
            assert r.age is not None or r.country is not None
            # country-centroid coords are deliberately 0.3; the 0.6
            # boost used to destroy that signal.
            assert r.confidence < 0.6, f"confidence must not be boosted, got {r.confidence}"

    def test_all_none_record_is_dropped(self):
        from rlpe.geology_extraction import GeologyRecord

        assert not GeologyRecord().has_geology_content()
        assert GeologyRecord(age="Late Triassic").has_geology_content()
        assert GeologyRecord(chemostrat="CIE").has_geology_content()
        assert GeologyRecord(latitude=41.5).has_geology_content()


# ---------------------------------------------------------------------------
# 0.2 — output-dir probe (pure function, no Qt)
# ---------------------------------------------------------------------------


class TestOutputDirProbe:
    def test_writable_dir_returns_none(self, tmp_path):
        from rlpe.gui.outdir_probe import probe_output_dir_writable

        assert probe_output_dir_writable(str(tmp_path)) is None

    def test_missing_dir_returns_error(self, tmp_path):
        from rlpe.gui.outdir_probe import probe_output_dir_writable

        err = probe_output_dir_writable(str(tmp_path / "nope"))
        assert err is not None

    def test_file_path_returns_error(self, tmp_path):
        from rlpe.gui.outdir_probe import probe_output_dir_writable

        f = tmp_path / "afile"
        f.write_text("x")
        assert probe_output_dir_writable(str(f)) is not None

    def test_probe_file_cleaned_up(self, tmp_path):
        from rlpe.gui.outdir_probe import probe_output_dir_writable

        probe_output_dir_writable(str(tmp_path))
        leftovers = list(tmp_path.glob(".rlpe_probe_*"))
        assert leftovers == [], f"probe temp file leaked: {leftovers}"


# ---------------------------------------------------------------------------
# 0.1 / 0.3 — GUI source guards (PySide6 may be absent; static checks)
# ---------------------------------------------------------------------------


class TestGuiSourceGuards:
    def test_pipeline_worker_imports_threading(self):
        """P0 regression: PipelineWorker used threading.Event() with no
        ``import threading`` — NameError on every GUI run start."""
        src = (_GUI / "pipeline_worker.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        imports = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imports |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert "threading" in imports, "pipeline_worker.py must import threading"
        assert "threading.Event()" in src

    def test_run_tab_binds_worker_into_thread_done_slot(self):
        """P0 regression: plain ``_on_thread_done`` connection let batch
        mode kill the NEXT job's worker. The slot must be bound per
        worker via functools.partial and take a worker argument."""
        src = (_GUI / "run_tab.py").read_text(encoding="utf-8")
        assert "functools.partial(self._on_thread_done, self._worker)" in src
        tree = ast.parse(src)
        done_def = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "_on_thread_done":
                    done_def = node
        assert done_def is not None, "_on_thread_done must exist"
        args = [a.arg for a in done_def.args.args]
        assert args and args[-1] == "worker", (
            f"_on_thread_done must accept the worker, got args={args}"
        )
        # the slot must not touch self._worker for quit/terminate
        assert "worker.isRunning()" in src
        assert "worker.quit()" in src

    def test_batch_dialog_uses_extracted_probe(self):
        src = (_GUI / "batch_dialog.py").read_text(encoding="utf-8")
        assert "probe_output_dir_writable" in src
        # the broken inline probe must be gone
        assert "os.getpid()" not in src or "_os_probe.getpid()" not in src


# ---------------------------------------------------------------------------
# 0.4 — GROBID↔OD fallback depth guard
# ---------------------------------------------------------------------------


class TestOdGrobidCycleGuard:
    @pytest.fixture
    def pipe(self, tmp_path):
        from unittest.mock import patch

        from rlpe.config import PipelineConfig

        with (
            patch("rlpe.pipeline.GrobidClient"),
            patch("rlpe.pipeline.OCRBackend"),
            patch("rlpe.pipeline.TaxonRecognizer"),
            patch("rlpe.pipeline.PanelSegmenter"),
        ):
            cfg = PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "work")
            from rlpe.pipeline import RadiolarianPipeline

            return RadiolarianPipeline(cfg)

    def test_depth_guard_blocks_second_fallback(self, pipe, tmp_path):
        """GROBID down + OD empty must terminate after ONE fallback hop
        instead of recursing GROBID→OD→GROBID→… forever."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4\n%%EOF")
        calls = {"od": 0, "grobid": 0}

        def fake_grobid(paper_id, pdf_path):
            calls["grobid"] += 1
            # simulate GROBID probe failure → falls back to OD
            if not pipe.config.extra.get("grobid_no_probe", False):
                pipe.config.extra["grobid_no_probe"] = False
                pipe.config.grobid_url = "http://127.0.0.1:1"  # unreachable
            # force the fast-fail path
            with pipe._grobid_lock:
                in_progress = paper_id in pipe._grobid_in_progress
            if pipe.config.extra.get("grobid_no_probe"):
                return []
            # let the real implementation run its probe; it will fail
            # fast (unreachable URL) and call _process_one_pdf_od
            return pipe._process_one_pdf_grobid_impl(paper_id, pdf_path)

        # Directly exercise the guard logic: the legitimate chain is
        # OD(1) → GROBID(2) → OD-retry(3), then the 4th hop is refused.
        # Audit 2026-09-07: cap raised from 3→4 so the OD-retry at depth 3
        # gets a real chance (GROBID-down environments were stamping every
        # such paper as _ingestion_od_cycle without processing).
        assert pipe._enter_od_grobid_guard("p1", "OD") is True
        assert pipe._enter_od_grobid_guard("p1", "GROBID") is True
        assert pipe._enter_od_grobid_guard("p1", "OD") is True, (
            "OD-retry at depth 3 is the legitimate fallback and must be allowed"
        )
        assert pipe._enter_od_grobid_guard("p1", "GROBID") is False, (
            "fourth nested fallback must be refused"
        )
        pipe._exit_od_grobid_guard()
        pipe._exit_od_grobid_guard()
        pipe._exit_od_grobid_guard()
        pipe._exit_od_grobid_guard()
        # depth back to 0 → allowed again
        assert pipe._enter_od_grobid_guard("p1", "OD") is True

    def test_cycle_stub_marks_ingestion_warning(self, pipe, tmp_path):
        stub = pipe._make_od_grobid_cycle_stub("p1", tmp_path / "a.pdf", "od")
        assert stub["metadata"]["ingestion_warning"] is True
        assert stub["metadata"]["extraction_source"] == "od_cycle"
