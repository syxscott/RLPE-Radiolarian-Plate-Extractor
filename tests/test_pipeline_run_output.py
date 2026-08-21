"""End-to-end test: ``RadiolarianPipeline.run()`` writes the canonical
``run_output.json`` data-package alongside ``matches.jsonl``.

This is the regression guard for the Phase-D product layer: the
pipeline side must emit a fully-shaped, JSON-serializable
``RunOutput`` payload — not just the per-row ``matches.jsonl`` — so
that downstream consumers (CSV/DwC-A exporters, ML splits, the web
UI) read from one validated bundle.

We do not exercise the real GROBID/OCR/segmentation chain here; the
heavier E2E tests under ``tests/test_e2e_*`` cover that path. This
test only verifies the ``run_output.json`` write step is reached,
its shape matches the published RunOutput, and the failure modes
of the write step (empty rows, broken converter) don't take the
``matches.jsonl`` with them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# These imports depend on the optional ``cv2`` chain pulled by the full
# pipeline. Skip the test module if the host can't import it rather than
# failing the suite on a missing optional dep.
pytest.importorskip("cv2")

from rlpe.config import PipelineConfig  # noqa: E402
from rlpe.pipeline import RadiolarianPipeline  # noqa: E402
from rlpe.schema_models import SCHEMA_VERSION, RunOutput  # noqa: E402


def _make_pdf(tmp_path: Path, name: str = "paper.pdf") -> Path:
    """Drop a placeholder PDF — the test pipeline never opens it."""
    p = tmp_path / name
    p.write_bytes(b"%PDF-1.4\n% minimal placeholder\n")
    return p


def _build_pipe_with_fake_match(monkeypatch, tmp_path: Path, matches_per_pdf):
    """Construct a RadiolarianPipeline whose _process_one_pdf returns
    a canned list of dict rows (one entry per PDF). Returns (pipe, pdf_paths)."""

    cfg = PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "work")
    pipe = RadiolarianPipeline(cfg)

    pdf_paths = [_make_pdf(tmp_path, f"paper_{i}.pdf") for i in range(len(matches_per_pdf))]
    # Replace the per-PDF processor with a stub that returns the canned rows
    # for the matching paper (matched by .name suffix).
    monkeypatch.setattr(
        pipe,
        "_process_one_pdf",
        lambda pdf: matches_per_pdf[int(pdf.stem.split("_")[-1]) - 1],
    )
    return pipe, pdf_paths


class TestPipelineRunOutput:
    def test_run_writes_run_output_json_when_there_are_rows(self, tmp_path, monkeypatch):
        """A non-empty run produces both matches.jsonl and run_output.json,
        and the run_output.json decodes back into a RunOutput-shaped
        payload with provenance + panels."""
        fake_match = {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "1",
            "species": "Actinomma leptodermum",
            "panel_path": "/tmp/p.png",
            "bbox": [10, 20, 110, 220],
            "confidence": 0.9,
            "label_text": "1",
            "caption_snippet": "Fig. 1. 1. Actinomma leptodermum",
            "ocr_text": "1",
            "metadata": {},
            "paper_metadata": {
                "title": "Sample paper",
                "authors": ["Doe, J."],
                "year": 2020,
                "journal": "J. Palaeo",
                "volume": "1",
                "issue": "1",
                "pages": "1-10",
                "doi": "10.0/abc",
                "abstract": None,
                "keywords": ["Radiolaria"],
                "publisher": "X",
                "page_count": 10,
                "source": "grobid",
                "confidence": 0.95,
            },
        }
        pipe, _ = _build_pipe_with_fake_match(monkeypatch, tmp_path, matches_per_pdf=[[fake_match]])

        rows = pipe.run()
        assert rows == [fake_match]

        manifest_dir = pipe.config.manifests_dir()
        matches_path = manifest_dir / "matches.jsonl"
        run_output_path = manifest_dir / "run_output.json"
        assert matches_path.exists()
        assert run_output_path.exists()

        payload = json.loads(run_output_path.read_text(encoding="utf-8"))
        # Payload validates against the published schema. If a converter
        # breaks or a Provenance field drifts, RunOutput.model_validate
        # raises here and the test fails — which is what we want.
        RunOutput.model_validate(payload)
        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["provenance"]["schema_version"] == SCHEMA_VERSION
        assert len(payload["panels"]) == 1
        assert payload["panels"][0]["panel_id"] == "1"
        assert payload["panels"][0]["species"] == "Actinomma leptodermum"
        # Provenance is fully populated, not just a shell
        for key in (
            "pipeline_version",
            "schema_version",
            "git_commit",
            "git_dirty",
            "config_snapshot",
            "input_sha256",
            "timestamp_utc",
            "host",
            "python_version",
        ):
            assert key in payload["provenance"], f"provenance missing {key}"

    def test_run_skips_run_output_json_when_no_rows(self, tmp_path, monkeypatch):
        """A run with no matches (zero PDFs, or zero matches per PDF) does
        not create a stray empty run_output.json. We must not pollute
        the output dir with a stale empty bundle from a no-op run."""
        pipe, _ = _build_pipe_with_fake_match(monkeypatch, tmp_path, matches_per_pdf=[[]])
        rows = pipe.run()
        assert rows == []
        manifest_dir = pipe.config.manifests_dir()
        matches_path = manifest_dir / "matches.jsonl"
        run_output_path = manifest_dir / "run_output.json"
        # matches.jsonl is created (even if empty) — that matches the
        # behaviour before this commit and downstream tools depend on it.
        assert matches_path.exists()
        # run_output.json must NOT be created when there's nothing to bundle.
        assert not run_output_path.exists()

    def test_run_output_failure_does_not_clobber_matches_jsonl(self, tmp_path, monkeypatch):
        """If the run_output.json write step raises, the matches.jsonl
        file must still be on disk and readable. Otherwise downstream
        consumers that only depend on matches.jsonl would silently
        lose all the row-level data because of a higher-layer failure.
        """
        fake_match = {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "2",
            "species": "Species two",
            "panel_path": None,
            "bbox": None,
            "confidence": 0.5,
            "label_text": None,
            "caption_snippet": None,
            "ocr_text": None,
            "metadata": {},
            "paper_metadata": None,
        }
        pipe, _ = _build_pipe_with_fake_match(monkeypatch, tmp_path, matches_per_pdf=[[fake_match]])

        # Force the run_output.json write path to explode. ``matches.jsonl``
        # is written first and must survive.
        def boom(_payload):
            raise RuntimeError("synthetic run_output.json failure")

        # Audit 2026-08-20: patch ``rlpe.pipeline._safe_write_json``
        # (the indirection wrapper, not the bare ``write_json`` name).
        # Under pytest-cov 7.x + Python 3.11 the specialising adaptive
        # interpreter (PEP 659) caches ``write_json`` at the
        # module-level import; ``monkeypatch.setattr("rlpe.pipeline.
        # write_json", ...)`` lands on the attribute but the cached
        # LOAD_GLOBAL bytecode keeps calling the original function.
        # ``_safe_write_json`` resolves via ``globals().get()`` on every
        # call, so patching it works reliably.
        monkeypatch.setattr("rlpe.pipeline._safe_write_json", boom, raising=True)

        rows = pipe.run()
        assert rows == [fake_match]

        manifest_dir = pipe.config.manifests_dir()
        matches_path = manifest_dir / "matches.jsonl"
        assert matches_path.exists()
        lines = [ln for ln in matches_path.read_text(encoding="utf-8").splitlines() if ln]
        assert len(lines) == 1
        recovered = json.loads(lines[0])
        assert recovered["panel_id"] == "2"
        assert recovered["species"] == "Species two"
        # run_output.json is allowed to be absent after a failure.
        assert not (manifest_dir / "run_output.json").exists()
        # llm_usage sidecar is independently guarded; matches.jsonl is
        # still safe even if both sidecar write paths fail.
