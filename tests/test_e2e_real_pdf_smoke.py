"""End-to-end smoke test on a real PDF.

This test exercises the FULL pipeline (PDF -> figures -> segments -> matches.jsonl)
on the committed Xiao Yifan 2017 micro-XCT PDF. It uses no LLM backend
(local_only is forced) and tolerates OCR / segmentation backends being
unavailable by graceful degradation.

What we assert:
  * pipeline.run() returns a list (possibly empty)
  * The pipeline writes a matches.jsonl under the work dir
  * If any rows are produced, each validates against the published
    PanelRecord schema
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "Xiao Yifan et al 2017a micro-XCT.pdf"
if not PDF.exists():
    pytest.skip("Xiao Yifan micro-XCT PDF not present", allow_module_level=True)

sys.path.insert(0, str(ROOT / "src"))

from rlpe.config import PipelineConfig  # noqa: E402
from rlpe.pipeline import RadiolarianPipeline  # noqa: E402
from rlpe.schema_models import PanelRecord  # noqa: E402


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """Provide a fresh work dir per test run."""
    d = tmp_path / "e2e_smoke"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_xiao_yifan_pipeline_runs_end_to_end(workdir: Path) -> None:
    # Make the PDF visible to the pipeline. The pipeline glob is
    # ``self.config.pdf_dir.glob("*.pdf")`` so we copy the PDF into
    # a fresh directory.
    pdf_dir = workdir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(PDF, pdf_dir / PDF.name)

    cfg = PipelineConfig(
        pdf_dir=pdf_dir,
        work_dir=workdir,
        ocr_backend="paddleocr",
        num_workers=1,  # serial so the test is deterministic
        save_intermediate=True,
        extra={
            "use_opendataloader": True,
            "data_outbound_policy": "local_only",
            "od_use_ocr": False,
        },
    )
    pipe = RadiolarianPipeline(cfg)
    rows = pipe.run()
    # rows may be empty if no figures / no OCR / no caption. We just
    # assert it ran without raising and the JSONL was written.
    jsonl = workdir / "output" / "manifests" / "matches.jsonl"
    assert jsonl.exists(), f"matches.jsonl not written at {jsonl}"
    with jsonl.open(encoding="utf-8") as f:
        file_rows = [json.loads(line) for line in f if line.strip()]
    assert len(file_rows) == len(rows)
    # If we have any rows, they should round-trip through the schema.
    # (When OpenDataLoader is unavailable rows are empty and the
    # schema check is vacuously true.)
    if not file_rows:
        # Without OCR / segmentation / OD installed, the pipeline
        # legitimately produced no matches on a non-plate micro-XCT
        # paper. We accept this as a ran end to end success.
        return
    # Use the schema __fields_set__ to strip non-schema keys instead
    # of hand-maintaining a list. This way every future addition to
    # the PanelRecord schema is automatically picked up here.
    allowed = set(PanelRecord.model_fields.keys())
    allowed_meta = set(PanelRecord.model_fields["metadata"].annotation.model_fields.keys())
    for row in file_rows:
        for k in list(row.keys()):
            if k not in allowed:
                row.pop(k, None)
        meta = row.get("metadata") or {}
        for k in list(meta.keys()):
            if k not in allowed_meta:
                meta.pop(k, None)
        row["metadata"] = meta
        PanelRecord.model_validate(row)
