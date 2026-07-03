"""OA corpus smoke driver.

Iterates over a directory of radiolarian-paper PDFs (e.g.
``放射虫论文_OA_download/``), runs ``RadiolarianPipeline`` on each
PDF in turn, and writes a per-paper JSONL summary that captures:

  - pdf                  filename
  - sha256               first 16 hex chars of the file's sha256
  - ok                   True if pipeline completed without raising
  - elapsed_s            wall-clock seconds for this PDF
  - error                repr(exc) when ok=False; else None
  - row_count            number of MatchResult rows the pipeline emitted
  - range_chart_detected_count  number of figures classified as range_chart
  - geo_vision_calls     MiniMax-M3 calls made for geo vision (proxy:
                         backend.cost_summary()["calls"] after the run)
  - geo_vision_cost_cny  cost accumulated during this PDF
  - run_output_path      absolute path of run_output.json if written
  - llm_usage_path       absolute path of llm_usage.json if written

Usage
-----
::

    # local-only mode — no MiniMax API calls
    python scripts/smoke_oa_corpus.py \\
        --corpus 放射虫论文_OA_download \\
        --out    work/oa_smoke_results.jsonl \\
        --limit  30

    # mock-LLM mode — uses FakeM3Backend, no real network
    python scripts/smoke_oa_corpus.py \\
        --corpus 放射虫论文_OA_download \\
        --out    work/oa_smoke_results.jsonl \\
        --with-mock-llm \\
        --limit  10

Constraints
-----------
* Never imports ``requests`` (asserted by test). All outbound MiniMax
  traffic must go through the real backend; the smoke driver is
  intentionally network-free unless the caller provides API creds.
* Always writes per-paper ``ok`` row, even on pipeline exception —
  the driver never crashes the whole batch on a single bad PDF.
* ``local_only`` is the default outbound policy to ensure CI / dev
  runs are zero-cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Ensure src/ is importable when invoked as ``python scripts/smoke_oa_corpus.py``.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

logger = logging.getLogger("smoke_oa_corpus")


# --------------------------------------------------------------------------- selection


def select_representative(
    corpus_dir: Path,
    n: int = 40,
    *,
    seed: int = 42,
) -> list[Path]:
    """Pick up to ``n`` PDFs from ``corpus_dir`` deterministically.

    Filters:

    * only files matching ``*.pdf`` (case-insensitive)
    * size > 0 (zero-byte files are download placeholders)
    * sorted alphabetically (stable)

    Selection is then a seeded random sample of ``min(n, len(filtered))``
    entries. Same seed → same selection across runs, which makes
    pytest-based assertions on output reproducible.
    """
    if not corpus_dir.exists() or not corpus_dir.is_dir():
        return []
    candidates: list[Path] = sorted(
        p
        for p in corpus_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf" and p.stat().st_size > 0
    )
    if not candidates:
        return []
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, k=min(n, len(candidates))))


# --------------------------------------------------------------------------- summary


@dataclass(slots=True)
class SmokeRow:
    """One row in the output JSONL."""

    pdf: str
    sha256: str
    ok: bool
    elapsed_s: float
    error: str | None
    row_count: int
    range_chart_detected_count: int = 0
    geo_vision_calls: int = 0
    geo_vision_cost_cny: float = 0.0
    run_output_path: str | None = None
    llm_usage_path: str | None = None


@dataclass(slots=True)
class SmokeSummary:
    ok_count: int = 0
    fail_count: int = 0
    mean_elapsed_s: float = 0.0
    total_cost_cny: float = 0.0
    total_rows: int = 0
    range_chart_total: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)


def summarize_results(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-paper rows into a run-level summary dict."""
    rows = list(rows)
    if not rows:
        return asdict(SmokeSummary())

    ok_rows = [r for r in rows if r.get("ok")]
    fail_rows = [r for r in rows if not r.get("ok")]
    summary = SmokeSummary(
        ok_count=len(ok_rows),
        fail_count=len(fail_rows),
        mean_elapsed_s=(sum(r.get("elapsed_s", 0.0) for r in rows) / len(rows)),
        total_cost_cny=sum(r.get("geo_vision_cost_cny", 0.0) for r in rows),
        total_rows=sum(r.get("row_count", 0) for r in rows),
        range_chart_total=sum(r.get("range_chart_detected_count", 0) for r in rows),
        errors=[
            {"pdf": r.get("pdf", "?"), "error": str(r.get("error"))}
            for r in fail_rows
            if r.get("error")
        ],
    )
    return asdict(summary)


# --------------------------------------------------------------------------- driver


def _sha256_short(path: Path, *, hex_chars: int = 16) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except OSError:
        return "0" * hex_chars
    return h.hexdigest()[:hex_chars]


def _make_pipeline(work_dir: Path, pdf_dir: Path, *, with_mock_llm: bool):
    """Construct a RadiolarianPipeline.

    Kept narrow: the smoke driver never overrides M3 cost/retry; it
    relies on whatever the user has configured via env / config.extra.
    ``with_mock_llm=True`` patches ``gemma_runtime.backend`` to a
    FakeM3Backend so cost counters increment deterministically without
    any network I/O.

    ``pdf_dir`` is the directory ``RadiolarianPipeline.run()`` globs for
    ``*.pdf`` — must contain exactly one PDF for a single-paper run so
    that ``matches.jsonl`` / ``run_output.json`` / ``llm_usage.json``
    reflect only this paper.
    """
    try:
        from rlpe.config import PipelineConfig
        from rlpe.pipeline import RadiolarianPipeline
    except ImportError as exc:
        # The pipeline imports cv2 at module top; if the env lacks it,
        # raise a clear error rather than the raw ModuleNotFoundError.
        raise RuntimeError(
            "smoke_oa_corpus requires the full rlpe runtime "
            "(cv2 + paddleocr or easyocr). Run inside the 'CV' conda env. "
            f"Underlying error: {exc}"
        ) from exc

    extra: dict[str, Any] = {
        # Default to local_only so the smoke run is zero-cost unless the
        # caller explicitly provides API credentials via env.
        "data_outbound_policy": "local_only",
        "use_geology_llm": False,
    }
    if with_mock_llm:
        from tests.fakes.fake_m3_backend import FakeM3Backend

        extra["MiniMax_api_key"] = "mock"
        extra["MiniMax_endpoint"] = "http://mock/"
        extra["MiniMax_model"] = "MiniMax-M3-mock"

    cfg = PipelineConfig(
        pdf_dir=pdf_dir,
        work_dir=work_dir,
        ocr_backend="none",
        use_gpu=False,
        num_workers=1,
        extra=extra,
    )
    pipeline = RadiolarianPipeline(cfg)
    if with_mock_llm and pipeline.gemma_runtime is not None:
        pipeline.gemma_runtime.backend = FakeM3Backend(
            api_key="mock",
            base_url="http://mock/",
            model="MiniMax-M3-mock",
        )
    return pipeline


def _run_one(
    pdf: Path,
    *,
    work_dir: Path,
    with_mock_llm: bool,
) -> SmokeRow:
    """Run the pipeline on a single PDF and capture the per-paper row."""
    # Each PDF gets its own sub-work-dir so matches.jsonl / run_output.json
    # don't collide across papers. ``paper_pdf_dir`` is the dir the
    # pipeline will glob for ``*.pdf``; we copy just this one PDF in so
    # ``RadiolarianPipeline.run()`` (which iterates ``pdf_dir.glob``) picks
    # it up. The single-PDF glob also makes ``run()`` write a
    # ``run_output.json`` / ``llm_usage.json`` for THIS paper only, which
    # is what we report in the smoke JSONL row.
    paper_work_dir = work_dir / pdf.stem
    paper_work_dir.mkdir(parents=True, exist_ok=True)
    paper_pdf_dir = paper_work_dir / "pdfs"
    paper_pdf_dir.mkdir(parents=True, exist_ok=True)
    target_pdf = paper_pdf_dir / pdf.name
    if not target_pdf.exists():
        target_pdf.write_bytes(pdf.read_bytes())

    # Pipeline is constructed with paper_pdf_dir as its pdf_dir so the
    # ``run()`` glob picks up exactly this one PDF; paper_work_dir is the
    # pipeline's work_dir where manifests/figures/etc. land.
    pipeline = _make_pipeline(paper_work_dir, paper_pdf_dir, with_mock_llm=with_mock_llm)

    t0 = time.monotonic()
    try:
        # ``run()`` writes matches.jsonl + run_output.json + llm_usage.json
        # to paper_work_dir/output/manifests/. Returns the per-paper rows.
        result_rows = pipeline.run()
    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.warning("Pipeline failed for %s: %s", pdf.name, exc)
        return SmokeRow(
            pdf=pdf.name,
            sha256=_sha256_short(pdf),
            ok=False,
            elapsed_s=elapsed,
            error=repr(exc),
            row_count=0,
        )

    elapsed = time.monotonic() - t0
    manifests_dir = paper_work_dir / "output" / "manifests"
    run_output_path = manifests_dir / "run_output.json"
    llm_usage_path = manifests_dir / "llm_usage.json"
    range_chart_count = sum(
        1
        for r in result_rows
        if (r.get("metadata") or {}).get("extraction_source") == "range_chart"
    )

    geo_calls = 0
    geo_cost = 0.0
    if pipeline.gemma_runtime is not None:
        backend = getattr(pipeline.gemma_runtime, "backend", None)
        if backend is not None:
            try:
                cost = backend.cost_summary()  # type: ignore[attr-defined]
                geo_calls = int(cost.get("calls", 0))
                geo_cost = float(cost.get("total_cost_cny", 0.0))
            except Exception:
                pass

    return SmokeRow(
        pdf=pdf.name,
        sha256=_sha256_short(pdf),
        ok=True,
        elapsed_s=elapsed,
        error=None,
        row_count=len(result_rows),
        range_chart_detected_count=range_chart_count,
        geo_vision_calls=geo_calls,
        geo_vision_cost_cny=geo_cost,
        run_output_path=str(run_output_path) if run_output_path.exists() else None,
        llm_usage_path=str(llm_usage_path) if llm_usage_path.exists() else None,
    )


def run_smoke(
    corpus_dir: Path,
    out_jsonl: Path,
    *,
    with_mock_llm: bool = False,
    limit: int = 40,
    seed: int = 42,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """Drive the smoke run.

    Parameters
    ----------
    corpus_dir : Path
        Directory containing the source PDFs.
    out_jsonl : Path
        Where to write the per-paper rows. Created if missing.
    with_mock_llm : bool
        If True, swap ``gemma_runtime.backend`` for ``FakeM3Backend``
        so M3 vision calls return canned JSON instead of HTTP I/O.
    limit : int
        Maximum number of PDFs to run.
    seed : int
        Seed for deterministic selection.
    work_dir : Path | None
        Where the pipeline emits intermediates. Defaults to a temp dir
        under ``work/oa_smoke_<timestamp>/``.
    """
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    selected = select_representative(corpus_dir, n=limit, seed=seed)
    if not selected:
        return summarize_results([])

    if work_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        work_dir = _REPO_ROOT / "work" / f"oa_smoke_{ts}"
    work_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for pdf in selected:
        row = _run_one(pdf, work_dir=work_dir, with_mock_llm=with_mock_llm)
        rows.append(asdict(row))
        with out_jsonl.open("a") as f:
            f.write(json.dumps(rows[-1]) + "\n")
        logger.info(
            "smoke: %s ok=%s rows=%d elapsed=%.1fs cost=%.4f",
            pdf.name,
            row.ok,
            row.row_count,
            row.elapsed_s,
            row.geo_vision_cost_cny,
        )
    return summarize_results(rows)


# --------------------------------------------------------------------------- CLI


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OA-corpus smoke driver for RadiolarianPipeline.")

    def _existing_dir(p: str) -> Path:
        path = Path(p)
        if not path.exists():
            parser.error(f"--corpus path does not exist: {p}")
        if not path.is_dir():
            parser.error(f"--corpus path is not a directory: {p}")
        return path

    parser.add_argument(
        "--corpus",
        type=_existing_dir,
        default=_REPO_ROOT / "放射虫论文_OA_download",
        help="Directory of PDF files to smoke-test.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "work" / "oa_smoke_results.jsonl",
        help="Path of output JSONL (one row per PDF).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Maximum number of PDFs to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic PDF selection.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Where to put pipeline intermediates. Default: work/oa_smoke_<ts>/",
    )
    parser.add_argument(
        "--with-mock-llm",
        action="store_true",
        help="Swap gemma_runtime.backend for FakeM3Backend (no network).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Truncate the output JSONL before writing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parse_args(argv)
    if args.reset and args.out.exists():
        args.out.unlink()
    summary = run_smoke(
        args.corpus,
        args.out,
        with_mock_llm=args.with_mock_llm,
        limit=args.limit,
        seed=args.seed,
        work_dir=args.work_dir,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
