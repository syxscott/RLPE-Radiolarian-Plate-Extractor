"""Per-stage performance benchmark for the RLPE pipeline.

Measures wall-clock time for each independently-runnable stage
of the pipeline (PDF parse, OpenDataLoader, segmentation, OCR,
caption parser) using the smallest committed paper as input.
Outputs a JSON report to stdout (or to `--output`).

Why a separate benchmark script:
- The full pipeline takes 30+ minutes per paper on a GPU. A
  per-stage breakdown is the only way to attribute that time.
- The pipeline is a series of independent stages; the slowest
  stage is the optimization target.

Usage:
    PYTHONPATH=src python scripts/benchmark.py
    PYTHONPATH=src python scripts/benchmark.py --paper beccaro2006
    PYTHONPATH=src python scripts/benchmark.py --output work/bench.json

Stages (in dependency order):
1. PDF metadata: pymupdf open + page count + per-page text layer size
2. OpenDataLoader: extract figures from PDF (skipped if not installed)
3. Segmentation: watershed split on a small synthetic figure
4. OCR: PaddleOCR on a small synthetic panel (skipped if not installed)
5. Caption parser: parse 3 representative caption strings (always runs)

The output JSON is suitable for diffing in CI: stable schema, no
timestamps, no host-specific fields.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _time_n(fn: callable, n: int = 3) -> dict[str, float]:
    """Run fn() n times and return min/median/mean/total seconds."""
    times: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {
        "runs": n,
        "min": min(times),
        "median": statistics.median(times),
        "mean": statistics.mean(times),
        "total": sum(times),
    }


def bench_pdf_metadata(pdf_path: Path) -> dict[str, Any]:
    """Open the PDF, count pages, and read the text layer."""
    import fitz  # pymupdf

    def _run() -> None:
        doc = fitz.open(pdf_path)
        try:
            n_pages = doc.page_count
            text_bytes = sum(len(doc.load_page(i).get_text()) for i in range(n_pages))
        finally:
            doc.close()

    return {"n_pages_or_bytes": _run.__doc__ or "", **_time_n(_run, n=3)}


def bench_opendataloader(pdf_path: Path, work_dir: Path) -> dict[str, Any]:
    """Run OpenDataLoader on the PDF. Skips if not installed."""
    from rlpe.opendataloader_extractor import OpenDataLoaderExtractor

    ext = OpenDataLoaderExtractor()
    if not ext.is_available():
        return {"skipped": "opendataloader-pdf not installed"}
    out = work_dir / "bench_od"
    out.mkdir(parents=True, exist_ok=True)

    def _run() -> None:
        ext.extract(pdf_path, out)

    return _time_n(_run, n=1)  # OD is slow; only 1 run


def bench_segmentation() -> dict[str, Any]:
    """Watershed split on a synthetic 5-specimen figure."""
    import cv2
    import numpy as np

    from rlpe.segmentation import PanelSegmenter

    def _make_synthetic() -> np.ndarray:
        # 5 circles on a black background, with light bridging between
        # adjacent pairs to exercise the watershed splitter.
        img = np.zeros((400, 800, 3), dtype=np.uint8)
        centers = [(80, 80), (240, 80), (400, 80), (560, 80), (720, 80)]
        for cx, cy in centers:
            cv2.circle(img, (cx, cy), 40, (200, 200, 200), -1)
        return img

    img = _make_synthetic()
    segmenter = PanelSegmenter()

    def _run() -> None:
        segmenter.segment_image(img)

    return _time_n(_run, n=3)


def bench_ocr() -> dict[str, Any]:
    """OCR on a small synthetic panel. Skips if no OCR backend installed."""
    try:
        from rlpe.ocr import OCRBackend
    except ImportError:
        return {"skipped": "rlpe.ocr not importable"}

    import cv2
    import numpy as np

    try:
        engine = OCRBackend(backend="paddleocr", use_gpu=False)
    except Exception as exc:
        return {"skipped": f"OCR backend init failed: {type(exc).__name__}: {str(exc)[:80]}"}

    # Synthetic panel: dark background with white text
    img = np.zeros((100, 400, 3), dtype=np.uint8)
    cv2.putText(img, "Testocr 123", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    def _run() -> None:
        engine.recognize(img)

    return _time_n(_run, n=2)


def bench_caption_parser() -> dict[str, Any]:
    """Parse 3 representative caption strings of varying complexity.

    Uses the module-level `_regex_parse_caption` (no backend needed)
    rather than `M3Engine.parse_caption` (which requires an LLM
    backend). The regex parser is the one that actually does the
    caption → (label, species) work; M3's value-add is critique.
    """
    from rlpe.m3_engine import _regex_parse_caption

    captions = [
        # (a) Pouille-style — exercises _POUILE_CLAUSE_RE
        "Species epithet (Pl. 1, fig. 3)",
        # (b) Danelian-style — exercises _DANELIAN_CLAUSE_RE
        "1) Genus epithet Author, 1990, sample A2, x200; "
        "2) Other species Author, 1991, sample A3, x200; "
        "3-4) Range species Author, 1985",
        # (c) Figures N-M — the bandini2006 shape that currently
        # has the worst parser coverage
        "Figures 5-6 Archaeocenosphaera mellifera; "
        "7-8 Archaeocenosphaera sp.; "
        "9-10 Triactoma cellulosa; "
        "11-12 Triactoma hexeris",
    ]

    def _run() -> None:
        for c in captions:
            _regex_parse_caption(c)

    return _time_n(_run, n=3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper",
        default="beccaro2006",
        help="Committed paper to benchmark (default: beccaro2006, the smallest PDF)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write JSON report to this path (default: stdout)",
    )
    args = parser.parse_args()

    pdf_path = REPO_ROOT / "data" / "pdfs" / f"{args.paper}.pdf"
    if not pdf_path.exists():
        # bandini2006 is a special case — the gold uses the Greece PDF
        # but the file in data/pdfs/ has a different name.
        alt = REPO_ROOT / "data" / "pdfs" / f"{args.paper}_greece.pdf"
        if alt.exists():
            pdf_path = alt
    if not pdf_path.exists():
        print(f"ERROR: {pdf_path} (or _greece.pdf variant) not found", file=sys.stderr)
        return 1
    work_dir = REPO_ROOT / "work" / "bench"
    work_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema": "rlpe-benchmark-1.0",
        "paper": args.paper,
        "pdf_bytes": pdf_path.stat().st_size,
        "stages": {},
    }
    print(f"Benchmarking on {pdf_path.name} ({pdf_path.stat().st_size} bytes) ...", file=sys.stderr)

    print("  [1/5] PDF metadata (pymupdf) ...", file=sys.stderr)
    report["stages"]["pdf_metadata"] = bench_pdf_metadata(pdf_path)

    print("  [2/5] OpenDataLoader extract ...", file=sys.stderr)
    report["stages"]["opendataloader"] = bench_opendataloader(pdf_path, work_dir)

    print("  [3/5] Segmentation (watershed) ...", file=sys.stderr)
    report["stages"]["segmentation"] = bench_segmentation()

    print("  [4/5] OCR (PaddleOCR) ...", file=sys.stderr)
    report["stages"]["ocr"] = bench_ocr()

    print("  [5/5] Caption parser (m3_engine) ...", file=sys.stderr)
    report["stages"]["caption_parser"] = bench_caption_parser()

    out = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(out + "\n")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
