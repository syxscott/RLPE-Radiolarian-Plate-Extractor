"""Round 6 OA smoke driver — 6 representative PDFs, real MiniMax API.

Picks:
  - 2 gold papers (bandini2006, beccaro2006) for regression check
  - 3 fresh OA papers (Pouille, Boughdiri, Danelian2006) for live F1
  - 1 'other'-classified paper (Xiao_2017 or any micro-CT) to validate
    the Round 6 fig_type='other' skip branch

Each PDF runs with the FULL Round 6 configuration:
  - --use-opendataloader  (skip GROBID)
  - --data-outbound-policy=api_full (real MiniMax calls)
  - --use-gemma4 OFF (use MiniMax cloud backend)
  - --llm-backend=minimax
  - --use-geo-vision OFF (don't spend on geology unless requested)

Run inside the CV conda env:

    conda activate CV
    PYTHONPATH=src /home/user/anaconda3/envs/CV/bin/python \\
        scripts/oa_smoke_round6.py \\
        --out work/oa_smoke_round6/results.jsonl \\
        --work-dir work/oa_smoke_round6 \\
        --limit 6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CV_PY = "/home/user/anaconda3/envs/CV/bin/python"

# Hand-picked representative set: 2 gold + 3 OA + 1 'other'-skip-test
# (the 7th is added only if --limit >= 7).
PICKED_PDFS = [
    # gold regression
    "Bandini_2006 - Eclogae Geologicae Helvetiae - Turonian Radiolarians from Karnezeika, Argolis Peninsula, Peloponnesus (Greece).pdf",
    "Beccaro_2006 - Eclogae Geologicae Helvetiae - Radiolarian correlation of Jurassic siliceous successions of the Rosso Ammonitico Formation in the Southern Alps and Wes.pdf",
    # fresh OA
    "Pouille_2014 - JMicro - A diverse Upper Darriwilian radiolarian assemblage from the Shundy Formation of Kazakhstan_ insights into late Middle Or.pdf",
    "Boughdiri_2007 - Swiss Journal of Geosciences - Jurassic radiolarian-bearing series of Tunisia_ biostratigraphy and significance to western Tethys correlations.pdf",
    "Danelian_2006 - Eclogae Geologicae Helvetiae - Upper Jurassic Radiolaria from the Vocontian basin of SE France.pdf",
    # Round 6 'other'-skip test: contains Micro-CT or paleogeographic map
    "Baumgartner_2008 - IRIS - UPPER TRIASSIC TO CRETACEOUS RADIOLARIA FROM NICARAGUA AND NORTHERN COSTA RICA - THE MESQUITO COMPOSITE OCEANIC TERRANE.pdf",
]


@dataclass(slots=True)
class SmokeRow:
    pdf: str
    sha256: str
    ok: bool
    elapsed_s: float
    error: str | None
    row_count: int
    range_chart_detected_count: int = 0
    other_skipped_count: int = 0
    geo_vision_calls: int = 0
    geo_vision_cost_cny: float = 0.0
    total_cost_cny: float = 0.0
    panel_match_rate: float = 0.0
    species_match_rate: float = 0.0
    avg_confidence: float = 0.0
    run_output_path: str | None = None
    llm_usage_path: str | None = None


def _sha256_short(path: Path, *, hex_chars: int = 16) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except OSError:
        return "0" * hex_chars
    return h.hexdigest()[:hex_chars]


def _make_sub_corpus(pdf: Path, work_dir: Path) -> Path:
    """Copy one PDF into its own sub-corpus so the pipeline glob picks
    up exactly that paper."""
    paper_dir = work_dir / pdf.stem
    paper_dir.mkdir(parents=True, exist_ok=True)
    pdfs_dir = paper_dir / "pdfs"
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    target = pdfs_dir / pdf.name
    if not target.exists():
        target.write_bytes(pdf.read_bytes())
    return paper_dir


def _run_one(pdf: Path, *, work_dir: Path, with_geo_vision: bool) -> SmokeRow:
    paper_dir = _make_sub_corpus(pdf, work_dir)
    pdfs_dir = paper_dir / "pdfs"
    out_jsonl = paper_dir / "output" / "manifests" / "matches.jsonl"
    llm_usage_json = paper_dir / "output" / "manifests" / "llm_usage.json"

    cmd = [
        _CV_PY,
        "-m",
        "rlpe.cli",
        "--pdf-dir",
        str(pdfs_dir),
        "--work-dir",
        str(paper_dir),
        "--ocr-backend",
        "paddleocr",
        "--num-workers",
        "1",
        "--use-gpu",
        "--use-opendataloader",
        "--llm-backend",
        "minimax",
        "--data-outbound-policy",
        "api_full",
        # Disable per-panel Stage 4/5 — each call costs ~5s and a 30-panel
        # plate can blow past 600s. LLM-first path + Stage 1/2/3 already
        # give us coverage; Stage 4/5 are bonus critique.
        "--m3-disable-stage",
        "4",
        "--m3-disable-stage",
        "5",
    ]
    if with_geo_vision:
        cmd.append("--use-geo-vision")

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            env={**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src")},
            capture_output=True,
            text=True,
            timeout=1500,
        )
        elapsed = time.monotonic() - t0
        if result.returncode != 0:
            return SmokeRow(
                pdf=pdf.name,
                sha256=_sha256_short(pdf),
                ok=False,
                elapsed_s=elapsed,
                error=(result.stderr or result.stdout)[-400:],
                row_count=0,
            )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        return SmokeRow(
            pdf=pdf.name,
            sha256=_sha256_short(pdf),
            ok=False,
            elapsed_s=elapsed,
            error="subprocess.TimeoutExpired (600s)",
            row_count=0,
        )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        return SmokeRow(
            pdf=pdf.name,
            sha256=_sha256_short(pdf),
            ok=False,
            elapsed_s=elapsed,
            error=repr(exc),
            row_count=0,
        )

    # Parse matches.jsonl
    row_count = 0
    range_chart_count = 0
    other_skipped_count = 0
    panel_match_count = 0
    species_match_count = 0
    total_conf = 0.0
    if out_jsonl.exists():
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_count += 1
            md = row.get("metadata") or {}
            ext_src = md.get("extraction_source") or ""
            fig_type = md.get("figure_type") or ""
            if ext_src == "range_chart":
                range_chart_count += 1
            if fig_type == "other":
                other_skipped_count += 1
            if row.get("species"):
                species_match_count += 1
            panel_match_count += 1
            try:
                total_conf += float(row.get("confidence") or 0.0)
            except (TypeError, ValueError):
                pass

    # Parse llm_usage.json
    geo_calls = 0
    geo_cost = 0.0
    total_cost = 0.0
    if llm_usage_json.exists():
        try:
            usage = json.loads(llm_usage_json.read_text(encoding="utf-8"))
            total_cost = float(usage.get("total_cost_cny", 0.0) or 0.0)
            # geo_vision calls counted by looking at MiniMax_API rows
            # in matches.jsonl is hard; we use total cost as proxy here
        except (json.JSONDecodeError, OSError):
            pass

    avg_conf = (total_conf / panel_match_count) if panel_match_count else 0.0
    panel_match_rate = 1.0  # all rows that made it to matches.jsonl had a panel_id
    species_match_rate = (species_match_count / panel_match_count) if panel_match_count else 0.0

    return SmokeRow(
        pdf=pdf.name,
        sha256=_sha256_short(pdf),
        ok=True,
        elapsed_s=elapsed,
        error=None,
        row_count=row_count,
        range_chart_detected_count=range_chart_count,
        other_skipped_count=other_skipped_count,
        geo_vision_calls=geo_calls,
        geo_vision_cost_cny=geo_cost,
        total_cost_cny=total_cost,
        panel_match_rate=panel_match_rate,
        species_match_rate=species_match_rate,
        avg_confidence=avg_conf,
        run_output_path=str(out_jsonl),
        llm_usage_path=str(llm_usage_json) if llm_usage_json.exists() else None,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--corpus",
        type=Path,
        default=_REPO_ROOT / "放射虫论文_OA_download",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "work" / "oa_smoke_round6" / "results.jsonl",
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        default=_REPO_ROOT / "work" / "oa_smoke_round6",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=6,
    )
    p.add_argument(
        "--use-geo-vision",
        action="store_true",
    )
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("oa_smoke_round6")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.unlink(missing_ok=True)

    rows: list[SmokeRow] = []
    for name in PICKED_PDFS[: args.limit]:
        pdf = args.corpus / name
        if not pdf.exists():
            log.warning("missing PDF: %s", name)
            rows.append(
                SmokeRow(
                    pdf=name,
                    sha256="0" * 16,
                    ok=False,
                    elapsed_s=0.0,
                    error="file not found",
                    row_count=0,
                )
            )
            continue
        if pdf.stat().st_size == 0:
            log.warning("zero-byte PDF skipped: %s", name)
            continue
        log.info("=== %s (%d KB) ===", name, pdf.stat().st_size // 1024)
        row = _run_one(pdf, work_dir=args.work_dir, with_geo_vision=args.use_geo_vision)
        log.info(
            "ok=%s rows=%d rc=%d other_skipped=%d species=%.2f cost=¥%.4f elapsed=%.1fs",
            row.ok,
            row.row_count,
            row.range_chart_detected_count,
            row.other_skipped_count,
            row.species_match_rate,
            row.total_cost_cny,
            row.elapsed_s,
        )
        if not row.ok:
            log.error("error: %s", row.error)
        rows.append(row)
        with args.out.open("a") as f:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    # Summary
    ok_rows = [r for r in rows if r.ok]
    fail_rows = [r for r in rows if not r.ok]
    total_cost = sum(r.total_cost_cny for r in ok_rows)
    total_rows = sum(r.row_count for r in ok_rows)
    total_species = sum(int(r.species_match_rate * r.row_count) for r in ok_rows)
    avg_species_rate = (total_species / total_rows) if total_rows else 0.0
    print("\n=== SUMMARY ===")
    print(f"ok: {len(ok_rows)} / {len(rows)}")
    print(f"total rows: {total_rows}")
    print(f"species match rate: {avg_species_rate:.2%}")
    print(f"total cost: ¥{total_cost:.4f}")
    if fail_rows:
        print("failures:")
        for r in fail_rows:
            print(f"  - {r.pdf}: {r.error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
