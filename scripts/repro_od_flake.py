# Repro harness for the intermittent OD zero-figure flake.
#
# Background: three pipeline runs consumed a byte-identical OD JSON
# (MD5 2d501f1f...); two produced 0 figures (leading to the misleading
# _ingestion_grobid_cycle stub) and one produced 4 figures. This script
# re-runs the pure figure-extraction path over that saved JSON in fresh
# subprocesses with varying PYTHONHASHSEED to test whether the pairing
# outcome is process-hash-order dependent.
#
# Usage: python scripts/repro_od_flake.py [json_path] [od_dir] [--seeds N]
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
DEFAULT_JSON = Path(
    r"D:\rlpe_out\llm_compare\run_GLM\od_output\1e618c2e20204797"
    r"\Afanasieva 2020c Revision of Spinodeflandrella Holdsworthell "
    r"Haplodiacanhtus [Paleo J 54(12) 1443-1455].json"
)
DEFAULT_OD_DIR = Path(r"D:\rlpe_out\llm_compare\run_GLM\od_output\1e618c2e20204797")
PAPER_ID = "1e618c2e20204797"

CHILD = r"""
import json, sys
sys.path.insert(0, %r)
from pathlib import Path
from rlpe.opendataloader_extractor import (
    OpenDataLoaderExtractor,
    _find_plate_captions,
)

data = json.load(open(%r, encoding="utf-8"))
od_dir = Path(%r)
ex = OpenDataLoaderExtractor.__new__(OpenDataLoaderExtractor)
ex.caption_window = 5
ex.cross_page_captions = True
ex.merge_gap_pt = 12
ex.output_dir = od_dir
ex.rescue_ocr_inprocess = False

kids = data.get("kids") or []
plates = _find_plate_captions(kids, caption_window=5, cross_page=True)
figs = ex._extract_figures(data, od_dir, %r) or []
unpaired = ex._extract_unpaired_captions(data, list(figs), od_dir, %r)
print(json.dumps({
    "plate_captions": len(plates),
    "figures": len(figs),
    "unpaired": len(unpaired),
    "fig_ids": sorted(f.figure_id for f in figs)[:8],
}))
"""


def run_one(seed: int, json_path: Path, od_dir: Path) -> dict:
    env = {
        "PYTHONHASHSEED": str(seed),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "PATH": os.environ.get("PATH", ""),
    }
    proc = subprocess.run(
        [sys.executable, "-c", CHILD % (str(SRC), str(json_path), str(od_dir), PAPER_ID, PAPER_ID)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        env=env,
    )
    if proc.returncode != 0:
        return {"seed": seed, "error": (proc.stderr or "")[-300:]}
    return {"seed": seed, **json.loads(proc.stdout.strip().splitlines()[-1])}


def main() -> None:
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    od_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OD_DIR
    seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 20

    results = [run_one(s, json_path, od_dir) for s in range(seeds)]
    ok = [r for r in results if "error" not in r]
    err = [r for r in results if "error" in r]

    counts = collections.Counter(r["figures"] for r in ok)
    plate_counts = collections.Counter(r["plate_captions"] for r in ok)
    print(f"seeds={seeds} ok={len(ok)} errors={len(err)}")
    print(f"figures distribution: {dict(counts)}")
    print(f"plate_captions distribution: {dict(plate_counts)}")
    if err:
        print("errors:")
        for r in err[:3]:
            print("  seed", r["seed"], r["error"])
    varying = len(counts) > 1
    print("VERDICT:", "HASH-ORDER DEPENDENT (varies by seed)" if varying
          else "deterministic across seeds — flake is NOT hash-order")
    flaky = [r for r in ok if r["figures"] == 0]
    if flaky:
        print("zero-figure seeds:", [r["seed"] for r in flaky])
        print("sample zero-figure detail:", json.dumps(flaky[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
