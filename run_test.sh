#!/bin/bash
# 一键跑测试 PDF：OpenDataLoader + MiniMax M3 + EasyOCR
# 用法: bash run_test.sh [pdf_path]
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
PDF="${1:-${ROOT}/Xiao Yifan et al 2017a micro-XCT.pdf}"
TEST_PDF="/tmp/rlpe_test_input.pdf"
WORK_DIR="${ROOT}/work"

# 准备
mkdir -p data/pdfs
cp "$PDF" "$TEST_PDF"

# 加载 env
[ -f .env ] && set -a && source .env && set +a
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

rm -rf "$WORK_DIR"

python3 << PYINNER
from dotenv import load_dotenv
load_dotenv("${ROOT}/.env", override=True)
import os, logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
val = os.environ.get("ANTHROPIC_API_KEY","")
print(f"[env] ANTHROPIC_API_KEY length: {len(val)}")
from pathlib import Path
from rlpe.config import PipelineConfig
from rlpe.pipeline import RadiolarianPipeline
cfg = PipelineConfig(
    pdf_dir=Path("${ROOT}/data/pdfs"),
    work_dir=Path("${WORK_DIR}"),
    grobid_url="http://localhost:8070",
    use_gpu=False,
    ocr_backend="easyocr",
    num_workers=1,
    render_dpi=120,
    min_panel_score=0.5,
    extra={
        "use_opendataloader": True,
        "use_gemma4": True,
        "llm_backend": "MiniMax",
        "MiniMax_fallback_default": "rules",
        "MiniMax_interactive": False,
        "gemma_conf_threshold": float("${GEMMA_CONF:-0.70}"),
    },
)
pipeline = RadiolarianPipeline(cfg)
rows = pipeline.run()
print(f"\n=== Final: {len(rows)} rows ===")
import json
out = Path("${WORK_DIR}")/"output/manifests/matches.jsonl"
n_species = sum(1 for r in rows if r.get('species'))
n_m3_used = sum(1 for r in rows if r.get('metadata',{}).get('gemma_used'))
print(f"  rows with species: {n_species}/{len(rows)}")
print(f"  rows with M3 result applied: {n_m3_used}/{len(rows)}")
print(f"  output: {out}")
PYINNER
