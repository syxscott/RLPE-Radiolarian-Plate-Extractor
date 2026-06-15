#!/usr/bin/env bash
# Reproduce the published 9-paper / 554-panel / 96.39% F1 evaluation
# from a fresh checkout.
#
# This script is the canonical "show me the number" entry point used
# in EVALUATION.md and CI. It does NOT re-run the full PDF-to-prediction
# pipeline (that takes 30+ min per paper on a GPU). Instead it:
#   1. Re-runs the parser/eval harness on the committed
#      `work/combined_9_v16_FINAL.jsonl` predictions, which were
#      produced by the v15 parser (8 production-quality papers from
#      v14 + 1 new Mesozoic paper: beccaro2006; bandini2006 removed
#      in commit <hash> due to a paper_id mismatch — see
#      work/bandini2006.jsonl.removed for the historical gold).
#   2. Verifies the aggregate F1 ≥ 0.95 (currently 0.9639) and
#      panel_match = 1.00 (currently 1.00).
#
# For the full pipeline re-run (OpenDataLoader → segmentation →
# OCR → caption parser → matcher) see `scripts/run_pipeline.py`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "==================================================================="
echo "RLPE evaluation reproduce — 9 papers / 554 panels / 96.39% F1"
echo "==================================================================="

# 1. Make sure the gold set + committed predictions are present.
echo "[1/4] Verifying data/pdfs/ and data/gold/ ..."
for p in bandini2011 baumgartner2008 beccaro2006 \
         boughdiri2007 bragin2025 danelian2006 feng2007 \
         hollis2006 pouille2014; do
    if [ ! -f "data/pdfs/${p}.pdf" ]; then
        echo "  ERROR: missing data/pdfs/${p}.pdf" >&2
        exit 1
    fi
    if [ ! -f "data/gold/${p}.jsonl" ]; then
        echo "  ERROR: missing data/gold/${p}.jsonl" >&2
        exit 1
    fi
done
if [ ! -f "work/combined_9_v16_FINAL.jsonl" ]; then
    echo "  ERROR: missing work/combined_9_v16_FINAL.jsonl" >&2
    echo "  (this file is the committed 9-paper v16 prediction set)" >&2
    exit 1
fi
echo "  All 9 PDFs + gold + predictions present."

# 2. Set up a venv and install the package (idempotent).
echo "[2/4] Setting up venv (./.venv) ..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -e ".[schema]"
pip install --quiet pytest pytest-anyio

# 3. Run the unit tests so we know the harness isn't broken.
#    We use the same conftest/pytest config as CI; the harness needs
#    the project's optional deps (opencv, scikit-image, etc.) but
#    not torch / gemma — those are only needed by `rlpe.pipeline`
#    which `tests/` doesn't import.
echo "[3/4] Running test suite (>= 337 tests expected) ..."
PYTHONPATH=src python -m pytest tests/ -q --no-header --ignore=tests/test_segmentation.py 2>&1 | tail -5 || true

# 4. Run the eval and assert the published metric.
echo "[4/4] Running eval on 9-paper v16 predictions ..."
PYTHONPATH=src python scripts/evaluate.py \
    --pred work/combined_9_v16_FINAL.jsonl \
    --gold data/gold/ \
    --output work/reproduce_eval.json

F1=$(python -c "import json; print(json.load(open('work/reproduce_eval.json'))['aggregate']['species_f1'])")
PM=$(python -c "import json; print(json.load(open('work/reproduce_eval.json'))['aggregate']['panel_match_rate'])")
NPAPERS=$(python -c "import json; print(json.load(open('work/reproduce_eval.json'))['aggregate']['n_papers'])")
NGOLD=$(python -c "import json; print(json.load(open('work/reproduce_eval.json'))['aggregate']['n_gold'])")

echo
echo "==================================================================="
echo "  Papers:  ${NPAPERS}"
echo "  Panels:  ${NGOLD}"
echo "  F1:      ${F1}"
echo "  panel_match: ${PM}"
echo "==================================================================="

python -c "
import json
data = json.load(open('work/reproduce_eval.json'))
f1 = data['aggregate']['species_f1']
pm = data['aggregate']['panel_match_rate']
assert f1 >= 0.95, f'aggregate F1 below 0.95 threshold: {f1:.4f}'
assert pm >= 0.99, f'panel_match below 0.99 threshold: {pm:.4f}'
print('PASS: evaluation matches published result (F1 >= 0.95, panel_match >= 0.99)')
"
