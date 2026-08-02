#!/usr/bin/env bash
# reproduce_eval.sh — re-run the full RLPE pipeline on a 3-paper
# subset and evaluate against gold/. End-to-end reproducible F1.
# Usage: ANTHROPIC_API_KEY=sk-... bash reproduce_eval.sh [--limit N]
#
# This is the canonical "show me the number" entry point. Unlike the
# previous version (which only re-ran the eval harness on a frozen
# prediction file), this script RE-RUNS the full PDF→predictions
# pipeline from scratch using the MiniMax M3 backend and the
# OpenDataLoader figure parser. Wall time on a GPU is ~5-15 min per
# paper; full 9-paper set takes 30+ min and burns ¥100+ in M3 calls,
# so we default to a 3-paper subset.
#
# Layout:
#   work/reproduce_eval/
#     predictions.jsonl          # concat of per-paper jsonl
#     per_paper/<name>/          # each paper's work-dir
#       output/manifests/matches.jsonl
#       output/manifests/llm_usage.json
#     eval.json                  # scripts/evaluate.py JSON dump
#     REPORT.md                  # scripts/evaluate.py markdown + our header
#     per_paper_metrics.tsv      # per-paper F1 table
#
# Exit code:
#   0 — every paper finished (ok or skip-warn) and eval ran
#   1 — prerequisites missing, no paper ran, or eval failed
#
# The previous version (eval on a frozen jsonl) is kept documented in
# git history (commits before this rewrite) for reference.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
LIMIT=3
PAPER_ARG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --limit) LIMIT="$2"; shift 2 ;;
        --paper) PAPER_ARG="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,12p' "$0" | sed 's/^# *//'
            exit 0
            ;;
        *) echo "ERROR: unknown arg: $1" >&2; exit 1 ;;
    esac
done

OUT_DIR="$REPO_ROOT/work/reproduce_eval"
PER_PAPER_DIR="$OUT_DIR/per_paper"
mkdir -p "$OUT_DIR" "$PER_PAPER_DIR"
PRED_FILE="$OUT_DIR/predictions.jsonl"
EVAL_JSON="$OUT_DIR/eval.json"
REPORT_MD="$OUT_DIR/REPORT.md"
PER_PAPER_TSV="$OUT_DIR/per_paper_metrics.tsv"
TIMING_LOG="$OUT_DIR/timing.log"
COST_LOG="$OUT_DIR/cost.log"
: > "$PRED_FILE"
: > "$TIMING_LOG"
: > "$COST_LOG"

# ---------------------------------------------------------------------------
# Pre-flight: env, deps, gold set
# ---------------------------------------------------------------------------
echo "==================================================================="
echo "RLPE reproduce_eval.sh — full pipeline re-run + eval (3-paper subset)"
echo "==================================================================="

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY is not set. The MiniMax M3 backend" >&2
    echo "       reads the M-series via the Anthropic-compatible endpoint," >&2
    echo "       so ANTHROPIC_API_KEY must be in the environment." >&2
    echo "       Re-run with:  ANTHROPIC_API_KEY=sk-... bash reproduce_eval.sh" >&2
    exit 1
fi
echo "  ANTHROPIC_API_KEY: set (length=${#ANTHROPIC_API_KEY})"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "ERROR: $PY not found in PATH" >&2
    exit 1
fi

if [ ! -d "$REPO_ROOT/src/rlpe" ]; then
    echo "ERROR: $REPO_ROOT/src/rlpe not found. Are you in the repo root?" >&2
    exit 1
fi

if [ ! -d "$REPO_ROOT/data/gold" ]; then
    echo "ERROR: $REPO_ROOT/data/gold/ missing. Run from a fresh checkout" >&2
    echo "       or restore data/gold/ from the original commit." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Pick the 3-paper subset.  Default: 3 representative papers that cover
#  - high-F1 (beccaro2006, single-plate, clean)
#  - multi-plate (bandini2011, 9 plates, 273 gold panels)
#  - short abstract (pouille2014, 6 gold panels, 1 plate)
# Plus optional override via --paper <short_name>.
# ---------------------------------------------------------------------------
DEFAULT_PAPERS=("beccaro2006" "bandini2011" "pouille2014")
if [ -n "$PAPER_ARG" ]; then
    PAPERS=("$PAPER_ARG")
else
    PAPERS=("${DEFAULT_PAPERS[@]}")
fi
# Trim to --limit
if [ "${#PAPERS[@]}" -gt "$LIMIT" ]; then
    PAPERS=("${PAPERS[@]:0:$LIMIT}")
fi

echo "  Papers: ${PAPERS[*]}"
echo "  Out:    $OUT_DIR"
echo

# ---------------------------------------------------------------------------
# Find each paper's PDF by stable_id match against gold
# ---------------------------------------------------------------------------
# We compute stable_id(pdf) for every PDF in OA download + data/pdfs/
# + service_work/<job>/pdfs/ and keep the first one whose hash matches
# the gold file's paper_id.  This makes the script robust against
# re-downloaded OA copies (different bytes → different paper_id) and
# also against operators who keep PDFs in a custom dir.
PYTHONPATH="$REPO_ROOT/src" "$PY" - <<'PYEOF' > "$OUT_DIR/paper_index.tsv"
import os, sys
sys.path.insert(0, 'src')
from pathlib import Path
import json
from rlpe.utils import stable_id

REPO = Path('/home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor')
GOLD_DIR = REPO / 'data' / 'gold'

# Build gold short-name -> paper_id map
gold_index: dict[str, str] = {}
for f in sorted(GOLD_DIR.glob('*.jsonl')):
    g = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if g:
        gold_index[f.stem] = g[0]['paper_id']

# Candidate dirs
candidate_dirs = [
    REPO / 'data' / 'pdfs',
    REPO / '放射虫论文_OA_download',
]
# service_work: every <job>/pdfs/*.pdf
sw = REPO / 'service_work'
if sw.is_dir():
    for sub in sw.iterdir():
        if (sub / 'pdfs').is_dir():
            candidate_dirs.append(sub / 'pdfs')

index: dict[str, list[tuple[str, str]]] = {k: [] for k in gold_index}
for d in candidate_dirs:
    if not d.is_dir():
        continue
    for p in d.glob('*.pdf'):
        try:
            sid = stable_id(p)
        except Exception:
            continue
        for short, gid in gold_index.items():
            if sid == gid:
                index[short].append((str(p), sid))

for short in gold_index:
    matches = index[short]
    if matches:
        for p, sid in matches:
            print(f"{short}\t{sid}\t{p}")
    else:
        print(f"{short}\t{gold_index[short]}\tMISSING")
PYEOF
echo "  Paper index:"
column -t -s $'\t' "$OUT_DIR/paper_index.tsv" | sed 's/^/    /'
echo

# ---------------------------------------------------------------------------
# Per-paper pipeline run
# ---------------------------------------------------------------------------
PER_PAPER_FAIL=0
for paper in "${PAPERS[@]}"; do
    pdf_path="$(awk -F'\t' -v p="$paper" '$1==p && $3!="MISSING"{print $3; exit}' "$OUT_DIR/paper_index.tsv")"
    gold_id="$(awk -F'\t' -v p="$paper" '$1==p{print $2; exit}' "$OUT_DIR/paper_index.tsv")"
    if [ -z "$pdf_path" ]; then
        echo "  [WARN] $paper: no PDF on disk with matching gold paper_id ($gold_id). Skipping."
        echo "         (the OA download may have been re-fetched and the bytes" \
             "no longer match the gold. Restore the original PDF or run" \
             "with --paper <other>.)"
        continue
    fi
    echo "---"
    echo "[run] $paper"
    echo "      PDF:  $pdf_path"
    echo "      gold: paper_id=$gold_id"

    paper_work="$PER_PAPER_DIR/$paper"
    rm -rf "$paper_work"
    mkdir -p "$paper_work/pdfs"
    # Copy (not symlink) so the per-paper work-dir is self-contained
    # and a re-run with the same PDFs produces the same paper_id.
    cp "$pdf_path" "$paper_work/pdfs/"
    out_jsonl="$paper_work/output/manifests/matches.jsonl"
    llm_usage_json="$paper_work/output/manifests/llm_usage.json"
    mkdir -p "$(dirname "$out_jsonl")"

    cmd=(
        "$PY" -m rlpe.cli
        --pdf-dir "$paper_work/pdfs"
        --work-dir "$paper_work"
        --ocr-backend paddleocr
        --num-workers 1
        --use-gpu
        --use-opendataloader
        --llm-backend minimax
        --data-outbound-policy api_full
        --MiniMax-fallback-default rules
        --min-panel-score 0.5
        --export-jsonl "$out_jsonl"
        --m3-disable-stage 4
        --m3-disable-stage 5
    )

    t0=$(date +%s)
    if timeout 1200 "${cmd[@]}" >"$paper_work/stdout.log" 2>"$paper_work/stderr.log"; then
        rc=0
    else
        rc=$?
    fi
    t1=$(date +%s)
    elapsed=$((t1 - t0))
    echo "      elapsed=${elapsed}s rc=$rc"

    if [ "$rc" -ne 0 ]; then
        echo "  [WARN] $paper: pipeline exit=$rc (continuing — see $paper_work/stderr.log)"
        PER_PAPER_FAIL=$((PER_PAPER_FAIL + 1))
        echo -e "$paper\tERROR\t$elapsed\t$rc" >> "$TIMING_LOG"
        continue
    fi

    if [ ! -s "$out_jsonl" ]; then
        echo "  [WARN] $paper: no predictions produced (matches.jsonl empty)"
        PER_PAPER_FAIL=$((PER_PAPER_FAIL + 1))
        echo -e "$paper\tEMPTY\t$elapsed\t0" >> "$TIMING_LOG"
        continue
    fi

    # Append to combined predictions file
    cat "$out_jsonl" >> "$PRED_FILE"
    n_rows=$(wc -l < "$out_jsonl")
    echo "      rows=$n_rows"
    echo -e "$paper\tOK\t$elapsed\t$n_rows" >> "$TIMING_LOG"

    # Pull cost (cny) from llm_usage.json if present
    if [ -f "$llm_usage_json" ]; then
        cost_cny=$("$PY" -c "import json; d=json.load(open('$llm_usage_json')); print(float(d.get('total_cost_cny', 0.0) or 0.0))" 2>/dev/null || echo "0.0")
        echo -e "$paper\t$cost_cny" >> "$COST_LOG"
    fi
done

# ---------------------------------------------------------------------------
# If we got nothing, fail loud — running eval on zero predictions is
# meaningless and would silently report 0% F1.
# ---------------------------------------------------------------------------
n_preds=$(wc -l < "$PRED_FILE" 2>/dev/null || echo 0)
if [ "${n_preds:-0}" -eq 0 ]; then
    echo
    echo "ERROR: no predictions produced by any paper. Eval cannot run." >&2
    echo "  Likely causes:" >&2
    echo "   - OA-downloaded PDFs do not match the gold paper_id (re-downloaded bytes)" >&2
    echo "   - MiniMax API rejected the dummy/test key" >&2
    echo "   - Network/OCR/segmentation pipeline error" >&2
    echo "  Inspect $PER_PAPER_DIR/*/stderr.log" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Run the eval harness
# ---------------------------------------------------------------------------
echo
echo "---"
echo "[eval] $PRED_FILE ($(wc -l < "$PRED_FILE") rows) vs data/gold/"
cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT/src" "$PY" scripts/evaluate.py \
    --pred "$PRED_FILE" \
    --gold "$REPO_ROOT/data/gold/" \
    --output "$EVAL_JSON" \
    --report "$REPORT_MD"

# scripts/evaluate.py writes <output>.md (so REPORT.md is overwritten
# by the eval — we want to keep ours, so prepend our header to the
# eval's per-paper table).
# Read the aggregate F1 etc. out of eval.json
agg_f1=$("$PY" -c "import json; d=json.load(open('$EVAL_JSON'))['aggregate']; print(d['species_f1'])" 2>/dev/null || echo "0.0")
agg_p=$("$PY" -c "import json; d=json.load(open('$EVAL_JSON'))['aggregate']; print(d['species_precision'])" 2>/dev/null || echo "0.0")
agg_r=$("$PY" -c "import json; d=json.load(open('$EVAL_JSON'))['aggregate']; print(d['species_recall'])" 2>/dev/null || echo "0.0")
agg_pm=$("$PY" -c "import json; d=json.load(open('$EVAL_JSON'))['aggregate']; print(d['panel_match_rate'])" 2>/dev/null || echo "0.0")
agg_n_papers=$("$PY" -c "import json; d=json.load(open('$EVAL_JSON'))['aggregate']; print(d['n_papers'])" 2>/dev/null || echo "0")
agg_n_gold=$("$PY" -c "import json; d=json.load(open('$EVAL_JSON'))['aggregate']; print(d['n_gold'])" 2>/dev/null || echo "0")

# Per-paper metrics: from the eval.json "papers" sub-dict
"$PY" - <<PYEOF > "$PER_PAPER_TSV"
import json
d = json.load(open("$EVAL_JSON"))
print("paper_id\tn_gold\tn_pred\tpanel_match\tspecies_p\tspecies_r\tspecies_f1\texact_match")
for pid in sorted(d.get("papers", {})):
    m = d["papers"][pid]
    print(f"{pid}\t{m.get('n_gold',0)}\t{m.get('n_pred_panels',0)}\t"
          f"{m.get('panel_match_rate',0):.4f}\t{m.get('species_precision',0):.4f}\t"
          f"{m.get('species_recall',0):.4f}\t{m.get('species_f1',0):.4f}\t"
          f"{m.get('exact_match_rate',0):.4f}")
PYEOF

# ---------------------------------------------------------------------------
# Prepend our reproducibility header to the eval REPORT.md
# ---------------------------------------------------------------------------
GIT_HEAD=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")
RUN_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
total_wall=$(awk '{s+=$3} END{print s+0}' "$TIMING_LOG")
total_cost=$(awk '{s+=$2} END{printf "%.4f", s+0}' "$COST_LOG")
{
    echo "# RLPE Reproduce-Eval Report"
    echo
    echo "## Reproducibility"
    echo
    echo "| Field | Value |"
    echo "| --- | --- |"
    echo "| Pipeline commit | \`$GIT_HEAD\` |"
    echo "| Run timestamp | $RUN_TS |"
    echo "| Wall time (sum per-paper) | ${total_wall}s |"
    echo "| MiniMax M3 cost (sum cny) | ¥$total_cost |"
    echo "| Papers requested | ${PAPERS[*]} |"
    echo "| Papers OK | $(awk -F'\t' '$2=="OK"' "$TIMING_LOG" | wc -l) |"
    echo "| Papers failed | $PER_PAPER_FAIL |"
    echo "| ANTHROPIC_API_KEY | present (length=${#ANTHROPIC_API_KEY}) |"
    echo "| LLM backend | minimax (MiniMax-M3) |"
    echo "| OCR backend | paddleocr (--use-gpu) |"
    echo "| PDF parser | OpenDataLoader (--use-opendataloader) |"
    echo "| M3 stages disabled | 4, 5 (LLM-first path keeps stages 1-3) |"
    echo "| MiniMax fallback | rules |"
    echo "| min-panel-score | 0.5 |"
    echo
    echo "## Per-paper F1 (from scripts/evaluate.py)"
    echo
    cat "$PER_PAPER_TSV" | column -t -s $'\t' | sed 's/^/| /; s/$/ |/'
    echo
    echo "## Per-paper timing"
    echo
    echo "| paper | status | elapsed_s | rows |"
    echo "| --- | --- | ---: | ---: |"
    awk -F'\t' '{printf "| %s | %s | %s | %s |\n",$1,$2,$3,$4}' "$TIMING_LOG"
    echo
    echo "## Per-paper M3 cost (cny)"
    echo
    echo "| paper | cost_cny |"
    echo "| --- | ---: |"
    awk -F'\t' 'NF==2{printf "| %s | %s |\n",$1,$2}' "$COST_LOG" || true
    if [ ! -s "$COST_LOG" ]; then
        echo "| (none) | 0.0 |"
    fi
    echo
    echo "## Aggregate (from scripts/evaluate.py)"
    echo
    echo "| Metric | Value |"
    echo "| --- | --- |"
    echo "| Papers evaluated | $agg_n_papers |"
    echo "| Gold panels | $agg_n_gold |"
    echo "| Species F1 | $agg_f1 |"
    echo "| Species precision | $agg_p |"
    echo "| Species recall | $agg_r |"
    echo "| Panel match rate | $agg_pm |"
    echo
    echo "## Eval harness output (raw)"
    echo
    cat "$REPORT_MD"
} > "$REPORT_MD.tmp"
mv "$REPORT_MD.tmp" "$REPORT_MD"

# ---------------------------------------------------------------------------
# Final stdout summary
# ---------------------------------------------------------------------------
echo
echo "==================================================================="
echo "  Wall time:       ${total_wall}s"
echo "  MiniMax cost:    ¥$total_cost"
echo "  Papers (OK):     $(awk -F'\t' '$2=="OK"' "$TIMING_LOG" | wc -l) / ${#PAPERS[@]}"
echo "  Predictions:     $(wc -l < "$PRED_FILE") rows"
echo "  Aggregate F1:    $agg_f1"
echo "  Panel match:     $agg_pm"
echo "  Report:          $REPORT_MD"
echo "  Eval JSON:       $EVAL_JSON"
echo "==================================================================="

if [ "$PER_PAPER_FAIL" -gt 0 ] && [ "$agg_n_papers" -eq 0 ]; then
    echo "ERROR: every requested paper failed; eval is empty. Exiting 1." >&2
    exit 1
fi
exit 0
