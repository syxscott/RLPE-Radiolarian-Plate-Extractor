# RLPE Reproducibility Guide

This document explains how to reproduce any RLPE result bit-for-bit,
and how to verify that an existing result is genuine.

## TL;DR — Reproducing the batch4_v2 E2E + post-fix evaluation

```bash
# 1. Clone and install
git clone <repo> && cd <repo>
pip install -e ".[schema]"
pip install -r requirements.txt

# 2. Make sure the input PDFs are in place
ls work/batch4_v2/pdfs/    # should list 4 PDFs

# 3. Run the pipeline
PYTHONPATH=src python -m rlpe.cli \
  --pdf-dir work/batch4_v2/pdfs \
  --work-dir work/batch4_v2 \
  --use-opendataloader

# 4. Apply the caption-parser fix to pouille
PYTHONPATH=src python scripts/reprocess_pouille.py \
  --input work/batch4_v2/results.jsonl \
  --output work/batch4_v2/results_pouille_fixed.jsonl

# 5. Re-build the gold set (idempotent; only needed if captions changed)
PYTHONPATH=src python scripts/build_gold_from_captions.py

# 6. Run baseline and post-fix evaluations
PYTHONPATH=src python scripts/run_evaluation.py \
  --predictions work/batch4_v2/results.jsonl \
  --gold-dir data/gold \
  --output-dir work/batch4_v2/eval_reports \
  --label baseline

PYTHONPATH=src python scripts/run_evaluation.py \
  --predictions work/batch4_v2/results_pouille_fixed.jsonl \
  --gold-dir data/gold \
  --output-dir work/batch4_v2/eval_reports \
  --label pouille_fix

# 7. Export all three downstream views
PYTHONPATH=src python -m rlpe.cli_export \
  --input work/batch4_v2/results_pouille_fixed.jsonl \
  --output-dir work/batch4_v2/exports \
  --include-unmatched
```

## Verifying an existing result

Every exported run carries a `provenance` block (top-level in the JSONL,
or in a sidecar `*.provenance.json` for older formats). The block contains:

| Field | What it tells you |
| --- | --- |
| `pipeline_version` | RLPE semver (e.g. `1.1.0`) |
| `schema_version` | Output JSON Schema semver (e.g. `1.0.0`) |
| `git_commit` | Short SHA of the running commit |
| `git_dirty` | True if uncommitted changes were present |
| `config_snapshot` | Resolved PipelineConfig dict |
| `input_sha256` | `{filename: hex_digest}` for every input PDF |
| `timestamp_utc` | ISO 8601 UTC of the run |
| `host` | `os/release/machine/node` |
| `python_version` | e.g. `3.11.15` |

To verify that a result is genuine:

1. `git checkout <commit_from_provenance.git_commit>` — the running commit
2. `pip install -e .` — pin the exact code
3. Recompute the input SHA-256: `sha256sum work/batch4_v2/pdfs/*.pdf`
4. Compare against `provenance.input_sha256`
5. Re-run the pipeline and diff the output

If the SHA-256s match and the outputs match, the result is reproducible
on the same host (and Python version). The schema-version guard test
(`tests/test_schema_published.py`) ensures the published JSON Schema
is in sync with the code at the time of the run.

## Verifying the gold set

The gold set is at `data/gold/{paper}.jsonl`. Each line is a JSON
object with `paper_id`, `figure_id`, `panel_id`, `species`. To verify
that a prediction set was evaluated against the right gold:

```bash
PYTHONPATH=src python -c "
from rlpe.evaluation import load_gold
gold = load_gold('data/gold/hollis2006.jsonl')
print(f'{len(gold)} gold panels for hollis2006')
for p in gold[:3]: print(p.to_dict())
"
```

The `tests/test_gold.py` suite asserts the gold files exist, parse
as JSONL, and contain a sane number of panels for each paper.

## Verifying the published schema

The published schema is at `schemas/rlpe-v1.0.0.json`. It is regenerated
by `python -m rlpe.schema_dump` and must match what Pydantic emits today.
The drift guard is `tests/test_schema_published.py`:

```bash
PYTHONPATH=src python -m pytest tests/test_schema_published.py -v
```

If the published file is out of sync, the test fails with a diff. To
re-sync, run `python -m rlpe.schema_dump` and commit the change.

## Test coverage

`pytest tests/` is the entry point. Current state (2026-06-06):

- 267 tests passing, 2 skipped (pyarrow / jsonschema optional deps)
- Categories: provenance, schema_models, schema_published, gold,
  evaluation_metrics, exporters, danelian_caption_parser,
  pouille_caption_parser, scale_bar, stratigraphy, fig_caption,
  placeholder_filter, association, nms_and_cross_figure, etc.

## What's NOT in scope for reproducibility

- The neural matcher (`--use-neural-matcher`) is non-deterministic
  when run on GPU; the GPU and CUDA version affect output.
- The Gemma 4 / M3 / Anthropic LLM backends are non-deterministic
  by design. Set `--MiniMax-temperature 0` to minimize variance, but
  the cloud API is still a moving target.
- The web UI uses Celery/Redis which has its own ordering guarantees.

These are flagged in the `provenance.config_snapshot` and in the
README under "Caveats".
