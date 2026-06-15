# RLPE Reproducible Scientific Deliverable — Design

**Date:** 2026-06-06
**Status:** Approved (user pre-approved full-scope auto-execute)
**Author:** Claude (brainstorming session)

## Goal

Take the existing RLPE pipeline (4-paper batch4_v2, 178 tests, 438 panels, 10.6%–78.9% species match) from "working prototype" to "reproducible scientific deliverable" that an external reviewer can verify, a journal can cite, and three downstream audiences can consume.

## Scope (user-confirmed: full delivery)

1. Versioned output schema
2. Provenance metadata (commit hash, config snapshot, input SHA256, host, timestamp)
3. Ground-truth evaluation harness on the 4-paper batch
4. Pouille association algorithm fix (10.6% → closer to 78.9%)
5. Darwin Core mapping (PBDB/GBIF ingest)
6. Three exporters: analysis (CSV/Parquet), ML (JSONL + splits), archive (DwC-A zip)
7. CHANGELOG.md, CITATION.cff, REPRODUCIBILITY.md, SCHEMA.md

## Approach: Contract-first

Order of work:
1. Provenance stamping (what run produced this output)
2. Migrate `types.py` dataclass → Pydantic v2 canonical models
3. Auto-generate JSON Schema, publish to `schemas/rlpe-v1.0.0.json`
4. Build ground-truth dataset (4 papers × panels × species)
5. Build evaluation harness (PRF, IoU, F1, by paper and aggregate)
6. Run baseline, record metrics
7. Fix pouille caption→panel association
8. Re-run, record improvement
9. Build 3 exporters
10. DwC-A mapping
11. CHANGELOG / CITATION.cff / REPRODUCIBILITY.md / SCHEMA.md
12. Commit everything

## Architecture

```
rlpe/
  types.py              ← Pydantic v2 canonical models (was dataclass)
  pipeline.py           ← unchanged
  exporters/
    analysis.py         ← CSV + Parquet + DwC field renaming
    ml.py               ← JSONL + train/val/test split
    archive.py          ← DwC-A zip (PBDB/GBIF ingest)
  evaluation/
    gold.py             ← ground-truth loader
    metrics.py          ← PRF/IoU/F1
    report.py           ← Markdown + JSON report writer
  provenance/
    stamp.py            ← git rev / config snap / SHA256 / host
  cli.py                ← adds --export-* and --evaluate
data/gold/              ← 4 papers × panels × species (manual annotation)
schemas/                ← published, versioned JSON Schemas
docs/                   ← SCHEMA.md, REPRODUCIBILITY.md, CHANGELOG.md, CITATION.cff
```

## Output schema (top-level)

```json
{
  "schema_version": "1.0.0",
  "provenance": {
    "pipeline_version": "1.1.0",
    "git_commit": "5e88953",
    "git_dirty": true,
    "config_snapshot": { ... },
    "input_sha256": { "bandini2011.pdf": "..." },
    "timestamp_utc": "2026-06-06T03:30:00Z",
    "host": "linux/x86_64"
  },
  "panels": [ ... ]
}
```

## Exporter contracts (three views, one source)

- **Analysis view** (paleontologists): flat CSV/Parquet with DwC field names (`scientificName`, `eventDate`, `locality`, `decimalLatitude`, `decimalLongitude`, `occurrenceID`)
- **ML view** (researchers): JSONL with HF datasets loaders, train/val/test split by paper (no leakage), confidence + label noise flag
- **Archive view** (DB operators): DwC-A zip with `meta.xml`, occurrence core, EMl extension

## Non-goals

- Real-time API rate-limiting
- Multi-tenant authentication
- Cloud-native deployment (the design is local-Python, deployment is a separate concern)
- New LLM backends (existing 4 stay)
- UI redesign (web/ stays as-is)

## Risks

- Pydantic v2 migration of `types.py` could subtly change `asdict()` semantics — mitigated by adding tests before migration
- Ground-truth annotation is time-consuming — start with the 4 batch papers; defer others
- Pouille association fix may be a non-trivial rewrite of `association.py` — bounded by the existing `match_panels` API contract

## Verification

- All 178 existing tests still pass
- New `tests/test_provenance.py`, `tests/test_schema_validation.py`, `tests/test_exporters.py`
- `pytest tests/` → 178+N green
- `python -m rlpe --evaluate` → baseline + post-fix metrics in `work/batch4_v2/eval_report.md`
- `python -m rlpe.cli --export-analysis` → CSV/Parquet in `work/batch4_v2/exports/analysis/`
- `python -m rlpe.cli --export-ml` → JSONL + splits
- `python -m rlpe.cli --export-archive` → DwC-A zip
