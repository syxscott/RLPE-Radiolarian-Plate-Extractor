# Stage 4.5 live smoke test on Bandini 2011

Date: 2026-08-17
Commit: <filled-after-commit>

## Per-panel run (`--m3-per-panel`)
- Total rows: 64
- m3_per_panel stamped: 0
- Overwrite rate: 0.00% (0 of 0 stamped)
- High-confidence overwrites (>= 0.55): 0
- Avg M3 latency: n/a (no rows reached Stage 4.5)
- Max M3 latency: n/a
- Pipeline exit code: 0

## Baseline run (`--no-m3-per-panel`)
- Total rows: 67
- m3_per_panel stamped: 0 (expected)
- Pipeline exit code: 0

## Observations / FAILURE mode

**The Stage 4.5 per-panel code path did NOT execute.** No row in
`work/stage4_5_bandini/manifests/matches.jsonl` carries the
`m3_per_panel` metadata key, even though every row has a populated
top-level `panel_path` (Stage 3 crops are present).

### Root cause: flag is set on the typed attribute, but the call site reads it from `config.extra`

- `src/rlpe/cli.py` (line 495) wires the flag into the constructor as a
  typed kwarg: `m3_per_panel_enabled=args.m3_per_panel`. The inline
  comment there explicitly says this is intentional ("Passed as real
  PipelineConfig fields (not ``extra``) because
  ``_apply_m3_per_panel_species_id`` reads the typed attributes").
- `src/rlpe/pipeline.py` (line 1722) guards the call with
  `self.config.extra.get("m3_per_panel_enabled", False)`. Because the
  CLI did NOT mirror the value into `config.extra`, that `.get(...)`
  call returns `False` and the per-panel branch is skipped on every
  paper, every run.
- Verified empirically:
  ```python
  cfg = PipelineConfig(pdf_dir="x", work_dir="y", m3_per_panel_enabled=True)
  cfg.m3_per_panel_enabled      # -> True
  cfg.extra.get("m3_per_panel_enabled")  # -> None
  ```
- Same wiring inconsistency likely affects the other
  `self.config.extra.get("m3_stage3", ...)` /
  `...get("m3_multi_plate_enrich", ...)` call sites, but those opt-in
  flags have never been pushed through `extra` either, so they have
  the same defect.

### Secondary observation (not blocking)

The run also logged recurring `[MiniMax] API error, falling back to
rule pipeline for 197ae8bef60db7fb/od_plate_197ae8bef60db7fb_pNNN_plNN`
warnings for every figure (Stage 4 MiniMax call). This is pre-existing —
even if the per-panel flag wiring were fixed, Stage 4.5 would still
fire the same M3 backend and likely hit the same fallback. Not retried
per task constraint.

## Recommended next step (do NOT execute here — task forbids src/ edits)

In `src/rlpe/pipeline.py` line 1722, change the guard from
`self.config.extra.get("m3_per_panel_enabled", False)` to
`bool(self.config.m3_per_panel_enabled)` so the typed attribute the CLI
sets is what the gate reads. Re-run after that fix to measure the real
overwrite rate.

## Artifacts (gitignored, NOT committed)

- `work/stage4_5_bandini/` — per-panel run output
  (`manifests/matches.jsonl`, `panels/`, `figures/`, `tei/`,
  `od_output/`)
- `work/stage4_5_bandini_sw/` — service work dir
- `work/stage4_5_bandini_baseline/` — baseline run output
- `work/stage4_5_bandini_baseline_sw/` — baseline service work dir
- `work/bandini_only/` — single-PDF staging dir (CLI takes `--pdf-dir`,
  not a positional PDF path; the task spec's CLI signature was out of
  date)
- `work/stage4_5_bandini_run.log` — per-panel run log (full)
- `work/stage4_5_bandini_baseline.log` — baseline run log (full)