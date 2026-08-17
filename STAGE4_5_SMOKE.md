# Stage 4.5 live smoke test on Bandini 2011

Date: 2026-08-17
Commit (smoke + fix): <filled-after-commit>
Pre-fix smoke commit: `654c0fc` (0% overwrite rate — flag wiring bug)

## Per-panel run (`--m3-per-panel`), post-fix

- Total rows: 86
- m3_per_panel stamped: **67** (78% of rows reached Stage 4.5)
- Rows with overwriting species (M3 confidence >= 0.55): **0**
- Avg M3 latency: 5.75s, max 19.63s
- Confidence distribution (stamped rows): min 0.00, mean 0.14, max 0.30
  (all values < 0.55 → regex species stays per the confidence-gated
  overwrite design)
- Pipeline exit code: 0
- Sample `metadata.m3_per_panel` payload (panel 1 of pl01):
  ```json
  {
    "species": "None",
    "label": "1",
    "confidence": 0.15,
    "reasoning": "The panel's visible label is '1' (numeric, not a letter). No candidate label–species pairs were supplied from the caption, and no same-page context is available, so no species assignment can be made from the provided metadata.",
    "alternative": null,
    "latency_sec": 6.5,
    "fallback_used": false,
    "image_sha": "acc5d6fedc079d7b"
  }
  ```

## Baseline run (`--no-m3-per-panel`, pre-fix smoke commit)

- Total rows: 67
- m3_per_panel stamped: 0 (expected)
- Pipeline exit code: 0

## Pre-fix bug (smoke commit `654c0fc`)

> **The Stage 4.5 per-panel code path never executed.** Every output row has
> `panel_path`, but **0/64 rows have `metadata.m3_per_panel`**.

Root cause: flag-wiring defect. The CLI set
`m3_per_panel_enabled` as a typed `PipelineConfig` attribute but the
gate in `pipeline.py` read it from `self.config.extra.get(...)`, which
the CLI never populated. The same defect affected the
`m3_stage3` / `m3_multi_plate_enrich` gates.

## Post-fix patch (committed together with this doc)

`src/rlpe/pipeline.py` — three gate lines changed:

- L1715: `self.config.extra.get("m3_stage3", False)` → `self.config.m3_stage3_enabled`
- L1722: `self.config.extra.get("m3_per_panel_enabled", False)` → `self.config.m3_per_panel_enabled`
- L1734: `self.config.extra.get("m3_multi_plate_enrich", False)` → `self.config.m3_multi_plate_enrich_enabled`
- L3701 (GROBID path): same fix for the `m3_stage3` gate duplicate

`src/rlpe/config.py` — two new typed attributes:

- `m3_stage3_enabled: bool = False` (with `__post_init__` coercion)
- `m3_multi_plate_enrich_enabled: bool = False` (with `__post_init__` coercion)

`src/rlpe/cli.py` — three changes:

1. `--m3-per-panel` / `--use-m3-stage-3` / `--m3-multi-plate-enrich` now
   populate typed PipelineConfig kwargs (no longer routed via `extra`).
2. Removed the `cfg.extra["m3_per_panel_enabled"] = ...` mirror hack
   (workaround for the mis-wired gates; no longer needed).
3. Implicit opt-in: enabling any of the three M3 vision flags also sets
   `cfg.extra["m3_enhanced_mode"] = True` (since the gates require
   `self.m3_engine` to be built). Explicit `--m3-enhanced-mode` / `--no-m3-enhanced-mode`
   still wins if both are passed.

`tests/test_stage4_5_m3_per_panel.py` — six new regression tests:

- `test_config_has_m3_gate_typed_attrs` — the three M3 gate flags exist
  as typed attributes on `PipelineConfig`.
- `test_pipeline_gates_use_typed_attrs_not_extra` — source-guard that
  the gates no longer read `config.extra.get("m3_per_panel_enabled"|...)`.
- `test_stage4_5_gate_fires_with_typed_attr` — end-to-end: setting the
  typed attr to True causes the method to stamp metadata.
- `test_stage4_5_gate_short_circuits_when_typed_attr_false` —
  companion: setting to False makes the method pass rows through
  untouched (proves the gate is what controls the path).
- `test_cli_no_longer_mirrors_m3_per_panel_into_extra` — guards against
  re-introducing the workaround mirror.
- Updated `test_cli_argparse_accepts_m3_per_panel_flags` to assert the
  new typed-kwarg wiring + the implicit `m3_enhanced_mode` opt-in.

`tests/test_round6_cli_flags.py` — updated one assertion in
`test_use_geo_vision_routes_into_extra` to reflect that
`--use-m3-stage-3` is now a typed attribute (no longer routed via
`extra`).

## Confirmations from the live re-run

1. **Stage 4.5 now fires.** 67/86 rows stamped `metadata.m3_per_panel`
   (vs 0/64 before). The fix is producing the right effect in
   production.
2. **`m3_stage3_enabled` and `m3_multi_plate_enrich_enabled` are also
   fixed** by the same patch — gates now read the typed attributes, so
   the Stage 3 bbox/crop enrichment and Round 7 multi-plate enrichment
   will fire when their corresponding CLI flags are passed.
3. **Zero high-confidence overwrites (>= 0.55).** M3 returned uniformly
   low confidence (mean 0.14, max 0.30) because the production data
   flow does not currently populate `caption_pairs` / `page_context`
   on the row dicts — Task 6/8 in the original smoke had already
   flagged this as a follow-up. The fix unblocks Stage 4.5; reaching
   the overwrite-rate target (30-70%) requires the Task 6/8 caption
   flow plumbing, which is out of scope for this fix.
4. **The "Stage 4 MiniMax rule-pipeline fallback" warning from the
   pre-fix run is no longer firing** in this run — the LLM/MiniMax
   species extraction went through cleanly this time. Stage 4.5 itself
   is a different code path (M3 engine `infer_panel`), and the
   `fallback_used=False` flag on every stamped row confirms it.

## Artifacts (gitignored, NOT committed)

- `work/output/manifests/matches.jsonl` — per-panel run output (86 rows)
- `work/output/figures/`, `work/output/panels/`, `work/output/tei/`,
  `work/output/od_output/` — per-figure crops + OpenDataLoader output
- `work/output/manifests/run_output.json` — full run record
- `work/output/manifests/llm_usage.json` — LLM cost ledger
- `work/bandini_only/Bandini_2011.pdf` — single-PDF staging dir
- `work/stage4_5_bandini_run.log` — per-panel run log (full)