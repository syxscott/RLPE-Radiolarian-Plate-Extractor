# RLPE Output Schema Reference

This document describes the canonical output shape of an RLPE run.
The machine-readable JSON Schema lives at `schemas/rlpe-v1.0.0.json`.
Newer schema versions (`v1.1.0`, `v1.2.0`, `v1.3.0`) are listed under "Schema versions" below.

## Top-level shape

```json
{
  "schema_version": "1.0.0",
  "provenance": { ... },
  "panels": [ ... ]
}
```

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `schema_version` | string | yes | Semver of the output schema. Bumped on any field change. |
| `provenance` | object | yes | See Provenance below. |
| `panels` | array of PanelRecord | yes | One entry per detected panel. |

## Schema versions

| Version | Status | Notable additions |
| --- | --- | --- |
| `1.0.0` | shipped | Initial release. `PanelRecord`, `ScaleBarRecord`, `GeologyLinkRecord`, `PaperMetadataRecord`. |
| `1.1.0` | shipped (2026-08-02) | Adds 3 `PanelRecord` fields: `confidence_interval_low`, `confidence_interval_high`, `image_verified`, `review_priority`. |
| `1.2.0` | shipped (2026-08-02) | Adds `MorphologyRecord` (Stage 6, opt-in via `--m3-stage-6`). |
| `1.3.0` | shipped (2026-09-05) | Adds `RunOutput.knowledge_graphs` / `RunOutput.range_charts` (paper-level views, previously computed but dropped) and `PanelMetadata.matched_location` (map→range-chart bridge result). `ScaleBarRecord.warning` and per-entry `GeologyLinkRecord.link_source` / `figure_id` are now actually populated by the converter. |

## Provenance

```json
{
  "pipeline_version": "1.1.0",
  "schema_version": "1.0.0",
  "git_commit": "5e88953",
  "git_dirty": true,
  "config_snapshot": { "pdf_dir": "...", "min_panel_score": 0.8, ... },
  "input_sha256": { "bandini2011.pdf": "..." },
  "timestamp_utc": "2026-06-06T03:30:00Z",
  "host": "linux/6.17.0/x86_64/<node>",
  "python_version": "3.11.15"
}
```

The provenance block makes the result reproducible. See
`REPRODUCIBILITY.md` for the verification protocol.

## PanelRecord

A single detected specimen panel.

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `paper_id` | string | yes | Internal paper hash. Stable across re-runs. |
| `figure_id` | string | yes | `od_plate_<hash>_p<NNN>_pl<NN>` or similar. |
| `panel_id` | string \| null | no | Label printed in the figure ("A", "1", "12b"). |
| `caption_panel_id` | string \| null | no | The panel label as derived from the caption text. |
| `printed_panel_id` | string \| null | no | **(pixel evidence)** The panel label as OCR-read from the panel image itself. Only stamped by the classical OCR path — the LLM-first path deliberately leaves it empty because a caption-derived id is NOT pixel evidence. |
| `canonical_panel_id` | string \| null | no | **(v1.3.0+ populated)** Unifying label: `printed_panel_id` (image evidence) wins, else `caption_panel_id`, else falls back to `panel_id`. |
| `species` | string \| null | no | Latin name from the caption parser / taxon recognizer. |
| `panel_path` | string \| null | no | Absolute path to the cropped panel PNG. |
| `bbox` | [int×4] \| null | no | `[x, y, w, h]` in PDF coordinates. |
| `confidence` | float | yes | Pipeline's confidence in the match, 0..1. |
| `label_text` | string \| null | no | OCR text of the panel label. |
| `caption_snippet` | string \| null | no | The figure caption (truncated to 240 chars in batch4_v2; full text in newer runs). |
| `ocr_text` | string \| null | no | OCR text inside the panel crop. |
| `metadata` | object | yes | Diagnostic metadata. See below. |
| `paper_metadata` | object \| null | no | Bibliographic metadata. See below. |
| `confidence_interval_low` | float \| null | no | **(v1.1.0+)** Wilson score 95% CI lower bound for `confidence`. |
| `confidence_interval_high` | float \| null | no | **(v1.1.0+)** Wilson score 95% CI upper bound for `confidence`. |
| `image_verified` | bool | no | **(v1.1.0+)** Human-review flag indicating the panel label/species was visually verified. Default `False`. |
| `review_priority` | int | no | **(v1.1.0+)** Auto-computed priority 0..2 (0 = low, 2 = urgent) based on confidence + unmatched-species signals. Used by the Web UI to triage human review. |

## PanelMetadata

| Field | Type | Notes |
| --- | --- | --- |
| `panel_score` | float \| null | Raw score from the panel segmenter. |
| `ocr_count` | int | Number of OCR tokens in the panel crop. |
| `taxon_count` | int | Number of taxon names extracted by TaxonNerD. |
| `figure_number` | string \| null | "1", "2", etc. |
| `page_index` | int \| null | 1-indexed page in the source PDF. |
| `matcher_used` | bool | True if the neural matcher was used. |
| `matcher_type` | string | "heuristic" \| "neural" \| "llm_first" — **(v1.3.0+ honest)** rows produced by the LLM-first path carry `extraction_method="llm_first"` and previously defaulted to a misleading `"heuristic"` label. |
| `matcher_conf` | float | 0..1 |
| `caption_pairs_used` | bool | True if caption-parser pairs drove the match. |
| `scale_bar` | object \| null | See ScaleBarRecord. |
| `geology_links` | array | List of GeologyLinkRecord. **(v1.3.0+)** per-entry `link_source` / `figure_id` provenance is populated by the converter. |
| `m3_diagnostic` | object | M3 stage output (when used). |
| `extraction_source` | string | "opendataloader" \| "grobid" \| "m3" |

## ScaleBarRecord

| Field | Type | Notes |
| --- | --- | --- |
| `value` | float \| null | The scale bar's reported length (e.g. 100). |
| `unit` | string \| null | "um", "mm", etc. |
| `source` | string \| null | "caption" \| "ocr" \| "image" |
| `pixel_length` | float \| null | Length in pixels (from image detection). |
| `um_per_px` | float \| null | Computed pixel-to-micrometer ratio. |
| `confidence` | float | 0..1 |

## GeologyLinkRecord

| Field | Type | Notes |
| --- | --- | --- |
| `age` | string \| null | Free-text age ("Late Jurassic"). |
| `chronostratigraphy` | string \| null | ICS age name. |
| `chronostratigraphy_rank` | string \| null | "age" \| "epoch" \| "period" |
| `formation` | string \| null | Lithostratigraphic formation. |
| `locality` | string \| null | Free-text locality. |
| `latitude` | float \| null | Decimal latitude. |
| `longitude` | float \| null | Decimal longitude. |
| `section_type` | string \| null | "systematic_paleontology" \| "geological_setting" \| ... |
| `section_title` | string \| null | Paper section heading. |
| `evidence_text` | string \| null | Quoted text supporting the link. |
| `confidence` | float | 0..1 |

## PaperMetadataRecord

| Field | Type | Notes |
| --- | --- | --- |
| `title` | string \| null | |
| `authors` | list of string | |
| `year` | int \| null | |
| `journal` | string \| null | |
| `volume` | string \| null | |
| `issue` | string \| null | |
| `pages` | string \| null | |
| `doi` | string \| null | |
| `abstract` | string \| null | |
| `keywords` | list of string | |
| `publisher` | string \| null | |
| `page_count` | int \| null | |
| `source` | string | "grobid" \| "opendataloader" \| "none" |
| `confidence` | float | 0..1 |

## MorphologyRecord (v1.2.0+)

Structured morphological description for one species. Emitted by
Stage 6 (M3 morphology extraction, opt-in via `--m3-stage-6`). For
each unique `(paper_id, species)` pair with an anchorable Description
or Diagnosis section, the pipeline emits ONE `MorphologyRecord`.
Fields the source text doesn't mention are left `null` — never `False`
/ `0` / `""` — so the JSONL export distinguishes "M3 said yes" from
"M3 had nothing to say".

| Field | Type | Notes |
| --- | --- | --- |
| `morphology_id` | string | Stable id for this record. |
| `paper_id` | string | Owning paper. |
| `species` | string | Target species. |
| `figure_id` | string \| null | Figure this species was illustrated on. |
| `page_index` | int \| null | Page where the Description/Diagnosis was located. |
| `cephalis_shape` | string \| null | Cephalis / dome shape description. |
| `thorax_shape` | string \| null | Thorax shape description. |
| `abdomen_shape` | string \| null | Abdomen shape description. |
| `apertural_structure` | string \| null | Aperture / opening description. |
| `num_segments` | int \| null | Number of segments reported. |
| `diagnostic_features` | list of string | Free-text bullets of distinctive features. |
| `evidence_text` | string \| null | Quoted source text supporting the record. |
| `confidence` | float | 0..1 |

## Validation rules

- `extra="forbid"`: unknown fields are rejected.
- `bbox`: if present, must be exactly 4 ints.
- `panels`: must be an array (possibly empty).
- `provenance.pipeline_version`: must be a non-empty semver string.
- `paper_id`, `figure_id`: non-empty strings.

## Versioning policy

- **Major version bump** (1.x → 2.x): removing or renaming a field,
  changing its type, or splitting a record into multiple.
- **Minor version bump** (1.0 → 1.1): adding a new optional field.
- **Patch version bump** (1.0.0 → 1.0.1): documentation-only changes.

Downstream consumers should pin to a major version and accept any
minor/patch upgrade.
