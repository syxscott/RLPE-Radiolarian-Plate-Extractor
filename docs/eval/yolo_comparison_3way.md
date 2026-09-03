# 3-Way YOLO Comparison on 5 v19 Papers

**Date:** 2026-09-03  
**Setup:** RTX 4090, ultralytics 8.4.106, torch 2.12.0+cu130, 150 DPI PDF render, DEFAULT model config.  
**Goal:** find best of-the-shelf detector for radiolarian figure panels.  
**Note:** Gold rows are panel observations (no bbox), so precision/recall are *panel-count proxies*, not IoU-based.

## Models

| Label | Weights | Size |
|---|---|---|
| A_yolo11x | `yolo11x.pt` | 109.1 MB |
| B_yolov8n | `yolov8n.pt` | 6.2 MB |
| C_radio_yolo_v1 | `radiolarian_yolo_v1.pt` | 6.0 MB |

## Per-paper, per-model raw results

| Paper | Model | Detections | Avg Conf | Time (s) | Gold Rows | P (proxy) | R (proxy) | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| bandini2011 | A_yolo11x | 270 | 0.929 | 1.04 | 273 | 1.00 | 0.99 | in range (0.99x of gold) |
| bandini2011 | B_yolov8n | 58 | 0.431 | 0.83 | 273 | 1.00 | 0.21 | under-detection (0.21x of gold) |
| bandini2011 | C_radio_yolo_v1 | 126 | 0.653 | 0.81 | 273 | 1.00 | 0.46 | under-detection (0.46x of gold) |
| baumgartner2008 | A_yolo11x | 99 | 0.934 | 0.51 | 61 | 0.62 | 1.00 | in range (1.62x of gold) |
| baumgartner2008 | B_yolov8n | 3 | 0.449 | 0.44 | 61 | 1.00 | 0.05 | under-detection (0.05x of gold) |
| baumgartner2008 | C_radio_yolo_v1 | 85 | 0.681 | 0.41 | 61 | 0.72 | 1.00 | in range (1.39x of gold) |
| beccaro2006 | A_yolo11x | 36 | 0.916 | 0.31 | 35 | 0.97 | 1.00 | in range (1.03x of gold) |
| beccaro2006 | B_yolov8n | 7 | 0.365 | 0.27 | 35 | 1.00 | 0.20 | under-detection (0.20x of gold) |
| beccaro2006 | C_radio_yolo_v1 | 37 | 0.715 | 0.25 | 35 | 0.95 | 1.00 | in range (1.06x of gold) |
| danelian2006 | A_yolo11x | 42 | 0.937 | 0.32 | 42 | 1.00 | 1.00 | in range (1.00x of gold) |
| danelian2006 | B_yolov8n | 4 | 0.338 | 0.24 | 42 | 1.00 | 0.10 | under-detection (0.10x of gold) |
| danelian2006 | C_radio_yolo_v1 | 36 | 0.675 | 0.24 | 42 | 1.00 | 0.86 | in range (0.86x of gold) |
| hollis2006 | A_yolo11x | 47 | 0.906 | 0.51 | 73 | 1.00 | 0.64 | in range (0.64x of gold) |
| hollis2006 | B_yolov8n | 9 | 0.510 | 0.42 | 73 | 1.00 | 0.12 | under-detection (0.12x of gold) |
| hollis2006 | C_radio_yolo_v1 | 62 | 0.681 | 0.40 | 73 | 1.00 | 0.85 | in range (0.85x of gold) |

## Per-paper A vs B vs C (det / time)

| Paper | A:yolo11x det/time | B:yolov8n det/time | C:radio_yolo_v1 det/time | gold rows |
|---|---|---|---|---:|
| bandini2011 | 270/1.0s | 58/0.8s | 126/0.8s | 273 |
| baumgartner2008 | 99/0.5s | 3/0.4s | 85/0.4s | 61 |
| beccaro2006 | 36/0.3s | 7/0.3s | 37/0.2s | 35 |
| danelian2006 | 42/0.3s | 4/0.2s | 36/0.2s | 42 |
| hollis2006 | 47/0.5s | 9/0.4s | 62/0.4s | 73 |

## Aggregate (across 5 papers)

| Model | Total dets | Avg conf | Total time | Avg P (proxy) | Avg R (proxy) |
|---|---:|---:|---:|---:|---:|
| A_yolo11x | 494 | 0.924 | 2.68 | 0.92 | 0.93 |
| B_yolov8n | 81 | 0.419 | 2.20 | 1.00 | 0.14 |
| C_radio_yolo_v1 | 346 | 0.681 | 2.12 | 0.93 | 0.83 |

## Interpretation

- **recall_proxy** = min(detections / gold_rows, 1.0) — proxies how well the model finds expected panels.
- **precision_proxy** = min(gold_rows / detections, 1.0) — penalizes over-detection; perfect if detections <= gold_rows.
- Gold rows = panel observations in `data/gold/<slug>.jsonl`; the user-curated ground truth has no bbox, so this is an approximate comparison only.
- **time** is wall-clock for one YOLO forward pass over ALL rendered pages (single batched call), excluding PDF→PNG rendering.
- DEFAULT config: no threshold/IoU tuning per model; whatever Ultralytics ships as `model(paths, ...)` defaults.
