# Phase B: Schematic Figure Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal**: 让系统能识别并提取示意图（schematic/diagram/reconstruction/phylogenetic）里的所有文字和概念关系，准确率 ≥ 95%。

**Architecture**: 
- 新增 4 个 figure type（schematic / diagram / reconstruction / phylogenetic）
- M3 vision prompt 提取 text elements + relationships + extracted facts
- 输出存到 `PanelMetadata.figure_schematic_data` 新字段

**Tech Stack**: Python 3.11, Pydantic v2, pytest, MiniMax M3

---

## Context

设计文档：[docs/superpowers/specs/2026-07-20-figure-extraction-design.md](../specs/2026-07-20-figure-extraction-design.md)

当前问题：
- 论文里的示意图现在被分类成 "plate" 然后错误处理
- 用户要求能提取示意图的所有文字 + 概念关系，95% 准确率

---

## Task Structure

### Task 1: 扩展 figure type 分类器
- Modify: `src/rlpe/range_chart_extractor.py`
- Test: `tests/test_phase64_schematic_classifier.py`

### Task 2: 新增 `PanelMetadata.figure_schematic_data` 字段
- Modify: `src/rlpe/schema_models.py`
- Modify: `schemas/rlpe-v1.0.0.json` (regenerate)
- Test: `tests/test_phase64_schematic_schema.py`

### Task 3: M3 schematic prompt + extract_schematic 方法
- Modify: `src/rlpe/m3_engine.py`
- Test: `tests/test_phase64_schematic_extract.py`

### Task 4: Pipeline 路由
- Modify: `src/rlpe/pipeline.py`
- Test: `tests/test_phase64_pipeline_schematic.py`

### Task 5: 导出到 JSONL/xlsx/DwC-A
- Modify: `src/rlpe/converters.py`, `src/rlpe/exporters/xlsx.py`, `src/rlpe/exporters/archive.py`
- Test: `tests/test_phase64_schematic_export.py`

### Task 6: 真实 paper E2E
- Create: `scripts/smoke_schematic.py`
- Test: `tests/test_phase64_schematic_smoke.py`

### Task 7: GUI Results tab 显示
- Modify: `src/rlpe/gui/results_tab.py`

### Task 8: 回归 + final commit

---

## Constraints

- **TDD strictly** — 每个 task: 写测试 → RED → 实现 → GREEN → commit
- **No live LLM calls** — mock M3 backend
- **No new deps**
- **Files touched**: m3_engine / range_chart_extractor / pipeline / schema_models / schema_dump / schemas / converters / exporters / gui/results_tab
- **Commit format**: `feat(Phase 64 Plan B.X): <description>` per task + final `phase 64 (Plan B): schematic figure extraction complete`
- **Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`**

---

## Verification

- `python -m pytest tests/ -q` → 1523+ tests pass
- `python -m rlpe.schema_dump --out schemas/rlpe-v1.0.0.json` → schema sync
- 5 篇真实 paper smoke test
