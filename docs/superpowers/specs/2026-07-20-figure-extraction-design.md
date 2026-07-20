# 精确物种-图像关联 + 示意图提取 设计文档

> **For agentic workers:** 本文档是设计阶段成果。Phase 8 之后才会写实施 plan。

**目标**: 把"图说文本 → 物种-地层-时代关联"做到 95% 准确率，覆盖示意图片。

**Architecture**: 3 个独立 phase，每阶段独立可测、可发布。

**Tech Stack**: Python 3.11, PySide6, FastAPI, Pydantic v2, pytest, MiniMax M3 (多模态), Anthropic Claude

---

## Context

2026-07-20 用户问"如何精确关联到具体物种和图像 + 提取示意图"。审计发现：
- 当前 6 种 figure type（plate/range_chart/strat_column/litholog_column/paleogeographic_map/map）能独立提取
- **关联靠 paper 级共享假设**——plate 上的物种是否产在 strat column 那个地层，**程序是猜的**
- **示意图（schematic/diagram/reconstruction）当前直接被当 plate 处理**或丢弃
- 用户要求：高准确率（95%）+ 公开发布 + 不限成本

**3 个阶段**（按用户选择优先级 B → A → C）：

---

## Phase B: 示意图提取（**最优先**，2-3 天）

### 范围
- 新增 figure type `schematic` / `diagram` / `reconstruction`
- M3 专门 prompt 提取所有文字 + 概念关系
- 输出存到新字段 `figure_schematic_data`，不进 species 字段

### 工作流
```
classifier 识别为 schematic → m3_engine.extract_schematic(image, caption)
  → 返回 {"text_elements": [...], "relationships": [...], "context": {...}}
  → 写到 panel row 的 metadata.figure_schematic_data
```

### M3 Prompt 输出 Schema
```json
{
  "figure_type": "schematic | diagram | reconstruction | phylogenetic",
  "text_elements": [
    {"text": "Late Triassic", "type": "age", "confidence": 0.98},
    {"text": "Tethys Ocean", "type": "geographic", "confidence": 0.95}
  ],
  "relationships": [
    {"from": "box1", "to": "box2", "label": "evolved into"}
  ],
  "extracted_facts": {
    "ages_mentioned": ["Late Triassic", "Carnian"],
    "geographic_names": ["Tethys", "Panthalassa"],
    "taxa_mentioned": ["Genus species"]
  },
  "confidence": 0.0-1.0
}
```

### 关键文件
- `src/rlpe/m3_engine.py` — 新增 `PROMPT_REGISTRY["schematic_geo"]` + `extract_schematic()`
- `src/rlpe/range_chart_extractor.py` — `classify_figure_type()` 加 schematic / diagram / reconstruction / phylogenetic 关键词
- `src/rlpe/pipeline.py` — 路由到 `extract_schematic()`
- `src/rlpe/schema_models.py` — `PanelMetadata.figure_schematic_data` 字段
- `tests/test_phase64_schematic.py` — 12+ 单元测试

### 验收标准
- 单元测试 100% 通过
- 5 篇真实论文（Boughdiri/Danelian/Pouille/Beccaro/Baumgartner）跑通，提取出非空 `figure_schematic_data`
- 字段正确出现在导出的 JSONL/xlsx/DwC-A 里

---

## Phase A: 精确物种-图像关联（4-5 天）

### 范围
- 利用 plate caption 里的 Sample ID / Loc 编号直接匹配 strat column / map 中的编号
- 让 M3 跨图推理（plate + strat + map 一起送）作为兜底

### 3 种关联策略（按置信度排序）

**策略 1: Sample ID 直接匹配（最高置信度）**
- Regex 提取 plate caption 里的 `Sample \w+`, `Loc. \w+`, `ID-\d+`
- 与 strat column caption 中的 Sample / Loc 编号匹配
- 命中 → 物种直接关联到该地层时代
- 置信度: 1.0（直接文本匹配）

**策略 2: Locality 共享关联（中置信度）**
- 同 paper 同一 Locality 字符串（"Tunisia", "Greece", "Sicily"）
- plate 物种 + 该 paper 的 strat column / map 提到同一 locality → 关联
- 置信度: 0.7

**策略 3: M3 跨图推理（兜底，低置信度）**
- 对未关联的 plate，把 paper 的所有 figure 摘要 + plate caption 一起发给 M3
- 问 M3："根据这些信息，plate 上的物种最可能产在哪个地层/时代？"
- 置信度: 0.3-0.6（M3 给出的）

### 工作流
```
for each panel:
  1. 提取 sample_id, locality, age from panel metadata
  2. 查找 paper-level shared geo_links (strat / map / range)
  3. 按策略 1 匹配 Sample ID → 置信度 1.0
  4. 未命中 → 按策略 2 匹配 Locality → 置信度 0.7
  5. 未命中 → 按策略 3 调 M3 → 置信度 0.3-0.6
  6. 把关联结果写进 panel.metadata.geology_links，标记 source (sample_match / locality_match / m3_inference)
```

### M3 推理 Prompt
```
你是一个放射虫古生物学专家。这是一篇论文的 figure 信息:
- Figure 3 (strat column): 形成于 Late Cretaceous, Italy, Sample S1
- Figure 5 (plate): 物种 A, B, C, all from Sample S1

请回答: 图 5 的物种 A 最可能产在哪个地层和时代?
按 JSON 输出: {"species": "A", "age": "Late Cretaceous", "formation": "Scaglia", "confidence": 0.X}
```

### 关键文件
- `src/rlpe/sample_id_extractor.py` — 新建：regex 提取 Sample ID / Loc
- `src/rlpe/cross_figure_linker.py` — 新建：3 策略关联
- `src/rlpe/m3_engine.py` — 新增 `cross_figure_inference()`
- `src/rlpe/pipeline.py` — 在所有 figure 提取完后调用 linker
- `tests/test_phase65_sample_linker.py` — 20+ 单元测试

### 验收标准
- 20+ 真实 paper 跑通
- 关联率: Sample ID 命中 30% / Locality 命中 40% / M3 推理命中 20% / 完全没关联 10%
- 总关联率 ≥ 90%
- 导出 JSONL 中每行 panel 至少有一个 `geology_links` 项（即使 conf=0）

---

## Phase C: 多模态视觉坐标精确定位（5-7 天，可选）

### 范围
- 让 M3 输出"plate 第 X 格 → strat column 第 Y 层"的视觉坐标关系
- 输出 95%+ 准确率

### 工作流
```
1. 把 plate + strat column 两张图打包发 M3
2. M3 输出: {"plate_cells": [{"cell": "1", "links_to_strat_layer": 3}], ...}
3. 程序把这个映射存到 cross_figure_visual_links 字段
4. 据此给每个 panel 物种赋予最具体的地层时代
```

### 关键文件
- `src/rlpe/m3_engine.py` — 新增 `cross_figure_visual_inference()`
- `src/rlpe/schema_models.py` — 新增 `cross_figure_visual_links` 字段
- `tests/test_phase66_visual_linker.py` — 10+ 单元测试

### 验收标准
- 20+ paper 验证，每个关联的 precision ≥ 95%
- 仅在 Phase A 关联率 < 90% 时才调 Phase C（cost 节约）

---

## 跨阶段依赖

```
B (示意图)  ────┐
A (关联策略)  ──┼─→  C (视觉坐标)
公共依赖: M3 vision API
```

A 必须在 C 前完成（C 是 A 的升级版）。

---

## 测试矩阵

| 类型 | 数量 | 范围 |
|---|---|---|
| 单元测试 | 50+ | regex / 分类 / prompt schema |
| 集成测试 | 20+ | 端到端 pipeline |
| 真实 paper E2E | 20-50 | Boughdiri/Danelian/Pouille/Beccaro/Baumgartner/Takahashi/Uchino 等 |
| DwC 导出验证 | 5 | schema 兼容、字段完整 |
| 性能测试 | 3 | 100 paper 批处理 < 24h |

---

## 风险与回退

1. **M3 prompt 漂移** — 不同时间 M3 输出格式可能变化。**缓解**: `parse_json` 容错 + schema 校验
2. **跨图推理成本失控** — 100 paper × 9 plate = 900 次 M3 调用。**缓解**: Phase C 仅在 A 关联率不足时启用
3. **示意图 prompt 误判** — schematic 容易被误判为 diagram。**缓解**: 6+ 关键词 + human review sample 20 篇
4. **关联到错误的物种** — Sample ID 冲突（不同 figure 用同一 ID）。**缓解**: 加 paper_id + figure_id 复合键

---

## 后续（Phase 8+）

- 把关联结果导出到 PBDB-compatible JSON
- 写论文："Multi-modal extraction of precise species-stratigraphy links from radiolarian literature"
- 在 GBIF 注册 RLPE 为 reference implementation

---

## 用户确认问题清单

✅ 目标 = 两个都要
✅ 成本 = 不限
✅ 准确率 = 95%
✅ 验证标准 = 20-50 paper 统计
✅ 用户类型 = 公开发布
✅ 优先级 = B → A → C
✅ M3 调用时机 = 每次 plate 都调一次
