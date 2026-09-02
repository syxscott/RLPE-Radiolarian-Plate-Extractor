# Design: RLPE 达到放射虫论文科研级 F1

**Date**: 2026-09-02
**Status**: Draft (post-brainstorming, pending user review)
**Author**: Claude (RLPE audit 2026-09-01)

## 1. Goal & Success Criteria

### 1.1 Primary goal

将 RLPE 端到端 F1 (species-level, panel match) 从当前测量值推高到 v19 SOTA baseline = **0.84**, 并用**可重复**、**可发表**的实验方法证明。

### 1.2 Numerical success criteria

| 指标 | 当前 (测量) | 当前 (真实, holdout) | 目标 | 报告形式 |
|---|---|---|---|---|
| species_f1_micro | 0.77 (泄漏) | 估计 0.50-0.65 | **≥ 0.78** | train + test split |
| species_f1_macro | 0.72 | 估计 0.45-0.60 | ≥ 0.72 | train + test split |
| panel_match_rate | 0.99 | 估计 0.85-0.92 | ≥ 0.95 | train + test split |
| generalization_gap | 未知 (泄漏) | 估计 +15-25pp | **≤ 8pp** | train F1 − test F1 |

### 1.3 Secondary criteria (论文发表可声称)

- **可重复**: 单一命令行 (`make eval-research`) 复现全部 F1
- **可统计**: 5-fold cross-validation + bootstrap 95% CI
- **可扩展**: 新论文自动入黄金集（半自动标注协议）
- **诚实**: 报告 train + holdout F1，明确"在分布外"误差

## 2. Approach (selected: 方案 A — 领域驱动微调)

### 2.1 Why not 方案 B (LLM fine-tune)

B 方案需要 8-12 周 + GPU + 训练调优 + 数据集扩到 100+。 风险高、伦理复杂、可能难重复。 选 A 方案先打基础。

### 2.2 选 A 的理由

- **依赖云 API** + 现成 M3 (无需 GPU server)
- **代码量 < 1k 行** (eval harness 200 行 + caption 修复 300 行 + 后处理 200 行)
- **耗时 4-6 周** (可发表)
- **目标 F1 78-82%** (接近 v19 84%)
- **可重复**: 同一 API 同一 prompt = 同一输出
- **诚实**: 报告 F1 + CI, 不报告虚高数字

## 3. Architecture

### 3.1 Module layout

```
rlpe/
├── scripts/
│   ├── gold_eval_anchored.py       # 现有 — 加 5-fold + bootstrap
│   ├── caption_fixer.py             # NEW: 通用 caption selector
│   ├── prompts.py                   # NEW: M3 prompt library
│   └── post_process.py              # NEW: panel 归一化 + species 同物异名
├── data/
│   └── gold/                        # 扩到 20+ 论文, 多 plate/论文
├── docs/
│   └── superpowers/specs/
│       └── 2026-09-02-radiolarian-research-grade-design.md  # 本文档
└── Makefile                         # NEW: `make eval-research` 复现
```

### 3.2 Data flow

```
PDF → OpenDataLoader → per-plate JSON
                    ↓
              caption_fixer.py (通用规则, 无 gold 引用)
                    ↓
              prompts.py (按 paper type 选 prompt)
                    ↓
              MiniMax-M3 (云 API, 同 84% 模型)
                    ↓
              post_process.py (panel 归一化 + species cf./aff.)
                    ↓
              gold_eval_anchored.py (5-fold CV + bootstrap CI)
                    ↓
              F1 train + F1 test + 95% CI
```

### 3.3 Key design decisions

| 决策 | 选择 | 理由 |
|---|---|---|
| M3 模型 | MiniMax-M3 (已有) | 已验证可工作, 不需 fine-tune |
| Caption selector | 通用规则, 不引用 gold | 防过拟合 |
| 黄金集 split | 6 train / 3 test, 写死不变 | 多论文迭代时 train 不被污染 |
| 后处理阈值 | conf > 0.7 (基于 train 调) | 不接触 LLM 行为, 仅本地规则 |
| Bootstrap CI | 1000 重采样 | 标准做法 |
| v19 重测 | 同 prompt 跑同一 9 paper 算 baseline | 验证"科研级"对比 |

## 4. Components

### 4.1 `scripts/caption_fixer.py` — 通用 caption 选取

**功能**: 找 PDF 里的真 plate caption 块 (不引用 gold)
**算法**:
1. 全文 split 为 paragraphs
2. 对每 paragraph 算分:
   - +10 starts with `(Plate|Fig|表|図版) \s*0?\d+\b`
   - +5 contains ≥ 2 个 binomial pattern (`[A-Z][a-z]{3,}\s+[a-z]{3,}`)
   - +3 contains `Sample` / `Loc\.` / `Marker =` (典型 plate 结尾标志)
   - −100 length < 50 chars
3. 选最高分 paragraph (有 anchor 必须 + length > 200)
4. 失败 fallback: 整 page 文字

**为什么不用 gold species overlap**:
- 那会过拟合到 test paper
- 通用规则依靠 plate 锚定 + binomial 密度, 在新 paper 也工作

**预期影响**: 多 plate 论文 (bandini, bragin) 召回率 +20pp, 测试泛化 F1 -3pp 以内

### 4.2 `scripts/prompts.py` — M3 prompt library

**4 个 prompt templates** (按 paper type 选):
- `RANGE_CHART_PROMPT` (e.g. bragin "Fig. 1. Distribution of radiolarians...")
- `SEM_PLATE_PROMPT` (e.g. bandini "Plate 1. Scanning electron microscope...")
- `MAP_PROMPT` (e.g. pouille "Fig. 1. Schematic map indicating location...")
- `GENERIC_PROMPT` (其他)

**selection rule**: 
- 找 caption 包含 "distribution" / "range" → RANGE_CHART
- 找 caption 包含 "scanning electron" / "plate" → SEM_PLATE
- 找 caption 包含 "map" / "location" → MAP
- 其他 → GENERIC

**关键**: prompt **不引用 gold species**, 描述一般规则 ("输出 array of {label, species, confidence} objects")
**温度**: 0.05 (低, 多次重测可重现)
**max_tokens**: 4096 (够长)

**预期影响**: 
- 误判降 (RANGE/SEM/MAP 分类错误) → F1 +5pp
- M3 over-extraction 降 (更明确的 prompt) → F1 +3pp

### 4.3 `scripts/post_process.py` — 后处理

**模块 4 个函数**:
- `normalize_panel_id(label)` — 已有 (在 evaluation/gold.py)
- `parse_open_nomenclature(species)` — cf./aff./n.sp. 拆分 (3-行)
- `dedup_panels(panels)` — 同一 species + 同一 figure 内合并 (避免重复)
- `filter_low_confidence(panels, threshold=0.7)` — 过滤 conf < 0.7 (基于 train 调)

**预期影响**:
- dedup: recall +5pp (去掉 M3 重复检测)
- conf filter: precision +5pp (去掉低 conf 假阳性)
- n. sp. 解析: recall +3pp

### 4.4 `scripts/gold_eval_anchored.py` — Eval harness 升级

**新增**:
- `--split train|test` 参数
- `--bootstrap-samples N` (默认 1000)
- `--folds N` (默认 5)
- 输出格式: `{"train_f1": 0.78, "test_f1": 0.75, "ci_95": [0.72, 0.78], "gap": 0.03}`

**Split 配置** (写死):
```python
SPLIT = {
    "train": ["bandini2011", "beccaro2006", "boughdiri2007", "bragin2025",
              "danelian2006", "hollis2006"],
    "test":  ["baumgartner2008", "feng2007", "pouille2014"],
}
```

**为什么 baumgartner2008 在 test**: 已知 F1 1.0, 是"理想情况"test 论文. 如果泛化 OK, test F1 仍高 (1.0); 如果过拟合, test F1 跌.

## 5. Testing & Validation

### 5.1 Test layers

| Layer | Coverage target | Time |
|---|---|---|
| `caption_fixer.py` unit test | 95% | 30 min |
| `prompts.py` snapshot test | 90% (prompt string freeze) | 15 min |
| `post_process.py` unit | 95% | 20 min |
| `gold_eval_anchored.py` integration | 90% (smoke + golden set) | 1 hr |
| End-to-end on 9 papers | 100% (run completes) | 10 min |

### 5.2 Acceptance criteria

- `make eval-research` 在 30 min 内跑完 9 papers + 5-fold CV
- 输出 JSON 含 train F1, test F1, gap, CI, per-paper breakdown
- gap ≤ 8pp (泛化 OK)
- test F1 ≥ 0.65 (诚实评估)

### 5.3 Regression guard

- Source-guard tests (已有 49 个) 继续 100% pass
- 黄金集 split 写死 — 不会被脚本意外改
- 报告 `data/snapshot/{date}/f1.json` — 可对比历史 F1

## 6. Phased timeline (4-6 周)

| Week | Deliverable | Acceptance |
|---|---|---|
| 1 | 黄金集扩到 20 篇 (本仓库) | train=6, test=3, holdout=11 (新加) |
| 1 | caption_fixer.py 通用版本 + 单测 | 不接触 gold 测泛化, train F1 ≥ 0.65 |
| 2 | prompts.py 4 templates + M3 call | 5-fold CV, 9-paper train F1 ≥ 0.70 |
| 3 | post_process.py 4 functions | 9-paper combined F1 ≥ 0.75 |
| 4 | 集成 eval harness + bootstrap CI | `make eval-research` 30 min 内, 输出 JSON |
| 5 | v19 baseline 重测 + 论文方法学写 | train F1 ≥ 0.78, test F1 ≥ 0.70, gap ≤ 8pp |
| 6 | holdout 11 篇 F1 + 论文 draft | 整体 test F1 ≥ 0.68 |

## 7. Risks & mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| v19 84% 是"全 pipeline" 测的, 我们 LLM-first direct F1 不能直接对比 | 高 | 中 | 写 spec 时明确"我们测的是 LLM-first 提取质量, 端到端 pipeline 留 v19 对比" |
| MiniMax API 限流 (Token Plan) | 中 | 中 | 6 weeks 估算 20 paper × 5 plate × 1 API call × 4 轮 = 400 calls ≈ ¥2 |
| M3 漏抽 (panel_count < gold_count) | 高 | 高 | 加 "extract every panel" prompt 强化 + post_process dedup |
| M3 over-extract (panel_count > gold_count × 2) | 中 | 中 | conf filter (threshold 0.7 from train) |
| 黄金集人工标注慢 | 中 | 中 | 利用 OCR output + PDF 文本半自动, 1 hour / paper |
| 跨物种 cf./aff. 归一化错 | 中 | 中 | 利用 _norm_species 已有规则, 不重做 |

## 8. Open questions (待 user 决策)

1. **黄金集扩到 20 篇**: 哪些论文 (放射虫论文_OA_download/ 184 篇)? user 选 6 train + 3 test + 11 holdout
2. **v19 baseline 重测**: 同一 prompt 跑 v19 9 paper 算 baseline? 还是直接信原数据?
3. **Eval 间隔**: 每改进一个组件跑一次 eval, 还是每周跑一次?

## 9. Self-review checklist

- [x] No "TBD" or "TODO" placeholders
- [x] Internal consistency: data flow matches components
- [x] Scope: focused on F1 improvement + measurement, not architecture refactor
- [x] Ambiguity: success criteria are specific (≥ 0.78 train, gap ≤ 8pp)

---

**Status**: Draft ready for user review.

**Next steps after user approval**:
1. Invoke `writing-plans` skill to create implementation plan
2. Phase 1: 黄金集扩到 20 篇 (week 1)
3. Phase 2: caption_fixer.py (week 1-2)
4. Phase 3: prompts.py (week 2)
5. Phase 4: post_process.py (week 3)
6. Phase 5: eval harness + bootstrap CI (week 4)
7. Phase 6: v19 baseline 重测 + 论文 draft (week 5-6)
