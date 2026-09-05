# RLPE — Radiolarian Literature Plate Extractor

放射虫文献图版提取流水线：从已发表文献 PDF 中提取图版（plate）、切分
panel、识别拉丁学名，并建立 panel–label–species 对应关系；同时抽取地质
年代（ICS 数值年龄）、产地坐标、比例尺、分布表与形态学描述，导出为
JSON / CSV / Excel / Darwin Core Archive。

> **状态声明（诚实优先）**：本项目的工程框架与溯源体系达到科研辅助工具
> 水准，但**物种–panel 对应的准确率尚未经过人工核验基准的正式标定**
> （见[评估真相](#评估真相)一节）。当前定位是**需要专家审校的高效预标注
> 器**，所有产出数据请配合 GUI/浏览器的审校流程使用。

---

## 目录

- [真实能力一览](#真实能力一览)
- [快速开始](#快速开始)
- [提取哪些数据](#提取哪些数据)
- [架构总览](#架构总览)
- [常用 CLI 参数](#常用-cli-参数)
- [Web 界面与 API](#web-界面与-api)
- [评估真相](#评估真相)
- [已知限制](#已知限制)
- [数据标注指南](#数据标注指南)
- [开发与测试](#开发与测试)

---

## 真实能力一览

以下数字全部来自本仓库内可复现的测试（非宣传值）：

| 维度 | 实测结果 | 来源 |
|---|---|---|
| 物种精确度 | **98%**（51 个提取物种中 50 个可在原文核验） | 2026-09-06 25 篇随机覆盖测试 |
| 论文级产出率 | **36%**（9/25）；在"确有文本层图注的图版论文"子集中 **~56%**（9/16） | 同上 |
| panel 召回 | 文本层良好的论文 ~81%；全页扫描版图版 **0**（不支持） | 3 篇深度对照 + 25 篇测试 |
| 地质年代 | ICS 2023 数值区间映射（如 Early Jurassic → 174.7–201.4 Ma） | `stratigraphy.py` 内嵌年代表 |
| 比例尺 | 三源合并（caption/OCR/视觉线段）+ 2x–10x 分歧检测 | v18 产物 641/913 行有值 |
| 成本 | ~¥0.16/篇（MiniMax M3，仅对可提取内容计费） | 25 篇实测 688 调用 ¥4.09 |
| 物种幻觉率 | ~2%（1/51） | 同上 |

**明确不支持/已知短板**（详见[已知限制](#已知限制)）：

- 全页扫描版图版（日刊老文献常见，如 Motoyama 1998）——OD 提取不到
  文本层图注，当前整篇为 0；
- 分子系统/综述类论文的图（进化树、地图）会被正确地不产出 panel；
- M3 Stage 2 偶发把真图版误判为 diagram（Munasri 案例待修）。

---

## 快速开始

### 安装

```bash
pip install -e .
# 可选依赖（按需）：
pip install 'anthropic>=0.40,<0.50' python-dotenv   # MiniMax M3 云端
pip install paddleocr==2.7.3 paddlepaddle==2.6.2    # OCR（或 easyocr）
pip install opendataloader-pdf                       # OD 前端（需 Java 11+）
```

### 方式一：Web 界面（推荐非技术用户）

```bash
python run_web_server.py        # 访问 http://localhost:8000
```

拖拽上传 PDF → 配置参数 → 实时进度 → 结果审查 → Excel 导出。
LLM API 失败时会弹出四选一回退菜单（gemma4 / rules / stop / retry）。

### 方式二：CLI

```bash
# 推荐：OpenDataLoader 前端 + MiniMax M3（云端多模态）
python -m rlpe \
  --pdf-dir data/pdfs --work-dir work/run \
  --use-opendataloader \
  --use-gemma4 --llm-backend MiniMax \
  --MiniMax-endpoint https://api.minimaxi.com/anthropic --MiniMax-model MiniMax-M3 \
  --i-understand-data-leaves-my-machine --data-outbound-policy api_full \
  --m3-enhanced-mode --use-geo-vision \
  --export-jsonl work/run/matches.jsonl
```

关键行为说明：

- `--use-gemma4 --llm-backend MiniMax` + `.env` 中的
  `ANTHROPIC_API_KEY`（MiniMax 兼容 Anthropic 协议）即启用云端 LLM；
  无 key 时自动退化为正则抽取（质量大幅下降，不再有 LLM 调用）。
- `--m3-enhanced-mode` 启用 M3 五阶段引擎；`--use-geo-vision`
  /`--m3-stage-6` 会隐式启用引擎，无需再传。
- `--data-outbound-policy` 默认 `api_redacted`（缩略图+脱敏 caption）；
  `api_full` 需要环境变量 `RLPE_DATA_OUTBOUND_OPT_IN=1` 显式同意。
- 断点续跑：每篇论文写 `_checkpoints/<paper_id>.done`；零行论文会
  发出 `_ingestion_zero_rows` 警告 stub（进入 `run_output.warnings`），
  删除对应 checkpoint 即可强制重试。

### 方式三：桌面 GUI

```bash
python main.py        # 或 rlpe-gui
```

PySide6 界面：运行/队列/结果三视图，结果页支持图片预览 + bbox 叠加 +
"Mark verified" 人工核验回流（经同一 API 写入 corrections.jsonl）。

### 方式四：Docker

```bash
docker build -t rlpe:dev .
docker run --rm -p 8000:8000 rlpe:dev
```

---

## 提取哪些数据

一次运行的规范产出（`run_output.json`，schema v1.3.0，
`schemas/rlpe-v1.3.0.json`）：

| 数据类 | 内容 | 可靠性 |
|---|---|---|
| **Panels** | panel 裁剪 PNG + bbox + label + 物种 + Wilson 95% CI + `image_verified` 人工核验位 + 四元 panel-id 溯源（caption/printed/index/canonical） | 核心，精确度高、召回随论文格式波动 |
| **Taxa** | verbatim/规范名、属+种拆分、cf./aff. 限定词、命名权威、科/目/纲（PaleoDB，opt-in）、ICZN | 高 |
| **Geology contexts** | age + ICS 数值区间（ma_top/base/mid）、Formation/Member/Group、岩性、biozone、古环境/氧化还原/化学地层/沉积相 | 规则+词表，可用 |
| **Localities** | 地名、国家、现代经纬度（来源三标注：正则/国家质心/PaleoDB） | 中（国家质心是不确定度 25km 的回退） |
| **Paleo coordinates** | Euler 极旋转古纬度（内置近似表或外接 GPlates 文件） | 中低（模型粗近似，诚实标注 `embedded-approximate`） |
| **Scale bars** | 数值+单位+um_per_px，三源合并 + 分歧警告 | 高 |
| **Samples** | 样品号（caption 正则，含 `B_DP2` 类带下划线码） | 高 |
| **Range charts** | 逐物种延限（FAD/LAD、Ma 轴）+ biozone（需 API key） | 中（有真实论文验证） |
| **Morphology**（opt-in `--m3-stage-6`） | 壳形/尺寸/孔/刺/口围结构化描述 | 中（null ≠ false 契约） |
| **Knowledge graphs** | 物种–样品–地质–产地关系图 | 规则构建 |
| **Provenance** | pipeline 版本、git commit、config 快照、输入 SHA256、UTC 时间戳 | 每次 run 强制 |

导出通道：JSONL（行级 `matches.jsonl`）、JSON（`run_output.json`）、
CSV、Excel 5-sheet、Darwin Core Archive（GBIF/PBDB 可载入）。

---

## 架构总览

```
PDF ──┬─ OpenDataLoader（默认推荐，进程内，含文本层图注配对）
      │    └─ 空图注 OCR 兜底 + 孤儿图版页 OCR 抢救（乱码/扫描容错）
      └─ GROBID（需服务，TEI 路径）
            （两者互为回退，深度守卫防循环，cap=4）
  ↓
逐 figure 处理（_process_region，三引擎并行）：
  1. LLM-first（默认）：一次 M3 视觉调用直出全部 panel + 物种
  2. M3 五阶段：caption 结构化 → 图版分类 → 视觉 bbox → 逐panel匹配 → 自批判
  3. 经典 CV：SAM2（若可用）/ OpenCV 分割 → OCR → 规则匹配
  ↓
富化链：cross-figure 链接 → 分布表地质回链 → 地图桥接 → geo vision
        → Stage3 bbox crops → Stage4.5 逐panel物种 → 多图版富集
        → Stage6 形态学 → 跨图链接器（sample/locality/M3 四策略）
  ↓
_finalize_rows：去重 → stub/非法行过滤 → canonical/sample 盖章
        → paleo 富化 → 人工修正回放
  ↓
matches.jsonl + run_output.json（schema v1.3.0）→ 导出器族
```

---

## 常用 CLI 参数

以下旗标经过真实性审计（2026-09-06，逐个追踪到行为）：

| 类别 | 旗标 | 说明 |
|---|---|---|
| 前端 | `--use-opendataloader` | OD 进程内前端（推荐）；不开则走 GROBID |
| LLM | `--use-gemma4 --llm-backend MiniMax\|llamacpp\|ollama\|transformers` | 后端选择；`--MiniMax-*` 族配置云端 |
| LLM | `--MiniMax-max-concurrent / --MiniMax-max-retries / --MiniMax-thinking-budget` | 并发/重试/思考预算 |
| M3 | `--m3-enhanced-mode`；`--use-m3-stage3`；`--m3-per-panel`；`--m3-multi-plate-enrich`；`--m3-stage-6`；`--m3-disable-stage N` | 五阶段与扩展阶段开关（前三者隐式启用引擎） |
| 地质 | `--use-geo-vision`（隐式启用引擎）；`--use-geology-llm`；`--use-paleodb --paleodb-offline` | 地质视觉/关系抽取/PaleoDB 分类补全 |
| OCR | `--ocr-backend paddleocr\|easyocr --ocr-lang` | 主 OCR；OD 路径另有 config-only 键 `od_use_ocr` |
| 可复现 | `--deterministic --deterministic-seed N` | temperature=0 + RNG 播种（2026-09-06 接线修复） |
| 性能 | `--num-workers`（1-32）`--render-dpi` | 并发与渲染精度 |
| 导出 | `--export-csv / --export-json / --export-jsonl` | 三通道独立触发 |
| 数据合规 | `--data-outbound-policy api_redacted\|api_full\|local_only` + `--i-understand-data-leaves-my-machine` | 出站策略（默认 api_redacted） |
| 视觉 | `--use-yolo-figures [--yolo-model-path]`（默认放射虫微调模型）；`--sam2-checkpoint` | YOLO 图版检测（GROBID 路径）/SAM2 分割（缺文件静默回退 OpenCV） |

<details>
<summary>已修复的虚假旗标（历史记录）</summary>

- `--deterministic`：实现存在但零调用——2026-09-06 接线修复；
- 裸 `--use-yolo-figures`：空路径覆盖默认值导致必炸 ValueError——已回退默认权重；
- `-q/--verbose`：只调了 `rlpe.cli` logger（无任何输出）——已改为包级 logger。

</details>

---

## Web 界面与 API

FastAPI 服务（`run_web_server.py`）端点清单见
`docs/openapi-1.1.0.json`（重新生成：
`PYTHONPATH=src python scripts/gen_openapi.py`）。核心端点：

- `POST /jobs/upload` → `GET /jobs/{id}/status`（轮询）→ `GET /jobs/{id}/result`
- `GET /jobs/{id}/stream`（SSE）/ `WS /ws/jobs/{id}`（服务端健全，前端暂用轮询）
- `GET /jobs/{id}/export.xlsx`（5-sheet，panels/geology 过滤真实生效）
- `GET/POST /jobs/{id}/MiniMax-fallback`（API 失败人工决策弹窗）
- `POST /review/correction`（人工核验回流：翻转 `image_verified` 位 +
  corrections.jsonl 持久化，GUI 与 Web 共用）
- `GET /system/llm-status`、`POST /system/test-llm`

安全：可选 `X-API-Key`、CORS 默认仅 loopback、非 loopback 绑定强制
设 key、上传 100MB 上限、路径穿越防护。

---

## 评估真相

**本节是本项目最重要的诚实声明。**

1. **string-match F1 ≠ 真实准确率**。当前 gold 集（9 篇 612 panels）由
   caption parser 自产，衡量的是解析器自洽性。2026-09-02 重测证明：此前
   声称 F1=0.84 的 checkpoint，无 gold 辅助的端到端 image-verified F1 仅
   **0.075**（train 0.097 / test 0.0，见 `docs/baselines/v19_re-measured.md`）。
2. **本轮修复后的定位数字**来自 25 篇随机覆盖测试（seed=88，
   `work/coverage25/COVERAGE_REPORT.md`）：论文级产出率 36%、物种精确度
   98%、panel 召回为最大短板。**这些数字同样未经人工 image-verified
   基准标定**，发表任何准确率主张前必须先建人工核验 gold 集。
3. 评估工具链：`scripts/evaluate.py`（string F1）、
   `scripts/evaluate_image_verified.py`（EasyOCR 图像验证）、
   `scripts/run_research_eval.py`（5 折 + bootstrap CI）。
4. 每份评估报告强制嵌入 `gold_provenance` 免责声明。

---

## 已知限制

| 问题 | 影响 | 状态 |
|---|---|---|
| 全页扫描图版不支持 | 日刊老文献整篇为 0 | 需整页 OCR，未实现 |
| M3 Stage 2 偶发误判图版为 diagram | Munasri 类论文 panel 全丢 | 已定位，待修 |
| caption 配对缺口 | Xiao 类论文部分图版无图注 | 部分由 rescue 覆盖 |
| SAM2 本环境初始化失败 | 分割退化为 OpenCV（panel 召回降） | 环境问题；YOLO 替代已验证 24/24（`work/yolo_compare/`） |
| `ocr_corrections` 纠错词典未接入生产 | OCR 物种名纠错仅测试可用 | 已审计标记 |
| `--taxon-model` 四个选项无实际差异 | TaxoNERD 1.5.x 忽略 model 参数 | 库限制 |
| 死模块待清理 | `metrics.py`/`batch.py`/`preprocess.py`/`matching.py`/`io.py`/`tei.py`/`bootstrap.py` 无调用方 | 已审计标记 |
| knowledge_graphs / range_charts 明细 | 已进 run_output（v1.3.0），Web 结果视图未展示 | 可经文件端点获取 |
| MCP/API 多 worker 部署 | FALLBACK_PENDING 进程内 | 单机部署无影响 |

审计轨迹：本仓库采用"审计 → 修复 → 源码守卫测试"闭环，历次审计
（2026-08-01/02、09-01、09-04、09-05、09-06）的修复以
`[audit 日期 标签]` 格式标注在 commit message 与代码注释中。

---

## 数据标注指南

标注是标定真实准确率的前提。推荐顺序与规范：

1. **优先标 panel bbox + label + species**（训练/评估的最小闭环）；
2. panel_id 规范：大写字母/数字**保持论文原样**，不要重排；
3. species 规范：完整学名，`sp.`/`cf.`/`aff.` **保留原样**，不确定不要
   补全；
4. 地质信息：age 用标准年代、formation 保留专名、坐标记清古/今；
5. 比例尺：caption 明示则直接记 `50 um`（μm/um 统一 um）。

工具推荐：Label Studio（bbox+文本+关系）、CVAT（图版框选）、
自定义 JSON（最终训练/评估数据，格式见 `SCHEMA.md`）。产出可通过
`POST /review/correction` 或 GUI "Mark verified" 直接回流为
`image_verified` 数据。

---

## 开发与测试

```bash
# 全量测试（约 4000+ 用例；GUI 测试在 PySide6≥6.11+py3.11 组合自动跳过）
PYTHONPATH=src python -m pytest tests/ -q

# 单项质量门（与 GitHub Actions 一致）
python -m ruff check src/ tests/ scripts/
python -m ruff format --check src/ tests/ scripts/
PYTHONPATH=src python -m mypy src/rlpe/ --ignore-missing-imports --no-strict-optional

# eval 冒烟门（注意：当前对冻结快照为已知红，见 work/ 与 ci.yml 注释）
PYTHONPATH=src python scripts/evaluate.py \
  --pred work/combined_9_v17_FINAL.jsonl --gold data/gold/ --output work/ci_eval.json
```

约定：修复必须带回归测试；涉及 prompt/守卫的修复加**源码守卫测试**；
commit message 用 `fix(scope): 描述 [audit 日期 标签]` 格式。

## 文档索引

- 输出 schema 详解：[SCHEMA.md](SCHEMA.md)（机器可读：
  `schemas/rlpe-v1.3.0.json`）
- 评估方法史：[EVALUATION.md](EVALUATION.md)
- 可复现性协议：[REPRODUCIBILITY.md](REPRODUCIBILITY.md)
- Web 架构：[WEB_ARCHITECTURE.md](WEB_ARCHITECTURE.md)、
  使用手册：[WEB_GUIDE.md](WEB_GUIDE.md)
- CLI vs Web 对比：[COMPARISON_CLI_VS_WEB.md](COMPARISON_CLI_VS_WEB.md)
- 变更日志：[CHANGELOG.md](CHANGELOG.md)

## License

见 [LICENSE](LICENSE)。引用格式见 [CITATION.cff](CITATION.cff)。
