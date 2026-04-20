# RLPE

Radiolarian Literature Plate Extractor（放射虫文献图版提取流水线）

这个项目的目标是：

> 从已发表文献 PDF 中，自动提取图版与图注，识别 panel 标签（A/B/1/2...），提取拉丁学名，并建立 panel-label-species 的对应关系。

项目已经实现规则流水线，并支持可选 Gemma 4 多模态后处理增强。

> 建议先做小样本验证：**20~50 篇 PDF**，确认图版裁剪、标签匹配、比例尺与地质信息后，再进入全量批处理。

---

## 快速导航

### 🎯 新手入门（强烈推荐）
- **[📘 Web 界面使用指南](WEB_GUIDE.md)** - **无需命令行！** 👈 从这里开始
- [快速开始（建议新用户按这个顺序）](#快速开始建议新用户按这个顺序)

### 📚 功能和技术文档
- [功能总览](#功能总览)
- [目录结构说明（核心模块）](#目录结构说明核心模块)
- [开启 Gemma 4 增强](#开启-gemma-4-增强推荐)
- [数据标注怎么做（详细版）](#数据标注怎么做详细版)
- [地质信息抽取与比例尺](#地质信息抽取与比例尺)
- [服务化与上传（骨架）](#服务化与上传骨架)
- [训练脚本](#训练脚本)
- [推荐实践（提升稳定性）](#推荐实践提升稳定性)
- [常见问题（FAQ）](#常见问题faq)

---

## 🌐 Web 界面（推荐给非技术用户）

> **如果你不熟悉命令行，请直接使用 Web 界面！**

### 启动 Web 服务器

#### Windows 用户
```bash
# 方式1：双击运行
run_web_server.bat

# 方式2：PowerShell/CMD
python run_web_server.py
```

#### Linux/macOS 用户
```bash
chmod +x run_web_server.sh
./run_web_server.sh
```

### 打开网页界面
访问: **http://localhost:8000**

### 功能
✅ 拖拽上传 PDF  
✅ 可视化配置参数  
✅ 实时任务监控  
✅ 交互式结果审查  
✅ 一键导出结果  
✅ 人工标注纠正  

**详细使用指南见**: [WEB_GUIDE.md](WEB_GUIDE.md)

---

## 一页总结

### 你要解决的问题

从论文 PDF 中提取：

- 图版（figure / plate）
- panel（A/B/1/2...）
- 图注（caption）
- 拉丁学名（species / genus）
- 地质信息（Age / Formation / Locality）
- 比例尺（scale bar）

### 你的运行模式

| 模式 | 适用场景 | 后端 | 备注 |
|---|---|---|---|
| 规则流水线 | 快速批处理 | CPU + OpenCV + GROBID | 默认可跑 |
| llama.cpp | 你当前环境 | http://127.0.0.1:8080 | 推荐 |
| Ollama | 本地 GGUF 管理 | http://127.0.0.1:11434 | 兼容性好 |
| Transformers | HF 权重 | 本地 GPU | 适合权重模型 |

### 推荐工作流

1. 先跑 10~20 篇 PDF
2. 看 `output/panels/` 的裁剪效果
3. 抽检 `matches.jsonl`
4. 标注 30~50 篇 gold 数据
5. 再启用 Gemma 和地质抽取
6. 最后做全量批处理与评估

---

## 功能总览

当前版本支持：

1. 批量读取 PDF
2. 调 GROBID 提取 figure/caption
3. 渲染页面并做跨页候选页匹配
4. 检测图版区域（figure region）
5. 分割 panel
6. OCR 识别标签文本
7. 抽取拉丁学名
8. panel-label-species 自动匹配
9. 导出 JSON/CSV/JSONL
10. 计算基础评估指标
11. 可选：Gemma 4 后处理增强（低置信度自动回退规则）
12. 正文地质信息抽取（Age / Formation / Locality）
13. 比例尺信息抽取（caption/OCR/视觉线段估计）
14. Species-Geology 关系链接与知识图谱输出
15. API服务与任务队列骨架（FastAPI + Celery）

---

## 目录结构说明（核心模块）

- 主流程：[src/rlpe/pipeline.py](src/rlpe/pipeline.py)
- GROBID解析：[src/rlpe/grobid.py](src/rlpe/grobid.py)
- 版面定位与跨页候选：[src/rlpe/layout.py](src/rlpe/layout.py)
- 图版分割：[src/rlpe/segmentation.py](src/rlpe/segmentation.py)
- OCR：[src/rlpe/ocr.py](src/rlpe/ocr.py)
- 规则匹配：[src/rlpe/association.py](src/rlpe/association.py)
- Gemma后处理：[src/rlpe/gemma_postprocess.py](src/rlpe/gemma_postprocess.py)
- LLM后端抽象（Transformers/Ollama）：[src/rlpe/llm_backends.py](src/rlpe/llm_backends.py)
- 地质信息抽取：[src/rlpe/geology_extraction.py](src/rlpe/geology_extraction.py)
- 比例尺抽取：[src/rlpe/scale_bar.py](src/rlpe/scale_bar.py)
- 评估：[src/rlpe/evaluation.py](src/rlpe/evaluation.py)
- 命令行入口：[src/rlpe/cli.py](src/rlpe/cli.py)
- API服务：[src/rlpe/api/app.py](src/rlpe/api/app.py)
- 队列任务：[src/rlpe/worker/tasks.py](src/rlpe/worker/tasks.py)

---

## 快速开始（建议新用户按这个顺序）

### 第1步：准备数据目录

建议使用下面的结构：

```text
project_root/
	data/
		pdfs/                 # 输入PDF
	work/                   # 中间与输出目录
```

建议先放 10~100 篇 PDF 试运行，确认无误后再全量跑。

### 推荐目录结构（更清晰）

```text
RLPE/
├── data/
│   ├── pdfs/                 # 输入 PDF
│   └── annotations/          # 标注 JSON / Label Studio 导出
├── work/                     # 中间产物与结果
├── scripts/                  # 训练、后处理、工具脚本
└── src/rlpe/                 # 主代码
```

### 第2步：确认 GROBID 服务可用

默认地址：`http://localhost:8070`

只要该地址可访问，RLPE就可以调用。

### 第3步：运行基础流程（不启用Gemma）

```bash
python scripts/run_pipeline.py \
	--pdf-dir /path/to/data/pdfs \
	--work-dir /path/to/work \
	--grobid-url http://localhost:8070 \
	--ocr-backend paddleocr \
	--num-workers 4 \
	--render-dpi 200 \
	--save-intermediate \
	--export-jsonl /path/to/work/output/matches.jsonl \
	--export-csv /path/to/work/output/matches.csv
```

### 第4步：检查输出

重点查看：

- `output/tei/`：GROBID解析结果
- `output/figures/`：PDF渲染页图与region
- `output/panels/`：裁剪后的panel图片
- `output/manifests/matches.jsonl`：最终结构化结果

### 第5步：建议先检查哪些字段

最先看这几个字段是否合理：

- `panel_id`
- `species`
- `confidence`
- `scale_bar`
- `geology_links`
- `matcher_type`

---

## 开启 Gemma 4 增强（推荐）

Gemma模块是在规则匹配之后运行，作为最终决策层。

- 高于阈值：采用Gemma结果
- 低于阈值：自动回退规则结果

### 方式A：Transformers后端（Hugging Face）

```bash
python scripts/run_pipeline.py \
	--pdf-dir /path/to/data/pdfs \
	--work-dir /path/to/work \
	--grobid-url http://localhost:8070 \
	--use-gemma4 \
	--gemma-model-path /home/user/models/gemma-4-E4B \
	--gemma-conf-threshold 0.70 \
	--gemma-prompt-lang zh
```

参数说明：

- 默认启用4bit量化
- 若要关闭4bit：`--gemma-no-4bit`
- 默认启用bfloat16
- 若要关闭bfloat16：`--gemma-no-bfloat16`

### 方式A2：llama.cpp 后端（推荐你当前环境）

你现在使用的是 llama.cpp，本地地址为 `http://127.0.0.1:8080`。RLPE 已支持通过该地址调用模型。

示例：

```bash
python scripts/run_pipeline.py \
	--pdf-dir /path/to/data/pdfs \
	--work-dir /path/to/work \
	--grobid-url http://localhost:8070 \
	--use-gemma4 \
	--llm-backend llamacpp \
	--llama-host http://127.0.0.1:8080 \
	--llama-model gemma-4-31b-it \
	--gemma-conf-threshold 0.70
```

说明：

- 如果你的 llama.cpp 服务器支持 OpenAI-compatible 接口，RLPE 会优先调用 `/v1/chat/completions`
- 如果不支持，代码会自动回退到 `/completion`
- panel 图像会以 base64 方式随请求发送（若模型/服务支持视觉输入）
- 如果是纯文本模型，系统会自动走文本回退路径

### 方式B：离线增强已有结果

```bash
python scripts/gemma_batch_postprocess.py \
	--input-jsonl /path/to/work/output/matches.jsonl \
	--output-jsonl /path/to/work/output/matches_gemma.jsonl \
	--backend llamacpp \
	--llama-host http://127.0.0.1:8080 \
	--llama-model gemma-4-31b-it \
	--model-path /home/user/models/gemma-4-E4B \
	--conf-threshold 0.70 \
	--prompt-lang zh \
	--use-4bit
```

### 方式C：Ollama后端（GGUF，推荐你的本地权重）

你的权重文件：`/home/user/shenyaxuan/ollama-models/gemma-4-31B-it-Q4_K_M.gguf`

在Ollama中注册模型后，主流程可直接调用：

```bash
python scripts/run_pipeline.py \
	--pdf-dir /path/to/data/pdfs \
	--work-dir /path/to/work \
	--grobid-url http://localhost:8070 \
	--use-gemma4 \
	--llm-backend ollama \
	--ollama-model gemma-4-31b-it \
	--ollama-host http://127.0.0.1:11434 \
	--gemma-conf-threshold 0.70
```

离线增强（Ollama）：

```bash
python scripts/gemma_batch_postprocess.py \
	--input-jsonl /path/to/work/output/matches.jsonl \
	--output-jsonl /path/to/work/output/matches_gemma.jsonl \
	--backend ollama \
	--ollama-model gemma-4-31b-it \
	--ollama-host http://127.0.0.1:11434 \
	--conf-threshold 0.70
```

---

## 一键开关 Gemma（配置方式）

建议统一通过 `PipelineConfig.extra` 控制：

```python
extra = {
	"use_gemma4": True,
	"llm_backend": "llamacpp",  # transformers | ollama | llamacpp
	"gemma_model_path": "/home/user/models/gemma-4-E4B",
	"llama_host": "http://127.0.0.1:8080",
	"llama_model": "gemma-4-31b-it",
	"ollama_model": "gemma-4-31b-it",
	"ollama_host": "http://127.0.0.1:11434",
	"gemma_conf_threshold": 0.70,
	"gemma_prompt_lang": "zh",
	"gemma_use_4bit": True,
	"gemma_bfloat16": True,
	"gemma_device_map": "auto"
}
```

- 开启：`use_gemma4=True`
- 关闭：`use_gemma4=False`

---

## 数据标注怎么做（详细版）

下面这部分是你后面训练 matcher、Taxon NER、以及做人工校验闭环的基础。

### 标注原则

- 先标“最稳定、最容易复用”的字段
- 不确定的内容不要硬补全
- 所有标注都要保留 `evidence_text`
- 一篇文章要保持同一套命名规范

### 最推荐的标注工具

| 工具 | 适合做什么 | 备注 |
|---|---|---|
| Label Studio | bbox、文本、关系 | 适合人工校验闭环 |
| CVAT | 图像框选 | 适合 panel / figure bbox |
| 自定义 JSON | 最终训练数据 | 便于直接喂给脚本 |

### 1. 标注目标有哪些

建议至少分成 6 类标注任务：

#### 1) Figure 标注
- 标整张图版/图版页的边界
- 适合后续训练 figure region 检测

字段建议：
- `paper_id`
- `figure_id`
- `bbox`
- `page_index`

#### 2) Panel 标注
- 标每个 panel 的边界框
- 例如 A、B、C、1、2、3

字段建议：
- `panel_id`
- `bbox`
- `page_index`
- `source_figure_id`

#### 3) Label 标注
- 标图中标签的位置和文本
- 例如 A、B、C、a、b、1、2

字段建议：
- `label_text`
- `bbox`
- `confidence`

#### 4) Species 标注
- 标每个 panel 对应的拉丁学名
- 例如 `Actinomma leptodermum`

字段建议：
- `species`
- `panel_id`
- `evidence_source`（caption / figure text / manual）

#### 5) 地质信息标注
- 年代（Age）
- 地层（Formation / Member / Group）
- 地点（Locality）

字段建议：
- `species`
- `age`
- `formation`
- `locality`
- `evidence_text`

#### 6) 比例尺标注
- `50 μm`、`100 μm` 等文字值
- 若有比例尺线段，再标线段框

字段建议：
- `scale_value`
- `scale_unit`
- `pixel_length`
- `bbox`（如果你能看到线段）

---

### 2. 推荐的标注顺序

不要一开始什么都标。建议按这个顺序：

#### 第一步：先标最关键的 3 项
1. `panel bbox`
2. `panel label`
3. `species`

这三项足够先训练/评估：
- panel 分割
- label/species 匹配
- OCR 后处理

#### 第二步：再加图注和跨页关系
4. `figure-caption` 对应关系
5. `caption` 中的 label–species 子句

#### 第三步：再加地质信息
6. `age`
7. `formation`
8. `locality`

#### 第四步：再加比例尺
9. `scale bar`

---

### 3. 一个最实用的标注 JSON 格式

建议每篇文章输出一个 JSON 文件，结构如下：

```json
{
	"paper_id": "xxx",
	"figures": [
		{
			"figure_id": "fig1",
			"page_index": 12,
			"figure_bbox": [100, 200, 1200, 1600],
			"caption": "...",
			"panels": [
				{
					"panel_id": "A",
					"bbox": [120, 220, 300, 250],
					"label_text": "A",
					"species": "Actinomma leptodermum"
				}
			]
		}
	],
	"geology": [
		{
			"species": "Actinomma leptodermum",
			"age": "Late Eocene",
			"formation": "X Formation",
			"locality": "Y Section",
			"evidence_text": "..."
		}
	],
	"scale": [
		{
			"figure_id": "fig1",
			"scale_value": 50,
			"scale_unit": "um",
			"pixel_length": 132
		}
	]
}
```

### 4.1 字段含义速查

| 字段 | 含义 | 是否必填 |
|---|---|---|
| `paper_id` | 论文唯一编号 | 是 |
| `figure_id` | 图版编号 | 是 |
| `page_index` | 页码 | 是 |
| `figure_bbox` | 整张图版范围 | 建议 |
| `panel_id` | 子图标签 | 是 |
| `bbox` | panel边界框 | 是 |
| `label_text` | 图中标签文本 | 建议 |
| `species` | 拉丁学名 | 是 |
| `age` | 地质年代 | 建议 |
| `formation` | 地层名 | 建议 |
| `locality` | 产地/剖面 | 建议 |
| `scale_value` | 比例尺数值 | 建议 |
| `scale_unit` | 比例尺单位 | 建议 |
| `evidence_text` | 证据原文 | 强烈建议 |

---

### 4. 标注规范怎么统一

建议你在标注前统一下面规则：

#### panel id 规范
- 大写字母优先：A、B、C
- 如果原文是数字，按原文：1、2、3
- 不要自己改顺序，尽量保持论文原始标注逻辑

#### species 规范
- 尽量写完整学名
- 如果原文只写 `sp.`、`cf.`、`aff.`，也保留原样
- 不要把不确定的名字强行补全

#### 地质信息规范
- `age` 尽量写标准地质年代
- `formation` 保留原始专名
- `locality` 保留原始剖面/地点名

#### 比例尺规范
- 若 caption 明确写了 `Scale bar = 50 μm`，直接记 50 和 `um`
- `μm`、`um` 统一成 `um`
- 如果你能量出线段像素长度，也记上

---

### 5. 怎么开始标第一批数据

建议你先做 30~50 篇文章的标注，原因是：

- 太少，训练不稳定
- 太多，前期费时

#### 推荐任务拆分

##### 任务A：人工快速审校 panel
- 只看 bbox 和 label 是否正确
- 先别管地质信息

##### 任务B：人工确认 species
- 检查 `panel -> species` 是否对

##### 任务C：补地质信息
- 从正文中找 age/formation/locality

##### 任务D：补比例尺
- caption 里有就标文字
- 图中有线段就标线段

### 5.1 标注优先级建议

优先级从高到低：

1. panel bbox
2. panel label
3. species
4. figure-caption 对应关系
5. geology（age / formation / locality）
6. scale bar

---

### 6. 给训练脚本准备什么数据

#### 神经图匹配器训练数据
对应 [scripts/train_matcher.py](scripts/train_matcher.py)

每条样本建议至少包括：

- `panel_features`
- `label_features`
- `species_features`
- `target_panel_label`
- `target_panel_species`

### 6.1 神经图匹配器样本建议格式

```json
{
	"paper_id": "xxx",
	"figure_id": "fig1",
	"panel_features": [0.12, 0.18, 0.20, 0.15, 0.22, 0.25, 0.03, 0.01, 0.91, 1.0, 0.0, 0.0],
	"label_features": [0.10, 0.16, 0.02, 0.02, 0.11, 0.17, 0.0004, 0.01, 0.95, 0.04, 1.0, 0.0],
	"species_features": [0.3, 0.2, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
	"target_panel_label": 0,
	"target_panel_species": 0
}
```

#### Taxon NER 训练数据
对应 [scripts/train_taxon_ner.py](scripts/train_taxon_ner.py)

格式建议：

```json
[
	{
		"tokens": ["Actinomma", "leptodermum", "was", "found", "in", "..."],
		"ner_tags": ["B-TAXON", "I-TAXON", "O", "O", "O", "O"]
	}
]
```

---

### 7. 标注时最容易犯的错

- 把图中的字母标签和正文里的引文编号混淆
- 把 caption 里的 species 和正文里另一个 species 混在一起
- 地质年代没统一写法
- `sp.`、`cf.`、`aff.` 被错误补全
- 一个 caption 对多个 panel 时没有拆分清楚

### 7.1 人工复核建议

建议每批标注后做一次快速抽检：

- 随机抽 10% 样本
- 检查 bbox 是否偏移
- 检查 species 是否串行
- 检查 geology 是否引用错段落
- 检查 scale bar 是否被 OCR 误读

---

### 8. 你最先应该做的事情

如果你现在就要动手，建议优先顺序是：

1. 先选 20 篇 PDF
2. 每篇只标 1~2 张 figure
3. 先做 panel bbox + label + species
4. 再补地质信息和比例尺
5. 保存成统一 JSON
6. 用这些数据跑训练脚本

---

## 工程检查状态

已更新：

- llama.cpp 后端（默认本地 `http://127.0.0.1:8080`）
- README 标注规范
- CLI 参数支持 llama.cpp

如果你愿意，我下一步可以继续给你补一份“**标注模板 JSON 文件**”和“**label-studio 标注字段映射表**”，这样你可以直接开始标。 

---

## 最后建议

如果你现在准备正式推进，建议按下面顺序执行：

1. 先跑 10 篇样例，确认流程
2. 用 Label Studio 标 30~50 篇 gold 数据
3. 用这批数据训练 matcher / taxon NER
4. 启用 llama.cpp 做语义后处理
5. 再做地质信息与比例尺抽取
6. 最后接入 API / 上传 / 人工修正闭环

## 评估与对比

### 基础评估

可使用 [scripts/evaluate.py](scripts/evaluate.py) 对预测与gold进行评估。

### Gemma增强前后对比

在 [src/rlpe/evaluation.py](src/rlpe/evaluation.py) 中使用 `compare_before_after()`，重点字段：

- `match_acc_before`
- `match_acc_after`
- `match_improvement`
- `gemma_confidence_mean`

此外，`metadata` 中新增：

- `scale_bar`（比例尺值、单位、um_per_px）
- `geology_links`（物种与地质信息链接）
- `matcher_type`（heuristic / neural-graph）

---

## 地质信息抽取与比例尺

地质信息来源：

1. TEI全文章节（Systematic Paleontology / Geological Setting）
2. 规则抽取（Age / Formation / Locality）
3. 可选LLM关系链接（`--use-geology-llm`）

比例尺来源：

1. Caption文本（如 `Scale bar = 50 μm`）
2. OCR文本（图中标注）
3. 视觉线段长度估计（Hough）

主流程启用示例：

```bash
python scripts/run_pipeline.py \
	--pdf-dir /path/to/data/pdfs \
	--work-dir /path/to/work \
	--use-gemma4 \
	--use-geology-llm
```

---

## 服务化与上传（骨架）

已提供：

- FastAPI 上传与结果查询接口：[src/rlpe/api/app.py](src/rlpe/api/app.py)
- Celery任务骨架：[src/rlpe/worker/tasks.py](src/rlpe/worker/tasks.py)

接口示例：

- `POST /jobs/upload`
- `GET /jobs/{job_id}/status`
- `GET /jobs/{job_id}/result`
- `POST /review/correction`（人工校验回流）

---

## 训练脚本

- 神经图匹配训练：[scripts/train_matcher.py](scripts/train_matcher.py)
- Taxon垂类NER微调：[scripts/train_taxon_ner.py](scripts/train_taxon_ner.py)

---

## 推荐实践（提升稳定性）

1. 先小规模试跑（10~100篇）
2. 人工抽检 `output/panels/` 与 `matches.jsonl`
3. 再扩大到1000篇以上
4. 对错误模式做针对性优化（OCR阈值、分割阈值、caption窗口）
5. 最后全量跑并做评估报告

---

## 常见问题（FAQ）

### 1) Gemma加载失败怎么办？

- 检查 `gemma_model_path` 是否正确
- 检查模型目录是否完整
- 检查 Transformers 版本是否支持该模型类型

### 2) 出现CUDA OOM怎么办？

- 降低并发
- 关闭Gemma或降低Gemma调用规模
- 关闭bfloat16或调整量化策略

### 3) Gemma输出不是JSON怎么办？

- 当前代码已做JSON解析保护
- 解析失败时会自动回退规则方法

### 4) panel裁剪效果不好怎么办？

- 检查 `output/figures/regions`
- 调整分割相关阈值与图片DPI

---

## 从头到尾的检查清单

1. GROBID可访问（`http://localhost:8070`）
2. 输入PDF目录有文件
3. 先跑小样本
4. 检查panel裁剪质量
5. 检查结果文件核心字段（`panel_id`、`species`、`confidence`）
6. 启用Gemma后检查 `metadata.gemma_confidence` 与回退标志
7. 用gold数据做前后对比，确认 `match_improvement`

---

## 代码检查状态

已完成：

- 工作区静态错误检查（Problems）通过
- 关键模块无语法报错

如需本地再做一次编译检查，可运行：

```bash
python -m compileall src scripts
```
