# RLPE

Radiolarian Literature Plate Extractor（放射虫文献图版提取流水线）

这个项目的目标是：

> 从已发表文献 PDF 中，自动提取图版与图注，识别 panel 标签（A/B/1/2...），提取拉丁学名，并建立 panel-label-species 的对应关系。

项目已经实现规则流水线，并支持可选 Gemma 4 多模态后处理增强。

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

---

## 目录结构说明（核心模块）

- 主流程：[src/rlpe/pipeline.py](src/rlpe/pipeline.py)
- GROBID解析：[src/rlpe/grobid.py](src/rlpe/grobid.py)
- 版面定位与跨页候选：[src/rlpe/layout.py](src/rlpe/layout.py)
- 图版分割：[src/rlpe/segmentation.py](src/rlpe/segmentation.py)
- OCR：[src/rlpe/ocr.py](src/rlpe/ocr.py)
- 规则匹配：[src/rlpe/association.py](src/rlpe/association.py)
- Gemma后处理：[src/rlpe/gemma_postprocess.py](src/rlpe/gemma_postprocess.py)
- 评估：[src/rlpe/evaluation.py](src/rlpe/evaluation.py)
- 命令行入口：[src/rlpe/cli.py](src/rlpe/cli.py)

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

---

## 开启 Gemma 4 增强（推荐）

Gemma模块是在规则匹配之后运行，作为最终决策层。

- 高于阈值：采用Gemma结果
- 低于阈值：自动回退规则结果

### 方式A：在主流程中直接开启

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

### 方式B：离线增强已有结果

```bash
python scripts/gemma_batch_postprocess.py \
	--input-jsonl /path/to/work/output/matches.jsonl \
	--output-jsonl /path/to/work/output/matches_gemma.jsonl \
	--model-path /home/user/models/gemma-4-E4B \
	--conf-threshold 0.70 \
	--prompt-lang zh \
	--use-4bit
```

---

## 一键开关 Gemma（配置方式）

建议统一通过 `PipelineConfig.extra` 控制：

```python
extra = {
	"use_gemma4": True,
	"gemma_model_path": "/home/user/models/gemma-4-E4B",
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

## 评估与对比

### 基础评估

可使用 [scripts/evaluate.py](scripts/evaluate.py) 对预测与gold进行评估。

### Gemma增强前后对比

在 [src/rlpe/evaluation.py](src/rlpe/evaluation.py) 中使用 `compare_before_after()`，重点字段：

- `match_acc_before`
- `match_acc_after`
- `match_improvement`
- `gemma_confidence_mean`

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
