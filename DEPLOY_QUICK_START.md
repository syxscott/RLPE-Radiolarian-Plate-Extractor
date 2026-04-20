# RLPE Web 界面 - 快速部署指南

## 📋 预备条件

- Python 3.11+
- GROBID 服务已运行 (http://localhost:8070)
- 必要的依赖已安装

## 🚀 5 分钟快速部署

### 1️⃣ 安装依赖

```bash
# 如果还没安装
pip install fastapi uvicorn pydantic

# 或一次性安装所有服务依赖
pip install -r requirements.txt
```

### 2️⃣ 启动 Web 服务器

**Windows**:
```bash
python run_web_server.py
```

**Linux/macOS**:
```bash
python3 run_web_server.py
```

### 3️⃣ 打开浏览器

输入地址: `http://localhost:8000`

**完成！** ✅

---

## 🎯 后续步骤

1. 在 **"📤 上传处理"** 选项卡中上传 PDF
2. 配置处理参数 (推荐使用默认值)
3. 点击 **"▶️ 开始处理"** 按钮
4. 在 **"📋 任务管理"** 中监控进度
5. 在 **"📊 结果查看"** 中审核识别结果

---

## 📖 完整文档

查看详细使用指南: [WEB_GUIDE.md](WEB_GUIDE.md)

---

## 🔧 高级配置

### 自定义端口

编辑 `run_web_server.py`:
```python
# 改这行
uvicorn.run(app, host="0.0.0.0", port=8000)  # ← 改成其他端口
```

### 配置 GROBID 地址

在 Web 界面的 **"⚙️ 处理配置"** 中修改 GROBID 服务地址。

### 启用 LLM 增强

1. 启动 LLM 服务 (Ollama/llama.cpp/Transformers)
2. 在 Web 界面勾选 **"启用 Gemma 4 后处理增强"**
3. 配置 LLM 后端地址
4. 开始处理

---

## 🐛 故障排查

| 问题 | 解决方案 |
|---|---|
| 网页打不开 | 确保服务器运行中，访问 `http://localhost:8000` |
| "无法连接" 警告 | 检查 GROBID 是否在 `http://localhost:8070` 运行 |
| 上传失败 | 检查浏览器控制台 (F12)，查看网络错误 |
| 处理卡住 | 查看服务器日志，检查磁盘空间 |

---

## 📞 需要帮助？

1. 查看完整文档: [WEB_GUIDE.md](WEB_GUIDE.md)
2. 查看 API 文档: http://localhost:8000/docs
3. 查看服务器日志了解具体错误信息

---

**祝使用愉快！** 🎉
