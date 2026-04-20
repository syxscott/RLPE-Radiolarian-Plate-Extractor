# RLPE Web 系统架构

## 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户浏览器                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  RLPE Web UI (HTML5 + CSS3 + JavaScript)                │  │
│  │  - 上传处理界面                                          │  │
│  │  - 任务管理                                              │  │
│  │  - 结果查看                                              │  │
│  │  - 标注纠正                                              │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP/REST API
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                    FastAPI Web 服务器                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  API 端点                                               │   │
│  │  • POST /jobs/upload - 上传 PDF                          │   │
│  │  • GET /jobs - 列出所有任务                              │   │
│  │  • GET /jobs/{id}/status - 查询任务状态                  │   │
│  │  • GET /jobs/{id}/result - 获取任务结果                  │   │
│  │  • POST /jobs/{id}/cancel - 取消任务                     │   │
│  │  • GET /results - 获取所有结果                           │   │
│  │  • POST /review/correction - 提交纠正                    │   │
│  │  • GET /system/info - 系统信息                           │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────┬────────────────────────────────────┬────────────┘
                 │                                    │
                 │ BackgroundTasks                    │ File I/O
                 │                                    │
         ┌───────▼────────┐              ┌────────────▼──────────┐
         │  任务处理线程  │              │  文件系统             │
         ├────────────────┤              ├──────────────────────┤
         │ RadiolarianPipeline           │  uploads/            │
         │ • GROBID 解析  │              │  service_work/       │
         │ • Panel 分割   │              │  ├── {job_id}/      │
         │ • OCR 识别    │              │  │   ├── pdfs/      │
         │ • 物种匹配    │              │  │   ├── output/    │
         │ • LLM 增强    │              │  │   └── ...        │
         └───────┬────────┘              │  └── corrections/    │
                 │                       └──────────────────────┘
                 │
         ┌───────▼──────────────┐
         │  外部服务依赖        │
         ├──────────────────────┤
         │ GROBID (PDF 提取)     │
         │ OpenCV (图像处理)     │
         │ PaddleOCR/EasyOCR    │
         │ TaxonERD (物种识别)   │
         │ SAM2 (Panel 分割)     │
         │ Gemma/Ollama (LLM)   │
         └──────────────────────┘
```

## 文件结构

```
web/
├── index.html              # 前端主页
├── css/
│   └── style.css          # 样式表 (响应式设计)
└── js/
    └── app.js             # 前端应用逻辑

src/rlpe/api/
├── __init__.py
└── app.py                 # FastAPI 应用 + 新增 API 端点

项目根目录/
├── run_web_server.py      # 服务器启动脚本
├── run_web_server.bat     # Windows 启动脚本
├── run_web_server.sh      # Linux/macOS 启动脚本
├── WEB_GUIDE.md          # Web 界面使用指南
└── DEPLOY_QUICK_START.md # 快速部署指南
```

## 数据流

### 1. 上传和处理流程

```
用户选择 PDF
    ↓
浏览器发送 POST /jobs/upload
    ↓
服务器保存文件到 uploads/
    ↓
创建 job_id, 初始状态 = "queued"
    ↓
返回 job_id 给浏览器
    ↓
后台线程执行 RadiolarianPipeline
    ↓
状态变为 "running", 更新 progress
    ↓
处理完成
    ↓
状态变为 "done" 或 "failed"
    ↓
浏览器定期轮询 GET /jobs/{id}/status
    ↓
任务完成时显示结果
```

### 2. 结果查询流程

```
用户点击 "结果查看" 选项卡
    ↓
浏览器发送 GET /results
    ↓
服务器从 RESULT_CACHE 中提取所有完成任务的结果
    ↓
返回 [ResultRecord, ...] 列表
    ↓
前端过滤、搜索、分页显示
    ↓
用户选择查看/纠正
```

### 3. 标注纠正流程

```
用户点击 "✏️ 纠正" 按钮
    ↓
弹出纠正表单
    ↓
用户填写正确信息并提交
    ↓
浏览器发送 POST /review/correction
    ↓
服务器将纠正记录追加到 corrections.jsonl
    ↓
返回成功状态
    ↓
显示成功提示
```

## API 端点详解

### 上传 PDF
```
POST /jobs/upload
Content-Type: multipart/form-data

Response:
{
  "job_id": "uuid",
  "status": "queued",
  "created_at": "2024-01-01T12:00:00",
  "filename": "paper.pdf",
  "progress": 0
}
```

### 列出所有任务
```
GET /jobs

Response:
[
  {
    "job_id": "uuid",
    "status": "done",
    "created_at": "2024-01-01T12:00:00",
    "filename": "paper.pdf",
    "progress": 100
  },
  ...
]
```

### 查询任务状态
```
GET /jobs/{job_id}/status

Response:
{
  "job_id": "uuid",
  "status": "running",
  "progress": 50,
  "detail": null
}
```

### 获取任务结果
```
GET /jobs/{job_id}/result

Response:
{
  "status": "done",
  "result": [
    {
      "paper_id": "abc123",
      "figure_id": "fig1",
      "panel_id": "A",
      "species": "Thecopsammatium sp.",
      "confidence": 0.95,
      "bbox": [100, 200, 300, 400],
      "panel_path": "/path/to/panel.png"
    },
    ...
  ]
}
```

### 获取所有结果
```
GET /results

Response:
[
  {
    "job_id": "uuid1",
    "paper_id": "abc123",
    "figure_id": "fig1",
    "species": "Thecopsammatium sp.",
    "confidence": 0.95,
    ...
  },
  ...
]
```

### 提交标注纠正
```
POST /review/correction
Content-Type: application/json

{
  "paper_id": "abc123",
  "figure_id": "fig1",
  "corrected_species": "Thecopsammatium bullatum",
  "corrected_label": "A",
  "reviewer": "Dr. Smith",
  "notes": "修正了物种鉴定"
}

Response:
{
  "status": "ok",
  "saved_to": "/path/to/corrections.jsonl"
}
```

### 获取系统信息
```
GET /system/info

Response:
{
  "version": "0.2.0",
  "python_version": "3.11.0",
  "grobid_url": "http://localhost:8070",
  "active_jobs": 2,
  "total_jobs": 15,
  "completed_jobs": 13,
  "failed_jobs": 0
}
```

## 前端组件说明

### JavaScript 主要函数

| 函数 | 功能 |
|---|---|
| `showNotification()` | 显示通知消息 |
| `addFiles()` | 添加文件到上传列表 |
| `loadJobs()` | 获取并显示任务列表 |
| `loadResults()` | 获取并显示结果 |
| `startJobPolling()` | 启动任务轮询 (每3秒) |
| `viewImage()` | 打开图像查看器 |
| `openCorrectionModal()` | 打开纠正表单 |

### CSS 响应式设计

- 桌面: 最大宽度 1400px
- 平板: 响应式网格布局
- 手机: 移动优先设计 (<768px)

## 性能考虑

### 客户端
- 图片懒加载
- 结果表格分页 (每页100条)
- 任务轮询间隔可配置 (默认3秒)

### 服务器
- 后台任务处理 (不阻塞 API)
- 结果缓存在内存中
- 支持多线程并发处理

### 优化建议
1. 考虑添加数据库存储长期结果
2. 实现 WebSocket 用于实时推送
3. 添加结果分页查询 (而非一次性返回)
4. 使用 Celery + Redis 处理大规模任务

## 安全考虑

### 当前实现
- ✅ CORS 已启用 (允许所有源)
- ✅ 文件验证 (仅接受 PDF)
- ⚠️ 无身份验证

### 生产环境建议
1. 添加 API 密钥认证
2. 限制 CORS 到特定域名
3. 实现上传文件大小限制
4. 添加 Rate Limiting
5. 使用 HTTPS
6. 扫描上传的 PDF 是否包含恶意内容

## 扩展可能性

1. **数据库持久化**: 将结果存储到 PostgreSQL/MongoDB
2. **用户管理**: 添加登录和权限控制
3. **高级搜索**: Elasticsearch 全文搜索结果
4. **批量导入**: 支持文件夹监控和自动处理
5. **API 文档**: 自动生成 OpenAPI/Swagger 文档
6. **监控和日志**: ELK Stack 集成
7. **分布式处理**: Celery 分布式任务队列
8. **Docker 容器化**: 便于部署和扩展

---

**版本**: 0.2.0  
**最后更新**: 2024年
