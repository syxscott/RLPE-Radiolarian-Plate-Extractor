# RLPE 在 Ubuntu 上的部署指南

本文档面向把 RLPE 部署到 **Ubuntu 22.04 / 24.04 LTS** 服务器的场景。

---

## 0. 硬件要求

| 配置 | 最低 | 推荐 |
|---|---|---|
| CPU | 2 核 | 4 核+ |
| RAM | 8 GB | 16 GB+ |
| 磁盘 | 20 GB | 50 GB+（cache 文件可能很大）|
| GPU | 无 | 可选（加速 OCR）；无 GPU 也能跑 |
| 网络 | 出网能访问 `api.minimaxi.com` | 同左 |

> **不需要独立显卡**——LLM 走 MiniMax M3 云端 API，OCR/分割都自动降级到 CPU。

---

## 1. 系统依赖

```bash
# 1.1 更新 apt
sudo apt update && sudo apt upgrade -y

# 1.2 安装基础工具
sudo apt install -y curl wget git build-essential ca-certificates

# 1.3 安装 Java 11+ (GROBID 依赖)
sudo apt install -y openjdk-17-jdk
java -version

# 1.4 (可选) 安装 Docker (用于跑 GROBID)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# 重新登录以生效
```

---

## 2. 安装 Miniconda

```bash
# 推荐 miniconda（轻量）
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash
# 重新登录
```

---

## 3. 部署项目

```bash
# 3.1 把项目复制到目标机器（用 git / scp / rsync 任意方式）
# 假设放到 /opt/rlpe
sudo mkdir -p /opt/rlpe
sudo chown $USER:$USER /opt/rlpe
cd /opt/rlpe
# rsync -avz user@source:/path/to/RLPE-Radiolarian-Plate-Extractor/ .

# 3.2 创建 conda 环境（自动装所有依赖）
conda env create -f environment.yml
# 如果环境名冲突或想重建：
# conda env remove -n rlpe && conda env create -f environment.yml

# 3.3 激活
conda activate rlpe
```

> 装完后所有依赖都在 `rlpe` 环境里。**不需要单独 `pip install`**。

---

## 4. 配置 API Key

```bash
# 4.1 复制模板
cp .env.example .env

# 4.2 编辑 .env，填入真实的 MiniMax Token Plan Key
nano .env
# 把 ANTHROPIC_API_KEY= 改成 ANTHROPIC_API_KEY=eyJhbGciOiJ...

# 4.3 验证 .env 没被误传到 git
cat .env
# 应该看到真实的 key，且首行不是 .gitignore 注释
```

---

## 5. 启动 GROBID（PDF 解析必需）

### 选项 A: Docker（推荐）

```bash
docker run -d \
    --name grobid \
    --restart unless-stopped \
    -p 8070:8070 \
    lfoppiano/grobid:0.8.0
# 等 30s，验证
curl http://localhost:8070/api/isalive
# 应该返回 true
```

### 选项 B: 直接跑 JAR

```bash
# 下载
wget https://github.com/kermitt2/grobid/releases/download/0.8.0/grobid-0.8.0.zip
unzip grobid-0.8.0.zip -d tools/
cd tools/grobid-0.8.0
./gradlew run  # 首次会下载大量依赖
```

---

## 6. 启动 RLPE 后端

```bash
# 6.1 启动 (开发模式，前台运行)
bash start_dev.sh web
# → 浏览器访问 http://localhost:8000

# 6.2 启动 (生产模式，多 worker)
bash start_dev.sh api
# → 监听 0.0.0.0:8000
# → 日志输出到 stdout
```

### 6.3 用 systemd 管理（生产推荐）

创建 `/etc/systemd/system/rlpe.service`：

```ini
[Unit]
Description=RLPE Backend (Radiolarian Plate Extractor)
After=network.target docker.service
Wants=docker.service

[Service]
Type=simple
User=rlpe
Group=rlpe
WorkingDirectory=/opt/rlpe
Environment="PATH=/home/rlpe/miniconda3/envs/rlpe/bin:/home/rlpe/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="RLPE_HOST=0.0.0.0"
Environment="RLPE_PORT=8000"
ExecStart=/home/rlpe/miniconda3/envs/rlpe/bin/python -m uvicorn rlpe.api.app:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启用：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now rlpe
sudo systemctl status rlpe
# 应该看到 active (running)
```

---

## 7. 防火墙

```bash
# 7.1 开放 8000 端口（仅 web 访问）
sudo ufw allow 8000/tcp
sudo ufw reload

# 7.2 (可选) 限制访问来源
sudo ufw allow from 192.168.1.0/24 to any port 8000
```

---

## 8. 反向代理 (Nginx, 可选)

如果想用 80/443 端口 + HTTPS，加 Nginx：

```bash
sudo apt install -y nginx
```

`/etc/nginx/sites-available/rlpe`：

```nginx
server {
    listen 80;
    server_name rlpe.example.com;
    
    client_max_body_size 256M;  # PDF 上传限制
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 600s;  # 长任务支持
    }
}
```

启用：

```bash
sudo ln -s /etc/nginx/sites-available/rlpe /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 9. 验证

```bash
# 9.1 健康检查
curl http://localhost:8000/health
# → {"status":"ok"}

# 9.2 GROBID
curl http://localhost:8070/api/isalive
# → true

# 9.3 MiniMax API
curl -X POST http://localhost:8000/admin/cache/clear \
  -H "Content-Type: application/json" \
  -d '{"scope":"all","delete_files":true}'

# 9.4 上传 PDF 测试
# → 浏览器 http://localhost:8000
# → 拖一个 PDF → 配置 MiniMax → 提交
# → 应该看到任务在处理
```

---

## 10. 常见 Ubuntu 部署坑

| 现象 | 原因 | 修复 |
|---|---|---|
| `paddlepaddle` 安装失败 | Linux ARM / 缺依赖 | 换 conda-forge：`conda install -c conda-forge paddlepaddle` |
| `libGL.so.1: cannot open` | OpenCV 缺系统库 | `sudo apt install -y libgl1 libglib2.0-0` |
| `pymupdf` 报 Illegal instruction | 旧 CPU 不支持 AVX2 | `pip install pymupdf==1.24.0`（更老版本）|
| GROBID 502 / 超时 | 启动慢，模型加载 | 启动后等 30s 再用 |
| Anthropic SDK SSL 错 | ca-certificates 旧 | `sudo apt install -y ca-certificates && sudo update-ca-certificates` |
| 8000 端口被占 | 别的服务 | `sudo lsof -i :8000` 查看 |
| conda activate 失败 | 没 init | `conda init bash` 然后重新登录 |

---

## 11. 性能调优（可选）

```bash
# 11.1 OCR CPU 模式跑多线程
export OMP_NUM_THREADS=4

# 11.2 限制单 PDF 处理时间
# 在 web UI 上传时配置 --num-workers 2 (默认 4，CPU 笔记本降到 2)

# 11.3 定期清理缓存（避免磁盘满）
# 添加 cron job
(crontab -l 2>/dev/null; echo "0 3 * * 0 curl -X POST http://localhost:8000/admin/cache/clear -H 'Content-Type: application/json' -d '{\"scope\":\"completed\",\"older_than_hours\":168,\"delete_files\":true}'") | crontab -
# 每周日凌晨 3 点清理 7 天前的已完成任务
```

---

## 12. 升级

```bash
cd /opt/rlpe
git pull  # 或者重新 rsync
conda activate rlpe
conda env update -f environment.yml --prune
sudo systemctl restart rlpe  # 如果用 systemd
```
