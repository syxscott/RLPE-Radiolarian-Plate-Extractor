#!/bin/bash
# ============================================================================
# RLPE 一键部署脚本 (Ubuntu 22.04 / 24.04 LTS)
# 用法: sudo bash deploy_ubuntu.sh
# 功能:
#   1. 安装系统依赖 (curl, git, java, ca-certs)
#   2. 安装 miniconda (如未装)
#   3. 创建 conda env 'rlpe'
#   4. 启动 GROBID via Docker (如已装 Docker)
#   5. 询问是否安装 systemd 服务
# ============================================================================
set -e

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  RLPE Ubuntu 部署脚本${NC}"
echo -e "${GREEN}============================================================${NC}"

# 0. 检查 root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}请用 sudo 运行: sudo bash $0${NC}"
    exit 1
fi

# 1. 系统依赖
echo -e "${YELLOW}[1/5] 安装系统依赖...${NC}"
apt update -qq
apt install -y curl wget git build-essential ca-certificates libgl1 libglib2.0-0

# 2. Java (GROBID 需要)
if ! command -v java &> /dev/null; then
    echo -e "${YELLOW}[2/5] 安装 OpenJDK 17...${NC}"
    apt install -y openjdk-17-jdk
else
    echo -e "${GREEN}[2/5] Java 已安装: $(java -version 2>&1 | head -1)${NC}"
fi

# 3. Miniconda
if ! command -v conda &> /dev/null; then
    echo -e "${YELLOW}[3/5] 安装 Miniconda...${NC}"
    if [ ! -d "$HOME/miniconda3" ]; then
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
        bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    fi
    export PATH="$HOME/miniconda3/bin:$PATH"
    conda init bash
else
    echo -e "${GREEN}[3/5] conda 已安装: $(conda --version)${NC}"
fi

# 4. 创建 conda env
echo -e "${YELLOW}[4/5] 创建 conda env 'rlpe'...${NC}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# bash 友好 (not requiring login shell)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
if conda env list | grep -q "^rlpe "; then
    echo "  env 'rlpe' 已存在，跳过创建（用 'conda env update' 更新）"
else
    conda env create -f "$SCRIPT_DIR/environment.yml"
fi

# 5. GROBID (optional)
echo -e "${YELLOW}[5/5] GROBID...${NC}"
if command -v docker &> /dev/null; then
    if docker ps -a 2>/dev/null | grep -q grobid; then
        echo "  GROBID 容器已存在"
    else
        read -p "  是否用 Docker 启动 GROBID? [y/N] " yn
        if [[ "$yn" =~ ^[Yy]$ ]]; then
            docker run -d --name grobid --restart unless-stopped -p 8070:8070 lfoppiano/grobid:0.8.0
            echo "  GROBID 启动中... 等 30 秒让模型加载"
            sleep 5
        fi
    fi
else
    echo "  Docker 未安装，跳过 GROBID。请手动："
    echo "  - 安装 Docker: curl -fsSL https://get.docker.com | sh"
    echo "  - 或下载 GROBID: https://github.com/kermitt2/grobid/releases"
fi

# 6. systemd 服务 (optional)
read -p "是否安装 systemd 服务 (生产模式)? [y/N] " yn
if [[ "$yn" =~ ^[Yy]$ ]]; then
    cat > /etc/systemd/system/rlpe.service << 'UNIT'
[Unit]
Description=RLPE Backend
After=network.target docker.service
Wants=docker.service

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=/opt/rlpe
Environment="PATH=/root/miniconda3/envs/rlpe/bin:/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="RLPE_HOST=0.0.0.0"
Environment="RLPE_PORT=8000"
ExecStart=/root/miniconda3/envs/rlpe/bin/python -m uvicorn rlpe.api.app:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT
    systemctl daemon-reload
    echo "  systemd 服务已创建。运行 'systemctl enable --now rlpe' 启动"
fi

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  部署完成！${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "下一步："
echo "  1. 配置 API Key:"
echo "     cp .env.example .env && nano .env"
echo ""
echo "  2. 启动后端 (开发):"
echo "     conda activate rlpe"
echo "     bash start_dev.sh web"
echo ""
echo "  3. 启动后端 (生产):"
echo "     systemctl enable --now rlpe"
echo ""
echo "  4. 验证:"
echo "     curl http://localhost:8000/health"
echo "     curl http://localhost:8070/api/isalive   # GROBID"
