#!/bin/bash
# Linux/macOS 启动脚本 - 启动 RLPE Web 服务器
# Start RLPE Web Server Script for Linux/macOS

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🔬 RLPE Web Server Launcher                               ║"
echo "║  放射虫图版提取系统 - Web 界面启动器                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 Python3"
    echo "❌ Error: Python3 not found"
    exit 1
fi

# 启动服务器
echo "⏳ 正在启动服务器... (Starting server...)"
echo ""
python3 run_web_server.py
