@echo off
REM Windows 批处理脚本 - 启动 RLPE Web 服务器
REM Start RLPE Web Server Script for Windows

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║  🔬 RLPE Web Server Launcher                               ║
echo ║  放射虫图版提取系统 - Web 界面启动器                         ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到 Python。请先安装 Python 3.11 或更高版本。
    echo ❌ Error: Python not found. Please install Python 3.11+
    pause
    exit /b 1
)

REM 启动服务器
echo ⏳ 正在启动服务器... (Starting server...)
echo.
python run_web_server.py

pause
