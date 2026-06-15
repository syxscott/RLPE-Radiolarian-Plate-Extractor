@echo off
REM ============================================================================
REM RLPE Backend Launcher (conda env: dev)
REM 用法: start_dev.bat [web|cli|api|grobid|test-api|shell]
REM ============================================================================
setlocal enabledelayedexpansion

set "CONDA_ENV=dev"
set "PROJECT_ROOT=%~dp0"
set "PYTHONPATH=%PROJECT_ROOT%src;%PYTHONPATH%"

REM 激活 conda dev 环境
call conda activate %CONDA_ENV% 2>/dev/null
if errorlevel 1 (
    echo X conda activate failed.  Ensure conda is in PATH.
    echo   Try: set PATH=D:\Program Files\ananconda3;%PATH%
    pause
    exit /b 1
)

REM 加载 .env
if exist "%PROJECT_ROOT%.env" (
    for /f "usebackq tokens=1,2 delims==" %%a in ("%PROJECT_ROOT%.env") do (
        if not "%%a"=="" set "%%a=%%b"
    )
)

set "MODE=%~1"
if "%MODE%"=="" set "MODE=web"

echo ============================================================
echo   RLPE Backend Launcher
echo   conda env : %CONDA_ENV%
echo   python    : 
call python --version
echo   mode      : %MODE%
echo   ANTHROPIC_API_KEY: %ANTHROPIC_API_KEY:~0,8%...
echo ============================================================

if /i "%MODE%"=="web" (
    python "%PROJECT_ROOT%run_web_server.py"
    goto :eof
)

if /i "%MODE%"=="api" (
    python -m uvicorn rlpe.api.app:app --host 0.0.0.0 --port 8000 --log-level info
    goto :eof
)

if /i "%MODE%"=="cli" (
    set "PDF_DIR=%~2"
    if "%PDF_DIR%"=="" set "PDF_DIR=data\pdfs"
    set "WORK_DIR=%~3"
    if "%WORK_DIR%"=="" set "WORK_DIR=work"
    python scripts\run_pipeline.py --pdf-dir "%PDF_DIR%" --work-dir "%WORK_DIR%" --use-gemma4 --llm-backend MiniMax %4 %5 %6 %7 %8 %9
    goto :eof
)

if /i "%MODE%"=="grobid" (
    if exist "%PROJECT_ROOT%tools\grobid\gradlew.bat" (
        cd /d "%PROJECT_ROOT%tools\grobid"
        call gradlew.bat run
    ) else (
        echo X GROBID not found.
        echo   Install Docker: docker run -d -p 8070:8070 lfoppiano/grobid:0.8.0
        echo   Or download GROBID:
        echo   https://github.com/kermitt2/grobid/releases
        pause
    )
    goto :eof
)

if /i "%MODE%"=="test-api" (
    python scripts\test_MiniMax_api.py
    goto :eof
)

if /i "%MODE%"=="shell" (
    cmd /k "set PYTHONPATH=%PYTHONPATH%"
    goto :eof
)

echo Usage: %~nx0 {web^|cli^|api^|grobid^|test-api^|shell}
echo   web       - Start Web server ^(default^)
echo   api       - Start pure API server
echo   cli       - Start CLI batch ^(needs --pdf-dir --work-dir^)
echo   grobid    - Start GROBID service
echo   test-api  - Test MiniMax M3 API
echo   shell     - Open dev shell
pause
