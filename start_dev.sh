#!/bin/bash
# ============================================================================
# RLPE Backend Launcher (conda env: rlpe)
# 用法：bash start_dev.sh [web|cli|api|grobid|test-api|shell]
# ============================================================================
set -e

CONDA_ENV="${RLPE_CONDA_ENV:-rlpe}"
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

# ------------------------------------------------------------
# Locate conda installation (works on Linux + macOS + WSL)
# ------------------------------------------------------------
_conda_init() {
    local conda_sh=""
    # Standard install locations, in priority order
    local candidates=(
        "${CONDA_SH}"                                                # user override
        "$HOME/miniconda3/etc/profile.d/conda.sh"
        "$HOME/anaconda3/etc/profile.d/conda.sh"
        "$HOME/miniforge3/etc/profile.d/conda.sh"
        "$HOME/mambaforge/etc/profile.d/conda.sh"
        "/opt/conda/etc/profile.d/conda.sh"
        "/opt/miniconda3/etc/profile.d/conda.sh"
        "/opt/anaconda3/etc/profile.d/conda.sh"
        "/usr/local/anaconda3/etc/profile.d/conda.sh"
        "/usr/local/miniconda3/etc/profile.d/conda.sh"
        "/root/miniconda3/etc/profile.d/conda.sh"
        "/root/anaconda3/etc/profile.d/conda.sh"
        # macOS arm64 (M1/M2)
        "/opt/homebrew/anaconda3/etc/profile.d/conda.sh"
        "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
    )
    for f in "${candidates[@]}"; do
        if [ -n "$f" ] && [ -f "$f" ]; then
            conda_sh="$f"
            break
        fi
    done
    if [ -n "$conda_sh" ]; then
        # shellcheck disable=SC1090
        source "$conda_sh"
    else
        echo "ERROR: conda not found. Tried:" >&2
        for f in "${candidates[@]:1}"; do
            echo "  $f" >&2
        done
        echo "" >&2
        echo "Set CONDA_SH=/path/to/conda.sh or install conda first." >&2
        exit 1
    fi
}
_conda_init

# Verify env exists
if ! conda env list | grep -qE "^\s*${CONDA_ENV}\s"; then
    echo "ERROR: conda env '${CONDA_ENV}' does not exist." >&2
    echo "  Create it with:  conda env create -f ${PROJECT_ROOT}/environment.yml" >&2
    echo "  Or set RLPE_CONDA_ENV=<existing-env-name>" >&2
    exit 1
fi

conda activate "${CONDA_ENV}"

# Load .env (silently skip if missing)
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${PROJECT_ROOT}/.env"
    set +a
fi

MODE="${1:-web}"
echo "============================================================"
echo "  RLPE Backend Launcher"
echo "  conda env  : ${CONDA_ENV}"
echo "  python     : $(python --version 2>&1)"
echo "  platform   : $(uname -srm 2>/dev/null || echo '?')"
echo "  project    : ${PROJECT_ROOT}"
echo "  mode       : ${MODE}"
echo "  API key    : ${ANTHROPIC_API_KEY:0:8}..."
echo "============================================================"

case "${MODE}" in
    web)
        exec python "${PROJECT_ROOT}/run_web_server.py"
        ;;
    api)
        HOST="${RLPE_HOST:-0.0.0.0}"
        PORT="${RLPE_PORT:-8000}"
        echo "Listening on http://${HOST}:${PORT}"
        exec python -m uvicorn rlpe.api.app:app \
            --host "${HOST}" --port "${PORT}" --log-level info
        ;;
    cli)
        PDF_DIR="${2:-data/pdfs}"
        WORK_DIR="${3:-work}"
        shift 3 2>/dev/null || shift $#
        exec python scripts/run_pipeline.py \
            --pdf-dir "${PDF_DIR}" \
            --work-dir "${WORK_DIR}" \
            --use-gemma4 \
            --llm-backend MiniMax \
            "$@"
        ;;
    grobid)
        if [ -d "${PROJECT_ROOT}/tools/grobid" ]; then
            cd "${PROJECT_ROOT}/tools/grobid"
            exec ./gradlew run
        else
            echo "GROBID not installed. Choose one:" >&2
            echo "  1) Docker: docker run -d -p 8070:8070 --name grobid lfoppiano/grobid:0.8.0" >&2
            echo "  2) Manual: wget https://github.com/kermitt2/grobid/releases/download/0.8.0/grobid-0.8.0.zip" >&2
            echo "            unzip -d tools/" >&2
            exit 1
        fi
        ;;
    test-api)
        exec python scripts/test_MiniMax_api.py
        ;;
    shell)
        echo "Entering ${CONDA_ENV} shell. PYTHONPATH=${PYTHONPATH}"
        exec "${SHELL:-bash}"
        ;;
    install)
        echo "Installing conda env '${CONDA_ENV}' from environment.yml ..."
        conda env create -f "${PROJECT_ROOT}/environment.yml" || \
        conda env update -f "${PROJECT_ROOT}/environment.yml" --prune
        echo "Done. Activate with: conda activate ${CONDA_ENV}"
        ;;
    *)
        cat <<USAGE
用法: $0 {web|api|cli|grobid|test-api|shell|install}
  web       - 启动 Web 服务 (默认)
  api       - 启动纯 API 服务 (uvicorn), 监听 0.0.0.0:8000
  cli       - 启动 CLI 批处理（需传 --pdf-dir --work-dir 等）
  grobid    - 启动 GROBID 服务
  test-api  - 测试 MiniMax M3 API 接入
  shell     - 激活 dev shell
  install   - 创建/更新 conda env (从 environment.yml)
USAGE
        exit 1
        ;;
esac
