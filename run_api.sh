#!/usr/bin/env bash
# ====================================================================
#  ClassTwin: Classroom Digital Twin (SCIC 2026)
#  API Service & Interactive Web Dashboard Launcher (Linux / macOS)
# ====================================================================

set -e

# Resolve repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Setup PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/src:${PYTHONPATH:-}"

# Detect Python executable
PYTHON_CMD=""
if [[ -f ".venv/bin/python" ]]; then
    PYTHON_CMD=".venv/bin/python"
elif [[ -f "venv/bin/python" ]]; then
    PYTHON_CMD="venv/bin/python"
elif [[ -f ".venv/Scripts/python.exe" ]]; then
    PYTHON_CMD=".venv/Scripts/python.exe"
elif [[ -f "venv/Scripts/python.exe" ]]; then
    PYTHON_CMD="venv/Scripts/python.exe"
fi

if [[ -z "$PYTHON_CMD" ]]; then
    for candidate in python3 python python.exe py; do
        if command -v "$candidate" &>/dev/null; then
            if "$candidate" -c "import fastapi" &>/dev/null 2>&1; then
                PYTHON_CMD="$candidate"
                break
            elif [[ -z "$PYTHON_CMD" ]]; then
                PYTHON_CMD="$candidate"
            fi
        fi
    done
fi

if [[ -z "$PYTHON_CMD" ]]; then
    echo "[ERROR] Python 3.10+ không tìm thấy trên hệ thống! Vui lòng cài đặt Python." >&2
    exit 1
fi

echo "===================================================================="
echo "  ClassTwin: Khởi chạy Digital Twin Service & Dashboard"
echo "  Python: $PYTHON_CMD"
echo "  Working Dir: $(pwd)"
echo "===================================================================="

show_help() {
    echo "===================================================================="
    echo "  ClassTwin run_api Launcher - Hướng dẫn sử dụng:"
    echo "===================================================================="
    echo "  ./run_api.sh                  : Chạy mặc định với chế độ Dev (--dev)"
    echo "  ./run_api.sh --dev            : Chạy kèm Vite dev server (port 3000)"
    echo "  ./run_api.sh --port 8000      : Chỉ chạy FastAPI backend (phục vụ frontend/dist)"
    echo "  ./run_api.sh --no-browser     : Không tự động mở trình duyệt"
    echo "  ./run_api.sh --host 0.0.0.0   : Đổi địa chỉ host lắng nghe"
    echo "===================================================================="
}

if [[ $# -gt 0 ]]; then
    case "$1" in
        help|--help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "[INFO] Chạy với tham số: $*"
            echo "===================================================================="
            exec "$PYTHON_CMD" scripts/run_api.py "$@"
            ;;
    esac
fi

# If no arguments provided, default to --dev
echo "[INFO] Chạy chế độ Development mặc định (--dev)"
echo "       FastAPI Backend : http://localhost:8000"
echo "       React Dashboard : http://localhost:3000"
echo "===================================================================="
exec "$PYTHON_CMD" scripts/run_api.py --dev
