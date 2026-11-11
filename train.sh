#!/usr/bin/env bash
# ====================================================================
#  ClassTwin: Classroom Digital Twin (SCIC 2026)
#  Training Launcher (Single Master Pipeline: scripts/pipeline.py)
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
            if "$candidate" -c "import torch" &>/dev/null 2>&1; then
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

show_help() {
    echo "===================================================================="
    echo "  ClassTwin Training Launcher - Hướng dẫn sử dụng:"
    echo "===================================================================="
    echo "  ./train.sh                  : Hiển thị menu tương tác hoặc chạy full pipeline"
    echo "  ./train.sh smoke / fast     : Chạy kiểm thử nhanh trên CPU (~10 giây)"
    echo "  ./train.sh preprocess       : Chỉ tiền xử lý và trích xuất priors 7 datasets"
    echo "  ./train.sh estimator        : Chỉ huấn luyện GRU-DKT state estimator"
    echo "  ./train.sh policy           : Huấn luyện chính sách CQL (Explicit)"
    echo "  ./train.sh irl              : Huấn luyện chính sách MaxEnt Inverse RL"
    echo "  ./train.sh eval             : Chạy benchmark đánh giá mô hình"
    echo "  ./train.sh [pipeline args]  : Truyền tham số trực tiếp vào scripts/pipeline.py"
    echo "===================================================================="
}

if [[ $# -gt 0 ]]; then
    case "$1" in
        help|--help|-h)
            show_help
            exit 0
            ;;
        smoke|fast|--smoke|--fast)
            echo "[INFO] Bắt đầu Fast Smoke Test (~10s)..."
            exec "$PYTHON_CMD" scripts/pipeline.py --stage all --num-shards 2 --max-students 50 --estimator-epochs 1 --policy-episodes 5 --policy-steps 10
            ;;
        preprocess|--preprocess)
            echo "[INFO] Bắt đầu Tiền xử lý đa nguồn (7 Datasets)..."
            exec "$PYTHON_CMD" scripts/pipeline.py --stage preprocess
            ;;
        estimator|dkt|--estimator)
            echo "[INFO] Bắt đầu Huấn luyện GRU-DKT State Estimator..."
            exec "$PYTHON_CMD" scripts/pipeline.py --stage estimator
            ;;
        policy|cql|--policy)
            shift
            echo "[INFO] Bắt đầu Huấn luyện Chính sách CQL (Explicit Reward)..."
            exec "$PYTHON_CMD" scripts/pipeline.py --stage policy --reward-mode explicit --policy-episodes 40 --policy-steps 150 "$@"
            ;;
        irl|policy-irl|--irl)
            shift
            echo "[INFO] Bắt đầu Huấn luyện Chính sách MaxEnt Inverse RL (IRL)..."
            exec "$PYTHON_CMD" scripts/pipeline.py --stage policy --reward-mode irl --policy-episodes 40 --policy-steps 150 "$@"
            ;;
        eval|evaluate)
            echo "[INFO] Bắt đầu Đánh giá Models & Policy Entropy..."
            exec "$PYTHON_CMD" scripts/pipeline.py --stage evaluate
            ;;
        *)
            echo "[INFO] Chạy pipeline với tham số: $*"
            exec "$PYTHON_CMD" scripts/pipeline.py "$@"
            ;;
    esac
fi

# Interactive menu if run without arguments in terminal
if [ -t 0 ]; then
    echo "===================================================================="
    echo "         ClassTwin: Classroom Digital Twin (SCIC 2026)"
    echo "                 Training Execution Launcher"
    echo "===================================================================="
    echo "  [1] Full Pipeline    - Toàn bộ chu trình (Preprocess + DKT + CQL + Eval)"
    echo "  [2] Fast Smoke Test  - Kiểm thử nhanh CPU (2 shards, 1 epoch, ~10s)"
    echo "  [3] Preprocess Only  - Trích xuất Parquet shards & priors 7 datasets"
    echo "  [4] Estimator Only   - Huấn luyện mô hình GRU-DKT"
    echo "  [5] Policy: CQL      - Huấn luyện Chính sách CQL (Explicit Reward)"
    echo "  [6] Policy: MaxEnt   - Huấn luyện Chính sách MaxEnt Inverse RL (IRL)"
    echo "  [7] Evaluate Only    - Chạy benchmark đánh giá Calibration & Entropy"
    echo "  [0] Thoát"
    echo "===================================================================="
    read -rp "Lựa chọn của bạn [1-7, mặc định 1]: " CHOICE
    CHOICE="${CHOICE:-1}"

    case "$CHOICE" in
        1)
            echo "[INFO] Bắt đầu Full Pipeline..."
            exec "$PYTHON_CMD" scripts/pipeline.py --stage all
            ;;
        2)
            echo "[INFO] Bắt đầu Fast Smoke Test (~10s)..."
            exec "$PYTHON_CMD" scripts/pipeline.py --stage all --num-shards 2 --max-students 50 --estimator-epochs 1 --policy-episodes 5 --policy-steps 10
            ;;
        3)
            echo "[INFO] Bắt đầu Tiền xử lý đa nguồn..."
            exec "$PYTHON_CMD" scripts/pipeline.py --stage preprocess
            ;;
        4)
            echo "[INFO] Bắt đầu Huấn luyện GRU-DKT..."
            exec "$PYTHON_CMD" scripts/pipeline.py --stage estimator
            ;;
        5)
            echo "[INFO] Bắt đầu Huấn luyện Chính sách CQL (Explicit Reward)..."
            exec "$PYTHON_CMD" scripts/pipeline.py --stage policy --reward-mode explicit --policy-episodes 40 --policy-steps 150
            ;;
        6)
            echo "[INFO] Bắt đầu Huấn luyện Chính sách MaxEnt Inverse RL (IRL)..."
            exec "$PYTHON_CMD" scripts/pipeline.py --stage policy --reward-mode irl --policy-episodes 40 --policy-steps 150
            ;;
        7)
            echo "[INFO] Bắt đầu Đánh giá Models & Policy Entropy..."
            exec "$PYTHON_CMD" scripts/pipeline.py --stage evaluate
            ;;
        0)
            echo "Tạm biệt!"
            exit 0
            ;;
        *)
            echo "[WARN] Lựa chọn không hợp lệ, chạy mặc định Full Pipeline..."
            exec "$PYTHON_CMD" scripts/pipeline.py --stage all
            ;;
    esac
else
    # Non-interactive / CI default
    echo "[INFO] Non-interactive mode: Bắt đầu Full Pipeline..."
    exec "$PYTHON_CMD" scripts/pipeline.py --stage all
fi
