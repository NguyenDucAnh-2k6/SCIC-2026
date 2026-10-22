#!/usr/bin/env bash
# ====================================================================
#  ClassTwin: Classroom Digital Twin (SCIC 2026)
#  Training Launcher (Full Pipeline & Policy Training) (Linux / macOS)
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
    echo "  ./train.sh                  : Hiện menu lựa chọn tương tác"
    echo "  ./train.sh smoke / fast     : Chạy kiểm thử nhanh trên CPU (~10 giây)"
    echo "  ./train.sh policy           : Huấn luyện chính sách CQL (Explicit)"
    echo "  ./train.sh irl              : Huấn luyện chính sách MaxEnt Inverse RL"
    echo "  ./train.sh eval             : Chạy benchmark đánh giá mô hình"
    echo "  ./train.sh [pipeline args]  : Truyền tham số trực tiếp vào pipeline.py"
    echo ""
    echo "  Ví dụ:"
    echo "    ./train.sh --num-shards 5 --estimator-epochs 3"
    echo "    ./train.sh --skip-preprocess --policy-episodes 50"
    echo "===================================================================="
}

# Handle arguments
if [[ $# -gt 0 ]]; then
    case "$1" in
        help|--help|-h)
            show_help
            exit 0
            ;;
        smoke|fast|--smoke|--fast)
            echo "[INFO] Bắt đầu Fast Smoke Test (~10s)..."
            exec "$PYTHON_CMD" scripts/pipeline.py --num-shards 2 --max-students 50 --estimator-epochs 1 --policy-episodes 5 --policy-steps 10
            ;;
        policy|cql|--policy)
            shift
            echo "[INFO] Bắt đầu Huấn luyện Chính sách CQL (Explicit Reward)..."
            exec "$PYTHON_CMD" scripts/train_policy.py --reward-mode explicit --episodes 40 --train-steps 150 "$@"
            ;;
        irl|policy-irl|--irl)
            shift
            echo "[INFO] Bắt đầu Huấn luyện Chính sách MaxEnt Inverse RL (IRL)..."
            exec "$PYTHON_CMD" scripts/train_policy.py --reward-mode irl --episodes 40 --train-steps 150 "$@"
            ;;
        eval|evaluate)
            echo "[INFO] Bắt đầu Đánh giá Models & Policy Entropy..."
            exec "$PYTHON_CMD" scripts/evaluate_models.py
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
    echo "  [1] Full Pipeline    - Toàn bộ chu trình (DKT + CQL + Manifest)"
    echo "  [2] Fast Smoke Test  - Kiểm thử nhanh CPU (2 shards, 1 epoch, ~10s)"
    echo "  [3] Policy: CQL      - Huấn luyện lại CQL (Explicit Reward)"
    echo "  [4] Policy: MaxEnt   - Huấn luyện lại Inverse RL (Latent Reward)"
    echo "  [5] Evaluate Models  - Đánh giá Entropy, Calibration & Policy"
    echo "  [0] Thoát"
    echo "===================================================================="
    read -rp "Lựa chọn của bạn [1-5, mặc định 1]: " CHOICE
    CHOICE="${CHOICE:-1}"

    case "$CHOICE" in
        1)
            echo "[INFO] Bắt đầu Full Pipeline..."
            exec "$PYTHON_CMD" scripts/pipeline.py
            ;;
        2)
            echo "[INFO] Bắt đầu Fast Smoke Test (~10s)..."
            exec "$PYTHON_CMD" scripts/pipeline.py --num-shards 2 --max-students 50 --estimator-epochs 1 --policy-episodes 5 --policy-steps 10
            ;;
        3)
            echo "[INFO] Bắt đầu Huấn luyện Chính sách CQL (Explicit Reward)..."
            exec "$PYTHON_CMD" scripts/train_policy.py --reward-mode explicit --episodes 40 --train-steps 150
            ;;
        4)
            echo "[INFO] Bắt đầu Huấn luyện Chính sách MaxEnt Inverse RL (IRL)..."
            exec "$PYTHON_CMD" scripts/train_policy.py --reward-mode irl --episodes 40 --train-steps 150
            ;;
        5)
            echo "[INFO] Bắt đầu Đánh giá Models & Policy Entropy..."
            exec "$PYTHON_CMD" scripts/evaluate_models.py
            ;;
        0)
            echo "Tạm biệt!"
            exit 0
            ;;
        *)
            echo "[WARN] Lựa chọn không hợp lệ, chạy mặc định Full Pipeline..."
            exec "$PYTHON_CMD" scripts/pipeline.py
            ;;
    esac
else
    # Non-interactive / CI default
    echo "[INFO] Non-interactive mode: Bắt đầu Full Pipeline..."
    exec "$PYTHON_CMD" scripts/pipeline.py
fi
