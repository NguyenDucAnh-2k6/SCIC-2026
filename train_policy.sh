#!/usr/bin/env bash
# ====================================================================
#  ClassTwin: Pedagogical Policy Training (CQL & Inverse RL)
# ====================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
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
    echo "[ERROR] Python không tìm thấy trên hệ thống!" >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    echo "[INFO] Huấn luyện CQL Policy mặc định (Explicit Reward, 40 episodes, 150 steps)..."
    exec "$PYTHON_CMD" scripts/train_policy.py --reward-mode explicit --episodes 40 --train-steps 150
else
    exec "$PYTHON_CMD" scripts/train_policy.py "$@"
fi
