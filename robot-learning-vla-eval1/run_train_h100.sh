#!/bin/bash
# Run SmolVLA eval1 training in the background.
# Survives SSH disconnection and laptop sleep/close.
# Usage: bash run_train_h100.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/train_h100_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/train_h100.pid"

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "[dry-run] Would launch: python scripts/train_eval1_smolvla_h100.py"
    echo "[dry-run] Log file    : $SCRIPT_DIR/$LOG_FILE"
    exit 0
fi

# Abort if a training job is already running
if [[ -f "$PID_FILE" ]]; then
    OLD_PID="$(cat "$PID_FILE")"
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[error] Training already running (PID $OLD_PID). Stop it first:"
        echo "         kill $OLD_PID"
        exit 1
    fi
fi

echo "[train] Starting training — output will NOT appear here."
echo "[train] Log  : $SCRIPT_DIR/$LOG_FILE"
echo "[train] Tail : tail -f $SCRIPT_DIR/$LOG_FILE"
echo ""

# nohup + setsid: double protection against SIGHUP when SSH session closes.
# 'disown' removes it from this shell's job table so closing the terminal
# doesn't send SIGHUP either.
nohup setsid python scripts/train_eval1_smolvla_h100.py \
    >> "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
disown "$TRAIN_PID"

echo "$TRAIN_PID" > "$PID_FILE"

echo "[train] PID  : $TRAIN_PID  (saved to $PID_FILE)"
echo "[train] Stop : kill $TRAIN_PID"
echo "[train] GPU  : watch -n2 nvidia-smi"
