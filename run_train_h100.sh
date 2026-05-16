#!/usr/bin/env bash
# =============================================================================
#  run_train_h100.sh — SmolVLA eval1 training launcher  [H100 edition]
#
#  GPU    : H100 (80 GB) · bf16 · batch 128 · 7 000 steps · ~20-40 min
#  Assumes setup_env.sh has already been run.
#  Launches training detached — survives closing VSCode / SSH / laptop lid.
#
#  Usage
#  -----
#    ./run_train_h100.sh              # start training
#    ./run_train_h100.sh --status     # check if still running
#    ./run_train_h100.sh --log        # tail live log
#    ./run_train_h100.sh --kill       # stop training
#
#  Monitor loss
#  ------------
#    W&B  : https://wandb.ai/<your-entity>/smolvla_banana_eval1
#    Local: ./run_train_h100.sh --log
# =============================================================================

set -euo pipefail

CONDA_ENV_NAME="lerobot"
LEROBOT_DIR="${HOME}/lerobot"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_eval1_h100.py"
LOG_FILE="${SCRIPT_DIR}/train_eval1_h100.log"
PID_FILE="${SCRIPT_DIR}/train_eval1_h100.pid"

# ── Conda activation ──────────────────────────────────────────────────────────
_activate_conda() {
    local CONDA_BASE=""
    for candidate in \
        "${HOME}/miniforge3" \
        "${HOME}/miniconda3" \
        "${HOME}/anaconda3" \
        "/opt/conda" \
        "/opt/miniforge3" \
        "/usr/local/miniforge3"
    do
        if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
            CONDA_BASE="$candidate"
            break
        fi
    done

    if [[ -z "$CONDA_BASE" ]]; then
        echo "❌  conda not found. Run ./setup_env.sh first."
        exit 1
    fi

    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
}

# ── Subcommands ───────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--status" ]]; then
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            ELAPSED=$(ps -o etime= -p "$PID" 2>/dev/null | tr -d ' ' || echo "unknown")
            echo "✅  Training RUNNING  |  PID $PID  |  Elapsed $ELAPSED"
            echo "    W&B : https://wandb.ai/home  →  smolvla_banana_eval1"
            echo "    Log : $LOG_FILE"
        else
            echo "⚠️   PID $PID not found — finished or crashed."
            echo "    Check: tail -50 $LOG_FILE"
        fi
    else
        echo "ℹ️   Not started yet. Run ./run_train.sh to begin."
    fi
    exit 0
fi

if [[ "${1:-}" == "--log" ]]; then
    [[ -f "$LOG_FILE" ]] && tail -f "$LOG_FILE" || echo "No log yet: $LOG_FILE"
    exit 0
fi

if [[ "${1:-}" == "--kill" ]]; then
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID" && echo "🛑  Stopped PID $PID"
        else
            echo "ℹ️   PID $PID not running."
        fi
        rm -f "$PID_FILE"
    else
        echo "ℹ️   No PID file found."
    fi
    exit 0
fi

# ── Pre-flight checks ─────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════"
echo "  SmolVLA eval1 · H100 · bf16 — launching training"
echo "═══════════════════════════════════════════════════════"

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "❌  train_eval1_h100.py not found at: $TRAIN_SCRIPT"
    exit 1
fi

if [[ ! -d "$LEROBOT_DIR" ]]; then
    echo "❌  LeRobot not found at $LEROBOT_DIR"
    echo "    Run ./setup_env.sh first."
    exit 1
fi

if [[ -f "$PID_FILE" ]]; then
    EXISTING_PID=$(cat "$PID_FILE")
    if kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "⚠️   Already running (PID $EXISTING_PID). Use --kill to stop first."
        exit 1
    fi
    rm -f "$PID_FILE"
fi

# ── Activate conda env ────────────────────────────────────────────────────────
_activate_conda
echo "      ✓ conda env '${CONDA_ENV_NAME}' active"

# ── Rotate old log ────────────────────────────────────────────────────────────
if [[ -f "$LOG_FILE" ]]; then
    mv "$LOG_FILE" "${LOG_FILE}.bak"
    echo "      Old log backed up → train_eval1_h100.log.bak"
fi

# ── Launch detached ───────────────────────────────────────────────────────────
cd "$LEROBOT_DIR"

nohup python "$TRAIN_SCRIPT" 2>&1 | tee "$LOG_FILE" &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"

sleep 3

if kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  ✅  Training started — safe to close VSCode/laptop"
    echo ""
    echo "  GPU     : H100 · bf16 · batch 128 · ~7 000 steps"
    echo "  PID     : $TRAIN_PID"
    echo "  Log     : $LOG_FILE"
    echo "  W&B     : https://wandb.ai/home  →  smolvla_banana_eval1"
    echo "  HF repo : https://huggingface.co/yukk1/test_eval1"
    echo "  Outputs : ${LEROBOT_DIR}/outputs/train/eval1_smolvla_h100"
    echo ""
    echo "  Commands:"
    echo "    Monitor log  :  ./run_train_h100.sh --log"
    echo "    Check status :  ./run_train_h100.sh --status"
    echo "    Stop         :  ./run_train_h100.sh --kill"
    echo "═══════════════════════════════════════════════════════"
else
    echo "❌  Process died immediately. Check:"
    echo "    tail -50 $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi
