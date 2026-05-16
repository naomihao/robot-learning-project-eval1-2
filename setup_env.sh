#!/usr/bin/env bash
# =============================================================================
#  setup_env.sh — LeRobot / SmolVLA environment setup
#
#  Follows: https://huggingface.co/docs/lerobot/installation
#  Run once on a fresh server before training.
#
#  Usage
#  -----
#    chmod +x setup_env.sh
#    ./setup_env.sh
# =============================================================================

set -euo pipefail

CONDA_ENV_NAME="lerobot"
LEROBOT_DIR="${HOME}/lerobot"

echo "═══════════════════════════════════════════════════════"
echo "  LeRobot environment setup"
echo "  Follows: https://huggingface.co/docs/lerobot/installation"
echo "═══════════════════════════════════════════════════════"

# ── Step 1: conda ─────────────────────────────────────────────────────────────
echo ""
echo "[1/4] Initialising conda..."

CONDA_BASE=""
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
    echo "❌  conda not found. Install miniforge first:"
    echo "    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-\$(uname)-\$(uname -m).sh"
    echo "    bash Miniforge3-\$(uname)-\$(uname -m).sh"
    exit 1
fi

# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
echo "      ✓ conda found at: $CONDA_BASE"

if ! conda env list | grep -q "^${CONDA_ENV_NAME}\s"; then
    echo "      Creating conda env '${CONDA_ENV_NAME}' (Python 3.12)..."
    conda create -y -n "${CONDA_ENV_NAME}" python=3.12
    echo "      ✓ Environment created"
else
    echo "      ✓ Environment '${CONDA_ENV_NAME}' already exists"
fi

conda activate "${CONDA_ENV_NAME}"
echo "      ✓ Activated: $(python --version)  |  $(which python)"

# ── Step 2: ffmpeg 7.X ────────────────────────────────────────────────────────
echo ""
echo "[2/4] Checking ffmpeg (requires 7.X; 8.X not yet supported)..."

FFMPEG_OK=false
if command -v ffmpeg &>/dev/null; then
    FFMPEG_VER=$(ffmpeg -version 2>&1 | head -1 | grep -oP 'version \K[0-9]+' | head -1 || echo "0")
    if [[ "$FFMPEG_VER" == "7" ]]; then
        FFMPEG_OK=true
        echo "      ✓ ffmpeg 7.X already installed"
    else
        echo "      ⚠️  ffmpeg ${FFMPEG_VER}.X found — reinstalling 7.1.1"
    fi
fi

if [[ "$FFMPEG_OK" == false ]]; then
    conda install -y ffmpeg=7.1.1 -c conda-forge
    echo "      ✓ ffmpeg 7.1.1 installed"
fi

# ── Step 3: lerobot[smolvla] from source ──────────────────────────────────────
echo ""
echo "[3/4] Setting up LeRobot from source..."

if [[ ! -d "$LEROBOT_DIR" ]]; then
    echo "      Cloning https://github.com/huggingface/lerobot ..."
    git clone https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
    echo "      ✓ Cloned to $LEROBOT_DIR"
else
    echo "      Repo exists — pulling latest (git pull)..."
    git -C "$LEROBOT_DIR" pull
    echo "      ✓ Up to date"
fi

pip install -e "${LEROBOT_DIR}[smolvla]" --quiet
echo "      ✓ lerobot[smolvla] installed (editable)"

# ── Step 4: credentials ───────────────────────────────────────────────────────
echo ""
echo "[4/4] Setting up credentials..."

if ! python -c "import wandb" 2>/dev/null; then
    pip install --quiet wandb
fi

# ── W&B ───────────────────────────────────────────────────────────────────────
if grep -q "WANDB_API_KEY" "${HOME}/.bashrc" 2>/dev/null; then
    echo "      ✓ WANDB_API_KEY already in ~/.bashrc"
else
    echo ""
    echo "      Get your W&B API key at: https://wandb.ai/authorize"
    read -r -p "      Paste WANDB_API_KEY: " WANDB_KEY
    if [[ -n "$WANDB_KEY" ]]; then
        echo "" >> "${HOME}/.bashrc"
        echo "# W&B API key (added by setup_env.sh)" >> "${HOME}/.bashrc"
        echo "export WANDB_API_KEY=${WANDB_KEY}" >> "${HOME}/.bashrc"
        export WANDB_API_KEY="${WANDB_KEY}"
        echo "      ✓ WANDB_API_KEY saved to ~/.bashrc"
    else
        echo "      ⚠️  No key entered — skipping W&B. Set manually later:"
        echo "          echo \"export WANDB_API_KEY=<key>\" >> ~/.bashrc"
    fi
fi

# Make sure the key is exported in the current shell too
# shellcheck source=/dev/null
source "${HOME}/.bashrc" 2>/dev/null || true

# ── HuggingFace ───────────────────────────────────────────────────────────────
if grep -q "HF_TOKEN" "${HOME}/.bashrc" 2>/dev/null; then
    echo "      ✓ HF_TOKEN already in ~/.bashrc"
else
    echo ""
    echo "      Get your HF token at: https://huggingface.co/settings/tokens"
    echo "      (needs Write permission to push checkpoints)"
    read -r -p "      Paste HF_TOKEN: " HF_KEY
    if [[ -n "$HF_KEY" ]]; then
        echo "" >> "${HOME}/.bashrc"
        echo "# HuggingFace token (added by setup_env.sh)" >> "${HOME}/.bashrc"
        echo "export HF_TOKEN=${HF_KEY}" >> "${HOME}/.bashrc"
        export HF_TOKEN="${HF_KEY}"
        echo "      ✓ HF_TOKEN saved to ~/.bashrc"
    else
        echo "      ⚠️  No token entered — skipping HF. Set manually later:"
        echo "          echo \"export HF_TOKEN=<token>\" >> ~/.bashrc"
    fi
fi

source "${HOME}/.bashrc" 2>/dev/null || true

# ── Verify both ───────────────────────────────────────────────────────────────
echo ""
if python -c "import wandb; assert wandb.api.api_key" 2>/dev/null; then
    echo "      ✓ W&B authenticated"
else
    echo "      ⚠️  W&B key not detected — check WANDB_API_KEY in ~/.bashrc"
fi

if python -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
    HF_USER=$(python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])")
    echo "      ✓ HuggingFace authenticated as: $HF_USER"
else
    echo "      ⚠️  HF token not detected — check HF_TOKEN in ~/.bashrc"
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ✅  Setup complete"
echo "  Keys saved to ~/.bashrc — no login needed on future runs"
echo "  Start training:  ./run_train.sh"
echo "═══════════════════════════════════════════════════════"
