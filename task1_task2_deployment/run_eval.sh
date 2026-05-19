#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EVAL_ROOT/.." && pwd)"

for bin_dir in \
  "$HOME/.conda/envs/lerobot/bin" \
  "$HOME/miniforge3/envs/lerobot/bin" \
  "$HOME/miniconda3/envs/lerobot/bin" \
  "$HOME/anaconda3/envs/lerobot/bin" \
  "$EVAL_ROOT/.venv/bin" \
  "$REPO_ROOT/.venv/bin" \
  "$REPO_ROOT/robot-learning-vla/.venv/bin"; do
  if [[ -d "$bin_dir" ]]; then
    export PATH="$bin_dir:$PATH"
  fi
done

PYTHON_BIN=""
if [[ -n "${LEROBOT_PYTHON:-}" ]]; then
  PYTHON_BIN="$LEROBOT_PYTHON"
else
  for python_bin in \
    "$HOME/.conda/envs/lerobot/bin/python" \
    "$HOME/miniforge3/envs/lerobot/bin/python" \
    "$HOME/miniconda3/envs/lerobot/bin/python" \
    "$HOME/anaconda3/envs/lerobot/bin/python" \
    "$EVAL_ROOT/.venv/bin/python" \
    "$REPO_ROOT/.venv/bin/python" \
    "$REPO_ROOT/robot-learning-vla/.venv/bin/python"; do
    if [[ -x "$python_bin" ]]; then
      PYTHON_BIN="$python_bin"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]] && command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/run_eval.py" "$@"
