#!/usr/bin/env bash
# Set up NVIDIA Alpamayo 2 Edge inside WSL2 (or any Linux box) for AlpaSandbox.
# Usage: bash scripts/setup_wsl.sh [install_dir]   (default: ~/alpamayo2-edge)
set -euo pipefail
DIR="${1:-$HOME/alpamayo2-edge}"

echo "==> Checking the GPU (needs the NVIDIA *Windows* driver when inside WSL)"
if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found. Update the NVIDIA driver on Windows; do not install one inside WSL." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "==> Installing uv (Python environment manager) if needed"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "==> Getting NVlabs/alpamayo2-edge into $DIR"
if [ ! -d "$DIR/.git" ]; then git clone https://github.com/NVlabs/alpamayo2-edge "$DIR"; fi
cd "$DIR"
export UV_PROJECT_ENVIRONMENT=.venv
uv sync --locked

echo "==> Hugging Face login (accept the model terms at https://huggingface.co/nvidia/Alpamayo2-Edge first)"
.venv/bin/hf auth whoami >/dev/null 2>&1 || .venv/bin/hf auth login

echo "==> Checking the model imports"
.venv/bin/python -c "import torch, alpamayo2_edge; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

cat <<MSG

Done. On Windows, run:
  alpasandbox doctor --model edge --runtime wsl --wsl-distro ${WSL_DISTRO_NAME:-Ubuntu-24.04} --wsl-workdir $DIR --wsl-venv .venv
  alpasandbox app    --model edge --runtime wsl --wsl-distro ${WSL_DISTRO_NAME:-Ubuntu-24.04} --wsl-workdir $DIR --wsl-venv .venv --backend carla
MSG
