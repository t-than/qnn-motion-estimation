#!/usr/bin/env bash
# Week 1 - system setup for Arch Linux.
# Installs build tools, media tools and a Python virtual environment with
# PyTorch (CUDA) for later weeks. Safe to re-run.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"

echo "==> Installing system packages (needs sudo)"
# base-devel/cmake/git: build HM
# ffmpeg: convert .y4m to raw .yuv, trim sequences
# p7zip: UVG sequences ship as .7z
# aria2: fast, resumable downloads
# python: scripts and training
sudo pacman -S --needed --noconfirm \
    base-devel cmake git ffmpeg p7zip unzip aria2 python python-pip

echo "==> Checking NVIDIA driver"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo "WARNING: nvidia-smi not found. Install the driver first, e.g."
    echo "         sudo pacman -S nvidia-open nvidia-utils   (then reboot)"
    echo "         (nvidia-open-dkms for non-stock kernels; see the Arch wiki 'NVIDIA' page)"
fi

echo "==> Creating Python virtual environment at ${VENV_DIR}"
python -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip
# The default Linux torch wheel on PyPI bundles the CUDA runtime;
# only the NVIDIA driver is needed on the system.
pip install numpy scipy pandas matplotlib tqdm torch torchvision bjontegaard

echo "==> Verifying PyTorch sees the GPU"
python - <<'EOF'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
EOF

echo
echo "Done. Activate the environment in new shells with:"
echo "  source ${VENV_DIR}/bin/activate"
