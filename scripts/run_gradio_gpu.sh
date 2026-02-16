#!/usr/bin/env bash
# Launch the Gradio web UI with GPU support (GTX 980 Ti / sm_52).
#
# The default venv uses CPU-only PyTorch 2.8.0. The GTX 980 Ti requires
# PyTorch 2.3.0+cu121 (last version with sm_52 support). This script
# swaps in the CUDA build, runs the Gradio app, then restores the CPU build.
#
# Usage:
#   ./scripts/run_gradio_gpu.sh [--share] [--port 7860]

set -euo pipefail
cd "$(dirname "$0")/.."

VENV_PYTHON=".venv/bin/python"
TORCH_CPU="torch==2.8.0+cpu"
TORCHVISION_CPU="torchvision==0.23.0+cpu"
TORCH_CUDA="torch==2.3.0+cu121"
TORCHVISION_CUDA="torchvision==0.18.0+cu121"
CPU_INDEX="https://download.pytorch.org/whl/cpu"
CUDA_INDEX="https://download.pytorch.org/whl/cu121"

restore_cpu() {
    echo ""
    echo "Restoring CPU PyTorch..."
    uv pip install "$TORCH_CPU" "$TORCHVISION_CPU" \
        --index-url "$CPU_INDEX" \
        --python "$VENV_PYTHON" --quiet
    uv sync --quiet
    echo "Restored to $TORCH_CPU"
}

echo "Installing CUDA PyTorch for GPU support..."
uv pip install "$TORCH_CUDA" "$TORCHVISION_CUDA" \
    --index-url "$CUDA_INDEX" \
    --python "$VENV_PYTHON" --quiet

# Restore CPU PyTorch on exit (Ctrl-C, error, or normal exit)
trap restore_cpu EXIT

echo "Launching Gradio with GPU support..."
$VENV_PYTHON uvq_web.py "$@"
