"""Shared fixtures for UVQ test suite."""

import os
import sys

import pytest
import torch

# Insert repo root into sys.path so that model modules (which use
# sys.path.append hacks to import video_reader) resolve correctly.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# Device fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device_cpu():
    return torch.device("cpu")


@pytest.fixture
def device_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


# ---------------------------------------------------------------------------
# Checkpoint path fixtures
# ---------------------------------------------------------------------------

UVQ1P5_CHECKPOINT_DIR = os.path.join(REPO_ROOT, "uvq1p5_pytorch", "checkpoints")
UVQ1P0_CHECKPOINT_DIR = os.path.join(REPO_ROOT, "uvq_pytorch", "checkpoint")
UVQ1P5_MLX_CHECKPOINT_DIR = os.path.join(REPO_ROOT, "uvq1p5_mlx", "checkpoints")


@pytest.fixture
def uvq1p5_checkpoint_dir():
    return UVQ1P5_CHECKPOINT_DIR


@pytest.fixture
def uvq1p0_checkpoint_dir():
    return UVQ1P0_CHECKPOINT_DIR


@pytest.fixture
def uvq1p5_mlx_checkpoint_dir():
    return UVQ1P5_MLX_CHECKPOINT_DIR


# ---------------------------------------------------------------------------
# Synthetic tensor fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_1080p_tensor():
    """Single-frame 1080p video tensor: (N, 1, 3, 1080, 1920)."""
    torch.manual_seed(42)
    return torch.randn(1, 1, 3, 1080, 1920)


@pytest.fixture
def synthetic_small_tensor():
    """Small tensor for quick forward-pass smoke tests: (N, 1, 3, 1080, 1920)
    with N=2 to test batching."""
    torch.manual_seed(42)
    return torch.randn(2, 1, 3, 1080, 1920)
