"""Convert UVQ 1.5 PyTorch .pth checkpoints to MLX .safetensors format.

Handles:
1. NCHW → NHWC weight transposition for Conv2d and LayerNorm 3D
2. Key remapping from PyTorch Sequential indexing to MLX named attributes:
   - ConvNormActivation `.0.conv.` → `.conv.conv.` (Conv2dSamePadding)
   - ConvNormActivation `.1.` (BN params) → `.bn.` (BatchNorm)
   - Drops `num_batches_tracked` keys

Usage:
    python scripts/convert_weights.py

Reads from uvq1p5_pytorch/checkpoints/ and writes to uvq1p5_mlx/checkpoints/.
"""

import os
import re

import numpy as np
import torch
from safetensors.numpy import save_file


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "uvq1p5_pytorch", "checkpoints")
DST_DIR = os.path.join(REPO_ROOT, "uvq1p5_mlx", "checkpoints")

CHECKPOINTS = {
    "content_net.pth": "content_net.safetensors",
    "distortion_net.pth": "distortion_net.safetensors",
    "aggregation_net.pth": "aggregation_net.safetensors",
}

# Keys in aggregation_net that are LayerNorm 3D parameters (shape 256,4,4)
LAYERNORM_3D_KEYS = {"ln1.weight", "ln1.bias"}

# BN parameter names
BN_PARAMS = {"weight", "bias", "running_mean", "running_var"}


def remap_key(key: str) -> str | None:
    """Remap a PyTorch state dict key to the MLX model parameter tree.

    PyTorch ConvNormActivation(Sequential) uses numeric indices:
      - .0.conv.weight for Conv2dSamePadding's Conv2d weight
      - .1.weight/.bias/.running_mean/.running_var for BatchNorm

    MLX Conv2dNormActivationSamePadding uses named attributes:
      - .conv.conv.weight for Conv2dSamePadding's Conv2d weight
      - .bn.weight/.bias/.running_mean/.running_var for BatchNorm

    Returns None for keys that should be dropped (num_batches_tracked).
    """
    # Drop num_batches_tracked (not used in MLX BatchNorm)
    if key.endswith(".num_batches_tracked"):
        return None

    # Remap ConvNormActivation conv child: .X.0.conv. → .X.conv.conv.
    # This matches patterns like features.0.0.conv.weight or
    # features.1.block.0.0.conv.weight
    key = re.sub(r"(\d+)\.0\.conv\.", r"\1.conv.conv.", key)

    # Remap ConvNormActivation BN child: .X.1.(bn_param) → .X.bn.(bn_param)
    # Only when the param name is a known BN parameter
    bn_params_pattern = "|".join(BN_PARAMS)
    key = re.sub(
        rf"(\d+)\.1\.({bn_params_pattern})",
        r"\1.bn.\2",
        key,
    )

    return key


def convert_checkpoint(src_path: str, dst_path: str, is_aggregation: bool = False):
    state_dict = torch.load(src_path, weights_only=True, map_location="cpu")
    converted = {}
    dropped = 0
    for key, tensor in state_dict.items():
        # Remap key for MLX parameter tree
        new_key = remap_key(key)
        if new_key is None:
            dropped += 1
            continue

        arr = tensor.numpy()
        ndim = arr.ndim
        if ndim == 4:
            # Conv2d: (O, I, H, W) → (O, H, W, I)
            arr = np.transpose(arr, (0, 2, 3, 1))
        elif ndim == 3 and is_aggregation and key in LAYERNORM_3D_KEYS:
            # LayerNorm([256, 4, 4]): (C, H, W) → (H, W, C) for NHWC
            arr = np.transpose(arr, (1, 2, 0))
        # 1D and 2D tensors stay as-is
        converted[new_key] = arr
    save_file(converted, dst_path)
    print(f"  {os.path.basename(src_path)} → {os.path.basename(dst_path)}"
          f"  ({len(converted)} tensors, {dropped} dropped)")


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    print("Converting UVQ 1.5 checkpoints to safetensors (NHWC, MLX keys)...")
    for src_name, dst_name in CHECKPOINTS.items():
        src_path = os.path.join(SRC_DIR, src_name)
        dst_path = os.path.join(DST_DIR, dst_name)
        is_agg = "aggregation" in src_name
        convert_checkpoint(src_path, dst_path, is_aggregation=is_agg)
    print("Done.")


if __name__ == "__main__":
    main()
