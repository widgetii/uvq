"""Grad-CAM visualization utilities for UVQ distortion branch.

Provides hook management, CAM computation, patch stitching, and overlay
rendering for generating spatial heatmaps of distortion predictions.

Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os

import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMHookManager:
  """Registers forward and backward hooks on a target layer to capture
  activations and gradients for Grad-CAM computation."""

  def __init__(self, target_layer: nn.Module):
    self.activations: torch.Tensor | None = None
    self.gradients: torch.Tensor | None = None
    self._fwd_hook = target_layer.register_forward_hook(self._forward_hook)
    self._bwd_hook = target_layer.register_full_backward_hook(
        self._backward_hook
    )

  def _forward_hook(self, module, input, output):
    self.activations = output.detach()

  def _backward_hook(self, module, grad_input, grad_output):
    self.gradients = grad_output[0].detach()

  def reset(self):
    """Clear cached activations and gradients between frames."""
    self.activations = None
    self.gradients = None

  def remove(self):
    """Remove registered hooks."""
    self._fwd_hook.remove()
    self._bwd_hook.remove()


def compute_gradcam(
    activations: torch.Tensor, gradients: torch.Tensor
) -> torch.Tensor:
  """Compute Grad-CAM heatmap from activations and gradients.

  Args:
    activations: Feature map tensor of shape (B, C, H, W).
    gradients: Gradient tensor of shape (B, C, H, W).

  Returns:
    CAM tensor of shape (B, H, W) normalized to [0, 1] per image.
  """
  weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
  cam = (weights * activations).sum(dim=1)  # (B, H, W)
  cam = F.relu(cam)

  # Per-image normalization to [0, 1]
  b = cam.shape[0]
  for i in range(b):
    cam_max = cam[i].max()
    if cam_max > 0:
      cam[i] = cam[i] / cam_max

  return cam


def stitch_patch_cams(
    patch_cams: list[np.ndarray],
    num_patches_y: int,
    num_patches_x: int,
    patch_height: int,
    patch_width: int,
) -> np.ndarray:
  """Stitch per-patch CAMs into a full-frame heatmap.

  Upsamples each patch CAM to (patch_height, patch_width) and arranges them
  in row-major grid order.

  Args:
    patch_cams: List of (H_cam, W_cam) numpy arrays in row-major order.
    num_patches_y: Number of patches vertically.
    num_patches_x: Number of patches horizontally.
    patch_height: Target height per patch in pixels.
    patch_width: Target width per patch in pixels.

  Returns:
    Full-frame heatmap as numpy array (frame_H, frame_W) in [0, 1].
  """
  frame_h = num_patches_y * patch_height
  frame_w = num_patches_x * patch_width
  full_cam = np.zeros((frame_h, frame_w), dtype=np.float32)

  for idx, cam in enumerate(patch_cams):
    row = idx // num_patches_x
    col = idx % num_patches_x

    # Upsample to patch size using bilinear interpolation
    cam_tensor = torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0)
    cam_upsampled = F.interpolate(
        cam_tensor, size=(patch_height, patch_width), mode="bilinear",
        align_corners=False,
    )
    cam_np = cam_upsampled.squeeze().numpy()

    y_start = row * patch_height
    x_start = col * patch_width
    full_cam[y_start : y_start + patch_height, x_start : x_start + patch_width] = cam_np

  # Re-normalize full frame to [0, 1]
  cam_max = full_cam.max()
  if cam_max > 0:
    full_cam = full_cam / cam_max

  return full_cam


def overlay_cam_on_frame(
    frame_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
  """Alpha-blend a CAM heatmap with the original frame.

  Args:
    frame_rgb: Original frame as uint8 (H, W, 3).
    cam: Heatmap as float (H, W) in [0, 1].
    alpha: Blending weight for the heatmap.

  Returns:
    Blended image as uint8 (H, W, 3).
  """
  colormap = matplotlib.cm.jet(cam)[:, :, :3]  # (H, W, 3) float in [0, 1]
  frame_float = frame_rgb.astype(np.float32) / 255.0
  blended = (1 - alpha) * frame_float + alpha * colormap
  blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
  return blended


def save_gradcam_frame(
    overlay: np.ndarray,
    output_dir: str,
    frame_index: int,
    model_version: str,
) -> str:
  """Save a Grad-CAM overlay frame as PNG.

  Args:
    overlay: Blended image as uint8 (H, W, 3).
    output_dir: Directory to save into.
    frame_index: Frame number for filename.
    model_version: Model version string for filename (e.g. "1.5").

  Returns:
    Path to the saved PNG file.
  """
  os.makedirs(output_dir, exist_ok=True)
  version_str = model_version.replace(".", "p")
  filename = f"gradcam_v{version_str}_frame_{frame_index:04d}.png"
  filepath = os.path.join(output_dir, filename)
  plt.imsave(filepath, overlay)
  return filepath
