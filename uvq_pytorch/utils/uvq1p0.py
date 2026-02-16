"""UVQ1.0 Pytorch model wrapper.

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
import sys
from typing import Any

import numpy as np
import torch

from . import aggregationnet
from . import compressionnet
from . import contentnet
from . import distortionnet

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils"
        )
    )
)
import video_reader


class UVQ1p0:
  """Wrapper class for UVQ 1.0 model inference."""

  def __init__(self, eval_mode=True):
    self.contentnet = contentnet.ContentNetInference(eval_mode=eval_mode)
    self.compressionnet = compressionnet.CompressionNetInference(
        eval_mode=eval_mode
    )
    self.distortionnet = distortionnet.DistortionNetInference(
        eval_mode=eval_mode
    )
    self.aggregationnet = aggregationnet.AggregationNetInference(
        pretrained=eval_mode
    )

  def cuda(self):
    """Moves all sub-network models to CUDA."""
    self.contentnet.model.cuda()
    self.compressionnet.model.cuda()
    self.distortionnet.model.cuda()
    for name in self.aggregationnet.models:
      self.aggregationnet.models[name].cuda()
    return self

  def infer(
      self,
      video_filename: str,
      video_length: int,
      transpose: bool = False,
      device: str = "cpu",
  ) -> dict[str, Any]:
    """Runs UVQ 1.0 inference on a video file.

    Args:
        video_filename: Path to the video file.
        video_length: Length of the video in seconds.
        transpose: Whether to transpose the video before processing.

    Returns:
        A dictionary containing the UVQ 1.0 scores and per-patch labels.
    """
    video_resized1, video_resized2 = self.load_video(
        video_filename, video_length, transpose
    )
    content_features, content_labels = (
        self.contentnet.get_labels_and_features_for_all_frames(
            video=video_resized2, device=device
        )
    )
    compression_features, compression_labels = (
        self.compressionnet.get_labels_and_features_for_all_frames(
            video=video_resized1, device=device,
        )
    )
    distortion_features, distortion_labels = (
        self.distortionnet.get_labels_and_features_for_all_frames(
            video=video_resized1, device=device,
        )
    )
    results = self.aggregationnet.predict(
        compression_features, content_features, distortion_features,
        device=device,
    )
    results["content_labels"] = content_labels                 # (T, 3862)
    results["compression_patch_labels"] = compression_labels  # (T, 4, 4, 1)
    results["distortion_patch_labels"] = distortion_labels    # (T, 2, 2, 26)
    return results

  def infer_gradcam(
      self,
      video_filename: str,
      video_length: int,
      transpose: bool,
      output_dir: str,
      device: str = "cpu",
  ) -> dict[str, float | list[str]]:
    """Runs UVQ 1.0 inference with Grad-CAM heatmap generation.

    Since the UVQ 1.0 pipeline breaks the gradient graph between feature
    extraction and aggregation, Grad-CAM backprops from the distortion
    classifier's output (sum of 26-class sigmoid probabilities) instead
    of the final quality score.

    Args:
        video_filename: Path to the video file.
        video_length: Length of the video in seconds.
        transpose: Whether to transpose the video before processing.
        output_dir: Directory to save Grad-CAM PNG overlays.

    Returns:
        A dictionary containing UVQ 1.0 scores and a list of saved
        Grad-CAM PNG file paths.
    """
    from gradcam import (
        GradCAMHookManager,
        compute_gradcam,
        overlay_cam_on_frame,
        save_gradcam_frame,
        stitch_patch_cams,
    )

    # Run normal inference to get quality scores
    results = self.infer(video_filename, video_length, transpose, device=device)

    # Re-load video for gradcam (720p, 5fps)
    video_720p, _ = self.load_video(video_filename, video_length, transpose)
    # video_720p shape: (num_seconds, fps, C=3, H=720, W=1280)

    num_patches_y = self.distortionnet.num_patches_y  # 2
    num_patches_x = self.distortionnet.num_patches_x  # 2
    patch_h = self.distortionnet.patch_height  # 360
    patch_w = self.distortionnet.patch_width   # 640

    # Hook the last conv layer of the distortion model
    target_layer = self.distortionnet.model.features[17]
    hook_mgr = GradCAMHookManager(target_layer)

    gradcam_files = []
    try:
      for k in range(video_720p.shape[0]):
        hook_mgr.reset()

        # Collect per-patch CAMs for this frame
        patch_cams = []
        for j in range(num_patches_y):
          for i in range(num_patches_x):
            hook_mgr.reset()

            patch_np = video_720p[
                k, 0, :,
                j * patch_h : (j + 1) * patch_h,
                i * patch_w : (i + 1) * patch_w,
            ]
            patch_tensor = torch.from_numpy(
                patch_np[np.newaxis].copy()
            ).float().to(device)

            # Forward without no_grad to build computation graph
            features, label_probs = self.distortionnet.model(patch_tensor)

            # Backward from sum of distortion class probabilities
            label_probs.sum().backward()

            cam = compute_gradcam(
                hook_mgr.activations, hook_mgr.gradients
            )  # (1, H_cam, W_cam)
            patch_cams.append(cam[0].cpu().numpy())

            self.distortionnet.model.zero_grad(set_to_none=True)

        # Stitch 2x2 patch CAMs into full 720p heatmap
        full_cam = stitch_patch_cams(
            patch_cams, num_patches_y, num_patches_x, patch_h, patch_w,
        )

        # Recover original frame pixels: [-1, 1] → [0, 255]
        frame_pixels = video_720p[k, 0]  # (C, H, W) numpy
        frame_rgb = (
            (frame_pixels.transpose(1, 2, 0) + 1) * 127.5
        ).clip(0, 255).astype(np.uint8)

        overlay = overlay_cam_on_frame(frame_rgb, full_cam)
        filepath = save_gradcam_frame(overlay, output_dir, k, "1.0")
        gradcam_files.append(filepath)
    finally:
      hook_mgr.remove()

    results["gradcam_files"] = gradcam_files
    return results

  def load_video(self, video_filename, video_length, transpose=False):
    video, video_small = video_reader.load_video_1p0(
        video_filename, video_length, transpose
    )
    video = video.transpose(0, 1, 4, 2, 3)
    video_small = video_small.transpose(0, 1, 4, 2, 3)
    return video, video_small
