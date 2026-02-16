"""MLX implementation of the UVQ 1.5 model.

Mirrors uvq1p5_pytorch/utils/uvq1p5.py but uses MLX for inference.

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

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import aggregationnet
from . import contentnet
from . import distortionnet

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', '..', 'utils'
        )
    )
)
import video_reader


class UVQ1p5Core(nn.Module):
  """UVQ 1.5 core model (MLX)."""

  def __init__(self, content_net, distortion_net, aggregation_net):
    super().__init__()
    self.content_net = content_net
    self.distortion_net = distortion_net
    self.aggregation_net = aggregation_net

  def __call__(self, video):
    content_features = self.content_net(video)
    distortion_features = self.distortion_net(video)
    pred_dict = self.aggregation_net(content_features, distortion_features)
    return pred_dict['uvq_1p5_features']


class UVQ1p5(nn.Module):
  """UVQ 1.5 model (MLX)."""

  def __init__(self, pretrained=True):
    super().__init__()
    ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")

    self.content_net = contentnet.ContentNet(
        model_path=os.path.join(ckpt_dir, "content_net.safetensors"),
        pretrained=pretrained,
    )
    self.distortion_net = distortionnet.DistortionNet(
        model_path=os.path.join(ckpt_dir, "distortion_net.safetensors"),
        pretrained=pretrained,
    )
    self.aggregation_net = aggregationnet.AggregationNet(
        model_path=os.path.join(ckpt_dir, "aggregation_net.safetensors"),
        pretrained=pretrained,
    )

    self.uvq1p5_core = UVQ1p5Core(
        self.content_net, self.distortion_net, self.aggregation_net
    )

  def infer(
      self,
      video_filename: str,
      video_length: int,
      transpose: bool,
      fps: int = 1,
      orig_fps: float | None = None,
      ffmpeg_path: str = "ffmpeg",
      device: str = "mlx",
  ) -> dict[str, Any]:
    """Runs UVQ 1.5 inference on a video file using MLX.

    Args:
      video_filename: Path to the video file.
      video_length: Length of the video in seconds.
      transpose: Whether to transpose the video.
      fps: Frames per second to sample.
      orig_fps: Original fps for frame index calculation.
      ffmpeg_path: Path to ffmpeg executable.
      device: Unused (always MLX).

    Returns:
      Dict with uvq1p5_score, per_frame_scores, and frame_indices.
    """
    video, _ = self.load_video(
        video_filename, video_length, transpose,
        fps=fps, ffmpeg_path=ffmpeg_path,
    )
    # video: numpy (num_seconds, fps, H, W, 3) in [-1, 1] — already NHWC
    num_seconds, read_fps, h, w, c = video.shape
    num_frames = num_seconds * read_fps
    # Reshape to (num_frames, 1, H, W, C) for the model
    video = video.reshape(num_frames, 1, h, w, c)

    batch_size = 24
    predictions = []
    for i in range(0, num_frames, batch_size):
      batch_np = video[i : i + batch_size]
      batch = mx.array(batch_np, dtype=mx.float32)
      pred = self.uvq1p5_core(batch)
      mx.eval(pred)
      predictions.append(pred)

    prediction = mx.concatenate(predictions, axis=0)
    video_score = mx.mean(prediction).item()
    frame_scores = np.array(prediction).flatten().tolist()

    if orig_fps:
      frame_indices = [
          int(round(i * orig_fps / fps)) for i in range(len(frame_scores))
      ]
    else:
      frame_indices = list(range(len(frame_scores)))

    return {
        "uvq1p5_score": video_score,
        "per_frame_scores": frame_scores,
        "frame_indices": frame_indices,
    }

  def load_video(
      self,
      video_filename: str,
      video_length: int,
      transpose: bool = False,
      fps: int = 1,
      ffmpeg_path: str = "ffmpeg",
  ) -> tuple[np.ndarray, int]:
    """Load and preprocess video. Returns numpy array in NHWC format."""
    video, num_real_frames = video_reader.load_video_1p5(
        video_filename,
        video_length,
        transpose,
        video_fps=fps,
        video_height=1080,
        video_width=1920,
        ffmpeg_path=ffmpeg_path,
    )
    # video_reader returns (S, F, H, W, 3) — already NHWC, no transpose needed
    return video, num_real_frames
