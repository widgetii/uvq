"""MLX implementation of the AggregationNet for UVQ 1.5.

Mirrors uvq1p5_pytorch/utils/aggregationnet.py but operates in NHWC format.

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

import mlx.core as mx
import mlx.nn as nn

from . import custom_nn_layers

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints", "aggregation_net.safetensors"
)

NUM_CHANNELS_PER_SUBNET = 128
NUM_FILTERS = 256


class AggregationNetCore(nn.Module):
  """Core aggregation network (NHWC)."""

  def __init__(self, subnets):
    super().__init__()
    self.subnets = subnets
    num_subnets = len(subnets)

    self.conv1 = nn.Conv2d(
        num_subnets * NUM_CHANNELS_PER_SUBNET,
        NUM_FILTERS,
        kernel_size=1,
        bias=True,
    )
    self.ln1 = custom_nn_layers.LayerNorm3D([256, 4, 4], eps=0.001)
    self.linear1 = nn.Linear(NUM_FILTERS, 1, bias=True)
    self.resizer = custom_nn_layers.AdaptiveAvgPool2d((4, 4))

  def __call__(self, content_features, distortion_features):
    """Forward pass.

    Args:
      content_features: (N, 4, 4, 128) NHWC
      distortion_features: (N, 24, 24, 128) NHWC

    Returns:
      Score tensor of shape (N, 1).
    """
    content_resized = self.resizer(content_features)
    distortion_resized = self.resizer(distortion_features)

    x = mx.concatenate([content_resized, distortion_resized], axis=-1)

    x = self.conv1(x)
    x = self.ln1(x)
    x = nn.relu(x)

    # MaxPool2d(4,4) on (N, 4, 4, 256) → (N, 1, 1, 256)
    x = mx.max(x, axis=(1, 2), keepdims=True)

    # Flatten: (N, 1, 1, 256) → (N, 256)
    x = mx.reshape(x, (x.shape[0], -1))
    x = self.linear1(x)
    x = mx.tanh(x) * 2 + 3
    return x


class AggregationNet(nn.Module):
  """AggregationNet wrapper that loads weights."""

  def __init__(self, model_path=MODEL_PATH, pretrained=True):
    super().__init__()
    self.model = AggregationNetCore(["content", "distortion"])
    if pretrained:
      self._load_weights(model_path)

  def _load_weights(self, model_path):
    weights = mx.load(model_path)
    self.model.load_weights(list(weights.items()))
    self.eval()

  def __call__(self, content_features, distortion_features):
    """Run aggregation.

    Args:
      content_features: (N, 4, 4, 128) NHWC
      distortion_features: (N, 24, 24, 128) NHWC

    Returns:
      dict with 'uvq_1p5_features' → mean score tensor of shape (N, 1).
    """
    r = self.model(content_features, distortion_features)
    return {"uvq_1p5_features": r}
