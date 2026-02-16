"""MLX implementation of the ContentNet used in UVQ 1.5.

Mirrors uvq1p5_pytorch/utils/contentnet.py but operates in NHWC format.

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
    os.path.dirname(__file__), "..", "checkpoints", "content_net.safetensors"
)

# ContentNet uses eps=0.001, momentum=0.99
_BN_EPS = 0.001
_BN_MOM = 0.99


class ContentNetCore(nn.Module):
  """EfficientNet-B0 based content feature extractor (NHWC)."""

  def __init__(self):
    super().__init__()
    sd_prob = [0.0] * 16  # stochastic_depth_prob_step = 0.0

    self.resize = custom_nn_layers.BilinearResize(256, 256)

    self.features = [
        custom_nn_layers.Conv2dNormActivationSamePadding(
            3, 32, kernel_size=3, stride=2, activation="silu",
            bn_eps=_BN_EPS, bn_momentum=_BN_MOM,
        ),
        custom_nn_layers.MBConvSamePadding(32, 1, 16, 3, 1, sd_prob[0], bn_eps=_BN_EPS, bn_momentum=_BN_MOM),
        custom_nn_layers.MBConvSamePadding(16, 6, 24, 3, 2, sd_prob[1], bn_eps=_BN_EPS, bn_momentum=_BN_MOM),
        custom_nn_layers.MBConvSamePadding(24, 6, 24, 3, 1, sd_prob[2], bn_eps=_BN_EPS, bn_momentum=_BN_MOM),
        custom_nn_layers.MBConvSamePadding(24, 6, 40, 5, 2, sd_prob[3], bn_eps=_BN_EPS, bn_momentum=_BN_MOM),
        custom_nn_layers.MBConvSamePadding(40, 6, 40, 5, 1, sd_prob[4], bn_eps=_BN_EPS, bn_momentum=_BN_MOM),
        custom_nn_layers.MBConvSamePadding(40, 6, 80, 3, 2, sd_prob[5], bn_eps=_BN_EPS, bn_momentum=_BN_MOM),
        custom_nn_layers.MBConvSamePadding(80, 6, 80, 3, 1, sd_prob[6], bn_eps=_BN_EPS, bn_momentum=_BN_MOM),
        custom_nn_layers.MBConvSamePadding(80, 6, 80, 3, 1, sd_prob[7], bn_eps=_BN_EPS, bn_momentum=_BN_MOM),
        custom_nn_layers.MBConvSamePadding(80, 6, 112, 5, 1, sd_prob[8], bn_eps=_BN_EPS, bn_momentum=_BN_MOM),
        custom_nn_layers.MBConvSamePadding(112, 6, 112, 5, 1, sd_prob[9], bn_eps=_BN_EPS, bn_momentum=_BN_MOM),
        custom_nn_layers.MBConvSamePadding(112, 6, 112, 5, 1, sd_prob[10], bn_eps=_BN_EPS, bn_momentum=_BN_MOM),
        custom_nn_layers.MBConvSamePadding(112, 6, 192, 5, 2, sd_prob[11], bn_eps=_BN_EPS, bn_momentum=_BN_MOM),
        custom_nn_layers.MBConvSamePadding(192, 6, 192, 5, 1, sd_prob[12], bn_eps=_BN_EPS, bn_momentum=_BN_MOM),
        custom_nn_layers.MBConvSamePadding(192, 6, 192, 5, 1, sd_prob[13], bn_eps=_BN_EPS, bn_momentum=_BN_MOM),
        custom_nn_layers.MBConvSamePadding(192, 6, 192, 5, 1, sd_prob[14], bn_eps=_BN_EPS, bn_momentum=_BN_MOM),
        custom_nn_layers.MBConvSamePadding(192, 6, 320, 3, 1, sd_prob[15], bn_eps=_BN_EPS, bn_momentum=_BN_MOM),
        custom_nn_layers.Conv2dSamePadding(320, 128, kernel_size=2, stride=1),
    ]
    self.avgpool = custom_nn_layers.AdaptiveAvgPool2d((4, 4))

  def __call__(self, x):
    x = self.resize(x)
    for layer in self.features:
      x = layer(x)
    return x


class ContentNet(nn.Module):
  """ContentNet wrapper that loads weights and runs inference on video frames."""

  def __init__(self, model_path=MODEL_PATH, pretrained=True):
    super().__init__()
    self.model = ContentNetCore()
    if pretrained:
      self._load_weights(model_path)

  def _load_weights(self, model_path):
    weights = mx.load(model_path)
    self.model.load_weights(list(weights.items()))
    self.eval()

  def __call__(self, video):
    """Process video frames through ContentNet.

    Args:
      video: (N, 1, H, W, C) NHWC tensor with values in [-1, 1].

    Returns:
      Features of shape (N, 4, 4, 128) in NHWC.
    """
    # Take first frame per second: (N, 1, H, W, C) → (N, H, W, C)
    input_video = video[:, 0]
    features = self.model(input_video)
    return features
