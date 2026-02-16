"""MLX implementation of the DistortionNet for UVQ 1.5.

Mirrors uvq1p5_pytorch/utils/distortionnet.py but operates in NHWC format.

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
    os.path.dirname(__file__), "..", "checkpoints", "distortion_net.safetensors"
)

# DistortionNet uses eps=0.001, momentum=0 (different from ContentNet!)
_BN_EPS = 0.001
_BN_MOM = 0.0

# Input video size
VIDEO_HEIGHT = 1080
VIDEO_WIDTH = 1920
VIDEO_CHANNELS = 3

# Input patch size
PATCH_HEIGHT = 360
PATCH_WIDTH = 640

# Output feature size
DIM_HEIGHT_FEATURE = 24
DIM_WIDTH_FEATURE = 24
DIM_CHANNEL_FEATURE = 128


class DistortionNetCore(nn.Module):
  """EfficientNet-B0 based distortion feature extractor (NHWC)."""

  def __init__(self):
    super().__init__()
    sd_step = 0.0125
    sd_prob = [x * sd_step for x in range(16)]

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
        custom_nn_layers.Conv2dSamePadding(
            320, DIM_CHANNEL_FEATURE, kernel_size=2, stride=1, bias=False
        ),
    ]
    # MaxPool2d (5, 13) on NHWC spatial dims
    # In PyTorch: nn.MaxPool2d(kernel_size=(5, 13), stride=1, padding=0)
    # then PermuteLayerNHWC (but we're already NHWC, so just MaxPool)
    # MLX doesn't have MaxPool2d, so we implement it manually

  def __call__(self, x):
    for layer in self.features:
      x = layer(x)
    # x shape: (N, H_feat, W_feat, 128) where H_feat and W_feat depend on
    # patch size. For 360x640 input: after stride-2 layers,
    # the spatial dims are about 12x21 → after Conv2dSamePadding(k=2,s=1)
    # it stays ~12x21. Then MaxPool2d(5,13) with stride=1, padding=0
    # produces (12-5+1, 21-13+1) = (8, 9) → actually the PyTorch version
    # produces patches that are then reassembled.

    # Manual MaxPool2d(5, 13) with no padding, stride=1
    n, h, w, c = x.shape
    kh, kw = 5, 13
    oh = h - kh + 1
    ow = w - kw + 1
    # Use sliding window via reshape
    # Collect all windows
    patches = []
    for i in range(oh):
      for j in range(ow):
        patches.append(x[:, i:i+kh, j:j+kw, :])
    # Stack and take max
    stacked = mx.stack(patches, axis=1)  # (N, oh*ow, kh, kw, C)
    stacked = mx.reshape(stacked, (n, oh, ow, kh, kw, c))
    features = mx.max(stacked, axis=(3, 4))  # (N, oh, ow, C)
    # Already NHWC, no permute needed
    return features


class DistortionNet(nn.Module):
  """DistortionNet wrapper with patch processing."""

  def __init__(
      self,
      model_path=MODEL_PATH,
      pretrained=True,
      video_height=VIDEO_HEIGHT,
      video_width=VIDEO_WIDTH,
      patch_height=PATCH_HEIGHT,
      patch_width=PATCH_WIDTH,
      feature_channels=DIM_CHANNEL_FEATURE,
      feature_height=DIM_HEIGHT_FEATURE,
      feature_width=DIM_WIDTH_FEATURE,
  ):
    super().__init__()
    self.model = DistortionNetCore()
    if pretrained:
      self._load_weights(model_path)

    self.num_patches_y = video_height // patch_height   # 3
    self.num_patches_x = video_width // patch_width     # 3
    self.patch_height = patch_height
    self.patch_width = patch_width
    self.feature_channels = feature_channels
    self.feature_height = feature_height
    self.feature_width = feature_width
    self.patch_feature_height = feature_height // self.num_patches_y  # 8
    self.patch_feature_width = feature_width // self.num_patches_x    # 8

  def _load_weights(self, model_path):
    weights = mx.load(model_path)
    self.model.load_weights(list(weights.items()))

  def __call__(self, video):
    """Process video frames through DistortionNet.

    Args:
      video: (N, 1, H, W, C) NHWC tensor in [-1, 1] where H=1080, W=1920, C=3.

    Returns:
      Features of shape (N, 24, 24, 128) in NHWC.
    """
    # Take first frame: (N, 1, H, W, C) → (N, H, W, C)
    video = video[:, 0]
    num_sec = video.shape[0]
    h, w, c = video.shape[1], video.shape[2], video.shape[3]

    # Split into patches: (N, H, W, C) → (N, py, ph, px, pw, C)
    video_reshaped = mx.reshape(
        video,
        (num_sec, self.num_patches_y, self.patch_height,
         self.num_patches_x, self.patch_width, c),
    )
    # → (N, py, px, ph, pw, C)
    video_reshaped = mx.transpose(video_reshaped, (0, 1, 3, 2, 4, 5))
    # → (N*py*px, ph, pw, C)
    batched_patches = mx.reshape(
        video_reshaped, (-1, self.patch_height, self.patch_width, c)
    )

    # Forward pass on all patches
    batch_features = self.model(batched_patches)

    # Reassemble: (N*py*px, pfh, pfw, C) → (N, py, px, pfh, pfw, C)
    feature = mx.reshape(
        batch_features,
        (num_sec, self.num_patches_y, self.num_patches_x,
         self.patch_feature_height, self.patch_feature_width,
         self.feature_channels),
    )
    # → (N, py, pfh, px, pfw, C)
    feature = mx.transpose(feature, (0, 1, 3, 2, 4, 5))
    # → (N, feature_height, feature_width, C)
    feature = mx.reshape(
        feature,
        (num_sec,
         self.num_patches_y * self.patch_feature_height,
         self.num_patches_x * self.patch_feature_width,
         self.feature_channels),
    )
    return feature
