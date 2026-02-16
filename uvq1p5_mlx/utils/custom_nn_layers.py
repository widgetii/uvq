"""Custom neural network layers for the UVQ 1.5 MLX implementation.

MLX-native ports of the custom layers in uvq1p5_pytorch/utils/custom_nn_layers.py.
All layers operate in NHWC data format (MLX default).

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

import mlx.core as mx
import mlx.nn as nn


class Conv2dSamePadding(nn.Module):
  """2D Convolution with TensorFlow-style 'same' padding (NHWC)."""

  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size,
      stride=1,
      dilation=1,
      groups=1,
      bias=False,
  ):
    super().__init__()
    self.kernel_size = (
        kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    )
    self.stride = stride if isinstance(stride, tuple) else (stride, stride)
    self.conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=self.kernel_size,
        stride=self.stride,
        padding=0,
        dilation=dilation,
        bias=bias,
    )

  def __call__(self, x):
    # x: (N, H, W, C) in NHWC
    ih, iw = x.shape[1], x.shape[2]
    kh, kw = self.kernel_size
    sh, sw = self.stride

    ph = max((ih - 1) // sh * sh + kh - ih, 0)
    pw = max((iw - 1) // sw * sw + kw - iw, 0)

    if ph > 0 or pw > 0:
      pad_top = ph // 2
      pad_bottom = ph - pad_top
      pad_left = pw // 2
      pad_right = pw - pad_left
      # mx.pad: list of (before, after) per dimension — N, H, W, C
      x = mx.pad(x, [(0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)])

    return self.conv(x)


class Conv2dNormActivationSamePadding(nn.Module):
  """Conv2dSamePadding + BatchNorm + optional activation."""

  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size=3,
      stride=1,
      groups=1,
      bn_eps=0.001,
      bn_momentum=0.99,
      activation="silu",
      bias=False,
  ):
    super().__init__()
    # ConvNormActivation in torchvision always sets bias=False when norm_layer
    # is provided, regardless of what the caller passes.
    self.conv = Conv2dSamePadding(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        groups=groups,
        bias=False,
    )
    self.bn = nn.BatchNorm(out_channels, eps=bn_eps, momentum=bn_momentum)
    if activation == "silu":
      self.activation = nn.SiLU()
    elif activation == "relu":
      self.activation = nn.ReLU()
    elif activation is None:
      self.activation = None
    else:
      raise ValueError(f"Unknown activation: {activation}")

  def __call__(self, x):
    x = self.conv(x)
    x = self.bn(x)
    if self.activation is not None:
      x = self.activation(x)
    return x


class SqueezeExcitation(nn.Module):
  """Squeeze-and-Excitation block (NHWC).

  Uses fc1/fc2 attribute names to match torchvision's SqueezeExcitation,
  which is what the PyTorch checkpoint keys use.
  """

  def __init__(self, input_channels, squeeze_channels):
    super().__init__()
    # torchvision SE uses Conv2d(1x1) internally, keyed as fc1/fc2
    self.fc1 = nn.Conv2d(input_channels, squeeze_channels, kernel_size=1, bias=True)
    self.fc2 = nn.Conv2d(squeeze_channels, input_channels, kernel_size=1, bias=True)

  def __call__(self, x):
    # x: (N, H, W, C)
    scale = mx.mean(x, axis=(1, 2), keepdims=True)  # (N, 1, 1, C)
    scale = nn.silu(self.fc1(scale))
    scale = mx.sigmoid(self.fc2(scale))
    return x * scale


class MBConvSamePadding(nn.Module):
  """Mobile inverted residual block with 'same' padding (NHWC).

  StochasticDepth is no-op at eval time, so it is omitted entirely.
  """

  def __init__(
      self,
      input_channels,
      expand_ratio,
      out_channels,
      kernel,
      stride,
      stochastic_depth_prob,
      bn_eps=0.001,
      bn_momentum=0.99,
  ):
    super().__init__()
    self.use_res_connect = stride == 1 and input_channels == out_channels

    expanded_channels = input_channels * expand_ratio

    layers = []
    # expand
    if expanded_channels != input_channels:
      layers.append(
          Conv2dNormActivationSamePadding(
              input_channels, expanded_channels, kernel_size=1, stride=1,
              bn_eps=bn_eps, bn_momentum=bn_momentum, activation="silu",
          )
      )

    # depthwise
    layers.append(
        Conv2dNormActivationSamePadding(
            expanded_channels, expanded_channels, kernel_size=kernel,
            stride=stride, groups=expanded_channels,
            bn_eps=bn_eps, bn_momentum=bn_momentum, activation="silu",
        )
    )

    # squeeze-excitation
    squeeze_channels = max(1, input_channels // 4)
    layers.append(SqueezeExcitation(expanded_channels, squeeze_channels))

    # project
    layers.append(
        Conv2dNormActivationSamePadding(
            expanded_channels, out_channels, kernel_size=1,
            bn_eps=bn_eps, bn_momentum=bn_momentum, activation=None,
        )
    )

    self.block = layers

  def __call__(self, x):
    result = x
    for layer in self.block:
      result = layer(result)
    if self.use_res_connect:
      result = result + x
    return result


class AdaptiveAvgPool2d(nn.Module):
  """Adaptive average pooling for NHWC tensors.

  Only supports cases where input size is evenly divisible by output size.
  """

  def __init__(self, output_size):
    super().__init__()
    if isinstance(output_size, int):
      output_size = (output_size, output_size)
    self.output_size = output_size

  def __call__(self, x):
    # x: (N, H, W, C)
    ih, iw = x.shape[1], x.shape[2]
    oh, ow = self.output_size
    if ih == oh and iw == ow:
      return x
    kh = ih // oh
    kw = iw // ow
    # Use reshaping for average pooling
    n, _, _, c = x.shape
    x = mx.reshape(x, (n, oh, kh, ow, kw, c))
    x = mx.mean(x, axis=(2, 4))
    return x


class BilinearResize(nn.Module):
  """Bilinear resize for NHWC tensors using numpy-based interpolation.

  Used for ContentNet's 1080x1920 → 256x256 resize.
  Falls back to a simple bilinear implementation to ensure numerical
  consistency across backends.
  """

  def __init__(self, target_height, target_width):
    super().__init__()
    self.target_height = target_height
    self.target_width = target_width

  def __call__(self, x):
    # x: (N, H, W, C)
    n, ih, iw, c = x.shape
    oh, ow = self.target_height, self.target_width
    if ih == oh and iw == ow:
      return x

    # Compute sampling grid (align_corners=False)
    h_scale = ih / oh
    w_scale = iw / ow

    # Source coordinates for each output pixel
    h_coords = mx.arange(oh).astype(mx.float32) * h_scale + (h_scale - 1) / 2
    w_coords = mx.arange(ow).astype(mx.float32) * w_scale + (w_scale - 1) / 2

    h_coords = mx.clip(h_coords, 0, ih - 1)
    w_coords = mx.clip(w_coords, 0, iw - 1)

    h0 = mx.floor(h_coords).astype(mx.int32)
    w0 = mx.floor(w_coords).astype(mx.int32)
    h1 = mx.minimum(h0 + 1, ih - 1)
    w1 = mx.minimum(w0 + 1, iw - 1)

    h_frac = h_coords - h0.astype(mx.float32)
    w_frac = w_coords - w0.astype(mx.float32)

    # Reshape for broadcasting: h → (oh, 1), w → (1, ow)
    h0 = h0[:, None]      # (oh, 1)
    h1 = h1[:, None]      # (oh, 1)
    w0 = w0[None, :]      # (1, ow)
    w1 = w1[None, :]      # (1, ow)
    h_frac = h_frac[:, None, None]  # (oh, 1, 1)
    w_frac = w_frac[None, :, None]  # (1, ow, 1)

    # Create index grids: (oh, ow)
    h0_grid = mx.broadcast_to(h0, (oh, ow))
    h1_grid = mx.broadcast_to(h1, (oh, ow))
    w0_grid = mx.broadcast_to(w0, (oh, ow))
    w1_grid = mx.broadcast_to(w1, (oh, ow))

    # Gather and interpolate per batch element
    # x: (N, H, W, C) — index with [h_grid, w_grid] for each batch
    top_left = x[:, h0_grid, w0_grid, :]      # (N, oh, ow, C)
    top_right = x[:, h0_grid, w1_grid, :]     # (N, oh, ow, C)
    bottom_left = x[:, h1_grid, w0_grid, :]   # (N, oh, ow, C)
    bottom_right = x[:, h1_grid, w1_grid, :]  # (N, oh, ow, C)

    top = top_left * (1 - w_frac) + top_right * w_frac
    bottom = bottom_left * (1 - w_frac) + bottom_right * w_frac
    result = top * (1 - h_frac) + bottom * h_frac

    return result


class LayerNorm3D(nn.Module):
  """LayerNorm for NHWC tensors with 3D normalized shape.

  Equivalent to PyTorch's nn.LayerNorm([C, H, W]) operating on NCHW input,
  but here we normalize over axes (1, 2, 3) of NHWC input with affine
  parameters of shape (H, W, C).
  """

  def __init__(self, normalized_shape_chw, eps=0.001):
    super().__init__()
    # normalized_shape_chw is [C, H, W] from PyTorch
    # In NHWC, the affine params are (H, W, C) — already transposed by
    # the conversion script
    c, h, w = normalized_shape_chw
    self.eps = eps
    self.weight = mx.ones((h, w, c))
    self.bias = mx.zeros((h, w, c))

  def __call__(self, x):
    # x: (N, H, W, C) — normalize over (H, W, C) = axes (1, 2, 3)
    mean = mx.mean(x, axis=(1, 2, 3), keepdims=True)
    var = mx.var(x, axis=(1, 2, 3), keepdims=True)
    x = (x - mean) / mx.sqrt(var + self.eps)
    x = x * self.weight + self.bias
    return x
