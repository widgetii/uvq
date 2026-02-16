"""Tests for uvq1p5_pytorch/utils/custom_nn_layers.py.

Verifies same-padding layers produce output_size = ceil(input_size / stride).
"""

import math

import pytest
import torch

from uvq1p5_pytorch.utils.custom_nn_layers import (
    Conv2dSamePadding,
    Conv3DSamePadding,
    Interpolate,
    MaxPool3dSame,
    MBConvSamePadding,
)


# ---------------------------------------------------------------------------
# Conv2dSamePadding
# ---------------------------------------------------------------------------

class TestConv2dSamePadding:
    @pytest.mark.parametrize(
        "h,w,kernel,stride",
        [
            (32, 32, 3, 1),
            (32, 32, 3, 2),
            (31, 31, 5, 2),
            (64, 128, 7, 2),
            (15, 15, 3, 1),
        ],
    )
    def test_output_spatial_size(self, h, w, kernel, stride):
        layer = Conv2dSamePadding(3, 16, kernel_size=kernel, stride=stride)
        x = torch.randn(1, 3, h, w)
        out = layer(x)
        expected_h = math.ceil(h / stride)
        expected_w = math.ceil(w / stride)
        assert out.shape[2] == expected_h
        assert out.shape[3] == expected_w

    def test_output_channels(self):
        layer = Conv2dSamePadding(3, 32, kernel_size=3, stride=1)
        x = torch.randn(1, 3, 16, 16)
        out = layer(x)
        assert out.shape[1] == 32


# ---------------------------------------------------------------------------
# Conv3DSamePadding
# ---------------------------------------------------------------------------

class TestConv3DSamePadding:
    @pytest.mark.parametrize(
        "d,h,w,kernel,stride",
        [
            (5, 16, 16, (3, 3, 3), (1, 1, 1)),
            (5, 16, 16, (3, 3, 3), (2, 2, 2)),
            (5, 90, 160, (3, 7, 7), (2, 2, 2)),
            (4, 12, 20, (1, 1, 1), (1, 1, 1)),
        ],
    )
    def test_output_spatial_size(self, d, h, w, kernel, stride):
        layer = Conv3DSamePadding(3, 8, kernel_size=kernel, stride=stride)
        x = torch.randn(1, 3, d, h, w)
        out = layer(x)
        expected_d = math.ceil(d / stride[0])
        expected_h = math.ceil(h / stride[1])
        expected_w = math.ceil(w / stride[2])
        assert out.shape[2] == expected_d
        assert out.shape[3] == expected_h
        assert out.shape[4] == expected_w


# ---------------------------------------------------------------------------
# MaxPool3dSame
# ---------------------------------------------------------------------------

class TestMaxPool3dSame:
    @pytest.mark.parametrize(
        "d,h,w,kernel,stride",
        [
            (5, 45, 80, (1, 3, 3), (1, 2, 2)),
            (5, 90, 160, (3, 3, 3), (2, 2, 2)),
            (3, 23, 40, (3, 3, 3), (1, 1, 1)),
            (2, 12, 20, (2, 2, 2), (2, 2, 2)),
        ],
    )
    def test_output_spatial_size(self, d, h, w, kernel, stride):
        layer = MaxPool3dSame(kernel_size=kernel, stride=stride)
        x = torch.randn(1, 8, d, h, w)
        out = layer(x)
        expected_d = math.ceil(d / stride[0])
        expected_h = math.ceil(h / stride[1])
        expected_w = math.ceil(w / stride[2])
        assert out.shape[2] == expected_d
        assert out.shape[3] == expected_h
        assert out.shape[4] == expected_w


# ---------------------------------------------------------------------------
# MBConvSamePadding
# ---------------------------------------------------------------------------

class TestMBConvSamePadding:
    @pytest.mark.parametrize(
        "in_ch,expand,out_ch,kernel,stride",
        [
            (32, 1, 16, 3, 1),
            (16, 6, 24, 3, 2),
            (24, 6, 40, 5, 2),
            (80, 6, 80, 3, 1),
        ],
    )
    def test_output_spatial_size(self, in_ch, expand, out_ch, kernel, stride):
        layer = MBConvSamePadding(in_ch, expand, out_ch, kernel, stride, 0.0)
        h, w = 32, 32
        x = torch.randn(1, in_ch, h, w)
        layer.eval()
        out = layer(x)
        expected_h = math.ceil(h / stride)
        expected_w = math.ceil(w / stride)
        assert out.shape[2] == expected_h
        assert out.shape[3] == expected_w
        assert out.shape[1] == out_ch


# ---------------------------------------------------------------------------
# Interpolate
# ---------------------------------------------------------------------------

class TestInterpolate:
    def test_upsample(self):
        layer = Interpolate(size=(64, 64), mode="bilinear", align_corners=False)
        x = torch.randn(1, 3, 16, 16)
        out = layer(x)
        assert out.shape == (1, 3, 64, 64)

    def test_downsample(self):
        layer = Interpolate(size=(8, 8), mode="bilinear", align_corners=False)
        x = torch.randn(1, 3, 32, 32)
        out = layer(x)
        assert out.shape == (1, 3, 8, 8)

    def test_scale_factor(self):
        layer = Interpolate(scale_factor=2.0, mode="bilinear", align_corners=False)
        x = torch.randn(1, 3, 10, 10)
        out = layer(x)
        assert out.shape == (1, 3, 20, 20)
