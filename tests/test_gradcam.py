"""Tests for Grad-CAM visualization utilities."""

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from utils.gradcam import (
    GradCAMHookManager,
    compute_gradcam,
    overlay_cam_on_frame,
    save_gradcam_frame,
    stitch_patch_cams,
)
from uvq_inference import setup_parser


class TestComputeGradcam:
    def test_output_shape(self):
        activations = torch.randn(4, 64, 12, 20)
        gradients = torch.randn(4, 64, 12, 20)
        cam = compute_gradcam(activations, gradients)
        assert cam.shape == (4, 12, 20)

    def test_output_range(self):
        activations = torch.randn(2, 32, 8, 8)
        gradients = torch.randn(2, 32, 8, 8)
        cam = compute_gradcam(activations, gradients)
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0

    def test_single_batch(self):
        activations = torch.randn(1, 16, 6, 10)
        gradients = torch.randn(1, 16, 6, 10)
        cam = compute_gradcam(activations, gradients)
        assert cam.shape == (1, 6, 10)

    def test_zero_gradients(self):
        activations = torch.randn(2, 16, 4, 4)
        gradients = torch.zeros(2, 16, 4, 4)
        cam = compute_gradcam(activations, gradients)
        assert cam.max() == 0.0


class TestStitchPatchCams:
    def test_3x3_grid(self):
        """Test stitching for UVQ 1.5 (3x3 patches, 1080p)."""
        patch_cams = [np.random.rand(12, 20).astype(np.float32) for _ in range(9)]
        result = stitch_patch_cams(patch_cams, 3, 3, 360, 640)
        assert result.shape == (1080, 1920)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_2x2_grid(self):
        """Test stitching for UVQ 1.0 (2x2 patches, 720p)."""
        patch_cams = [np.random.rand(8, 8).astype(np.float32) for _ in range(4)]
        result = stitch_patch_cams(patch_cams, 2, 2, 360, 640)
        assert result.shape == (720, 1280)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_1x1_grid(self):
        patch_cams = [np.random.rand(4, 4).astype(np.float32)]
        result = stitch_patch_cams(patch_cams, 1, 1, 100, 200)
        assert result.shape == (100, 200)


class TestOverlayCamOnFrame:
    def test_output_shape_and_dtype(self):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        cam = np.random.rand(480, 640).astype(np.float32)
        overlay = overlay_cam_on_frame(frame, cam)
        assert overlay.shape == (480, 640, 3)
        assert overlay.dtype == np.uint8

    def test_zero_alpha(self):
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        cam = np.random.rand(100, 100).astype(np.float32)
        overlay = overlay_cam_on_frame(frame, cam, alpha=0.0)
        np.testing.assert_array_equal(overlay, frame)

    def test_output_range(self):
        frame = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        cam = np.random.rand(50, 50).astype(np.float32)
        overlay = overlay_cam_on_frame(frame, cam)
        assert overlay.min() >= 0
        assert overlay.max() <= 255


class TestSaveGradcamFrame:
    def test_save_creates_file(self):
        overlay = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = save_gradcam_frame(overlay, tmpdir, 5, "1.5")
            assert os.path.exists(filepath)
            assert filepath.endswith("gradcam_v1p5_frame_0005.png")

    def test_save_creates_directory(self):
        overlay = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "nested", "dir")
            filepath = save_gradcam_frame(overlay, output_dir, 0, "1.0")
            assert os.path.exists(filepath)
            assert "gradcam_v1p0_frame_0000.png" in filepath


class TestHookManager:
    def test_captures_activations_and_gradients(self):
        """Verify hooks capture correct shapes on a simple model."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )
        target_layer = model[2]  # second conv
        hook_mgr = GradCAMHookManager(target_layer)

        x = torch.randn(2, 3, 8, 8, requires_grad=True)
        out = model(x)
        out.sum().backward()

        assert hook_mgr.activations is not None
        assert hook_mgr.gradients is not None
        assert hook_mgr.activations.shape == (2, 32, 8, 8)
        assert hook_mgr.gradients.shape == (2, 32, 8, 8)

        hook_mgr.remove()

    def test_reset_clears_state(self):
        model = nn.Linear(4, 2)
        hook_mgr = GradCAMHookManager(model)

        x = torch.randn(1, 4, requires_grad=True)
        out = model(x)
        out.sum().backward()

        assert hook_mgr.activations is not None
        hook_mgr.reset()
        assert hook_mgr.activations is None
        assert hook_mgr.gradients is None

        hook_mgr.remove()

    def test_remove_stops_capturing(self):
        conv = nn.Conv2d(3, 8, 3, padding=1)
        hook_mgr = GradCAMHookManager(conv)
        hook_mgr.remove()

        x = torch.randn(1, 3, 4, 4, requires_grad=True)
        out = conv(x)
        out.sum().backward()

        # After remove, hooks should not have fired
        assert hook_mgr.activations is None
        assert hook_mgr.gradients is None


class TestGradcamCliArgs:
    def test_gradcam_flag_default(self):
        parser = setup_parser()
        args = parser.parse_args(["video.mp4"])
        assert args.gradcam is False
        assert args.gradcam_output == "gradcam_output"

    def test_gradcam_flag_enabled(self):
        parser = setup_parser()
        args = parser.parse_args(["video.mp4", "--gradcam"])
        assert args.gradcam is True

    def test_gradcam_output_custom(self):
        parser = setup_parser()
        args = parser.parse_args([
            "video.mp4", "--gradcam", "--gradcam_output", "/tmp/my_cams"
        ])
        assert args.gradcam_output == "/tmp/my_cams"
