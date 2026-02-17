"""RKNN implementation of the UVQ 1.5 model for Rockchip RK3588S NPU.

Mirrors uvq1p5_mlx/utils/uvq1p5.py but uses RKNN-Lite for inference.
Content and distortion nets run on the NPU; the aggregation net runs on
CPU via ONNX Runtime as a workaround for driver-level bugs on older
rknpu drivers (e.g. 0.8.2 on BSP 5.10 kernels).

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

import cv2
import numpy as np
import onnxruntime as ort

from rknnlite.api import RKNNLite

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', '..', 'utils'
        )
    )
)
import video_reader


class UVQ1p5:
    """UVQ 1.5 model using RKNN-Lite for RK3588S NPU inference.

    Content and distortion feature extractors run on the NPU via RKNN-Lite.
    The aggregation net runs on CPU via ONNX Runtime because RKNN produces
    incorrect results for this model on older NPU drivers (0.8.x).
    """

    def __init__(self, models_dir=None, onnx_dir=None):
        if models_dir is None:
            models_dir = os.path.join(
                os.path.dirname(__file__), "..", "models"
            )
        if onnx_dir is None:
            onnx_dir = os.path.join(
                os.path.dirname(__file__), "..", "..",
                "uvq1p5_web", "public", "models"
            )

        # NPU models for feature extraction
        self.content_net = RKNNLite(verbose=False)
        self.distortion_net = RKNNLite(verbose=False)

        ret = self.content_net.load_rknn(
            os.path.join(models_dir, "content_net.rknn"))
        if ret != 0:
            raise RuntimeError(f"Failed to load content_net.rknn (ret={ret})")

        ret = self.distortion_net.load_rknn(
            os.path.join(models_dir, "distortion_net.rknn"))
        if ret != 0:
            raise RuntimeError(
                f"Failed to load distortion_net.rknn (ret={ret})")

        for net, name in [
            (self.content_net, "content_net"),
            (self.distortion_net, "distortion_net"),
        ]:
            ret = net.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
            if ret != 0:
                raise RuntimeError(
                    f"Failed to init runtime for {name} (ret={ret})")

        # Aggregation via ONNX Runtime on CPU (workaround for NPU driver bug)
        agg_onnx_path = os.path.join(onnx_dir, "aggregation_net.onnx")
        if not os.path.isfile(agg_onnx_path):
            raise RuntimeError(
                f"aggregation_net.onnx not found at {agg_onnx_path}. "
                f"Run scripts/export_onnx.py first.")
        self.aggregation_sess = ort.InferenceSession(
            agg_onnx_path, providers=["CPUExecutionProvider"])

    def release(self):
        """Release RKNN resources."""
        self.content_net.release()
        self.distortion_net.release()

    def _process_frame(self, frame_nchw):
        """Process a single frame through the three-net pipeline.

        Args:
            frame_nchw: (1, 3, 1080, 1920) float32 in [-1, 1].

        Returns:
            Score as float.
        """
        frame = frame_nchw[0]  # (3, 1080, 1920)

        # --- Content branch: resize to 256x256 ---
        # Transpose to HWC for cv2.resize, then back to NCHW
        frame_hwc = np.transpose(frame, (1, 2, 0))  # (1080, 1920, 3)
        content_hwc = cv2.resize(
            frame_hwc, (256, 256), interpolation=cv2.INTER_LINEAR)
        content_input = np.transpose(
            content_hwc, (2, 0, 1))[np.newaxis]  # (1, 3, 256, 256)

        content_feat = self.content_net.inference(
            inputs=[content_input],
            data_format='nchw')[0]  # (1, 128, 8, 8)

        # --- Distortion branch: 3x3 patch grid ---
        # frame: (3, 1080, 1920) -> split into 9 patches of (3, 360, 640)
        c, h, w = frame.shape
        patches = frame.reshape(c, 3, 360, 3, 640)
        patches = np.transpose(patches, (1, 3, 0, 2, 4))  # (3, 3, 3, 360, 640)
        patches = patches.reshape(9, c, 360, 640)  # (9, 3, 360, 640)

        # Run each patch individually (batch=1 per inference call)
        patch_feats = []
        for i in range(9):
            patch_input = patches[i:i+1]  # (1, 3, 360, 640)
            feat = self.distortion_net.inference(
                inputs=[patch_input],
                data_format='nchw')[0]  # (1, 128, 8, 8)
            patch_feats.append(feat)
        patch_feats = np.concatenate(patch_feats, axis=0)  # (9, 128, 8, 8)

        # Reassemble 3x3 grid: (9, 128, 8, 8) -> (1, 128, 24, 24)
        patch_feats = patch_feats.reshape(3, 3, 128, 8, 8)
        patch_feats = np.transpose(patch_feats, (2, 0, 3, 1, 4))
        distortion_feat = patch_feats.reshape(1, 128, 24, 24)

        # --- Aggregation (ONNX Runtime on CPU) ---
        score = self.aggregation_sess.run(
            None,
            {"content": content_feat, "distortion": distortion_feat},
        )[0]  # (1, 1)

        return float(score.flatten()[0])

    def infer(
        self,
        video_filename: str,
        video_length: int,
        transpose: bool,
        fps: int = 1,
        orig_fps: float | None = None,
        ffmpeg_path: str = "ffmpeg",
        device: str = "rknn",
    ) -> dict[str, Any]:
        """Runs UVQ 1.5 inference on a video file using RKNN-Lite.

        Args:
            video_filename: Path to the video file.
            video_length: Length of the video in seconds.
            transpose: Whether to transpose the video.
            fps: Frames per second to sample.
            orig_fps: Original fps for frame index calculation.
            ffmpeg_path: Path to ffmpeg executable.
            device: Unused (always RKNN).

        Returns:
            Dict with uvq1p5_score, per_frame_scores, and frame_indices.
        """
        video, _ = video_reader.load_video_1p5(
            video_filename,
            video_length,
            transpose,
            video_fps=fps,
            video_height=1080,
            video_width=1920,
            ffmpeg_path=ffmpeg_path,
        )
        # video: numpy (num_seconds, fps, H, W, 3) in [-1, 1] — NHWC
        num_seconds, read_fps, h, w, c = video.shape
        num_frames = num_seconds * read_fps
        video = video.reshape(num_frames, h, w, c)

        frame_scores = []
        for i in range(num_frames):
            # Convert NHWC -> NCHW for the RKNN models
            frame_nhwc = video[i]  # (H, W, C)
            frame_nchw = np.transpose(frame_nhwc, (2, 0, 1))[
                np.newaxis]  # (1, 3, 1080, 1920)
            score = self._process_frame(frame_nchw)
            frame_scores.append(score)

        video_score = float(np.mean(frame_scores))

        if orig_fps:
            frame_indices = [
                int(round(i * orig_fps / fps))
                for i in range(len(frame_scores))
            ]
        else:
            frame_indices = list(range(len(frame_scores)))

        return {
            "uvq1p5_score": video_score,
            "per_frame_scores": frame_scores,
            "frame_indices": frame_indices,
        }
