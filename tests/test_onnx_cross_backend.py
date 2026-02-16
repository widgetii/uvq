"""Cross-backend consistency tests: PyTorch CPU vs ONNX Runtime CPU."""

import os
import sys

import numpy as np
import pytest

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

onnx_installed = pytest.mark.skipif(
    not HAS_ORT,
    reason="onnxruntime is not installed",
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_MODELS_DIR = os.path.join(REPO_ROOT, "uvq1p5_web", "public", "models")

onnx_models_exist = pytest.mark.skipif(
    not os.path.isfile(os.path.join(ONNX_MODELS_DIR, "content_net.onnx")),
    reason="ONNX models not exported (run scripts/export_onnx.py first)",
)


@onnx_installed
@onnx_models_exist
@pytest.mark.slow
@pytest.mark.onnx
class TestONNXCrossBackendConsistency:
    @pytest.fixture(scope="class")
    def pytorch_model(self):
        from uvq1p5_pytorch.utils.uvq1p5 import UVQ1p5
        m = UVQ1p5()
        m.eval()
        return m

    @pytest.fixture(scope="class")
    def onnx_sessions(self):
        content_sess = ort.InferenceSession(
            os.path.join(ONNX_MODELS_DIR, "content_net.onnx"),
            providers=["CPUExecutionProvider"],
        )
        distortion_sess = ort.InferenceSession(
            os.path.join(ONNX_MODELS_DIR, "distortion_net.onnx"),
            providers=["CPUExecutionProvider"],
        )
        aggregation_sess = ort.InferenceSession(
            os.path.join(ONNX_MODELS_DIR, "aggregation_net.onnx"),
            providers=["CPUExecutionProvider"],
        )
        return content_sess, distortion_sess, aggregation_sess

    def _run_onnx_pipeline(self, onnx_sessions, video_nchw):
        """Run the full ONNX pipeline mimicking the PyTorch data flow."""
        content_sess, distortion_sess, aggregation_sess = onnx_sessions

        # video_nchw: (N, 1, 3, 1080, 1920)
        frame = video_nchw[:, 0]  # (N, 3, 1080, 1920)

        # Content: resize to 256x256 (using torch for consistency)
        import torch
        import torch.nn.functional as F

        frame_t = torch.from_numpy(frame).float()
        content_input = F.interpolate(
            frame_t, size=(256, 256), mode="bilinear", align_corners=False,
        ).numpy()

        content_feat = content_sess.run(
            None, {"input": content_input}
        )[0]  # (N, 128, 8, 8)

        # Distortion: split into 3x3 patches
        n, c, h, w = frame.shape
        patches = frame.reshape(n, c, 3, 360, 3, 640)
        patches = np.transpose(patches, (0, 2, 4, 1, 3, 5))  # (N, 3, 3, C, 360, 640)
        patches = patches.reshape(-1, c, 360, 640)  # (N*9, 3, 360, 640)

        patch_feats = distortion_sess.run(
            None, {"input": patches}
        )[0]  # (N*9, 128, 8, 8)

        # Reassemble 3x3 grid
        patch_feats = patch_feats.reshape(n, 3, 3, 128, 8, 8)
        # -> (N, 128, 3*8, 3*8) = (N, 128, 24, 24)
        patch_feats = np.transpose(patch_feats, (0, 3, 1, 4, 2, 5))
        distortion_feat = patch_feats.reshape(n, 128, 24, 24)

        # Aggregation
        score = aggregation_sess.run(
            None, {"content": content_feat, "distortion": distortion_feat}
        )[0]  # (N, 1)

        return score.mean()

    def test_synthetic_score_consistency(self, pytorch_model, onnx_sessions):
        """PyTorch CPU and ONNX Runtime CPU should produce similar scores."""
        import torch

        # Generate shared random input as numpy (NHWC initially, like the MLX test)
        np.random.seed(42)
        data = np.random.randn(1, 1, 1080, 1920, 3).astype(np.float32)

        # PyTorch: expects NCHW -> (N, 1, C, H, W)
        pt_input = torch.from_numpy(data.transpose(0, 1, 4, 2, 3)).float()
        with torch.inference_mode():
            pt_pred = pytorch_model.uvq1p5_core(pt_input)
        pt_score = pt_pred.item()

        # ONNX Runtime: same NCHW layout
        onnx_score = self._run_onnx_pipeline(
            onnx_sessions, data.transpose(0, 1, 4, 2, 3)
        )

        assert pt_score == pytest.approx(float(onnx_score), abs=0.01), (
            f"PyTorch CPU score {pt_score} vs ONNX Runtime score {onnx_score}"
        )
