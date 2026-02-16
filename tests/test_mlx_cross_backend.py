"""Cross-backend consistency tests: PyTorch CPU vs MLX."""

import sys

import numpy as np
import pytest

mlx_required = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="MLX requires macOS (Apple Silicon)",
)

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

mlx_installed = pytest.mark.skipif(
    not HAS_MLX,
    reason="mlx is not installed",
)


@mlx_required
@mlx_installed
@pytest.mark.slow
@pytest.mark.mlx
class TestCrossBackendConsistency:
    @pytest.fixture(scope="class")
    def pytorch_model(self):
        from uvq1p5_pytorch.utils.uvq1p5 import UVQ1p5
        m = UVQ1p5()
        m.eval()
        return m

    @pytest.fixture(scope="class")
    def mlx_model(self):
        from uvq1p5_mlx.utils.uvq1p5 import UVQ1p5
        return UVQ1p5()

    def test_synthetic_score_consistency(self, pytorch_model, mlx_model):
        """PyTorch CPU and MLX should produce similar scores on synthetic data."""
        import torch

        # Generate shared random input as numpy
        np.random.seed(42)
        data = np.random.randn(1, 1, 1080, 1920, 3).astype(np.float32)

        # PyTorch: expects NCHW → (N, 1, C, H, W)
        pt_input = torch.from_numpy(data.transpose(0, 1, 4, 2, 3)).float()
        with torch.inference_mode():
            pt_pred = pytorch_model.uvq1p5_core(pt_input)
        pt_score = pt_pred.item()

        # MLX: expects NHWC → (N, 1, H, W, C), data is already NHWC
        mlx_input = mx.array(data, dtype=mx.float32)
        mlx_pred = mlx_model.uvq1p5_core(mlx_input)
        mx.eval(mlx_pred)
        mlx_score = mlx_pred.item()

        assert pt_score == pytest.approx(mlx_score, abs=0.01), (
            f"PyTorch CPU score {pt_score} vs MLX score {mlx_score}"
        )
