"""Integration tests for UVQ 1.5 MLX model — loads real checkpoints."""

import os
import sys

import pytest

mlx_required = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="MLX requires macOS (Apple Silicon)",
)

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

mlx_installed = pytest.mark.skipif(
    not HAS_MLX,
    reason="mlx is not installed",
)

MLX_CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "uvq1p5_mlx", "checkpoints",
)


# ---------------------------------------------------------------------------
# Checkpoint existence
# ---------------------------------------------------------------------------

class TestMLXCheckpointFiles:
    @pytest.mark.parametrize("name", [
        "content_net.safetensors",
        "distortion_net.safetensors",
        "aggregation_net.safetensors",
    ])
    def test_checkpoint_exists(self, name):
        path = os.path.join(MLX_CHECKPOINT_DIR, name)
        assert os.path.isfile(path), f"Missing checkpoint: {path}"


# ---------------------------------------------------------------------------
# Sub-network loading
# ---------------------------------------------------------------------------

@mlx_required
@mlx_installed
@pytest.mark.slow
@pytest.mark.mlx
class TestMLXSubNetworkLoading:
    def test_content_net_loads(self):
        from uvq1p5_mlx.utils.contentnet import ContentNet
        net = ContentNet()
        assert isinstance(net, nn.Module)

    def test_distortion_net_loads(self):
        from uvq1p5_mlx.utils.distortionnet import DistortionNet
        net = DistortionNet()
        assert isinstance(net, nn.Module)

    def test_aggregation_net_loads(self):
        from uvq1p5_mlx.utils.aggregationnet import AggregationNet
        net = AggregationNet()
        assert isinstance(net, nn.Module)


# ---------------------------------------------------------------------------
# Forward pass with synthetic tensors
# ---------------------------------------------------------------------------

@mlx_required
@mlx_installed
@pytest.mark.slow
@pytest.mark.mlx
class TestMLXForwardPass:
    @pytest.fixture(scope="class")
    def model(self):
        from uvq1p5_mlx.utils.uvq1p5 import UVQ1p5
        return UVQ1p5()

    def test_output_shape(self, model):
        """Single-frame input should produce shape (1, 1)."""
        mx.random.seed(42)
        x = mx.random.normal((1, 1, 1080, 1920, 3))
        pred = model.uvq1p5_core(x)
        mx.eval(pred)
        assert pred.shape == (1, 1), f"Expected (1,1), got {pred.shape}"

    def test_batch_output_shape(self, model):
        """Two-frame batch should produce shape (2, 1)."""
        mx.random.seed(42)
        x = mx.random.normal((2, 1, 1080, 1920, 3))
        pred = model.uvq1p5_core(x)
        mx.eval(pred)
        assert pred.shape == (2, 1), f"Expected (2,1), got {pred.shape}"

    def test_score_range(self, model):
        """Output should be in [1, 5] (tanh*2+3)."""
        mx.random.seed(42)
        x = mx.random.normal((1, 1, 1080, 1920, 3))
        pred = model.uvq1p5_core(x)
        mx.eval(pred)
        score = pred.item()
        assert 1.0 <= score <= 5.0, f"Score {score} outside [1,5]"

    def test_deterministic(self, model):
        """Same input should produce same output."""
        mx.random.seed(123)
        x = mx.random.normal((1, 1, 1080, 1920, 3))
        mx.eval(x)
        pred1 = model.uvq1p5_core(x)
        mx.eval(pred1)
        pred2 = model.uvq1p5_core(x)
        mx.eval(pred2)
        assert pred1.item() == pytest.approx(pred2.item(), abs=1e-6)

    def test_finite_output(self, model):
        """Output should not contain NaN or Inf."""
        mx.random.seed(42)
        x = mx.random.normal((1, 1, 1080, 1920, 3))
        pred = model.uvq1p5_core(x)
        mx.eval(pred)
        assert mx.isnan(pred).sum().item() == 0
        assert mx.isinf(pred).sum().item() == 0
