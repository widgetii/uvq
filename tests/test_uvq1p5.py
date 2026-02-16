"""Integration tests for UVQ 1.5 model — loads real checkpoints."""

import os

import pytest
import torch
import torch.nn as nn

from tests.conftest import UVQ1P5_CHECKPOINT_DIR


# ---------------------------------------------------------------------------
# Checkpoint existence
# ---------------------------------------------------------------------------

class TestCheckpointFiles:
    @pytest.mark.parametrize("name", [
        "content_net.pth",
        "distortion_net.pth",
        "aggregation_net.pth",
    ])
    def test_checkpoint_exists(self, name):
        path = os.path.join(UVQ1P5_CHECKPOINT_DIR, name)
        assert os.path.isfile(path), f"Missing checkpoint: {path}"


# ---------------------------------------------------------------------------
# Sub-network loading
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestSubNetworkLoading:
    def test_content_net_loads(self):
        from uvq1p5_pytorch.utils.contentnet import ContentNet
        net = ContentNet()
        assert isinstance(net, nn.Module)

    def test_distortion_net_loads(self):
        from uvq1p5_pytorch.utils.distortionnet import DistortionNet
        model_path = os.path.join(UVQ1P5_CHECKPOINT_DIR, "distortion_net.pth")
        net = DistortionNet(model_path=model_path)
        assert isinstance(net, nn.Module)

    def test_aggregation_net_loads(self):
        from uvq1p5_pytorch.utils.aggregationnet import AggregationNet
        model_path = os.path.join(UVQ1P5_CHECKPOINT_DIR, "aggregation_net.pth")
        net = AggregationNet(model_path=model_path)
        assert isinstance(net, nn.Module)


# ---------------------------------------------------------------------------
# Full model loading
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestFullModel:
    @pytest.fixture(scope="class")
    def model(self):
        from uvq1p5_pytorch.utils.uvq1p5 import UVQ1p5
        return UVQ1p5()

    def test_is_nn_module(self, model):
        assert isinstance(model, nn.Module)

    def test_has_core(self, model):
        assert hasattr(model, "uvq1p5_core")

    def test_eval_mode(self, model):
        assert not model.uvq1p5_core.training


# ---------------------------------------------------------------------------
# Forward pass with synthetic tensors
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestForwardPass:
    @pytest.fixture(scope="class")
    def model(self):
        from uvq1p5_pytorch.utils.uvq1p5 import UVQ1p5
        m = UVQ1p5()
        m.eval()
        return m

    def test_output_shape(self, model):
        """Single-frame input should produce a scalar-like prediction per frame."""
        x = torch.randn(1, 1, 3, 1080, 1920)
        with torch.inference_mode():
            pred = model.uvq1p5_core(x)
        assert pred.shape == (1, 1)

    def test_batch_output_shape(self, model):
        """Two-frame batch."""
        x = torch.randn(2, 1, 3, 1080, 1920)
        with torch.inference_mode():
            pred = model.uvq1p5_core(x)
        assert pred.shape == (2, 1)

    def test_score_range(self, model):
        """AggregationNet uses tanh*2+3, so output is in [1, 5]."""
        x = torch.randn(1, 1, 3, 1080, 1920)
        with torch.inference_mode():
            pred = model.uvq1p5_core(x)
        score = pred.item()
        assert 1.0 <= score <= 5.0, f"Score {score} outside [1,5]"

    def test_deterministic(self, model):
        """Same input should produce same output."""
        torch.manual_seed(123)
        x = torch.randn(1, 1, 3, 1080, 1920)
        with torch.inference_mode():
            pred1 = model.uvq1p5_core(x).item()
            pred2 = model.uvq1p5_core(x).item()
        assert pred1 == pytest.approx(pred2, abs=1e-6)

    def test_finite_output(self, model):
        x = torch.randn(1, 1, 3, 1080, 1920)
        with torch.inference_mode():
            pred = model.uvq1p5_core(x)
        assert torch.isfinite(pred).all()
