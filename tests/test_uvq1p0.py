"""Integration tests for UVQ 1.0 model — loads real checkpoints."""

import os

import pytest
import torch.nn as nn

from tests.conftest import UVQ1P0_CHECKPOINT_DIR


# ---------------------------------------------------------------------------
# Checkpoint existence
# ---------------------------------------------------------------------------

class TestCheckpointFiles:
    @pytest.mark.parametrize("name", [
        "contentnet_pytorch.pt",
        "compressionnet_pytorch_statedict.pt",
        "distortionnet_pytorch_statedict.pt",
        "contentnet_labels.csv",
    ])
    def test_main_checkpoint_exists(self, name):
        path = os.path.join(UVQ1P0_CHECKPOINT_DIR, name)
        assert os.path.isfile(path), f"Missing checkpoint: {path}"

    def test_aggregation_model_count(self):
        """There should be 5 folds x 7 feature combos = 35 aggregation .pt files."""
        agg_dir = os.path.join(UVQ1P0_CHECKPOINT_DIR, "aggregationnet_models")
        pt_files = [f for f in os.listdir(agg_dir) if f.endswith(".pt")]
        assert len(pt_files) == 35, f"Expected 35 aggregation models, found {len(pt_files)}"


# ---------------------------------------------------------------------------
# Sub-network loading
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestSubNetworkLoading:
    def test_contentnet_loads(self):
        from uvq_pytorch.utils.contentnet import ContentNetInference
        net = ContentNetInference()
        assert isinstance(net.model, nn.Module)

    def test_compressionnet_loads(self):
        from uvq_pytorch.utils.compressionnet import CompressionNetInference
        net = CompressionNetInference()
        assert isinstance(net.model, nn.Module)

    def test_distortionnet_loads(self):
        from uvq_pytorch.utils.distortionnet import DistortionNetInference
        net = DistortionNetInference()
        assert isinstance(net.model, nn.Module)

    def test_aggregationnet_loads(self):
        from uvq_pytorch.utils.aggregationnet import AggregationNetInference
        net = AggregationNetInference()
        assert len(net.models) == 35


# ---------------------------------------------------------------------------
# Full model loading
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestFullModel:
    @pytest.fixture(scope="class")
    def model(self):
        from uvq_pytorch.utils.uvq1p0 import UVQ1p0
        return UVQ1p0()

    def test_loads_without_error(self, model):
        assert model is not None

    def test_is_not_nn_module(self, model):
        """UVQ1p0 is a plain class, NOT nn.Module.

        This documents a known issue: calling model.cuda() in
        uvq_inference.py:177 will raise AttributeError because UVQ1p0
        does not inherit from nn.Module and therefore has no .cuda() method.
        """
        assert not isinstance(model, nn.Module)

    def test_has_subnets(self, model):
        assert hasattr(model, "contentnet")
        assert hasattr(model, "compressionnet")
        assert hasattr(model, "distortionnet")
        assert hasattr(model, "aggregationnet")
