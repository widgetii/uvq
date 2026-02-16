"""GPU tests — all gated on CUDA availability."""

import pytest
import torch
import torch.nn as nn

gpu_required = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


@gpu_required
@pytest.mark.gpu
class TestUVQ1p5GPU:
    @pytest.fixture(scope="class")
    def model(self):
        from uvq1p5_pytorch.utils.uvq1p5 import UVQ1p5
        m = UVQ1p5()
        m.eval()
        return m

    def test_model_moves_to_cuda(self, model):
        model_cuda = model.cuda()
        # Check that parameters are on CUDA
        param = next(model_cuda.parameters())
        assert param.device.type == "cuda"
        # Move back to CPU to not affect other tests
        model.cpu()

    def test_forward_pass_on_cuda(self, model):
        model.cuda()
        x = torch.randn(1, 1, 3, 1080, 1920, device="cuda")
        with torch.inference_mode():
            pred = model.uvq1p5_core(x)
        score = pred.item()
        assert 1.0 <= score <= 5.0, f"Score {score} outside [1,5]"
        assert torch.isfinite(pred).all()
        model.cpu()

    def test_cpu_gpu_consistency(self, model):
        """CPU and GPU should produce the same score within tolerance."""
        torch.manual_seed(42)
        x = torch.randn(1, 1, 3, 1080, 1920)

        model.cpu()
        with torch.inference_mode():
            cpu_score = model.uvq1p5_core(x).item()

        model.cuda()
        with torch.inference_mode():
            gpu_score = model.uvq1p5_core(x.cuda()).item()

        model.cpu()
        assert cpu_score == pytest.approx(gpu_score, abs=0.01), (
            f"CPU score {cpu_score} vs GPU score {gpu_score}"
        )


@gpu_required
@pytest.mark.gpu
class TestUVQ1p0GPU:
    @pytest.fixture(scope="class")
    def model(self):
        from uvq_pytorch.utils.uvq1p0 import UVQ1p0
        m = UVQ1p0()
        return m

    def test_model_moves_to_cuda(self, model):
        model.cuda()
        # Check that each sub-network's parameters are on CUDA
        param = next(model.contentnet.model.parameters())
        assert param.device.type == "cuda"
        param = next(model.compressionnet.model.parameters())
        assert param.device.type == "cuda"
        param = next(model.distortionnet.model.parameters())
        assert param.device.type == "cuda"
        for name, agg_model in model.aggregationnet.models.items():
            param = next(agg_model.parameters())
            assert param.device.type == "cuda", f"Aggregation model {name} not on CUDA"
            break  # just check the first one
        # Move back to CPU
        model.contentnet.model.cpu()
        model.compressionnet.model.cpu()
        model.distortionnet.model.cpu()
        for agg_model in model.aggregationnet.models.values():
            agg_model.cpu()

    def test_contentnet_forward_on_cuda(self, model):
        import numpy as np
        model.contentnet.model.cuda()
        frame = np.random.randn(3, 496, 496).astype(np.float32)
        features, labels = model.contentnet.predict_and_get_features(frame, device="cuda")
        assert features.shape == (1, 16, 16, 100)
        assert labels.shape == (3862,)
        model.contentnet.model.cpu()

    def test_compressionnet_forward_on_cuda(self, model):
        import numpy as np
        model.compressionnet.model.cuda()
        patch = np.random.randn(1, 3, 5, 180, 320).astype(np.float32)
        features, labels = model.compressionnet.predict_and_get_features(patch, device="cuda")
        assert features.shape[0] == 1
        model.compressionnet.model.cpu()

    def test_distortionnet_forward_on_cuda(self, model):
        import numpy as np
        model.distortionnet.model.cuda()
        frame = np.random.randn(1, 3, 360, 640).astype(np.float32)
        features, labels = model.distortionnet.predict_and_get_features(frame, device="cuda")
        assert features.shape[0] == 1
        model.distortionnet.model.cpu()

    def test_aggregationnet_forward_on_cuda(self, model):
        import numpy as np
        model.cuda()
        compression_features = np.random.randn(1, 16, 16, 100).astype(np.float32)
        content_features = np.random.randn(1, 16, 16, 100).astype(np.float32)
        distortion_features = np.random.randn(1, 16, 16, 100).astype(np.float32)
        results = model.aggregationnet.predict(
            compression_features, content_features, distortion_features, device="cuda"
        )
        assert "compression_content_distortion" in results
        # Move back
        model.contentnet.model.cpu()
        model.compressionnet.model.cpu()
        model.distortionnet.model.cpu()
        for agg_model in model.aggregationnet.models.values():
            agg_model.cpu()
