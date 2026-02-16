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
class TestUVQ1p0GPUBug:
    def test_uvq1p0_cuda_raises(self):
        """UVQ1p0 is not nn.Module, so .cuda() should raise AttributeError.

        This documents the bug at uvq_inference.py:177.
        """
        from uvq_pytorch.utils.uvq1p0 import UVQ1p0
        model = UVQ1p0()
        with pytest.raises(AttributeError):
            model.cuda()
