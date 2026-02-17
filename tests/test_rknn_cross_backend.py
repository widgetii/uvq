"""Cross-backend consistency tests: PyTorch (pre-computed reference) vs RKNN-Lite NPU.

Unlike the ONNX cross-backend test which runs both backends live, this test
is designed to run on the target device (Orange Pi 5 Plus / RK3588S) where
PyTorch is not installed.  The expected score is read from reference.json,
which is generated on the PC by scripts/export_rknn.py alongside the .rknn
model files.
"""

import glob
import json
import os

import numpy as np
import pytest

try:
    from rknnlite.api import RKNNLite
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False

rknn_installed = pytest.mark.skipif(
    not HAS_RKNN,
    reason="rknn-toolkit-lite2 is not installed",
)

npu_available = pytest.mark.skipif(
    not glob.glob("/dev/rknpu*"),
    reason="NPU driver not loaded (no /dev/rknpu* device)",
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RKNN_MODELS_DIR = os.path.join(REPO_ROOT, "uvq1p5_rknn", "models")

rknn_models_exist = pytest.mark.skipif(
    not os.path.isfile(os.path.join(RKNN_MODELS_DIR, "content_net.rknn")),
    reason="RKNN models not exported (run scripts/export_rknn.py first)",
)

reference_exists = pytest.mark.skipif(
    not os.path.isfile(os.path.join(RKNN_MODELS_DIR, "reference.json")),
    reason="reference.json not found (run scripts/export_rknn.py first)",
)


def _load_rknn_session(model_name):
    """Load a single RKNN model and init its runtime.

    Returns the RKNNLite instance, or calls pytest.skip if the NPU driver
    is not available (e.g. mainline kernel without rknpu.ko).
    """
    rknn = RKNNLite(verbose=False)
    ret = rknn.load_rknn(os.path.join(RKNN_MODELS_DIR, model_name))
    assert ret == 0, f"Failed to load {model_name}"
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    if ret != 0:
        rknn.release()
        pytest.skip("NPU driver not available (rknpu.ko not loaded)")
    return rknn


@rknn_installed
@npu_available
@rknn_models_exist
@reference_exists
@pytest.mark.slow
@pytest.mark.rknn
class TestRKNNCrossBackendConsistency:
    """Compare RKNN NPU output against a PyTorch reference score.

    The reference was generated on the PC during scripts/export_rknn.py using
    the same numpy seed and pre-processed inputs (content 256x256, distortion
    patches 360x640).  No PyTorch is required on the device.
    """

    @pytest.fixture(scope="class")
    def reference(self):
        ref_path = os.path.join(RKNN_MODELS_DIR, "reference.json")
        with open(ref_path) as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def rknn_sessions(self):
        content = _load_rknn_session("content_net.rknn")
        distortion = _load_rknn_session("distortion_net.rknn")
        aggregation = _load_rknn_session("aggregation_net.rknn")
        yield content, distortion, aggregation
        content.release()
        distortion.release()
        aggregation.release()

    def _run_rknn_pipeline(self, rknn_sessions, content_input, patches_input):
        """Run the three RKNN models with pre-processed inputs."""
        content_sess, distortion_sess, aggregation_sess = rknn_sessions

        # Content features
        content_feat = content_sess.inference(
            inputs=[content_input],
            data_format='nchw')[0]  # (1, 128, 8, 8)

        # Distortion: run each patch individually (batch=1)
        patch_feats = []
        for i in range(patches_input.shape[0]):
            feat = distortion_sess.inference(
                inputs=[patches_input[i:i+1]],
                data_format='nchw')[0]  # (1, 128, 8, 8)
            patch_feats.append(feat)
        patch_feats = np.concatenate(patch_feats, axis=0)  # (9, 128, 8, 8)

        # Reassemble 3x3 grid: (9, 128, 8, 8) -> (1, 128, 24, 24)
        patch_feats = patch_feats.reshape(3, 3, 128, 8, 8)
        patch_feats = np.transpose(patch_feats, (2, 0, 3, 1, 4))
        distortion_feat = patch_feats.reshape(1, 128, 24, 24)

        # Aggregation
        score = aggregation_sess.inference(
            inputs=[content_feat, distortion_feat],
            data_format='nchw')[0]  # (1, 1)

        return float(score.flatten()[0])

    def test_score_matches_reference(self, reference, rknn_sessions):
        """RKNN NPU score should match pre-computed PyTorch reference."""
        # Regenerate the same deterministic inputs used by export_rknn.py
        np.random.seed(reference["numpy_seed"])
        content_input = np.random.randn(
            *reference["content_input_shape"]).astype(np.float32)
        patches_input = np.random.randn(
            *reference["patches_input_shape"]).astype(np.float32)

        rknn_score = self._run_rknn_pipeline(
            rknn_sessions, content_input, patches_input)
        ref_score = reference["score"]

        assert rknn_score == pytest.approx(ref_score, abs=0.05), (
            f"RKNN score {rknn_score} vs PyTorch reference {ref_score}"
        )
