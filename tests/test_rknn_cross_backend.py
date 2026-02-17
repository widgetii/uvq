"""Cross-backend consistency tests: PyTorch (pre-computed reference) vs RKNN-Lite NPU.

Unlike the ONNX cross-backend test which runs both backends live, this test
is designed to run on the target device (Orange Pi 5 Plus / RK3588S) where
PyTorch is not installed.  The expected score is read from reference.json,
which is generated on the PC by scripts/export_rknn.py alongside the .rknn
model files.

The test uses the same hybrid pipeline as the production code: content and
distortion nets on NPU via RKNN-Lite, aggregation net on CPU via ONNX Runtime.
"""

import json
import os

import numpy as np
import pytest

try:
    from rknnlite.api import RKNNLite
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

rknn_installed = pytest.mark.skipif(
    not HAS_RKNN,
    reason="rknn-toolkit-lite2 is not installed",
)

ort_installed = pytest.mark.skipif(
    not HAS_ORT,
    reason="onnxruntime is not installed",
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RKNN_MODELS_DIR = os.path.join(REPO_ROOT, "uvq1p5_rknn", "models")
ONNX_MODELS_DIR = os.path.join(REPO_ROOT, "uvq1p5_web", "public", "models")

rknn_models_exist = pytest.mark.skipif(
    not os.path.isfile(os.path.join(RKNN_MODELS_DIR, "content_net.rknn")),
    reason="RKNN models not exported (run scripts/export_rknn.py first)",
)

onnx_agg_exists = pytest.mark.skipif(
    not os.path.isfile(os.path.join(ONNX_MODELS_DIR, "aggregation_net.onnx")),
    reason="aggregation_net.onnx not found (run scripts/export_onnx.py first)",
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
@ort_installed
@rknn_models_exist
@onnx_agg_exists
@reference_exists
@pytest.mark.slow
@pytest.mark.rknn
class TestRKNNCrossBackendConsistency:
    """Compare hybrid RKNN+ONNX pipeline against a PyTorch reference score.

    Content and distortion nets run on the NPU via RKNN-Lite.  The
    aggregation net runs on CPU via ONNX Runtime (workaround for NPU driver
    bugs on older rknpu drivers like 0.8.2).

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
        yield content, distortion
        content.release()
        distortion.release()

    @pytest.fixture(scope="class")
    def aggregation_sess(self):
        return ort.InferenceSession(
            os.path.join(ONNX_MODELS_DIR, "aggregation_net.onnx"),
            providers=["CPUExecutionProvider"],
        )

    def _run_pipeline(self, rknn_sessions, aggregation_sess,
                      content_input, patches_input):
        """Run content/distortion on NPU, aggregation on CPU."""
        content_sess, distortion_sess = rknn_sessions

        # Content features (NPU)
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

        # Aggregation (CPU via ONNX Runtime)
        score = aggregation_sess.run(
            None,
            {"content": content_feat, "distortion": distortion_feat},
        )[0]  # (1, 1)

        return float(score.flatten()[0])

    def test_score_matches_reference(self, reference, rknn_sessions,
                                     aggregation_sess):
        """Hybrid RKNN+ONNX score should match pre-computed PyTorch reference."""
        # Regenerate the same deterministic inputs used by export_rknn.py
        np.random.seed(reference["numpy_seed"])
        content_input = np.random.randn(
            *reference["content_input_shape"]).astype(np.float32)
        patches_input = np.random.randn(
            *reference["patches_input_shape"]).astype(np.float32)

        score = self._run_pipeline(
            rknn_sessions, aggregation_sess,
            content_input, patches_input)
        ref_score = reference["score"]

        assert score == pytest.approx(ref_score, abs=0.05), (
            f"Hybrid RKNN+ONNX score {score} vs PyTorch reference {ref_score}"
        )
