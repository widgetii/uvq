"""Export UVQ 1.5 PyTorch models to ONNX format.

Produces three ONNX models:
  - content_net.onnx:     (N, 3, 256, 256)  -> (N, 128, 8, 8)
  - distortion_net.onnx:  (N, 3, 360, 640)  -> (N, 128, 8, 8)
  - aggregation_net.onnx: (N, 128, 8, 8) + (N, 128, 24, 24) -> (N, 1)

Also generates reference.json with a synthetic test case for cross-runtime
validation (PyTorch vs ONNX Runtime vs browser).

Usage:
    uv run python scripts/export_onnx.py [--output_dir uvq1p5_web/public/models]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# Ensure repo root is importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from uvq1p5_pytorch.utils.uvq1p5 import UVQ1p5


# ---------------------------------------------------------------------------
# ONNX wrapper modules (strip NHWC permutes, dict inputs, F.interpolate)
# ---------------------------------------------------------------------------

class ContentNetCoreONNX(nn.Module):
    """Content feature extractor without F.interpolate.

    Browser/Node.js callers resize to 256x256 before invoking this model.
    Input:  (N, 3, 256, 256)   NCHW float32 in [-1, 1]
    Output: (N, 128, 8, 8)     NCHW float32
    """

    def __init__(self, core):
        super().__init__()
        self.features = core.features

    def forward(self, x):
        return self.features(x)


class DistortionNetCoreONNX(nn.Module):
    """Distortion feature extractor for a single patch.

    Keeps EfficientNet features + MaxPool2d, strips PermuteLayerNHWC.
    Input:  (N, 3, 360, 640)   NCHW float32 in [-1, 1]
    Output: (N, 128, 8, 8)     NCHW float32
    """

    def __init__(self, core):
        super().__init__()
        self.features = core.features
        self.maxpool = nn.MaxPool2d(kernel_size=(5, 13), stride=1, padding=0)

    def forward(self, x):
        x = self.features(x)
        return self.maxpool(x)


class AggregationNetCoreONNX(nn.Module):
    """Aggregation net taking two NCHW tensors directly (no dict wrapper).

    Input:  content    (N, 128, 8, 8)   NCHW float32
            distortion (N, 128, 24, 24) NCHW float32
    Output: (N, 1)  quality score in [1, 5]
    """

    def __init__(self, core):
        super().__init__()
        self.resizer = core.resizer
        self.conv1 = core.conv1
        self.ln1 = core.ln1
        self.relu1 = core.relu1
        self.maxpool1 = core.maxpool1
        self.linear1 = core.linear1
        self.tanh = core.tanh

    def forward(self, content, distortion):
        content = self.resizer(content)
        distortion = self.resizer(distortion)
        x = torch.cat([content, distortion], dim=1)
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = nn.Flatten()(x)
        x = self.linear1(x)
        x = self.tanh(x) * 2 + 3
        return x


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def verify_onnx(onnx_path, dummy_inputs, pytorch_outputs, atol=5e-3):
    """Verify ONNX model matches PyTorch output using onnxruntime."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    if isinstance(dummy_inputs, dict):
        feed = {k: v.numpy() for k, v in dummy_inputs.items()}
    else:
        input_name = sess.get_inputs()[0].name
        feed = {input_name: dummy_inputs.numpy()}

    ort_outputs = sess.run(None, feed)
    pt_np = pytorch_outputs.detach().numpy()

    max_diff = np.max(np.abs(ort_outputs[0] - pt_np))
    print(f"  Max diff: {max_diff:.2e} (atol={atol})")
    assert max_diff < atol, f"ONNX verification failed: max_diff={max_diff}"
    print("  OK")


def export_model(model, dummy_inputs, onnx_path, input_names, output_names,
                 dynamic_axes):
    """Export a PyTorch model to ONNX."""
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    if isinstance(dummy_inputs, dict):
        args = tuple(dummy_inputs.values())
    else:
        args = (dummy_inputs,)

    torch.onnx.export(
        model,
        args,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  Exported: {onnx_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export UVQ 1.5 to ONNX")
    parser.add_argument(
        "--output_dir",
        default=os.path.join(REPO_ROOT, "uvq1p5_web", "public", "models"),
        help="Directory for ONNX model files",
    )
    args = parser.parse_args()

    print("Loading PyTorch UVQ 1.5 model...")
    uvq = UVQ1p5(eval_mode=True, pretrained=True)

    # --- Content Net ---
    print("\nExporting ContentNet...")
    content_onnx = ContentNetCoreONNX(uvq.content_net.model)
    content_onnx.eval()

    dummy_content = torch.randn(1, 3, 256, 256)
    with torch.inference_mode():
        pt_content_out = content_onnx(dummy_content)
    print(f"  PyTorch output shape: {pt_content_out.shape}")

    content_path = os.path.join(args.output_dir, "content_net.onnx")
    export_model(
        content_onnx,
        dummy_content,
        content_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    verify_onnx(content_path, dummy_content, pt_content_out)

    # --- Distortion Net ---
    print("\nExporting DistortionNet...")
    distortion_onnx = DistortionNetCoreONNX(uvq.distortion_net.model)
    distortion_onnx.eval()

    dummy_distortion = torch.randn(1, 3, 360, 640)
    with torch.inference_mode():
        pt_distortion_out = distortion_onnx(dummy_distortion)
    print(f"  PyTorch output shape: {pt_distortion_out.shape}")

    distortion_path = os.path.join(args.output_dir, "distortion_net.onnx")
    export_model(
        distortion_onnx,
        dummy_distortion,
        distortion_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    verify_onnx(distortion_path, dummy_distortion, pt_distortion_out)

    # --- Aggregation Net ---
    print("\nExporting AggregationNet...")
    agg_onnx = AggregationNetCoreONNX(uvq.aggregation_net.model)
    agg_onnx.eval()

    dummy_content_feat = torch.randn(1, 128, 8, 8)
    dummy_distortion_feat = torch.randn(1, 128, 24, 24)
    with torch.inference_mode():
        pt_agg_out = agg_onnx(dummy_content_feat, dummy_distortion_feat)
    print(f"  PyTorch output shape: {pt_agg_out.shape}")

    agg_path = os.path.join(args.output_dir, "aggregation_net.onnx")
    export_model(
        agg_onnx,
        {"content": dummy_content_feat, "distortion": dummy_distortion_feat},
        agg_path,
        input_names=["content", "distortion"],
        output_names=["output"],
        dynamic_axes={
            "content": {0: "batch"},
            "distortion": {0: "batch"},
            "output": {0: "batch"},
        },
    )
    verify_onnx(
        agg_path,
        {"content": dummy_content_feat, "distortion": dummy_distortion_feat},
        pt_agg_out,
    )

    # --- Generate reference.json for cross-runtime validation ---
    print("\nGenerating reference.json...")
    torch.manual_seed(42)
    np.random.seed(42)

    ref_content_input = torch.randn(1, 3, 256, 256)
    ref_patches_input = torch.randn(9, 3, 360, 640)

    with torch.inference_mode():
        ref_content_feat = content_onnx(ref_content_input)
        ref_distortion_patches = distortion_onnx(ref_patches_input)

        # Reassemble 3x3 grid: (9, 128, 8, 8) -> (1, 128, 24, 24)
        patches = ref_distortion_patches.reshape(3, 3, 128, 8, 8)
        patches = patches.permute(2, 0, 3, 1, 4)  # (128, 3, 8, 3, 8)
        ref_distortion_feat = patches.reshape(1, 128, 24, 24)

        ref_score = agg_onnx(ref_content_feat, ref_distortion_feat)

    reference = {
        "content_input_shape": list(ref_content_input.shape),
        "patches_input_shape": list(ref_patches_input.shape),
        "content_feat_shape": list(ref_content_feat.shape),
        "distortion_feat_shape": list(ref_distortion_feat.shape),
        "score": ref_score.item(),
        "content_input_sum": ref_content_input.sum().item(),
        "patches_input_sum": ref_patches_input.sum().item(),
        "torch_seed": 42,
        "numpy_seed": 42,
    }

    ref_path = os.path.join(args.output_dir, "reference.json")
    with open(ref_path, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"  Score: {reference['score']:.6f}")
    print(f"  Saved: {ref_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
