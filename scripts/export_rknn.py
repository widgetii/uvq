"""Export UVQ 1.5 ONNX models to RKNN format for Rockchip RK3588S NPU.

Converts the three ONNX models produced by export_onnx.py into .rknn files:
  - content_net.rknn:     (1, 3, 256, 256)  -> (1, 128, 8, 8)
  - distortion_net.rknn:  (1, 3, 360, 640)  -> (1, 128, 8, 8)
  - aggregation_net.rknn: (1, 128, 8, 8) + (1, 128, 24, 24) -> (1, 1)

Prerequisites:
  - ONNX models must exist (run scripts/export_onnx.py first)
  - rknn-toolkit2 must be installed (x86_64 PC, from Rockchip GitHub releases)

Usage:
    python scripts/export_rknn.py [--onnx_dir ...] [--output_dir ...] [--quantize --dataset ...]

Board setup (Orange Pi 5 Plus / RK3588S):
  1. NPU driver: switch to vendor BSP kernel or install out-of-tree rknpu.ko
  2. pip install rknn-toolkit-lite2  (aarch64 wheel from Rockchip GitHub)
  3. sudo apt install ffmpeg
  4. Copy .rknn files from PC to uvq1p5_rknn/models/
"""

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def convert_model(rknn_cls, onnx_path, rknn_path, input_shapes,
                  input_names=None, quantize=False, dataset_path=None):
    """Convert a single ONNX model to RKNN format."""
    rknn = rknn_cls(verbose=False)

    # No mean/std normalization — inputs are already float32 in [-1, 1]
    rknn.config(target_platform="rk3588")

    # RKNN doesn't support dynamic axes — fix the batch dimension via
    # input_size_list.  For multi-input models we also pass input names.
    size_list = [list(s) for s in input_shapes]
    print(f"  Loading ONNX: {onnx_path}")
    ret = rknn.load_onnx(
        model=onnx_path,
        inputs=input_names,
        input_size_list=size_list,
    )
    if ret != 0:
        raise RuntimeError(f"Failed to load ONNX model: {onnx_path} (ret={ret})")

    print(f"  Building RKNN (quantize={quantize})...")
    ret = rknn.build(do_quantization=quantize, dataset=dataset_path)
    if ret != 0:
        raise RuntimeError(f"Failed to build RKNN model (ret={ret})")

    print(f"  Exporting: {rknn_path}")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        raise RuntimeError(f"Failed to export RKNN model (ret={ret})")

    size_mb = os.path.getsize(rknn_path) / (1024 * 1024)
    print(f"  OK ({size_mb:.1f} MB)")

    # Verify with simulator if available
    try:
        ret = rknn.init_runtime(target=None)  # simulator mode
        if ret == 0:
            # Run a quick sanity check with random input
            feeds = [np.random.randn(*s).astype(np.float32) for s in input_shapes]
            outputs = rknn.inference(inputs=feeds)
            print(f"  Simulator check: output shape(s) = "
                  f"{[o.shape for o in outputs]}")
        rknn.release()
    except Exception as e:
        print(f"  Simulator check skipped: {e}")
        rknn.release()


def main():
    parser = argparse.ArgumentParser(
        description="Export UVQ 1.5 ONNX models to RKNN format")
    parser.add_argument(
        "--onnx_dir",
        default=os.path.join(REPO_ROOT, "uvq1p5_web", "public", "models"),
        help="Directory containing ONNX models from export_onnx.py",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(REPO_ROOT, "uvq1p5_rknn", "models"),
        help="Directory for RKNN model files",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable INT8 quantization (requires --dataset)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to calibration dataset text file (required if --quantize)",
    )
    args = parser.parse_args()

    if args.quantize and not args.dataset:
        parser.error("--dataset is required when --quantize is set")

    # Check ONNX models exist
    model_names = ["content_net.onnx", "distortion_net.onnx", "aggregation_net.onnx"]
    for name in model_names:
        path = os.path.join(args.onnx_dir, name)
        if not os.path.isfile(path):
            print(f"Error: ONNX model not found: {path}")
            print("Run scripts/export_onnx.py first.")
            sys.exit(1)

    try:
        from rknn.api import RKNN
    except ImportError:
        print("Error: rknn-toolkit2 is not installed.")
        print("Install from: https://github.com/airockchip/rknn-toolkit2")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Content Net ---
    print("\nConverting ContentNet...")
    convert_model(
        RKNN,
        os.path.join(args.onnx_dir, "content_net.onnx"),
        os.path.join(args.output_dir, "content_net.rknn"),
        input_shapes=[(1, 3, 256, 256)],
        input_names=["input"],
        quantize=args.quantize,
        dataset_path=args.dataset,
    )

    # --- Distortion Net ---
    print("\nConverting DistortionNet...")
    convert_model(
        RKNN,
        os.path.join(args.onnx_dir, "distortion_net.onnx"),
        os.path.join(args.output_dir, "distortion_net.rknn"),
        input_shapes=[(1, 3, 360, 640)],
        input_names=["input"],
        quantize=args.quantize,
        dataset_path=args.dataset,
    )

    # --- Aggregation Net ---
    print("\nConverting AggregationNet...")
    convert_model(
        RKNN,
        os.path.join(args.onnx_dir, "aggregation_net.onnx"),
        os.path.join(args.output_dir, "aggregation_net.rknn"),
        input_shapes=[(1, 128, 8, 8), (1, 128, 24, 24)],
        input_names=["content", "distortion"],
        quantize=args.quantize,
        dataset_path=args.dataset,
    )

    # --- Generate reference.json for on-device validation ---
    # Uses numpy RNG (reproducible on aarch64 without PyTorch) to generate
    # deterministic inputs, then runs them through PyTorch to get the expected
    # score.  The on-device test regenerates the same inputs from the seed and
    # compares the RKNN output against this saved reference.
    print("\nGenerating reference.json...")
    import json
    import torch

    from scripts.export_onnx import (
        ContentNetCoreONNX,
        DistortionNetCoreONNX,
        AggregationNetCoreONNX,
    )
    from uvq1p5_pytorch.utils.uvq1p5 import UVQ1p5

    uvq = UVQ1p5(eval_mode=True, pretrained=True)

    content_onnx = ContentNetCoreONNX(uvq.content_net.model)
    content_onnx.eval()
    distortion_onnx = DistortionNetCoreONNX(uvq.distortion_net.model)
    distortion_onnx.eval()
    agg_onnx = AggregationNetCoreONNX(uvq.aggregation_net.model)
    agg_onnx.eval()

    np.random.seed(42)
    ref_content_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
    ref_patches_input = np.random.randn(9, 3, 360, 640).astype(np.float32)

    with torch.inference_mode():
        ref_content_feat = content_onnx(
            torch.from_numpy(ref_content_input))
        ref_patch_feats = distortion_onnx(
            torch.from_numpy(ref_patches_input))

        # Reassemble 3x3 grid: (9, 128, 8, 8) -> (1, 128, 24, 24)
        pf = ref_patch_feats.reshape(3, 3, 128, 8, 8)
        pf = pf.permute(2, 0, 3, 1, 4)  # (128, 3, 8, 3, 8)
        ref_distortion_feat = pf.reshape(1, 128, 24, 24)

        ref_score = agg_onnx(ref_content_feat, ref_distortion_feat)

    reference = {
        "numpy_seed": 42,
        "content_input_shape": list(ref_content_input.shape),
        "patches_input_shape": list(ref_patches_input.shape),
        "score": ref_score.item(),
    }

    ref_path = os.path.join(args.output_dir, "reference.json")
    with open(ref_path, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"  Score: {reference['score']:.6f}")
    print(f"  Saved: {ref_path}")

    print("\nDone. Copy .rknn files + reference.json to the device:")
    print(f"  scp {args.output_dir}/*.rknn {ref_path} "
          f"user@orangepi:path/to/uvq1p5_rknn/models/")


if __name__ == "__main__":
    main()
