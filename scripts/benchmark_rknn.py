"""Benchmark UVQ 1.5 RKNN pipeline on RK3588 NPU.

Measures per-stage and per-frame timing for comparison with other backends
in PERFORMANCE.md. Uses synthetic inputs matching the real pipeline shapes.
"""
import json
import time

import numpy as np

MODEL_DIR = "/home/orangepi/uvq/models"
N_WARMUP = 3
N_FRAMES = 20


def benchmark():
    # --- Model Loading ---
    from rknnlite.api import RKNNLite
    import onnxruntime as ort

    t0 = time.perf_counter()

    c_rknn = RKNNLite(verbose=False)
    c_rknn.load_rknn(f"{MODEL_DIR}/content_net.rknn")
    c_rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

    d_rknn = RKNNLite(verbose=False)
    d_rknn.load_rknn(f"{MODEL_DIR}/distortion_net.rknn")
    d_rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_1)

    agg_sess = ort.InferenceSession(
        f"{MODEL_DIR}/aggregation_net.onnx",
        providers=["CPUExecutionProvider"],
    )

    t_load = time.perf_counter() - t0
    print(f"Model loading: {t_load:.3f} s")

    # --- Generate synthetic frames ---
    # Simulate 1080p frames (1920x1080) resized to model input format:
    #   content: (1, 3, 256, 256)
    #   distortion: 9 patches of (1, 3, 360, 640)
    np.random.seed(42)
    content_frames = [
        np.random.randn(1, 3, 256, 256).astype(np.float32)
        for _ in range(N_FRAMES)
    ]
    distortion_patches = [
        np.random.randn(9, 3, 360, 640).astype(np.float32)
        for _ in range(N_FRAMES)
    ]

    # --- Warmup ---
    print(f"Warming up ({N_WARMUP} frames)...")
    for i in range(N_WARMUP):
        cf = c_rknn.inference(inputs=[content_frames[0]])[0]
        for j in range(9):
            d_rknn.inference(inputs=[distortion_patches[0][j:j+1]])

    # --- Benchmark ---
    print(f"Benchmarking {N_FRAMES} frames...")
    content_times = []
    distortion_times = []
    agg_times = []
    total_times = []

    for i in range(N_FRAMES):
        t_frame_start = time.perf_counter()

        # Content net
        t0 = time.perf_counter()
        content_feat = c_rknn.inference(inputs=[content_frames[i]])[0]
        content_times.append(time.perf_counter() - t0)

        # Distortion net (9 patches)
        t0 = time.perf_counter()
        patch_feats = []
        for j in range(9):
            pf = d_rknn.inference(inputs=[distortion_patches[i][j:j+1]])[0]
            patch_feats.append(pf)
        patch_feats = np.concatenate(patch_feats, axis=0)

        # Reassemble 3x3 grid: (9,128,8,8) -> (1,128,24,24)
        pf = patch_feats.reshape(3, 3, 128, 8, 8)
        pf = pf.transpose(2, 0, 3, 1, 4).reshape(1, 128, 24, 24)
        distortion_times.append(time.perf_counter() - t0)

        # Aggregation net (ONNX Runtime on CPU)
        t0 = time.perf_counter()
        score = agg_sess.run(
            None, {"content": content_feat, "distortion": pf}
        )[0]
        agg_times.append(time.perf_counter() - t0)

        total_times.append(time.perf_counter() - t_frame_start)

    # --- Results ---
    def ms_stats(times):
        arr = np.array(times) * 1000
        return arr.mean(), np.median(arr), arr.std(), arr.min(), arr.max()

    print(f"\n{'='*60}")
    print(f"UVQ 1.5 RKNN Benchmark — {N_FRAMES} frames")
    print(f"{'='*60}")

    headers = ["Stage", "Mean (ms)", "Median (ms)", "Std", "Min", "Max"]
    rows = []
    for name, times in [
        ("content_net (NPU)", content_times),
        ("distortion_net (NPU, 9 patches)", distortion_times),
        ("aggregation_net (CPU/ONNX)", agg_times),
        ("total_pipeline", total_times),
    ]:
        mean, med, std, mn, mx = ms_stats(times)
        rows.append((name, mean, med, std, mn, mx))

    # Print table
    print(f"\n{'Stage':<35} {'Mean':>8} {'Median':>8} {'Std':>6} {'Min':>8} {'Max':>8}")
    print("-" * 80)
    for name, mean, med, std, mn, mx in rows:
        print(f"{name:<35} {mean:>7.1f}  {med:>7.1f}  {std:>5.1f}  {mn:>7.1f}  {mx:>7.1f}")

    print(f"\nModel loading: {t_load*1000:.0f} ms")

    total_inference = sum(total_times)
    print(f"Total inference ({N_FRAMES} frames): {total_inference:.3f} s")
    print(f"Average throughput: {N_FRAMES/total_inference:.1f} fps")

    # Verify score is reasonable
    print(f"\nLast frame score: {score.flatten()[0]:.4f}")

    c_rknn.release()
    d_rknn.release()


if __name__ == "__main__":
    benchmark()
