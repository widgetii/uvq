# Performance Benchmarks

Benchmark numbers for UVQ inference pipelines on a single workstation.

## Hardware

### x86 Workstation

| Component | Spec |
|-----------|------|
| CPU | Intel Core i7-6700K @ 4.00 GHz (4 cores / 8 threads) |
| RAM | 32 GB DDR4 |
| GPU | NVIDIA GeForce GTX 980 Ti (6 GB VRAM, sm_52) |
| Storage | NVMe SSD |
| OS | Arch Linux 6.12, Python 3.12 |

### Apple Silicon

| Component | Spec |
|-----------|------|
| SoC | Apple M4 (10-core CPU / 10-core GPU) |
| RAM | 16 GB unified memory |
| Storage | NVMe SSD |
| OS | macOS 26.3, Python 3.12 |

## Test Video

All benchmarks use the same 20-second 720p gaming clip (`Gaming_720P-25aa_orig.mp4`, 1280x720, 30 fps, ~11 MB) from the YouTube-UGC dataset. 1080p benchmarks use `Gaming_1080P-0ef8_orig.mp4` (1920x1080, ~80 MB).

## UVQ 1.5 -- CPU

Input: 720p video, 1 fps sampling, batch_size=24.

| Stage | Time (s) | % |
|-------|----------|---|
| probe | 0.194 | 0.6% |
| video_decode | 3.576 | 11.1% |
| tensor_prep | 0.075 | 0.2% |
| model_loading | 0.126 | 0.4% |
| forward_pass | 28.354 | 87.7% |
| scoring | <0.001 | 0.0% |
| **TOTAL** | **32.326** | |

Score: **3.034**

### UVQ 1.5 CPU -- 1080p native input

| Stage | Time (s) | % |
|-------|----------|---|
| probe | 0.238 | 0.8% |
| video_decode | 2.985 | 9.5% |
| tensor_prep | 0.083 | 0.3% |
| model_loading | 0.124 | 0.4% |
| forward_pass | 27.998 | 89.1% |
| scoring | <0.001 | 0.0% |
| **TOTAL** | **31.428** | |

Score: **3.914**

The 1080p video decodes slightly faster (no upscale needed) but the forward pass time is similar since the model always operates on 1080p frames.

## UVQ 1.0 -- CPU

Input: 720p video, 5 fps sampling, dual-stream (720p + 496x496).

| Stage | Time (s) | % |
|-------|----------|---|
| probe | 0.199 | 0.3% |
| video_decode | 3.876 | 6.0% |
| model_loading | 0.565 | 0.9% |
| contentnet_forward | 2.557 | 4.0% |
| compressionnet_forward | 47.778 | 74.0% |
| distortionnet_forward | 9.439 | 14.6% |
| aggregation | 0.123 | 0.2% |
| **TOTAL** | **64.537** | |

Combined score: **3.236**

CompressionNet dominates at 74% of total time, processing 16 patches (4x4 grid) per second of video through a 3D Inception network.

## UVQ 1.5 -- GPU (GTX 980 Ti)

PyTorch 2.3.0+cu121 (last version supporting sm_52).

### 720p input (batch_size=4)

| Stage | Time (s) | % |
|-------|----------|---|
| probe | 0.130 | 2.3% |
| video_decode | 3.571 | 63.5% |
| tensor_prep | 0.078 | 1.4% |
| model_loading | 0.212 | 3.8% |
| gpu_transfer | 0.078 | 1.4% |
| forward_pass | 1.555 | 27.7% |
| scoring | <0.001 | 0.0% |
| **TOTAL** | **5.624** | |

### 1080p input (batch_size=1)

| Stage | Time (s) | % |
|-------|----------|---|
| probe | 0.155 | 3.0% |
| video_decode | 2.965 | 57.2% |
| tensor_prep | 0.137 | 2.6% |
| model_loading | 0.179 | 3.4% |
| gpu_transfer | 0.078 | 1.5% |
| forward_pass | 1.673 | 32.3% |
| scoring | <0.001 | 0.0% |
| **TOTAL** | **5.187** | |

Batch size is limited by the 6 GB VRAM on the GTX 980 Ti. Larger GPUs can use batch_size=24.

## UVQ 1.5 -- MLX (Apple M4)

MLX backend using Apple Silicon GPU acceleration. Input: 1 fps sampling, batch_size=24.

### 720p input

| Stage | Time (s) | % |
|-------|----------|---|
| probe | 0.630 | 2.0% |
| video_decode | 11.564 | 36.7% |
| model_loading | 0.040 | 0.1% |
| forward_pass | 19.272 | 61.1% |
| scoring | 0.034 | 0.1% |
| **TOTAL** | **31.540** | |

Score: **3.034**

### 1080p native input

| Stage | Time (s) | % |
|-------|----------|---|
| probe | 0.490 | 1.8% |
| video_decode | 11.767 | 43.1% |
| model_loading | 0.073 | 0.3% |
| forward_pass | 14.947 | 54.8% |
| scoring | 0.019 | 0.1% |
| **TOTAL** | **27.295** | |

Score: **3.923**

Model loading is near-instant (~0.04s) thanks to the safetensors format. Video decode is slower than x86 because FFmpeg runs on the efficiency cores. Scores match the PyTorch CPU backend within 0.01 tolerance.

## Comparison

### UVQ 1.5 vs UVQ 1.0 (CPU, 720p)

| Model | Time (s) | Score | Relative |
|-------|----------|-------|----------|
| UVQ 1.5 | 32.0 | 3.034 | **1.0x** |
| UVQ 1.0 | 63.6 | 3.236 | 2.0x slower |

### UVQ 1.5 CPU vs GPU vs MLX (720p)

| Device | Time (s) | Forward Pass (s) | Relative |
|--------|----------|-------------------|----------|
| CPU (i7-6700K) | 32.3 | 28.4 | 1.0x |
| MLX (Apple M4) | 31.5 | 19.3 | 1.0x (1.5x forward) |
| GPU (GTX 980 Ti) | 5.6 | 1.6 | **5.8x faster** |

The MLX backend forward pass is **1.5x faster** than i7-6700K CPU, but end-to-end time is similar due to slower FFmpeg decode on macOS. The forward pass alone is **18.2x faster** on the discrete GPU. End-to-end GPU speedup is lower (5.8x) because video decoding (FFmpeg) now dominates at 64% of total GPU pipeline time.

## Key Observations

1. **CPU bottleneck is model inference.** The forward pass accounts for 88% (UVQ 1.5) and 93% (UVQ 1.0, combined subnets) of CPU pipeline time.

2. **GPU bottleneck shifts to video decoding.** With GPU inference at 1.6s vs CPU at 28.4s, FFmpeg decode becomes the dominant stage (64% of GPU pipeline time).

3. **MLX provides moderate forward pass speedup.** The M4's GPU accelerates inference 1.5x vs x86 CPU, but FFmpeg decode on macOS is slower, keeping end-to-end times comparable.

4. **UVQ 1.0 CompressionNet is expensive.** The 3D Inception network processing 16 patches per second takes 48s alone -- more than the entire UVQ 1.5 pipeline.

5. **Model loading is fast.** UVQ 1.5 loads in ~0.1-0.2s (30 MB, PyTorch) or ~0.04s (safetensors, MLX). UVQ 1.0 in ~0.6s (169 MB). Neither is a bottleneck.

6. **VRAM limits batch size.** The GTX 980 Ti (6 GB) can only fit batch_size=4 for 720p and batch_size=1 for 1080p. Modern GPUs with 8+ GB should handle batch_size=24. MLX uses unified memory so batch_size=24 works on 16 GB M4.

## Reproducing

```bash
# CPU benchmarks (both models)
uv run pytest -m perf -v -s

# GPU benchmarks require a CUDA-capable GPU and a compatible PyTorch build
uv run python uvq_inference.py <video> --model_version 1.5 --device cuda

# MLX benchmarks (macOS Apple Silicon only)
uv sync --group mlx
uv run pytest -m "perf and mlx" -v -s
```
