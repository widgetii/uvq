# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UVQ (Universal Video Quality) is Google's no-reference perceptual video quality assessment framework with PyTorch implementations for UVQ 1.0 and UVQ 1.5 models. It processes user-generated video content and outputs quality scores on a 0-5 scale.

## Commands

### Install dependencies
```bash
uv sync
```
FFmpeg and FFprobe must also be available on PATH (or specified via `--ffmpeg_path`/`--ffprobe_path`).

### Run inference
```bash
# UVQ 1.5 (recommended)
uv run python uvq_inference.py <video_file> --model_version 1.5

# UVQ 1.0 (legacy)
uv run python uvq_inference.py <video_file> --model_version 1.0

# GPU acceleration
uv run python uvq_inference.py <video_file> --model_version 1.5 --device cuda

# MLX (Apple Silicon) acceleration
uv sync --group mlx
uv run python uvq_inference.py <video_file> --model_version 1.5 --device mlx

# Batch mode (pass a .txt file listing video paths)
uv run python uvq_inference.py video_list.txt --model_version 1.5 --output batch_results.txt

# WebGPU browser inference (export ONNX models first)
uv sync --group onnx
uv run python scripts/export_onnx.py
cd uvq1p5_web && npm install && npm run dev
```

### Run tests
```bash
uv run pytest tests/ -v --timeout=120 -k "not gpu and not perf"
```

## Architecture

### Data Flow

```
Video → probe.py (FFprobe metadata) → video_reader.py (FFmpeg decode + resize + normalize)
      → Model (ContentNet + DistortionNet → AggregationNet) → quality score
```

### Key Entry Point

`uvq_inference.py` — CLI dispatcher that routes to `run_single_inference()` or `run_batch_inference()` based on whether the input is a single video file or a .txt list.

### Two Model Versions

**UVQ 1.5** (`uvq1p5_pytorch/`) — Default, lighter model (~30MB checkpoints). Uses two EfficientNet-B0 branches (content + distortion) fused by an aggregation net. Input: 1080p, configurable fps (default 1). Output: single `uvq1p5_score`. Also available as an MLX backend (`uvq1p5_mlx/`) for Apple Silicon acceleration via `--device mlx`, and as a WebGPU/WASM browser backend (`uvq1p5_web/`) via ONNX Runtime Web.

**UVQ 1.0** (`uvq_pytorch/`) — Legacy, larger model (~169MB checkpoints). Uses three separate networks (compression via 3D Inception, content via EfficientNet, distortion) with a multi-combination aggregation net. Input: 720p + 496x496 downsampled, 5 fps. Output: multiple scores (compression, content, distortion, and combinations).

### Shared Utilities

- `utils/video_reader.py` — FFmpeg subprocess decoding to raw RGB24, resize, normalize to [-1, 1]. Separate load functions per model version (`load_video_1p0`, `load_video_1p5`).
- `utils/probe.py` — FFprobe wrappers for duration, frame rate, dimensions, frame count.

### TensorFlow Compatibility Layer

Both model versions have `custom_nn_layers.py` modules that emulate TensorFlow's "same" padding in PyTorch (`Conv2dSamePadding`, `Conv3DSamePadding`, `MBConvSamePadding`, etc.). This is critical because the pre-trained weights were originally trained in TensorFlow.

### Web/ONNX Backend

`uvq1p5_web/` — Browser-based inference using ONNX Runtime Web with WebGPU. Models are exported from PyTorch via `scripts/export_onnx.py` to three ONNX files (content_net, distortion_net, aggregation_net) stored in `uvq1p5_web/public/models/`. The `uvq1p5_web/src/uvq.js` engine is framework-agnostic and works with both `onnxruntime-web` (browser) and `onnxruntime-node` (Node.js CLI testing).

**WebGPU constraints in onnxruntime-web:**
- Pin `onnxruntime-web` to **1.21.x**. Versions >=1.22 introduce aggressive GPU buffer destruction that causes "Buffer used in submit while destroyed" errors with multiple sessions on a shared WebGPU device. The `onnxruntime-web/webgpu` subpath doesn't exist in <=1.20.
- WebGPU sessions must be created **sequentially** (no `Promise.all`); the EP rejects concurrent session creation.
- **Skip warmup** for the WebGPU backend — dummy inference triggers the same buffer errors. The first real inference compiles shader pipelines on demand.
- Always **dispose output tensors** after reading their `.data` to release GPU buffers (see `uvq.js` methods). Leaking output tensors causes stale buffer references.

### Pre-trained Weights

Checkpoints are committed directly in `uvq1p5_pytorch/checkpoints/` and `uvq_pytorch/checkpoint/`. The `models/` directory contains TensorFlow baseline models (not used by the PyTorch inference code). ONNX models in `uvq1p5_web/public/models/` are generated (not committed) — run `scripts/export_onnx.py` to create them.
