# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UVQ (Universal Video Quality) is Google's no-reference perceptual video quality assessment framework with PyTorch implementations for UVQ 1.0 and UVQ 1.5 models. It processes user-generated video content and outputs quality scores on a 0-5 scale.

## Commands

### Install dependencies
```bash
pip install -r requirements.txt
```
FFmpeg and FFprobe must also be available on PATH (or specified via `--ffmpeg_path`/`--ffprobe_path`).

### Run inference
```bash
# UVQ 1.5 (recommended)
python uvq_inference.py <video_file> --model_version 1.5

# UVQ 1.0 (legacy)
python uvq_inference.py <video_file> --model_version 1.0

# GPU acceleration
python uvq_inference.py <video_file> --model_version 1.5 --device cuda

# Batch mode (pass a .txt file listing video paths)
python uvq_inference.py video_list.txt --model_version 1.5 --output batch_results.txt
```

There are no tests, linters, or build steps configured in this repository.

## Architecture

### Data Flow

```
Video → probe.py (FFprobe metadata) → video_reader.py (FFmpeg decode + resize + normalize)
      → Model (ContentNet + DistortionNet → AggregationNet) → quality score
```

### Key Entry Point

`uvq_inference.py` — CLI dispatcher that routes to `run_single_inference()` or `run_batch_inference()` based on whether the input is a single video file or a .txt list.

### Two Model Versions

**UVQ 1.5** (`uvq1p5_pytorch/`) — Default, lighter model (~30MB checkpoints). Uses two EfficientNet-B0 branches (content + distortion) fused by an aggregation net. Input: 1080p, configurable fps (default 1). Output: single `uvq1p5_score`.

**UVQ 1.0** (`uvq_pytorch/`) — Legacy, larger model (~169MB checkpoints). Uses three separate networks (compression via 3D Inception, content via EfficientNet, distortion) with a multi-combination aggregation net. Input: 720p + 496x496 downsampled, 5 fps. Output: multiple scores (compression, content, distortion, and combinations).

### Shared Utilities

- `utils/video_reader.py` — FFmpeg subprocess decoding to raw RGB24, resize, normalize to [-1, 1]. Separate load functions per model version (`load_video_1p0`, `load_video_1p5`).
- `utils/probe.py` — FFprobe wrappers for duration, frame rate, dimensions, frame count.

### TensorFlow Compatibility Layer

Both model versions have `custom_nn_layers.py` modules that emulate TensorFlow's "same" padding in PyTorch (`Conv2dSamePadding`, `Conv3DSamePadding`, `MBConvSamePadding`, etc.). This is critical because the pre-trained weights were originally trained in TensorFlow.

### Pre-trained Weights

Checkpoints are committed directly in `uvq1p5_pytorch/checkpoints/` and `uvq_pytorch/checkpoint/`. The `models/` directory contains TensorFlow baseline models (not used by the PyTorch inference code).
