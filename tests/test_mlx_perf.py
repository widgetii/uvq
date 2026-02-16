"""Performance tests for MLX UVQ 1.5 inference pipeline.

Run with:
    pytest -m "perf and mlx" -v -s
"""

import math
import os
import sys
import time
import urllib.request
from contextlib import contextmanager

import numpy as np
import pytest

mlx_required = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="MLX requires macOS (Apple Silicon)",
)

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

mlx_installed = pytest.mark.skipif(
    not HAS_MLX,
    reason="mlx is not installed",
)

from utils import probe
from utils.video_reader import load_video_1p5

# ---------------------------------------------------------------------------
# Demo video download
# ---------------------------------------------------------------------------

GCS_BASE = "https://storage.googleapis.com/ugc-dataset/vp9_compressed_videos/"
DEMO_CACHE_DIR = "/tmp/uvq_demo_cache"


def _download_demo(filename):
    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)
    local_path = os.path.join(DEMO_CACHE_DIR, filename)
    if os.path.exists(local_path):
        return local_path
    url = GCS_BASE + filename
    print(f"  Downloading {filename} ...")
    urllib.request.urlretrieve(url, local_path)
    return local_path


@pytest.fixture(scope="session")
def demo_video_720p():
    return _download_demo("Gaming_720P-25aa_orig.mp4")


@pytest.fixture(scope="session")
def demo_video_1080p():
    return _download_demo("Gaming_1080P-0ef8_orig.mp4")


# ---------------------------------------------------------------------------
# Timing infrastructure
# ---------------------------------------------------------------------------

@contextmanager
def _time_block(label, timings_dict):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    timings_dict[label] = elapsed


def _format_timing_table(timings, total=None):
    if total is None:
        total = sum(timings.values())
    lines = []
    max_label = max(len(k) for k in timings) if timings else 10
    header = f"  {'Stage':<{max_label}}  {'Time (s)':>10}  {'%':>6}"
    lines.append(header)
    lines.append("  " + "-" * (max_label + 20))
    for label, elapsed in timings.items():
        pct = elapsed / total * 100 if total > 0 else 0
        lines.append(f"  {label:<{max_label}}  {elapsed:>10.3f}  {pct:>5.1f}%")
    lines.append("  " + "-" * (max_label + 20))
    lines.append(f"  {'TOTAL':<{max_label}}  {total:>10.3f}  100.0%")
    return "\n".join(lines)


def _get_video_meta(video_path):
    dimensions = probe.get_dimensions(video_path)
    duration = probe.get_video_duration(video_path)
    orig_fps = probe.get_r_frame_rate(video_path)
    video_length = math.ceil(duration) if duration else 0
    transpose = bool(dimensions and dimensions[0] < dimensions[1])
    return video_length, transpose, orig_fps


# ===========================================================================
# MLX UVQ 1.5 — Stage-level profiling
# ===========================================================================

@mlx_required
@mlx_installed
@pytest.mark.perf
@pytest.mark.slow
@pytest.mark.mlx
class TestMLXPerfStages:
    """Profile individual stages of the MLX UVQ 1.5 pipeline."""

    def test_model_loading(self):
        timings = {}
        with _time_block("UVQ1p5_MLX()", timings):
            from uvq1p5_mlx.utils.uvq1p5 import UVQ1p5
            model = UVQ1p5()
        print(f"\n  MLX model loaded")
        print(_format_timing_table(timings))

    def test_forward_pass(self, demo_video_720p):
        from uvq1p5_mlx.utils.uvq1p5 import UVQ1p5
        model = UVQ1p5()

        video_length, transpose, _ = _get_video_meta(demo_video_720p)
        video, _ = load_video_1p5(
            demo_video_720p, video_length, transpose, video_fps=1,
        )
        # video: (S, F, H, W, 3) — already NHWC
        num_seconds, fps, h, w, c = video.shape
        num_frames = num_seconds * fps
        video = video.reshape(num_frames, 1, h, w, c)

        timings = {}
        batch_size = 24
        predictions = []
        with _time_block("forward_pass", timings):
            for i in range(0, num_frames, batch_size):
                batch = mx.array(video[i:i+batch_size], dtype=mx.float32)
                pred = model.uvq1p5_core(batch)
                mx.eval(pred)
                predictions.append(pred)
            prediction = mx.concatenate(predictions, axis=0)
        score = mx.mean(prediction).item()
        print(f"\n  Frames: {num_frames}, score: {score:.3f}")
        print(_format_timing_table(timings))
        assert 1.0 <= score <= 5.0

    def test_end_to_end(self, demo_video_720p):
        timings = {}

        with _time_block("probe", timings):
            video_length, transpose, orig_fps = _get_video_meta(demo_video_720p)

        with _time_block("video_decode", timings):
            video, _ = load_video_1p5(
                demo_video_720p, video_length, transpose, video_fps=1,
            )

        with _time_block("model_loading", timings):
            from uvq1p5_mlx.utils.uvq1p5 import UVQ1p5
            model = UVQ1p5()

        num_seconds, fps, h, w, c = video.shape
        num_frames = num_seconds * fps
        video = video.reshape(num_frames, 1, h, w, c)

        batch_size = 24
        predictions = []
        with _time_block("forward_pass", timings):
            for i in range(0, num_frames, batch_size):
                batch = mx.array(video[i:i+batch_size], dtype=mx.float32)
                pred = model.uvq1p5_core(batch)
                mx.eval(pred)
                predictions.append(pred)
            prediction = mx.concatenate(predictions, axis=0)

        with _time_block("scoring", timings):
            video_score = mx.mean(prediction).item()
            frame_scores = np.array(prediction).flatten().tolist()

        total = sum(timings.values())
        print(f"\n  === MLX UVQ 1.5 End-to-End (720p input) ===")
        print(f"  Score: {video_score:.3f}, frames: {len(frame_scores)}")
        print(_format_timing_table(timings, total))
        assert 1.0 <= video_score <= 5.0


@mlx_required
@mlx_installed
@pytest.mark.perf
@pytest.mark.slow
@pytest.mark.mlx
class TestMLXPerf1080p:
    def test_end_to_end(self, demo_video_1080p):
        timings = {}

        with _time_block("probe", timings):
            video_length, transpose, orig_fps = _get_video_meta(demo_video_1080p)

        with _time_block("video_decode", timings):
            video, _ = load_video_1p5(
                demo_video_1080p, video_length, transpose, video_fps=1,
            )

        with _time_block("model_loading", timings):
            from uvq1p5_mlx.utils.uvq1p5 import UVQ1p5
            model = UVQ1p5()

        num_seconds, fps, h, w, c = video.shape
        num_frames = num_seconds * fps
        video = video.reshape(num_frames, 1, h, w, c)

        batch_size = 24
        predictions = []
        with _time_block("forward_pass", timings):
            for i in range(0, num_frames, batch_size):
                batch = mx.array(video[i:i+batch_size], dtype=mx.float32)
                pred = model.uvq1p5_core(batch)
                mx.eval(pred)
                predictions.append(pred)
            prediction = mx.concatenate(predictions, axis=0)

        with _time_block("scoring", timings):
            video_score = mx.mean(prediction).item()
            frame_scores = np.array(prediction).flatten().tolist()

        total = sum(timings.values())
        print(f"\n  === MLX UVQ 1.5 End-to-End (1080p native input) ===")
        print(f"  Score: {video_score:.3f}, frames: {len(frame_scores)}")
        print(_format_timing_table(timings, total))
        assert 1.0 <= video_score <= 5.0
