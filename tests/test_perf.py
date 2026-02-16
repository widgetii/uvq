"""Performance tests for UVQ inference pipeline.

Downloads real demo videos from GCS (YouTube-UGC dataset) and profiles
each stage of the UVQ 1.5 and UVQ 1.0 inference pipelines independently.

Run with:
    pytest -m perf -v -s           # all perf tests (-s needed for timing output)
    pytest -m perf -v -s -k "1p5"  # UVQ 1.5 only
    pytest -m perf -v -s -k "1p0"  # UVQ 1.0 only
"""

import math
import os
import time
import urllib.request
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from utils import probe
from utils.video_reader import load_video_1p0, load_video_1p5

# ---------------------------------------------------------------------------
# Demo video download (copied from uvq_web.py to avoid module-level imports)
# ---------------------------------------------------------------------------

GCS_BASE = "https://storage.googleapis.com/ugc-dataset/vp9_compressed_videos/"
DEMO_CACHE_DIR = "/tmp/uvq_demo_cache"


def _download_demo(filename):
    """Download a demo video from GCS, caching in /tmp."""
    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)
    local_path = os.path.join(DEMO_CACHE_DIR, filename)
    if os.path.exists(local_path):
        return local_path
    url = GCS_BASE + filename
    print(f"  Downloading {filename} ...")
    urllib.request.urlretrieve(url, local_path)
    return local_path


# ---------------------------------------------------------------------------
# Session-scoped video fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def demo_video_720p():
    """Gaming_720P-25aa_orig.mp4 — ~11 MB, smallest demo video."""
    return _download_demo("Gaming_720P-25aa_orig.mp4")


@pytest.fixture(scope="session")
def demo_video_1080p():
    """Gaming_1080P-0ef8_orig.mp4 — ~80 MB, 1080p demo video."""
    return _download_demo("Gaming_1080P-0ef8_orig.mp4")


# ---------------------------------------------------------------------------
# Timing infrastructure
# ---------------------------------------------------------------------------

@contextmanager
def _time_block(label, timings_dict):
    """Context manager that records elapsed wall-clock time into *timings_dict*."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    timings_dict[label] = elapsed


def _format_timing_table(timings, total=None):
    """Return an ASCII table with stage name, time, and percentage of total."""
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


# ---------------------------------------------------------------------------
# Helper: extract metadata used by both pipelines
# ---------------------------------------------------------------------------

def _get_video_meta(video_path):
    """Return (video_length, transpose, orig_fps) from ffprobe."""
    dimensions = probe.get_dimensions(video_path)
    duration = probe.get_video_duration(video_path)
    orig_fps = probe.get_r_frame_rate(video_path)
    video_length = math.ceil(duration) if duration else 0
    transpose = bool(dimensions and dimensions[0] < dimensions[1])
    return video_length, transpose, orig_fps


# ===========================================================================
# UVQ 1.5 — Stage-level profiling
# ===========================================================================

@pytest.mark.perf
@pytest.mark.slow
class TestUVQ1p5PerfStages:
    """Profile individual stages of the UVQ 1.5 pipeline."""

    def test_probe(self, demo_video_720p):
        """Time the 4 ffprobe calls separately."""
        timings = {}
        path = demo_video_720p
        with _time_block("get_dimensions", timings):
            dims = probe.get_dimensions(path)
        with _time_block("get_video_duration", timings):
            dur = probe.get_video_duration(path)
        with _time_block("get_r_frame_rate", timings):
            fps = probe.get_r_frame_rate(path)
        with _time_block("get_nb_frames", timings):
            nb = probe.get_nb_frames(path)
        print(f"\n  ffprobe results: dims={dims} dur={dur:.2f}s fps={fps} frames={nb}")
        print(_format_timing_table(timings))
        assert dims is not None
        assert dur is not None and dur > 0

    def test_video_decode(self, demo_video_720p):
        """Time FFmpeg decode + resize to 1080p + normalize."""
        video_length, transpose, _ = _get_video_meta(demo_video_720p)
        timings = {}
        with _time_block("load_video_1p5", timings):
            video, num_real_frames = load_video_1p5(
                demo_video_720p, video_length, transpose, video_fps=1,
            )
        print(f"\n  Video shape: {video.shape}, real frames: {num_real_frames}")
        print(_format_timing_table(timings))
        assert video.shape[0] == video_length

    def test_tensor_prep(self, demo_video_720p):
        """Time numpy transpose + torch.from_numpy().float()."""
        video_length, transpose, _ = _get_video_meta(demo_video_720p)
        video, _ = load_video_1p5(
            demo_video_720p, video_length, transpose, video_fps=1,
        )
        timings = {}
        with _time_block("np.transpose", timings):
            video_t = video.transpose(0, 1, 4, 2, 3)
        with _time_block("torch.from_numpy", timings):
            tensor = torch.from_numpy(video_t).float()
        print(f"\n  Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
        print(_format_timing_table(timings))
        assert tensor.shape[2] == 3  # channels dim

    def test_model_loading(self):
        """Time UVQ1p5() constructor (3 checkpoints, ~30 MB)."""
        timings = {}
        with _time_block("UVQ1p5()", timings):
            from uvq1p5_pytorch.utils.uvq1p5 import UVQ1p5
            model = UVQ1p5()
        print(f"\n  Model loaded, training={model.uvq1p5_core.training}")
        print(_format_timing_table(timings))
        assert not model.uvq1p5_core.training

    def test_forward_pass(self, demo_video_720p):
        """Time batched inference through UVQ1p5Core (batch_size=24)."""
        from uvq1p5_pytorch.utils.uvq1p5 import UVQ1p5
        model = UVQ1p5()
        model.eval()

        video_length, transpose, _ = _get_video_meta(demo_video_720p)
        video, _ = load_video_1p5(
            demo_video_720p, video_length, transpose, video_fps=1,
        )
        video_t = video.transpose(0, 1, 4, 2, 3)
        video_tensor = torch.from_numpy(video_t).float()
        num_seconds, fps, c, h, w = video_tensor.shape
        num_frames = num_seconds * fps
        video_tensor = video_tensor.reshape(num_frames, 1, c, h, w)

        timings = {}
        batch_size = 24
        predictions = []
        with _time_block("forward_pass", timings):
            with torch.inference_mode():
                for i in range(0, num_frames, batch_size):
                    batch = video_tensor[i : i + batch_size]
                    pred = model.uvq1p5_core(batch)
                    predictions.append(pred)
            prediction = torch.cat(predictions, dim=0)
        score = torch.mean(prediction).item()
        print(f"\n  Frames: {num_frames}, score: {score:.3f}")
        print(_format_timing_table(timings))
        assert 1.0 <= score <= 5.0

    def test_scoring(self, demo_video_720p):
        """Time the final scoring step (torch.mean + tolist) — expected negligible."""
        from uvq1p5_pytorch.utils.uvq1p5 import UVQ1p5
        model = UVQ1p5()
        model.eval()

        video_length, transpose, orig_fps = _get_video_meta(demo_video_720p)
        video, _ = load_video_1p5(
            demo_video_720p, video_length, transpose, video_fps=1,
        )
        video_t = video.transpose(0, 1, 4, 2, 3)
        video_tensor = torch.from_numpy(video_t).float()
        num_seconds, fps, c, h, w = video_tensor.shape
        num_frames = num_seconds * fps
        video_tensor = video_tensor.reshape(num_frames, 1, c, h, w)

        with torch.inference_mode():
            prediction = model.uvq1p5_core(video_tensor)

        timings = {}
        with _time_block("torch.mean", timings):
            video_score = torch.mean(prediction).item()
        with _time_block("tolist", timings):
            frame_scores = prediction.cpu().numpy().flatten().tolist()
        with _time_block("frame_indices", timings):
            if orig_fps:
                frame_indices = [
                    int(round(i * orig_fps / 1)) for i in range(len(frame_scores))
                ]
            else:
                frame_indices = list(range(len(frame_scores)))
        print(f"\n  Score: {video_score:.3f}, frames: {len(frame_scores)}")
        print(_format_timing_table(timings))
        assert video_score > 0

    def test_end_to_end(self, demo_video_720p):
        """Full UVQ 1.5 pipeline with per-stage breakdown table."""
        timings = {}

        with _time_block("probe", timings):
            video_length, transpose, orig_fps = _get_video_meta(demo_video_720p)

        with _time_block("video_decode", timings):
            video, _ = load_video_1p5(
                demo_video_720p, video_length, transpose, video_fps=1,
            )

        with _time_block("tensor_prep", timings):
            video_t = video.transpose(0, 1, 4, 2, 3)
            video_tensor = torch.from_numpy(video_t).float()

        with _time_block("model_loading", timings):
            from uvq1p5_pytorch.utils.uvq1p5 import UVQ1p5
            model = UVQ1p5()
            model.eval()

        num_seconds, fps, c, h, w = video_tensor.shape
        num_frames = num_seconds * fps
        video_tensor = video_tensor.reshape(num_frames, 1, c, h, w)

        batch_size = 24
        predictions = []
        with _time_block("forward_pass", timings):
            with torch.inference_mode():
                for i in range(0, num_frames, batch_size):
                    batch = video_tensor[i : i + batch_size]
                    pred = model.uvq1p5_core(batch)
                    predictions.append(pred)
                prediction = torch.cat(predictions, dim=0)

        with _time_block("scoring", timings):
            video_score = torch.mean(prediction).item()
            frame_scores = prediction.cpu().numpy().flatten().tolist()

        total = sum(timings.values())
        print(f"\n  === UVQ 1.5 End-to-End (720p input) ===")
        print(f"  Score: {video_score:.3f}, frames: {len(frame_scores)}")
        print(_format_timing_table(timings, total))
        assert 1.0 <= video_score <= 5.0


# ===========================================================================
# UVQ 1.0 — Stage-level profiling
# ===========================================================================

@pytest.mark.perf
@pytest.mark.slow
class TestUVQ1p0PerfStages:
    """Profile individual stages of the UVQ 1.0 pipeline."""

    def test_probe(self, demo_video_720p):
        """Time ffprobe calls for UVQ 1.0."""
        timings = {}
        path = demo_video_720p
        with _time_block("get_dimensions", timings):
            dims = probe.get_dimensions(path)
        with _time_block("get_video_duration", timings):
            dur = probe.get_video_duration(path)
        with _time_block("get_r_frame_rate", timings):
            fps = probe.get_r_frame_rate(path)
        with _time_block("get_nb_frames", timings):
            nb = probe.get_nb_frames(path)
        print(f"\n  ffprobe results: dims={dims} dur={dur:.2f}s fps={fps} frames={nb}")
        print(_format_timing_table(timings))
        assert dims is not None

    def test_video_decode(self, demo_video_720p):
        """Time FFmpeg producing TWO outputs (720p + 496x496) at 5fps."""
        video_length, transpose, _ = _get_video_meta(demo_video_720p)
        timings = {}
        with _time_block("load_video_1p0", timings):
            video, video_small = load_video_1p0(
                demo_video_720p, video_length, transpose, video_fps=5,
            )
        print(f"\n  Video shape: {video.shape}, small: {video_small.shape}")
        print(_format_timing_table(timings))
        assert video.shape[0] == video_length
        assert video_small.shape[0] == video_length

    def test_model_loading(self):
        """Time UVQ1p0() constructor (4 main + 35 aggregation models, ~169 MB)."""
        timings = {}
        with _time_block("UVQ1p0()", timings):
            from uvq_pytorch.utils.uvq1p0 import UVQ1p0
            model = UVQ1p0()
        print(f"\n  Model loaded, aggregation models: {len(model.aggregationnet.models)}")
        print(_format_timing_table(timings))
        assert len(model.aggregationnet.models) == 35

    def test_contentnet_forward(self, demo_video_720p):
        """Time ContentNet: 1 forward pass per second of video."""
        from uvq_pytorch.utils.contentnet import ContentNetInference
        net = ContentNetInference()

        video_length, transpose, _ = _get_video_meta(demo_video_720p)
        _, video_small = load_video_1p0(
            demo_video_720p, video_length, transpose, video_fps=5,
        )
        video_small = video_small.transpose(0, 1, 4, 2, 3)

        timings = {}
        with _time_block("contentnet_all_frames", timings):
            features, labels = net.get_labels_and_features_for_all_frames(video_small)
        print(f"\n  Features shape: {features.shape}, labels shape: {labels.shape}")
        print(f"  Seconds processed: {video_small.shape[0]}")
        print(_format_timing_table(timings))
        assert features.shape[0] == video_small.shape[0]

    def test_compressionnet_forward(self, demo_video_720p):
        """Time CompressionNet: 16 patches (4x4) per second of video."""
        from uvq_pytorch.utils.compressionnet import CompressionNetInference
        net = CompressionNetInference()

        video_length, transpose, _ = _get_video_meta(demo_video_720p)
        video, _ = load_video_1p0(
            demo_video_720p, video_length, transpose, video_fps=5,
        )
        video = video.transpose(0, 1, 4, 2, 3)

        timings = {}
        with _time_block("compressionnet_all_frames", timings):
            features, labels = net.get_labels_and_features_for_all_frames(video)
        num_patches = net.num_patches_x * net.num_patches_y
        print(f"\n  Features shape: {features.shape}, labels shape: {labels.shape}")
        print(f"  Patches per second: {num_patches}, seconds: {video.shape[0]}")
        print(_format_timing_table(timings))
        assert features.shape[0] == video.shape[0]

    def test_distortionnet_forward(self, demo_video_720p):
        """Time DistortionNet: 4 patches (2x2) per second of video."""
        from uvq_pytorch.utils.distortionnet import DistortionNetInference
        net = DistortionNetInference()

        video_length, transpose, _ = _get_video_meta(demo_video_720p)
        video, _ = load_video_1p0(
            demo_video_720p, video_length, transpose, video_fps=5,
        )
        video = video.transpose(0, 1, 4, 2, 3)

        timings = {}
        with _time_block("distortionnet_all_frames", timings):
            features, labels = net.get_labels_and_features_for_all_frames(video)
        num_patches = net.num_patches_x * net.num_patches_y
        print(f"\n  Features shape: {features.shape}, labels shape: {labels.shape}")
        print(f"  Patches per second: {num_patches}, seconds: {video.shape[0]}")
        print(_format_timing_table(timings))
        assert features.shape[0] == video.shape[0]

    def test_aggregation(self, demo_video_720p):
        """Time the 35-model aggregation ensemble prediction."""
        from uvq_pytorch.utils.uvq1p0 import UVQ1p0
        model = UVQ1p0()

        video_length, transpose, _ = _get_video_meta(demo_video_720p)
        video, video_small = load_video_1p0(
            demo_video_720p, video_length, transpose, video_fps=5,
        )
        video = video.transpose(0, 1, 4, 2, 3)
        video_small = video_small.transpose(0, 1, 4, 2, 3)

        # Get features from each subnet first
        content_features, _ = model.contentnet.get_labels_and_features_for_all_frames(video_small)
        compression_features, _ = model.compressionnet.get_labels_and_features_for_all_frames(video)
        distortion_features, _ = model.distortionnet.get_labels_and_features_for_all_frames(video)

        timings = {}
        with _time_block("aggregation_predict", timings):
            results = model.aggregationnet.predict(
                compression_features, content_features, distortion_features,
            )
        print(f"\n  Aggregation results: {results}")
        print(_format_timing_table(timings))
        assert "compression_content_distortion" in results

    def test_end_to_end(self, demo_video_720p):
        """Full UVQ 1.0 pipeline with per-stage breakdown table."""
        timings = {}

        with _time_block("probe", timings):
            video_length, transpose, _ = _get_video_meta(demo_video_720p)

        with _time_block("video_decode", timings):
            video, video_small = load_video_1p0(
                demo_video_720p, video_length, transpose, video_fps=5,
            )
            video = video.transpose(0, 1, 4, 2, 3)
            video_small = video_small.transpose(0, 1, 4, 2, 3)

        with _time_block("model_loading", timings):
            from uvq_pytorch.utils.uvq1p0 import UVQ1p0
            model = UVQ1p0()

        with _time_block("contentnet_forward", timings):
            content_features, _ = model.contentnet.get_labels_and_features_for_all_frames(video_small)

        with _time_block("compressionnet_forward", timings):
            compression_features, _ = model.compressionnet.get_labels_and_features_for_all_frames(video)

        with _time_block("distortionnet_forward", timings):
            distortion_features, _ = model.distortionnet.get_labels_and_features_for_all_frames(video)

        with _time_block("aggregation", timings):
            results = model.aggregationnet.predict(
                compression_features, content_features, distortion_features,
            )

        total = sum(timings.values())
        score = results["compression_content_distortion"]
        print(f"\n  === UVQ 1.0 End-to-End (720p input) ===")
        print(f"  Combined score: {score:.3f}")
        print(f"  All scores: {results}")
        print(_format_timing_table(timings, total))
        assert 0 < score <= 5.0


# ===========================================================================
# UVQ 1.5 with native 1080p video
# ===========================================================================

@pytest.mark.perf
@pytest.mark.slow
class TestUVQ1p5Perf1080p:
    """End-to-end UVQ 1.5 with a native 1080p video input."""

    def test_end_to_end(self, demo_video_1080p):
        timings = {}

        with _time_block("probe", timings):
            video_length, transpose, orig_fps = _get_video_meta(demo_video_1080p)

        with _time_block("video_decode", timings):
            video, _ = load_video_1p5(
                demo_video_1080p, video_length, transpose, video_fps=1,
            )

        with _time_block("tensor_prep", timings):
            video_t = video.transpose(0, 1, 4, 2, 3)
            video_tensor = torch.from_numpy(video_t).float()

        with _time_block("model_loading", timings):
            from uvq1p5_pytorch.utils.uvq1p5 import UVQ1p5
            model = UVQ1p5()
            model.eval()

        num_seconds, fps, c, h, w = video_tensor.shape
        num_frames = num_seconds * fps
        video_tensor = video_tensor.reshape(num_frames, 1, c, h, w)

        batch_size = 24
        predictions = []
        with _time_block("forward_pass", timings):
            with torch.inference_mode():
                for i in range(0, num_frames, batch_size):
                    batch = video_tensor[i : i + batch_size]
                    pred = model.uvq1p5_core(batch)
                    predictions.append(pred)
                prediction = torch.cat(predictions, dim=0)

        with _time_block("scoring", timings):
            video_score = torch.mean(prediction).item()
            frame_scores = prediction.cpu().numpy().flatten().tolist()

        total = sum(timings.values())
        print(f"\n  === UVQ 1.5 End-to-End (1080p native input) ===")
        print(f"  Score: {video_score:.3f}, frames: {len(frame_scores)}")
        print(_format_timing_table(timings, total))
        assert 1.0 <= video_score <= 5.0


# ===========================================================================
# Side-by-side comparison
# ===========================================================================

@pytest.mark.perf
@pytest.mark.slow
class TestPerfComparison:
    """Run both pipelines on the same video and print comparison."""

    def test_uvq1p5_vs_uvq1p0(self, demo_video_720p):
        # --- UVQ 1.5 ---
        t0 = time.perf_counter()

        video_length, transpose, orig_fps = _get_video_meta(demo_video_720p)
        video, _ = load_video_1p5(
            demo_video_720p, video_length, transpose, video_fps=1,
        )
        video_t = video.transpose(0, 1, 4, 2, 3)
        video_tensor = torch.from_numpy(video_t).float()

        from uvq1p5_pytorch.utils.uvq1p5 import UVQ1p5
        model_1p5 = UVQ1p5()
        model_1p5.eval()

        num_seconds, fps, c, h, w = video_tensor.shape
        num_frames = num_seconds * fps
        video_tensor = video_tensor.reshape(num_frames, 1, c, h, w)

        batch_size = 24
        predictions = []
        with torch.inference_mode():
            for i in range(0, num_frames, batch_size):
                batch = video_tensor[i : i + batch_size]
                pred = model_1p5.uvq1p5_core(batch)
                predictions.append(pred)
            prediction = torch.cat(predictions, dim=0)

        score_1p5 = torch.mean(prediction).item()
        time_1p5 = time.perf_counter() - t0

        # --- UVQ 1.0 ---
        t0 = time.perf_counter()

        video_1p0, video_small = load_video_1p0(
            demo_video_720p, video_length, transpose, video_fps=5,
        )
        video_1p0 = video_1p0.transpose(0, 1, 4, 2, 3)
        video_small = video_small.transpose(0, 1, 4, 2, 3)

        from uvq_pytorch.utils.uvq1p0 import UVQ1p0
        model_1p0 = UVQ1p0()

        content_features, _ = model_1p0.contentnet.get_labels_and_features_for_all_frames(video_small)
        compression_features, _ = model_1p0.compressionnet.get_labels_and_features_for_all_frames(video_1p0)
        distortion_features, _ = model_1p0.distortionnet.get_labels_and_features_for_all_frames(video_1p0)
        results_1p0 = model_1p0.aggregationnet.predict(
            compression_features, content_features, distortion_features,
        )
        score_1p0 = results_1p0["compression_content_distortion"]
        time_1p0 = time.perf_counter() - t0

        # --- Print comparison ---
        speedup = time_1p0 / time_1p5 if time_1p5 > 0 else float("inf")
        print(f"\n  === UVQ 1.5 vs UVQ 1.0 Comparison ===")
        print(f"  Video: {os.path.basename(demo_video_720p)}")
        print(f"  {'Model':<12} {'Time (s)':>10} {'Score':>8}")
        print(f"  {'-'*32}")
        print(f"  {'UVQ 1.5':<12} {time_1p5:>10.3f} {score_1p5:>8.3f}")
        print(f"  {'UVQ 1.0':<12} {time_1p0:>10.3f} {score_1p0:>8.3f}")
        print(f"  {'-'*32}")
        print(f"  UVQ 1.5 is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} than UVQ 1.0")

        assert 1.0 <= score_1p5 <= 5.0
        assert 0 < score_1p0 <= 5.0
