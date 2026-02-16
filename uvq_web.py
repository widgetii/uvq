"""UVQ Web Interface — Gradio app for video quality assessment.

Runs both UVQ 1.5 and UVQ 1.0 models and presents visual quality information.
Demo videos come from the YouTube-UGC dataset on Google Cloud Storage.

Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import json
import math
import os
import urllib.request

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import torch

from utils import probe
from uvq1p5_pytorch.utils import uvq1p5
from uvq_pytorch.utils import uvq1p0

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Demo videos from YouTube-UGC dataset
# ---------------------------------------------------------------------------
GCS_BASE = "https://storage.googleapis.com/ugc-dataset/vp9_compressed_videos/"
DEMO_CACHE_DIR = "/tmp/uvq_demo_cache"

# Each entry maps a display name to the filename under GCS_BASE.
# These are H.264 originals from the YouTube-UGC dataset, verified to exist.
DEMO_VIDEOS = {
    "Gaming 720p — 11 MB (Gaming_720P-25aa)": "Gaming_720P-25aa_orig.mp4",
    "Sports 720p — 17 MB (Sports_720P-07d0)": "Sports_720P-07d0_orig.mp4",
    "Gaming 720p — 18 MB (Gaming_720P-0fdb)": "Gaming_720P-0fdb_orig.mp4",
    "Gaming 720p — 51 MB (Gaming_720P-103a)": "Gaming_720P-103a_orig.mp4",
    "Sports 720p — 51 MB (Sports_720P-0104)": "Sports_720P-0104_orig.mp4",
    "Gaming 720p — 68 MB (Gaming_720P-0fba)": "Gaming_720P-0fba_orig.mp4",
    "Gaming 1080p — 80 MB (Gaming_1080P-0ef8)": "Gaming_1080P-0ef8_orig.mp4",
    "Gaming 1080p — 88 MB (Gaming_1080P-0ce6)": "Gaming_1080P-0ce6_orig.mp4",
}

# ---------------------------------------------------------------------------
# Model singletons (loaded once at startup)
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading UVQ 1.5 model on {device}...")
model_1p5 = uvq1p5.UVQ1p5()
if device == "cuda":
    model_1p5.cuda()

print("Loading UVQ 1.0 model on CPU...")
model_1p0 = uvq1p0.UVQ1p0()

# ---------------------------------------------------------------------------
# Demo video download helper
# ---------------------------------------------------------------------------

def download_demo(filename, progress=None):
    """Download a demo video from GCS, caching in /tmp."""
    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)
    local_path = os.path.join(DEMO_CACHE_DIR, filename)
    if os.path.exists(local_path):
        return local_path
    url = GCS_BASE + filename
    if progress is not None:
        progress(0, desc=f"Downloading {filename}...")
    urllib.request.urlretrieve(url, local_path)
    return local_path


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def extract_metadata(video_path):
    """Extract video metadata via ffprobe."""
    dimensions = probe.get_dimensions(video_path)
    duration = probe.get_video_duration(video_path)
    fps = probe.get_r_frame_rate(video_path)
    nb_frames = probe.get_nb_frames(video_path)
    width, height = dimensions if dimensions else (None, None)
    return {
        "width": width,
        "height": height,
        "duration": round(duration, 2) if duration else None,
        "fps": fps,
        "nb_frames": nb_frames,
    }


def run_inference(video_path, progress=gr.Progress()):
    """Run both UVQ 1.5 and UVQ 1.0 on a video. Returns outputs for the UI."""

    progress(0.05, desc="Extracting video metadata...")
    meta = extract_metadata(video_path)

    if meta["duration"] is None:
        raise gr.Error("Could not determine video duration. Is this a valid video file?")

    video_length = math.ceil(meta["duration"])
    if video_length == 0:
        raise gr.Error("Video duration is 0 seconds.")

    transpose = False
    if meta["width"] and meta["height"] and meta["width"] < meta["height"]:
        transpose = True

    orig_fps = meta["fps"]
    sample_fps = 1  # UVQ 1.5 default

    # --- UVQ 1.5 ---
    progress(0.10, desc="Running UVQ 1.5 inference...")
    results_1p5 = model_1p5.infer(
        video_path,
        video_length,
        transpose,
        fps=sample_fps,
        orig_fps=orig_fps,
        device=device,
    )

    # --- UVQ 1.0 ---
    progress(0.55, desc="Running UVQ 1.0 inference...")
    results_1p0 = model_1p0.infer(video_path, video_length, transpose)
    results_1p0 = {k: float(v) for k, v in results_1p0.items()}

    progress(0.90, desc="Generating visualizations...")

    # --- Build outputs ---
    score_1p5 = results_1p5["uvq1p5_score"]
    score_1p0_combined = results_1p0["compression_content_distortion"]

    meta_md = (
        f"| Property | Value |\n|---|---|\n"
        f"| Resolution | {meta['width']}x{meta['height']} |\n"
        f"| Duration | {meta['duration']}s |\n"
        f"| Frame rate | {meta['fps']} fps |\n"
        f"| Frame count | {meta['nb_frames']} |"
    )

    temporal_plot = make_temporal_plot(results_1p5, sample_fps)
    dimensions_plot = make_dimensions_plot(results_1p0)

    full_json = json.dumps(
        {"uvq1p5": results_1p5, "uvq1p0": results_1p0},
        indent=2,
        default=str,
    )

    progress(1.0, desc="Done")
    return (
        video_path,       # video player
        meta_md,          # metadata markdown
        f"{score_1p5:.3f}",   # UVQ 1.5 score
        f"{score_1p0_combined:.3f}",  # UVQ 1.0 combined score
        temporal_plot,    # temporal chart
        dimensions_plot,  # dimensions bar chart
        full_json,        # raw JSON
    )


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def make_temporal_plot(results_1p5, fps):
    """Line plot of per-frame UVQ 1.5 scores over time."""
    scores = results_1p5["per_frame_scores"]
    avg = results_1p5["uvq1p5_score"]
    seconds = [i / fps for i in range(len(scores))]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(seconds, scores, color="#2563eb", linewidth=1.5, label="Per-frame score")
    ax.axhline(y=avg, color="#dc2626", linestyle="--", linewidth=1, label=f"Average ({avg:.3f})")
    ax.fill_between(seconds, scores, alpha=0.15, color="#2563eb")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Quality Score")
    ax.set_title("UVQ 1.5 — Temporal Quality")
    ax.set_ylim(1, 5)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_dimensions_plot(results_1p0):
    """Horizontal bar chart of all 7 UVQ 1.0 quality dimensions."""
    display_order = [
        ("compression", "Compression"),
        ("content", "Content"),
        ("distortion", "Distortion"),
        ("compression_content", "Compression + Content"),
        ("compression_distortion", "Compression + Distortion"),
        ("content_distortion", "Content + Distortion"),
        ("compression_content_distortion", "Combined (all 3)"),
    ]

    labels = []
    values = []
    for key, label in display_order:
        if key in results_1p0:
            labels.append(label)
            values.append(results_1p0[key])

    colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4", "#ef4444"]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    bars = ax.barh(labels, values, color=colors[: len(values)], height=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 5.5)
    ax.set_xlabel("Quality Score")
    ax.set_title("UVQ 1.0 — Quality Dimensions")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gradio UI callbacks
# ---------------------------------------------------------------------------

def analyze_upload(video_file, progress=gr.Progress()):
    """Callback for the Upload tab."""
    if video_file is None:
        raise gr.Error("Please upload a video first.")
    return run_inference(video_file, progress)


def analyze_demo(demo_name, progress=gr.Progress()):
    """Callback for the Demo tab."""
    if not demo_name:
        raise gr.Error("Please select a demo video.")
    filename = DEMO_VIDEOS[demo_name]
    local_path = download_demo(filename, progress)
    return run_inference(local_path, progress)


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------

def build_app():
    with gr.Blocks(title="UVQ: Universal Video Quality Assessment") as demo:
        gr.Markdown(
            "# UVQ: Universal Video Quality Assessment\n"
            "Analyze video quality using Google's UVQ 1.5 and UVQ 1.0 models simultaneously. "
            "Scores range from 1 (lowest) to 5 (highest). "
            "[Research article](https://arxiv.org/abs/2206.11733)"
        )

        # Outputs (shared across both tabs)
        with gr.Row(visible=False) as results_row:
            pass  # placeholder — actual outputs below

        # --- Input tabs ---
        with gr.Tabs():
            with gr.TabItem("Upload Video"):
                upload_input = gr.Video(label="Upload a video file")
                upload_btn = gr.Button("Analyze", variant="primary")

            with gr.TabItem("Demo Videos (YouTube-UGC)"):
                demo_dropdown = gr.Dropdown(
                    choices=list(DEMO_VIDEOS.keys()),
                    label="Select a demo video",
                )
                demo_btn = gr.Button("Load & Analyze", variant="primary")

        # --- Results section ---
        gr.Markdown("---")
        with gr.Row():
            video_player = gr.Video(label="Video", interactive=False)
            meta_display = gr.Markdown(label="Metadata")

        with gr.Row():
            score_1p5_display = gr.Textbox(label="UVQ 1.5 Score", interactive=False)
            score_1p0_display = gr.Textbox(label="UVQ 1.0 Combined Score", interactive=False)

        temporal_chart = gr.Plot(label="Temporal Quality (UVQ 1.5)")
        dimensions_chart = gr.Plot(label="Quality Dimensions (UVQ 1.0)")

        with gr.Accordion("Full JSON Output", open=False):
            json_output = gr.Code(language="json", label="Raw results")

        outputs = [
            video_player,
            meta_display,
            score_1p5_display,
            score_1p0_display,
            temporal_chart,
            dimensions_chart,
            json_output,
        ]

        upload_btn.click(fn=analyze_upload, inputs=[upload_input], outputs=outputs)
        demo_btn.click(fn=analyze_demo, inputs=[demo_dropdown], outputs=outputs)

    return demo


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UVQ Web Interface")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio URL")
    args = parser.parse_args()

    demo = build_app()
    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
