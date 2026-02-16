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
import tempfile
import urllib.request

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import probe
from uvq1p5_pytorch.utils import uvq1p5
from uvq_pytorch.utils import uvq1p0

matplotlib.use("Agg")

# Keys returned by UVQ 1.0 that are not scalar scores
_NON_SCORE_KEYS = {"gradcam_files", "content_labels", "compression_patch_labels", "distortion_patch_labels"}

# 26-class distortion labels used by UVQ 1.0's distortion network.
# Known names from the UVQ blog post are placed at likely positions;
# the rest are placeholders that can be updated if the full mapping is published.
# Index 0 is "non-distortion" (unknown / no distortion detected).
# Indices 1-25 follow the KADID-10k database ordering, confirmed by
# UVQ co-author Yilin Wang in https://github.com/google/uvq/issues/4
_DISTORTION_CLASS_NAMES = [
    "Non-distortion",
    "Gaussian blur",
    "Lens blur",
    "Motion blur",
    "Color diffusion",
    "Color shift",
    "Color quantization",
    "Color saturation 1",
    "Color saturation 2",
    "JPEG2000 compression",
    "JPEG compression",
    "White noise",
    "White noise in color",
    "Impulse noise",
    "Multiplicative noise",
    "Denoise",
    "Brighten",
    "Darken",
    "Mean shift",
    "Jitter",
    "Non-eccentricity patch",
    "Pixelate",
    "Quantization",
    "Color block",
    "High sharpen",
    "Contrast change",
]

def make_diagnostic_report(contentnet, content_labels, compression_patch_labels,
                           distortion_patch_labels, scores):
    """Build a Markdown diagnostic report summarizing UVQ 1.0 network outputs.

    Args:
        contentnet: ContentNetInference instance (for label_probabilities_to_text).
        content_labels: (T, 3862) content class probabilities per frame.
        compression_patch_labels: (T, 4, 4, 1) compression level per patch.
        distortion_patch_labels: (T, 2, 2, 26) distortion probabilities per patch.
        scores: dict of the 7 UVQ 1.0 quality scores.

    Returns:
        Markdown string with ContentNet, DistortionNet, CompressionNet, and
        score summaries.
    """
    lines = []

    # --- ContentNet ---
    avg_content = np.mean(content_labels, axis=0)  # (3862,)
    names, probs, _ = contentnet.label_probabilities_to_text(avg_content, top_n=5)
    content_items = [f"{n} ({p:.3f})" for n, p in zip(names, probs)]
    lines.append("### ContentNet (CT)")
    lines.append(", ".join(content_items))
    lines.append("")

    # --- DistortionNet ---
    # Average across time and patches → (26,)
    avg_dist = np.mean(distortion_patch_labels, axis=(0, 1, 2))
    sorted_idx = np.argsort(avg_dist)[::-1]
    dist_items = []
    for idx in sorted_idx:
        if len(dist_items) >= 5:
            break
        if idx == 0:
            continue  # skip "Non-distortion"
        dist_items.append(
            f"{_DISTORTION_CLASS_NAMES[idx]} ({avg_dist[idx]:.3f})"
        )
    lines.append("### DistortionNet (DT)")
    lines.append(", ".join(dist_items))
    lines.append("")

    # --- CompressionNet ---
    mean_comp = float(np.mean(compression_patch_labels))
    if mean_comp < 0.2:
        comp_desc = "low"
    elif mean_comp < 0.4:
        comp_desc = "medium-low"
    elif mean_comp < 0.6:
        comp_desc = "medium"
    elif mean_comp < 0.8:
        comp_desc = "medium-high"
    else:
        comp_desc = "high"
    lines.append("### CompressionNet (CP)")
    lines.append(f"Mean compression level: {mean_comp:.3f} ({comp_desc})")
    lines.append("")

    # --- Quality scores ---
    ct = scores.get("content", 0)
    dt = scores.get("distortion", 0)
    cp = scores.get("compression", 0)
    combined = scores.get("compression_content_distortion", 0)
    lines.append("### Predicted Quality Scores")
    lines.append(
        f"(CT, DT, CP) = ({ct:.3f}, {dt:.3f}, {cp:.3f}), "
        f"(CT+DT+CP) = {combined:.3f}"
    )

    return "\n".join(lines)


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

print(f"Loading UVQ 1.0 model on {device}...")
model_1p0 = uvq1p0.UVQ1p0()
if device == "cuda":
    model_1p0.cuda()

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


def run_inference(video_path, enable_gradcam=False, progress=gr.Progress()):
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
    gradcam_1p5 = None
    gradcam_1p0 = None

    if enable_gradcam:
        gradcam_dir_1p5 = tempfile.mkdtemp(prefix="uvq_gradcam_1p5_")
        progress(0.10, desc="Running UVQ 1.5 inference with Grad-CAM...")
        results_1p5 = model_1p5.infer_gradcam(
            video_path,
            video_length,
            transpose,
            output_dir=gradcam_dir_1p5,
            fps=sample_fps,
            orig_fps=orig_fps,
            device=device,
        )
        gradcam_1p5 = results_1p5.get("gradcam_files", [])
    else:
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
    if enable_gradcam:
        gradcam_dir_1p0 = tempfile.mkdtemp(prefix="uvq_gradcam_1p0_")
        progress(0.55, desc="Running UVQ 1.0 inference with Grad-CAM...")
        results_1p0 = model_1p0.infer_gradcam(
            video_path, video_length, transpose, output_dir=gradcam_dir_1p0,
            device=device,
        )
        gradcam_1p0 = results_1p0.get("gradcam_files", [])
        # Ensure numeric values for score display (gradcam results may include lists/arrays)
        results_1p0_scores = {
            k: float(v) for k, v in results_1p0.items() if k not in _NON_SCORE_KEYS
        }
    else:
        progress(0.55, desc="Running UVQ 1.0 inference...")
        results_1p0 = model_1p0.infer(video_path, video_length, transpose, device=device)
        results_1p0_scores = {
            k: float(v) for k, v in results_1p0.items() if k not in _NON_SCORE_KEYS
        }

    # Extract per-patch and content labels (always present from UVQ 1.0)
    content_labels = results_1p0.get("content_labels")
    compression_patch_labels = results_1p0.get("compression_patch_labels")
    distortion_patch_labels = results_1p0.get("distortion_patch_labels")

    progress(0.90, desc="Generating visualizations...")

    # --- Build outputs ---
    score_1p5 = results_1p5["uvq1p5_score"]
    score_1p0_combined = results_1p0_scores["compression_content_distortion"]

    meta_md = (
        f"| Property | Value |\n|---|---|\n"
        f"| Resolution | {meta['width']}x{meta['height']} |\n"
        f"| Duration | {meta['duration']}s |\n"
        f"| Frame rate | {meta['fps']} fps |\n"
        f"| Frame count | {meta['nb_frames']} |"
    )

    temporal_plot = make_temporal_plot(results_1p5, sample_fps)
    dimensions_plot = make_dimensions_plot(results_1p0_scores)
    compression_plot = make_compression_patch_plot(compression_patch_labels)
    distortion_plot = make_distortion_patch_plot(distortion_patch_labels)

    diagnostic_md = make_diagnostic_report(
        model_1p0.contentnet, content_labels,
        compression_patch_labels, distortion_patch_labels,
        results_1p0_scores,
    )

    full_json = json.dumps(
        {"uvq1p5": results_1p5, "uvq1p0": results_1p0_scores},
        indent=2,
        default=str,
    )

    progress(1.0, desc="Done")
    return (
        video_path,       # video player
        meta_md,          # metadata markdown
        f"{score_1p5:.3f}",   # UVQ 1.5 score
        f"{score_1p0_combined:.3f}",  # UVQ 1.0 combined score
        diagnostic_md,    # UVQ 1.0 diagnostic report
        temporal_plot,    # temporal chart
        dimensions_plot,  # dimensions bar chart
        full_json,        # raw JSON
        gradcam_1p5,      # Grad-CAM gallery for UVQ 1.5
        gradcam_1p0,      # Grad-CAM gallery for UVQ 1.0
        compression_plot, # compression patch heatmap
        distortion_plot,  # distortion patch heatmap + bar chart
    )


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def make_compression_patch_plot(compression_labels):
    """Heatmap of per-patch compression severity (4x4 grid, time-averaged)."""
    if compression_labels is None:
        return None
    # compression_labels shape: (T, 4, 4, 1)
    avg = np.mean(compression_labels, axis=0)[:, :, 0]  # (4, 4)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(avg, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="equal")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{avg[i, j]:.2f}", ha="center", va="center",
                    fontsize=10, color="black", fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(["Left", "", "", "Right"])
    ax.set_yticklabels(["Top", "", "", "Bottom"])
    ax.set_title(
        "Compression Artifacts — 4\u00d74 Spatial Grid (time-averaged)\n"
        "0.0 = no compression artifacts, 1.0 = severe compression",
        fontsize=10,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Compression Level")
    fig.tight_layout()
    return fig


def make_distortion_patch_plot(distortion_labels):
    """Two-panel figure: 2x2 mean-distortion heatmap + top-10 class bar chart."""
    if distortion_labels is None:
        return None
    # distortion_labels shape: (T, 2, 2, 26)
    avg = np.mean(distortion_labels, axis=0)  # (2, 2, 26)

    # Left panel: mean distortion probability per patch
    mean_per_patch = np.mean(avg, axis=2)  # (2, 2)

    # Right panel: top-10 distortion classes (averaged over all patches and time)
    class_means = np.mean(avg, axis=(0, 1))  # (26,)
    top_indices = np.argsort(class_means)[::-1][:10]
    top_names = [_DISTORTION_CLASS_NAMES[i] for i in top_indices]
    top_values = class_means[top_indices]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5),
                                   gridspec_kw={"width_ratios": [1, 1.8]})

    # Left: 2x2 heatmap
    im = ax1.imshow(mean_per_patch, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="equal")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f"{mean_per_patch[i, j]:.3f}", ha="center", va="center",
                     fontsize=11, color="black", fontweight="bold")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Left", "Right"])
    ax1.set_yticklabels(["Top", "Bottom"])
    ax1.set_title("Mean Distortion per Region", fontsize=10)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # Right: horizontal bar chart
    y_pos = np.arange(len(top_names))
    ax2.barh(y_pos, top_values, color="#f59e0b", height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_names)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean Probability")
    ax2.set_title("Top-10 Most Active Distortion Classes", fontsize=10)
    ax2.set_xlim(0, max(top_values.max() * 1.2, 0.1))
    for i, v in enumerate(top_values):
        ax2.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax2.grid(True, axis="x", alpha=0.3)

    fig.suptitle(
        "Distortion Detection — 2\u00d72 Spatial Grid (time-averaged)\n"
        "Probability of visual artifacts per region. Higher = more distortion detected.",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


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

def analyze_upload(video_file, enable_gradcam, progress=gr.Progress()):
    """Callback for the Upload tab."""
    if video_file is None:
        raise gr.Error("Please upload a video first.")
    return run_inference(video_file, enable_gradcam, progress)


def analyze_demo(demo_name, enable_gradcam, progress=gr.Progress()):
    """Callback for the Demo tab."""
    if not demo_name:
        raise gr.Error("Please select a demo video.")
    filename = DEMO_VIDEOS[demo_name]
    local_path = download_demo(filename, progress)
    return run_inference(local_path, enable_gradcam, progress)


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
                upload_gradcam = gr.Checkbox(
                    label="Generate Grad-CAM heatmaps", value=True,
                )
                upload_btn = gr.Button("Analyze", variant="primary")

            with gr.TabItem("Demo Videos (YouTube-UGC)"):
                demo_dropdown = gr.Dropdown(
                    choices=list(DEMO_VIDEOS.keys()),
                    label="Select a demo video",
                )
                demo_gradcam = gr.Checkbox(
                    label="Generate Grad-CAM heatmaps", value=True,
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

        gr.Markdown(
            "Scores range from **1** (lowest, e.g. severely blurry) to **5** "
            "(highest, e.g. professionally produced). The combined score fuses "
            "content, compression, and distortion assessments."
        )

        with gr.Accordion("UVQ 1.0 Diagnostic Report", open=True):
            diagnostic_report = gr.Markdown(label="Diagnostic Report")

        temporal_chart = gr.Plot(label="Temporal Quality (UVQ 1.5)")
        dimensions_chart = gr.Plot(label="Quality Dimensions (UVQ 1.0)")

        with gr.Accordion("Per-Patch Spatial Quality (UVQ 1.0)", open=True):
            gr.Markdown(
                "UVQ 1.0 divides each frame into a spatial grid and assesses "
                "each tile independently, allowing localization of quality issues. "
                "The compression network uses a **4\u00d74 grid** and the distortion "
                "network uses a **2\u00d72 grid**. Values shown are time-averaged "
                "across all frames."
            )
            compression_patch_chart = gr.Plot(
                label="Compression Artifacts (4\u00d74 grid)",
            )
            distortion_patch_chart = gr.Plot(
                label="Distortion Analysis (2\u00d72 grid)",
            )

        with gr.Accordion("Grad-CAM Heatmaps", open=True):
            gradcam_gallery_1p5 = gr.Gallery(
                label="Grad-CAM: Distortion Heatmap (UVQ 1.5)", columns=2,
            )
            gradcam_gallery_1p0 = gr.Gallery(
                label="Grad-CAM: Distortion Heatmap (UVQ 1.0)", columns=2,
            )

        with gr.Accordion("Full JSON Output", open=False):
            json_output = gr.Code(language="json", label="Raw results")

        outputs = [
            video_player,
            meta_display,
            score_1p5_display,
            score_1p0_display,
            diagnostic_report,
            temporal_chart,
            dimensions_chart,
            json_output,
            gradcam_gallery_1p5,
            gradcam_gallery_1p0,
            compression_patch_chart,
            distortion_patch_chart,
        ]

        upload_btn.click(
            fn=analyze_upload,
            inputs=[upload_input, upload_gradcam],
            outputs=outputs,
        )
        demo_btn.click(
            fn=analyze_demo,
            inputs=[demo_dropdown, demo_gradcam],
            outputs=outputs,
        )

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
