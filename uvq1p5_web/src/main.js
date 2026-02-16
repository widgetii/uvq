/**
 * UVQ 1.5 browser entry point.
 *
 * Loads ONNX models via onnxruntime-web with WebGPU (or WASM fallback),
 * extracts video frames via <video> + canvas, runs per-frame inference,
 * and displays the quality score.
 */

import * as ort from "onnxruntime-web/webgpu";
import { UVQ } from "./uvq.js";
import { preprocessFrame } from "./preprocessing.js";

const $ = (sel) => document.querySelector(sel);

const statusEl = $("#status");
const errorEl = $("#error-banner");
const fileArea = $("#file-input-area");
const fileInput = $("#video-file");
const progressContainer = $("#progress-bar-container");
const progressBar = $("#progress-bar");
const resultsEl = $("#results");
const scoreEl = $("#score");
const timingEl = $("#timing");
const frameScoresEl = $("#frame-scores");
const videoEl = $("#video-el");

function setStatus(msg) { statusEl.textContent = msg; }
function showError(msg) { errorEl.textContent = msg; errorEl.style.display = "block"; }

async function init() {
  // Check WebGPU
  const hasWebGPU = !!(navigator.gpu);
  const backend = hasWebGPU ? "webgpu" : "wasm";

  if (!hasWebGPU) {
    setStatus("WebGPU not available, falling back to WASM (slower).");
  }

  // Configure ort
  const sessionOptions = { executionProviders: [backend] };

  setStatus(`Loading models (${backend} backend)...`);
  const uvq = new UVQ(ort);

  try {
    await uvq.load(
      "models/content_net.onnx",
      "models/distortion_net.onnx",
      "models/aggregation_net.onnx",
      sessionOptions,
    );
  } catch (e) {
    showError(`Failed to load models: ${e.message}`);
    setStatus("Model loading failed.");
    return;
  }

  setStatus("Warming up...");
  try {
    await uvq.warmup();
  } catch (e) {
    // Warmup failure is non-fatal — first real inference will be slower
    console.warn("Warmup failed:", e);
  }

  setStatus("Ready. Select a video file to assess.");
  fileArea.style.display = "block";

  fileInput.addEventListener("change", async () => {
    const file = fileInput.files[0];
    if (!file) return;
    await processVideo(uvq, file);
  });
}

async function processVideo(uvq, file) {
  resultsEl.style.display = "none";
  progressContainer.style.display = "block";
  progressBar.value = 0;
  frameScoresEl.innerHTML = "";

  const url = URL.createObjectURL(file);
  videoEl.src = url;

  // Wait for metadata
  await new Promise((resolve, reject) => {
    videoEl.onloadedmetadata = resolve;
    videoEl.onerror = () => reject(new Error("Failed to load video"));
  });

  const duration = Math.floor(videoEl.duration);
  if (duration < 1) {
    showError("Video is shorter than 1 second.");
    progressContainer.style.display = "none";
    return;
  }

  setStatus(`Processing ${duration} frame(s)...`);

  const canvas = new OffscreenCanvas(videoEl.videoWidth, videoEl.videoHeight);
  const ctx = canvas.getContext("2d");

  const scores = [];
  const t0 = performance.now();

  for (let t = 0; t < duration; t++) {
    // Seek to middle of each second for more representative frame
    videoEl.currentTime = t + 0.5;
    await new Promise((resolve) => { videoEl.onseeked = resolve; });

    // Capture frame
    ctx.drawImage(videoEl, 0, 0);

    // Preprocess
    const { content, patches } = preprocessFrame(canvas);

    // Infer
    const score = await uvq.infer(content, patches);
    scores.push(score);

    // Update progress
    progressBar.value = ((t + 1) / duration) * 100;
    setStatus(`Processing frame ${t + 1} / ${duration}...`);
  }

  const elapsed = performance.now() - t0;
  URL.revokeObjectURL(url);

  // Display results
  const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
  scoreEl.textContent = avgScore.toFixed(3);
  timingEl.textContent = `${duration} frames in ${(elapsed / 1000).toFixed(1)}s (${(elapsed / duration).toFixed(0)} ms/frame)`;

  frameScoresEl.innerHTML = scores
    .map((s, i) => `<tr><td>${i}</td><td>${s.toFixed(4)}</td></tr>`)
    .join("");

  resultsEl.style.display = "block";
  progressContainer.style.display = "none";
  setStatus("Done.");
}

init().catch((e) => {
  console.error(e);
  showError(e.message);
});
