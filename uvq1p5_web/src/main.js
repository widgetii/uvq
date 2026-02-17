/**
 * UVQ 1.5 browser entry point.
 *
 * Loads ONNX models via onnxruntime-web with WebGPU and WASM backends,
 * extracts video frames via <video> + canvas, runs per-frame inference
 * on both backends, and displays a performance comparison.
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
const frameScoresHeadEl = $("#frame-scores-head");
const frameScoresEl = $("#frame-scores");
const videoEl = $("#video-el");

function setStatus(msg) { statusEl.textContent = msg; }
function showError(msg) { errorEl.textContent = msg; errorEl.style.display = "block"; }

const MODEL_PATHS = [
  "models/content_net.onnx",
  "models/distortion_net.onnx",
  "models/aggregation_net.onnx",
];

async function loadBackend(name, { warmup = true } = {}) {
  const uvq = new UVQ(ort);
  await uvq.load(...MODEL_PATHS, { executionProviders: [name] });
  if (warmup) await uvq.warmup();
  return uvq;
}

async function init() {
  const backends = {}; // { name: UVQ instance }

  // WebGPU requires a secure context (HTTPS or localhost)
  const hasWebGPU = window.isSecureContext && !!navigator.gpu;
  if (!hasWebGPU) {
    if (!window.isSecureContext) {
      console.warn("WebGPU unavailable: page is not in a secure context (use HTTPS or localhost).");
    } else {
      console.warn("WebGPU unavailable: navigator.gpu not found in this browser.");
    }
  }

  // Load WebGPU backend (skip warmup — first real inference compiles pipelines)
  if (hasWebGPU) {
    setStatus("Loading models (webgpu backend)...");
    try {
      backends.webgpu = await loadBackend("webgpu", { warmup: false });
    } catch (e) {
      console.warn("WebGPU backend failed:", e);
    }
  }

  // Always load WASM backend
  setStatus("Loading models (wasm backend)...");
  try {
    backends.wasm = await loadBackend("wasm");
  } catch (e) {
    if (!backends.webgpu) {
      showError(`Failed to load models: ${e.message}`);
      setStatus("Model loading failed.");
      return;
    }
  }

  const names = Object.keys(backends);
  if (names.length === 0) {
    showError("No backend available.");
    setStatus("Model loading failed.");
    return;
  }
  const compareMode = names.length === 2;
  setStatus(`Ready (${names.join(" + ")}). Select a video file to assess.`);
  fileArea.style.display = "block";

  fileInput.addEventListener("change", async () => {
    const file = fileInput.files[0];
    if (!file) return;
    if (compareMode) {
      await processVideoCompare(backends.webgpu, backends.wasm, file);
    } else {
      await processVideoSingle(backends[names[0]], names[0], file);
    }
  });
}

async function decodeFrame(videoEl, ctx, t) {
  videoEl.currentTime = t + 0.5;
  await new Promise((resolve) => { videoEl.onseeked = resolve; });
  ctx.drawImage(videoEl, 0, 0);
}

async function processVideoCompare(uvqGpu, uvqWasm, file) {
  resultsEl.style.display = "none";
  progressContainer.style.display = "block";
  progressBar.value = 0;
  frameScoresEl.innerHTML = "";

  const url = URL.createObjectURL(file);
  videoEl.src = url;

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

  setStatus(`Processing ${duration} frame(s) on both backends...`);

  const canvas = new OffscreenCanvas(videoEl.videoWidth, videoEl.videoHeight);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  const frames = [];
  const t0 = performance.now();

  for (let t = 0; t < duration; t++) {
    // Decode once
    const decodeStart = performance.now();
    await decodeFrame(videoEl, ctx, t);
    const { content, patches } = preprocessFrame(canvas);
    const decodeTime = performance.now() - decodeStart;

    // Infer WebGPU
    const gpuStart = performance.now();
    const gpuScore = await uvqGpu.infer(content, patches);
    const gpuTime = performance.now() - gpuStart;

    // Infer WASM
    const wasmStart = performance.now();
    const wasmScore = await uvqWasm.infer(content, patches);
    const wasmTime = performance.now() - wasmStart;

    frames.push({ gpuScore, wasmScore, decodeTime, gpuTime, wasmTime });

    progressBar.value = ((t + 1) / duration) * 100;
    setStatus(`Processing frame ${t + 1} / ${duration}...`);
  }

  const elapsed = performance.now() - t0;
  URL.revokeObjectURL(url);

  // Display results
  const avgGpu = frames.reduce((a, f) => a + f.gpuScore, 0) / frames.length;
  const totalGpuInfer = frames.reduce((a, f) => a + f.gpuTime, 0);
  const totalWasmInfer = frames.reduce((a, f) => a + f.wasmTime, 0);
  const speedup = totalWasmInfer / totalGpuInfer;

  scoreEl.textContent = avgGpu.toFixed(3);
  timingEl.textContent =
    `${duration} frames in ${(elapsed / 1000).toFixed(1)}s — ` +
    `WebGPU total: ${(totalGpuInfer / 1000).toFixed(1)}s, ` +
    `WASM total: ${(totalWasmInfer / 1000).toFixed(1)}s, ` +
    `speedup: ${speedup.toFixed(1)}x`;

  frameScoresHeadEl.innerHTML = `<tr><th>Second</th><th>Score (WebGPU)</th><th>Score (WASM)</th><th>Decode</th><th>Infer (WebGPU)</th><th>Infer (WASM)</th><th>Speedup</th></tr>`;
  frameScoresEl.innerHTML = frames
    .map((f, i) => {
      const sp = f.wasmTime / f.gpuTime;
      return `<tr>` +
        `<td>${i}</td>` +
        `<td>${f.gpuScore.toFixed(4)}</td>` +
        `<td>${f.wasmScore.toFixed(4)}</td>` +
        `<td>${f.decodeTime.toFixed(0)} ms</td>` +
        `<td>${f.gpuTime.toFixed(0)} ms</td>` +
        `<td>${f.wasmTime.toFixed(0)} ms</td>` +
        `<td>${sp.toFixed(1)}x</td>` +
        `</tr>`;
    })
    .join("");

  resultsEl.style.display = "block";
  progressContainer.style.display = "none";
  setStatus("Done.");
}

async function processVideoSingle(uvq, backendName, file) {
  resultsEl.style.display = "none";
  progressContainer.style.display = "block";
  progressBar.value = 0;
  frameScoresEl.innerHTML = "";

  const url = URL.createObjectURL(file);
  videoEl.src = url;

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

  setStatus(`Processing ${duration} frame(s) (${backendName})...`);

  const canvas = new OffscreenCanvas(videoEl.videoWidth, videoEl.videoHeight);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  const frames = [];
  const t0 = performance.now();

  for (let t = 0; t < duration; t++) {
    const decodeStart = performance.now();
    await decodeFrame(videoEl, ctx, t);
    const { content, patches } = preprocessFrame(canvas);
    const decodeTime = performance.now() - decodeStart;

    const inferStart = performance.now();
    const score = await uvq.infer(content, patches);
    const inferTime = performance.now() - inferStart;

    frames.push({ score, decodeTime, inferTime });

    progressBar.value = ((t + 1) / duration) * 100;
    setStatus(`Processing frame ${t + 1} / ${duration}...`);
  }

  const elapsed = performance.now() - t0;
  URL.revokeObjectURL(url);

  const avgScore = frames.reduce((a, f) => a + f.score, 0) / frames.length;
  scoreEl.textContent = avgScore.toFixed(3);
  timingEl.textContent = `${duration} frames in ${(elapsed / 1000).toFixed(1)}s (${(elapsed / duration).toFixed(0)} ms/frame, ${backendName})`;

  frameScoresHeadEl.innerHTML = `<tr><th>Second</th><th>Score</th><th>Decode time</th><th>Inference time</th></tr>`;
  frameScoresEl.innerHTML = frames
    .map((f, i) => `<tr><td>${i}</td><td>${f.score.toFixed(4)}</td><td>${f.decodeTime.toFixed(0)} ms</td><td>${f.inferTime.toFixed(0)} ms</td></tr>`)
    .join("");

  resultsEl.style.display = "block";
  progressContainer.style.display = "none";
  setStatus("Done.");
}

init().catch((e) => {
  console.error(e);
  showError(e.message);
});
