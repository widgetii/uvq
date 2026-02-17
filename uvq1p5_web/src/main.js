/**
 * UVQ 1.5 browser entry point.
 *
 * Loads ONNX models via onnxruntime-web with WebGPU and WASM backends,
 * extracts video frames via a pluggable decoder (video element or WebCodecs),
 * runs per-frame inference on both backends, and displays a performance comparison.
 */

import * as ort from "onnxruntime-web/webgpu";
import { UVQ } from "./uvq.js";
import { preprocessFrame } from "./preprocessing.js";
import { VideoElementDecoder, WebCodecsDecoder } from "./decoder.js";

const $ = (sel) => document.querySelector(sel);

const statusEl = $("#status");
const errorEl = $("#error-banner");
const fileArea = $("#file-input-area");
const fileInput = $("#video-file");
const decodeMethodEl = $("#decode-method");
const progressContainer = $("#progress-bar-container");
const progressBar = $("#progress-bar");
const resultsEl = $("#results");
const scoreEl = $("#score");
const timingEl = $("#timing");
const frameScoresHeadEl = $("#frame-scores-head");
const frameScoresEl = $("#frame-scores");

function setStatus(msg) { statusEl.textContent = msg; }
function showError(msg) { errorEl.textContent = msg; errorEl.style.display = "block"; }
function clearError() { errorEl.style.display = "none"; }

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

function createDecoder(method) {
  return method === "webcodecs" ? new WebCodecsDecoder() : new VideoElementDecoder();
}

function getDecoderLabel(method) {
  return method === "webcodecs" ? "WebCodecs" : "<video> element";
}

async function init() {
  const backends = {}; // { name: UVQ instance }

  // Disable WebCodecs option if not supported
  const hasWebCodecs = typeof VideoDecoder !== "undefined";
  if (!hasWebCodecs) {
    const wcOption = decodeMethodEl.querySelector('option[value="webcodecs"]');
    wcOption.disabled = true;
    wcOption.textContent += " (not supported)";
  }

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
    clearError();

    const method = decodeMethodEl.value;

    // WebCodecs only supports MP4/MOV containers (mp4box.js limitation)
    if (method === "webcodecs") {
      const ext = file.name.split(".").pop().toLowerCase();
      if (!["mp4", "m4v", "mov"].includes(ext)) {
        showError("WebCodecs decoder requires MP4/MOV files. Select a different file or switch to <video> element decode.");
        return;
      }
    }

    if (compareMode) {
      await processVideoCompare(backends.webgpu, backends.wasm, file, method);
    } else {
      await processVideoSingle(backends[names[0]], names[0], file, method);
    }
  });
}

async function processVideoCompare(uvqGpu, uvqWasm, file, method) {
  resultsEl.style.display = "none";
  progressContainer.style.display = "block";
  progressBar.value = 0;
  frameScoresEl.innerHTML = "";

  const decoder = createDecoder(method);
  const decoderLabel = getDecoderLabel(method);
  let info;
  try {
    setStatus(`Initializing decoder (${decoderLabel})...`);
    info = await decoder.init(file);
  } catch (e) {
    showError(`Decoder init failed: ${e.message}`);
    progressContainer.style.display = "none";
    return;
  }

  const { duration } = info;
  if (duration < 1) {
    showError("Video is shorter than 1 second.");
    decoder.dispose();
    progressContainer.style.display = "none";
    return;
  }

  setStatus(`Processing ${duration} frame(s) on both backends (${decoderLabel})...`);

  const frames = [];
  const t0 = performance.now();

  for (let t = 0; t < duration; t++) {
    // Decode once
    const decodeStart = performance.now();
    const canvas = await decoder.decodeFrame(t);
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
  decoder.dispose();

  // Display results
  const avgGpu = frames.reduce((a, f) => a + f.gpuScore, 0) / frames.length;
  const totalGpuInfer = frames.reduce((a, f) => a + f.gpuTime, 0);
  const totalWasmInfer = frames.reduce((a, f) => a + f.wasmTime, 0);
  const speedup = totalWasmInfer / totalGpuInfer;

  scoreEl.textContent = avgGpu.toFixed(3);
  timingEl.textContent =
    `${duration} frames in ${(elapsed / 1000).toFixed(1)}s — ` +
    `decode: ${decoderLabel} (init: ${(decoder.initTime / 1000).toFixed(2)}s), ` +
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

async function processVideoSingle(uvq, backendName, file, method) {
  resultsEl.style.display = "none";
  progressContainer.style.display = "block";
  progressBar.value = 0;
  frameScoresEl.innerHTML = "";

  const decoder = createDecoder(method);
  const decoderLabel = getDecoderLabel(method);
  let info;
  try {
    setStatus(`Initializing decoder (${decoderLabel})...`);
    info = await decoder.init(file);
  } catch (e) {
    showError(`Decoder init failed: ${e.message}`);
    progressContainer.style.display = "none";
    return;
  }

  const { duration } = info;
  if (duration < 1) {
    showError("Video is shorter than 1 second.");
    decoder.dispose();
    progressContainer.style.display = "none";
    return;
  }

  setStatus(`Processing ${duration} frame(s) (${backendName}, ${decoderLabel})...`);

  const frames = [];
  const t0 = performance.now();

  for (let t = 0; t < duration; t++) {
    const decodeStart = performance.now();
    const canvas = await decoder.decodeFrame(t);
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
  decoder.dispose();

  const avgScore = frames.reduce((a, f) => a + f.score, 0) / frames.length;
  scoreEl.textContent = avgScore.toFixed(3);
  timingEl.textContent =
    `${duration} frames in ${(elapsed / 1000).toFixed(1)}s ` +
    `(${(elapsed / duration).toFixed(0)} ms/frame, ${backendName}, ` +
    `decode: ${decoderLabel}, init: ${(decoder.initTime / 1000).toFixed(2)}s)`;

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
