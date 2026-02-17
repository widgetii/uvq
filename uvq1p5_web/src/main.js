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

async function loadBackend(name, { warmup = true, onProgress } = {}) {
  const uvq = new UVQ(ort);
  await uvq.load(...MODEL_PATHS, { executionProviders: [name] }, onProgress);
  if (warmup) await uvq.warmup();
  return uvq;
}

/**
 * Initialize both decoders. VideoElementDecoder always works;
 * WebCodecsDecoder is best-effort (MP4 only, WebCodecs-capable browsers).
 * Shows progress bar during WebCodecs batch decode.
 */
async function initDecoders(file, { setStatus, progressBar, progressContainer }) {
  const vidDecoder = new VideoElementDecoder();
  const info = await vidDecoder.init(file);

  let wcDecoder = null;
  if (typeof VideoDecoder !== "undefined") {
    const ext = file.name.split(".").pop().toLowerCase();
    if (["mp4", "m4v", "mov"].includes(ext)) {
      try {
        wcDecoder = new WebCodecsDecoder();
        setStatus("Decoding video with WebCodecs...");
        progressContainer.style.display = "block";
        progressBar.value = 0;
        await wcDecoder.init(file, {
          onProgress: (p) => { progressBar.value = p * 100; },
        });
      } catch (e) {
        console.warn("WebCodecs decoder init failed:", e.message);
        wcDecoder = null;
      }
    }
  }

  return { vidDecoder, wcDecoder, info };
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

  progressContainer.style.display = "block";
  progressBar.value = 0;

  const MODEL_NAMES = ["content_net", "distortion_net", "aggregation_net"];
  const totalSteps = (hasWebGPU ? 3 : 0) + 3; // 3 models per backend
  let completedSteps = 0;

  function onModelProgress(loaded, _total, backendName) {
    completedSteps++;
    setStatus(`Loading ${MODEL_NAMES[loaded - 1]} (${backendName})...`);
    progressBar.value = (completedSteps / totalSteps) * 100;
  }

  // Load WebGPU backend (skip warmup — first real inference compiles pipelines)
  if (hasWebGPU) {
    setStatus("Loading models (webgpu)...");
    try {
      backends.webgpu = await loadBackend("webgpu", {
        warmup: false,
        onProgress: (i, n) => onModelProgress(i, n, "webgpu"),
      });
    } catch (e) {
      console.warn("WebGPU backend failed:", e);
    }
  }

  // Always load WASM backend
  setStatus("Loading models (wasm)...");
  try {
    backends.wasm = await loadBackend("wasm", {
      onProgress: (i, n) => onModelProgress(i, n, "wasm"),
    });
  } catch (e) {
    if (!backends.webgpu) {
      showError(`Failed to load models: ${e.message}`);
      setStatus("Model loading failed.");
      progressContainer.style.display = "none";
      return;
    }
  }

  progressContainer.style.display = "none";

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

    if (compareMode) {
      await processVideoCompare(backends.webgpu, backends.wasm, file);
    } else {
      await processVideoSingle(backends[names[0]], names[0], file);
    }
  });
}

async function processVideoCompare(uvqGpu, uvqWasm, file) {
  resultsEl.style.display = "none";
  progressContainer.style.display = "block";
  progressBar.value = 0;
  frameScoresEl.innerHTML = "";

  let vidDecoder, wcDecoder, info;
  try {
    setStatus("Initializing decoders...");
    ({ vidDecoder, wcDecoder, info } = await initDecoders(file, { setStatus, progressBar, progressContainer }));
  } catch (e) {
    showError(`Decoder init failed: ${e.message}`);
    progressContainer.style.display = "none";
    return;
  }

  const { duration } = info;
  if (duration < 1) {
    showError("Video is shorter than 1 second.");
    vidDecoder.dispose();
    if (wcDecoder) wcDecoder.dispose();
    progressContainer.style.display = "none";
    return;
  }

  setStatus(`Processing ${duration} frame(s) on both backends...`);
  progressContainer.style.display = "block";
  progressBar.value = 0;

  const frames = [];
  const t0 = performance.now();

  for (let t = 0; t < duration; t++) {
    const canvas = await vidDecoder.decodeFrame(t);
    const { content, patches } = preprocessFrame(canvas);

    // Infer WebGPU
    const gpuStart = performance.now();
    const gpuScore = await uvqGpu.infer(content, patches);
    const gpuTime = performance.now() - gpuStart;

    // Infer WASM
    const wasmStart = performance.now();
    const wasmScore = await uvqWasm.infer(content, patches);
    const wasmTime = performance.now() - wasmStart;

    frames.push({ gpuScore, wasmScore, gpuTime, wasmTime });

    progressBar.value = ((t + 1) / duration) * 100;
    setStatus(`Processing frame ${t + 1} / ${duration}...`);
  }

  const elapsed = performance.now() - t0;
  vidDecoder.dispose();
  if (wcDecoder) wcDecoder.dispose();

  // Display results
  const avgGpu = frames.reduce((a, f) => a + f.gpuScore, 0) / frames.length;
  const totalGpuInfer = frames.reduce((a, f) => a + f.gpuTime, 0);
  const totalWasmInfer = frames.reduce((a, f) => a + f.wasmTime, 0);
  const inferSpeedup = totalWasmInfer / totalGpuInfer;

  scoreEl.textContent = avgGpu.toFixed(3);
  let timing = `${duration} frames in ${(elapsed / 1000).toFixed(1)}s`;
  if (wcDecoder) {
    timing += ` — WebCodecs decode: ${(wcDecoder.decodeTime / 1000).toFixed(2)}s (batch)`;
  }
  timing +=
    ` | WebGPU infer: ${(totalGpuInfer / 1000).toFixed(1)}s, ` +
    `WASM infer: ${(totalWasmInfer / 1000).toFixed(1)}s (${inferSpeedup.toFixed(1)}x)`;
  timingEl.textContent = timing;

  frameScoresHeadEl.innerHTML = `<tr><th>Second</th><th>Score (WebGPU)</th><th>Score (WASM)</th><th>Infer (WebGPU)</th><th>Infer (WASM)</th><th>Speedup</th></tr>`;
  frameScoresEl.innerHTML = frames
    .map((f, i) => {
      const sp = f.wasmTime / f.gpuTime;
      return `<tr>` +
        `<td>${i}</td>` +
        `<td>${f.gpuScore.toFixed(4)}</td>` +
        `<td>${f.wasmScore.toFixed(4)}</td>` +
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

  let vidDecoder, wcDecoder, info;
  try {
    setStatus("Initializing decoders...");
    ({ vidDecoder, wcDecoder, info } = await initDecoders(file, { setStatus, progressBar, progressContainer }));
  } catch (e) {
    showError(`Decoder init failed: ${e.message}`);
    progressContainer.style.display = "none";
    return;
  }

  const { duration } = info;
  if (duration < 1) {
    showError("Video is shorter than 1 second.");
    vidDecoder.dispose();
    if (wcDecoder) wcDecoder.dispose();
    progressContainer.style.display = "none";
    return;
  }

  setStatus(`Processing ${duration} frame(s) (${backendName})...`);
  progressContainer.style.display = "block";
  progressBar.value = 0;

  const frames = [];
  const t0 = performance.now();

  for (let t = 0; t < duration; t++) {
    const canvas = await vidDecoder.decodeFrame(t);

    const inferStart = performance.now();
    const { content, patches } = preprocessFrame(canvas);
    const score = await uvq.infer(content, patches);
    const inferTime = performance.now() - inferStart;

    frames.push({ score, inferTime });

    progressBar.value = ((t + 1) / duration) * 100;
    setStatus(`Processing frame ${t + 1} / ${duration}...`);
  }

  const elapsed = performance.now() - t0;
  vidDecoder.dispose();
  if (wcDecoder) wcDecoder.dispose();

  const avgScore = frames.reduce((a, f) => a + f.score, 0) / frames.length;

  scoreEl.textContent = avgScore.toFixed(3);
  let timing =
    `${duration} frames in ${(elapsed / 1000).toFixed(1)}s ` +
    `(${(elapsed / duration).toFixed(0)} ms/frame, ${backendName})`;
  if (wcDecoder) {
    timing += ` — WebCodecs decode: ${(wcDecoder.decodeTime / 1000).toFixed(2)}s (batch)`;
  }
  timingEl.textContent = timing;

  frameScoresHeadEl.innerHTML = `<tr><th>Second</th><th>Score</th><th>Inference</th></tr>`;
  frameScoresEl.innerHTML = frames
    .map((f, i) => `<tr><td>${i}</td><td>${f.score.toFixed(4)}</td><td>${f.inferTime.toFixed(0)} ms</td></tr>`)
    .join("");

  resultsEl.style.display = "block";
  progressContainer.style.display = "none";
  setStatus("Done.");
}

init().catch((e) => {
  console.error(e);
  showError(e.message);
});
