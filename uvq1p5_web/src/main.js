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

async function loadBackend(name, { warmup = true } = {}) {
  const uvq = new UVQ(ort);
  await uvq.load(...MODEL_PATHS, { executionProviders: [name] });
  if (warmup) await uvq.warmup();
  return uvq;
}

/**
 * Initialize both decoders. VideoElementDecoder always works;
 * WebCodecsDecoder is best-effort (MP4 only, WebCodecs-capable browsers).
 */
async function initDecoders(file) {
  const vidDecoder = new VideoElementDecoder();
  const info = await vidDecoder.init(file);

  let wcDecoder = null;
  if (typeof VideoDecoder !== "undefined") {
    const ext = file.name.split(".").pop().toLowerCase();
    if (["mp4", "m4v", "mov"].includes(ext)) {
      try {
        wcDecoder = new WebCodecsDecoder();
        await wcDecoder.init(file);
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
    ({ vidDecoder, wcDecoder, info } = await initDecoders(file));
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

  const frames = [];
  const t0 = performance.now();

  for (let t = 0; t < duration; t++) {
    // Decode with <video> element
    const vidStart = performance.now();
    const vidCanvas = await vidDecoder.decodeFrame(t);
    const vidTime = performance.now() - vidStart;

    // Decode with WebCodecs (timing only if available)
    let wcTime = null;
    if (wcDecoder) {
      const wcStart = performance.now();
      await wcDecoder.decodeFrame(t);
      wcTime = performance.now() - wcStart;
    }

    const { content, patches } = preprocessFrame(vidCanvas);

    // Infer WebGPU
    const gpuStart = performance.now();
    const gpuScore = await uvqGpu.infer(content, patches);
    const gpuTime = performance.now() - gpuStart;

    // Infer WASM
    const wasmStart = performance.now();
    const wasmScore = await uvqWasm.infer(content, patches);
    const wasmTime = performance.now() - wasmStart;

    frames.push({ gpuScore, wasmScore, vidTime, wcTime, gpuTime, wasmTime });

    progressBar.value = ((t + 1) / duration) * 100;
    setStatus(`Processing frame ${t + 1} / ${duration}...`);
  }

  const elapsed = performance.now() - t0;
  vidDecoder.dispose();
  if (wcDecoder) wcDecoder.dispose();

  // Display results
  const avgGpu = frames.reduce((a, f) => a + f.gpuScore, 0) / frames.length;
  const totalVidDecode = frames.reduce((a, f) => a + f.vidTime, 0);
  const totalWcDecode = wcDecoder ? frames.reduce((a, f) => a + f.wcTime, 0) : null;
  const totalGpuInfer = frames.reduce((a, f) => a + f.gpuTime, 0);
  const totalWasmInfer = frames.reduce((a, f) => a + f.wasmTime, 0);
  const inferSpeedup = totalWasmInfer / totalGpuInfer;

  scoreEl.textContent = avgGpu.toFixed(3);
  let timing =
    `${duration} frames in ${(elapsed / 1000).toFixed(1)}s — ` +
    `decode: <video> ${(totalVidDecode / 1000).toFixed(2)}s`;
  if (totalWcDecode != null) {
    const decodeSpeedup = totalVidDecode / totalWcDecode;
    timing += `, WebCodecs ${(totalWcDecode / 1000).toFixed(2)}s (${decodeSpeedup.toFixed(1)}x)`;
  }
  timing +=
    ` | WebGPU infer: ${(totalGpuInfer / 1000).toFixed(1)}s, ` +
    `WASM infer: ${(totalWasmInfer / 1000).toFixed(1)}s (${inferSpeedup.toFixed(1)}x)`;
  timingEl.textContent = timing;

  const wcHead = wcDecoder ? `<th>Decode (WebCodecs)</th>` : "";
  frameScoresHeadEl.innerHTML = `<tr><th>Second</th><th>Score (WebGPU)</th><th>Score (WASM)</th><th>Decode (&lt;video&gt;)</th>${wcHead}<th>Infer (WebGPU)</th><th>Infer (WASM)</th><th>Speedup</th></tr>`;
  frameScoresEl.innerHTML = frames
    .map((f, i) => {
      const sp = f.wasmTime / f.gpuTime;
      const wcCol = wcDecoder ? `<td>${f.wcTime.toFixed(0)} ms</td>` : "";
      return `<tr>` +
        `<td>${i}</td>` +
        `<td>${f.gpuScore.toFixed(4)}</td>` +
        `<td>${f.wasmScore.toFixed(4)}</td>` +
        `<td>${f.vidTime.toFixed(0)} ms</td>` +
        wcCol +
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
    ({ vidDecoder, wcDecoder, info } = await initDecoders(file));
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

  const frames = [];
  const t0 = performance.now();

  for (let t = 0; t < duration; t++) {
    // Decode with <video> element
    const vidStart = performance.now();
    const vidCanvas = await vidDecoder.decodeFrame(t);
    const vidTime = performance.now() - vidStart;

    // Decode with WebCodecs (timing only if available)
    let wcTime = null;
    if (wcDecoder) {
      const wcStart = performance.now();
      await wcDecoder.decodeFrame(t);
      wcTime = performance.now() - wcStart;
    }

    const inferStart = performance.now();
    const { content, patches } = preprocessFrame(vidCanvas);
    const score = await uvq.infer(content, patches);
    const inferTime = performance.now() - inferStart;

    frames.push({ score, vidTime, wcTime, inferTime });

    progressBar.value = ((t + 1) / duration) * 100;
    setStatus(`Processing frame ${t + 1} / ${duration}...`);
  }

  const elapsed = performance.now() - t0;
  vidDecoder.dispose();
  if (wcDecoder) wcDecoder.dispose();

  const avgScore = frames.reduce((a, f) => a + f.score, 0) / frames.length;
  const totalVidDecode = frames.reduce((a, f) => a + f.vidTime, 0);
  const totalWcDecode = wcDecoder ? frames.reduce((a, f) => a + f.wcTime, 0) : null;

  scoreEl.textContent = avgScore.toFixed(3);
  let timing =
    `${duration} frames in ${(elapsed / 1000).toFixed(1)}s ` +
    `(${(elapsed / duration).toFixed(0)} ms/frame, ${backendName}) — ` +
    `decode: <video> ${(totalVidDecode / 1000).toFixed(2)}s`;
  if (totalWcDecode != null) {
    const decodeSpeedup = totalVidDecode / totalWcDecode;
    timing += `, WebCodecs ${(totalWcDecode / 1000).toFixed(2)}s (${decodeSpeedup.toFixed(1)}x)`;
  }
  timingEl.textContent = timing;

  const wcHead = wcDecoder ? `<th>Decode (WebCodecs)</th>` : "";
  frameScoresHeadEl.innerHTML = `<tr><th>Second</th><th>Score</th><th>Decode (&lt;video&gt;)</th>${wcHead}<th>Inference</th></tr>`;
  frameScoresEl.innerHTML = frames
    .map((f, i) => {
      const wcCol = wcDecoder ? `<td>${f.wcTime.toFixed(0)} ms</td>` : "";
      return `<tr><td>${i}</td><td>${f.score.toFixed(4)}</td><td>${f.vidTime.toFixed(0)} ms</td>${wcCol}<td>${f.inferTime.toFixed(0)} ms</td></tr>`;
    })
    .join("");

  resultsEl.style.display = "block";
  progressContainer.style.display = "none";
  setStatus("Done.");
}

init().catch((e) => {
  console.error(e);
  showError(e.message);
});
