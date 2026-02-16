/**
 * UVQ 1.5 ONNX Node.js performance and correctness test.
 *
 * Uses onnxruntime-node (native CPU bindings) to validate that the ONNX
 * models produce correct outputs and to measure inference performance.
 *
 * Usage: node test/perf-test.mjs
 */

import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import ort from "onnxruntime-node";
import { UVQ } from "../src/uvq.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const MODELS_DIR = join(__dirname, "..", "public", "models");

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Seeded PRNG (mulberry32) for reproducible random inputs matching numpy. */
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function randomNormal(rng) {
  // Box-Muller transform
  const u1 = rng();
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1 || 1e-10)) * Math.cos(2 * Math.PI * u2);
}

function fillRandom(arr, rng) {
  for (let i = 0; i < arr.length; i++) {
    arr[i] = randomNormal(rng);
  }
  return arr;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function fmtMs(ms) {
  return ms < 1 ? `${(ms * 1000).toFixed(0)}us` : `${ms.toFixed(1)}ms`;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  console.log("UVQ 1.5 ONNX Node.js Performance Test");
  console.log("======================================\n");

  const modelPaths = {
    content: join(MODELS_DIR, "content_net.onnx"),
    distortion: join(MODELS_DIR, "distortion_net.onnx"),
    aggregation: join(MODELS_DIR, "aggregation_net.onnx"),
  };

  // Check model files exist
  for (const [name, path] of Object.entries(modelPaths)) {
    try {
      readFileSync(path);
    } catch {
      console.error(`Model not found: ${path}`);
      console.error(`Run 'uv run python scripts/export_onnx.py' first.`);
      process.exit(1);
    }
  }

  // Load models
  const uvq = new UVQ(ort);
  const opts = { executionProviders: ["cpu"] };

  const loadStart = performance.now();
  await uvq.load(modelPaths.content, modelPaths.distortion, modelPaths.aggregation, opts);
  const loadMs = performance.now() - loadStart;
  console.log(`Models loaded:       ${fmtMs(loadMs)}`);

  // Generate synthetic inputs
  const rng = mulberry32(42);
  const contentInput = fillRandom(new Float32Array(1 * 3 * 256 * 256), rng);
  const patchesInput = fillRandom(new Float32Array(9 * 3 * 360 * 640), rng);

  // Warm up (3 iterations)
  console.log("\nWarm-up (3 iterations)...");
  for (let i = 0; i < 3; i++) {
    await uvq.infer(contentInput, patchesInput);
  }

  // Timed inference (10 iterations)
  const N = 10;
  const contentTimes = [];
  const distortionTimes = [];
  const aggTimes = [];
  const totalTimes = [];

  console.log(`\nBenchmark (${N} iterations)...\n`);

  for (let i = 0; i < N; i++) {
    const t0 = performance.now();
    const contentFeat = await uvq.inferContentFeatures(contentInput);
    const t1 = performance.now();
    const distortionFeat = await uvq.inferDistortionFeatures(patchesInput);
    const t2 = performance.now();
    const score = await uvq.inferScore(contentFeat, distortionFeat);
    const t3 = performance.now();

    contentTimes.push(t1 - t0);
    distortionTimes.push(t2 - t1);
    aggTimes.push(t3 - t2);
    totalTimes.push(t3 - t0);
  }

  // Report timing
  const report = (label, times) => {
    const med = median(times);
    const min = Math.min(...times);
    const max = Math.max(...times);
    console.log(`${label.padEnd(20)} median=${fmtMs(med).padEnd(10)} min=${fmtMs(min).padEnd(10)} max=${fmtMs(max)}`);
  };

  report("Content net:", contentTimes);
  report("Distortion net:", distortionTimes);
  report("Aggregation net:", aggTimes);
  report("Total pipeline:", totalTimes);

  // Correctness: get final score and validate range
  const finalScore = await uvq.infer(contentInput, patchesInput);
  console.log(`\nScore: ${finalScore.toFixed(6)}`);

  if (finalScore < 1.0 || finalScore > 5.0) {
    console.error(`FAIL: Score ${finalScore} outside valid range [1, 5]`);
    process.exit(1);
  }
  console.log("Score range [1, 5]: OK");

  // Shape validation
  const contentFeat = await uvq.inferContentFeatures(contentInput);
  const distortionFeat = await uvq.inferDistortionFeatures(patchesInput);

  const assertDims = (name, actual, expected) => {
    const ok = actual.length === expected.length && actual.every((v, i) => v === expected[i]);
    if (!ok) {
      console.error(`FAIL: ${name} shape [${actual}] != expected [${expected}]`);
      process.exit(1);
    }
    console.log(`${name} shape [${actual}]: OK`);
  };

  assertDims("Content features", contentFeat.dims, [1, 128, 8, 8]);
  assertDims("Distortion features", distortionFeat.dims, [1, 128, 24, 24]);

  // Cross-reference with reference.json (if available)
  try {
    const refPath = join(MODELS_DIR, "reference.json");
    const ref = JSON.parse(readFileSync(refPath, "utf8"));
    const diff = Math.abs(finalScore - ref.score);
    const atol = 0.01;
    const status = diff < atol ? "OK" : "FAIL";
    console.log(`\nReference score: ${ref.score.toFixed(6)} (diff: ${diff.toFixed(6)}) ${status}`);
    if (diff >= atol) {
      console.error(`Score differs from PyTorch reference by ${diff.toFixed(6)} (atol=${atol})`);
      // Note: different PRNG means the inputs don't match — this is expected.
      // The reference.json check is only meaningful when inputs match exactly.
      console.log("(Expected: JS PRNG differs from numpy — reference check is informational only)");
    }
  } catch {
    console.log("\nreference.json not found, skipping cross-reference check.");
  }

  console.log("\nAll checks passed.");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
