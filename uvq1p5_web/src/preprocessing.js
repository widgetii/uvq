/**
 * Preprocessing utilities for UVQ 1.5 browser inference.
 *
 * Handles canvas-based resize and normalization to produce NCHW Float32Arrays
 * matching the PyTorch pipeline's expected inputs.
 */

/**
 * Resize a video frame to 256x256 and normalize to [-1, 1] in NCHW layout.
 *
 * @param {ImageBitmap|HTMLCanvasElement|OffscreenCanvas} source - Source frame
 * @returns {Float32Array} Shape: (1, 3, 256, 256)
 */
export function resizeAndNormalizeForContent(source) {
  const W = 256, H = 256;
  const canvas = new OffscreenCanvas(W, H);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(source, 0, 0, W, H);
  const imageData = ctx.getImageData(0, 0, W, H);
  return rgbaToNCHW(imageData.data, W, H, 1);
}

/**
 * Resize a video frame to 1920x1080 and extract 9 patches (3x3 grid of 640x360).
 * Each patch is normalized to [-1, 1] in NCHW layout.
 *
 * @param {ImageBitmap|HTMLCanvasElement|OffscreenCanvas} source - Source frame
 * @returns {Float32Array} Shape: (9, 3, 360, 640)
 */
export function extractDistortionPatches(source) {
  const FULL_W = 1920, FULL_H = 1080;
  const PATCH_W = 640, PATCH_H = 360;
  const GRID_X = 3, GRID_Y = 3;
  const NUM_PATCHES = GRID_X * GRID_Y;

  const canvas = new OffscreenCanvas(FULL_W, FULL_H);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(source, 0, 0, FULL_W, FULL_H);

  const patchSize = 3 * PATCH_H * PATCH_W;
  const result = new Float32Array(NUM_PATCHES * patchSize);

  for (let py = 0; py < GRID_Y; py++) {
    for (let px = 0; px < GRID_X; px++) {
      const x0 = px * PATCH_W;
      const y0 = py * PATCH_H;
      const imageData = ctx.getImageData(x0, y0, PATCH_W, PATCH_H);
      const patchNCHW = rgbaToNCHW(imageData.data, PATCH_W, PATCH_H, 1);

      const patchIdx = py * GRID_X + px;
      result.set(patchNCHW, patchIdx * patchSize);
    }
  }

  return result;
}

/**
 * Convert RGBA pixel data to NCHW Float32Array normalized to [-1, 1].
 *
 * @param {Uint8ClampedArray} rgba - RGBA pixel data (4 bytes per pixel)
 * @param {number} width
 * @param {number} height
 * @param {number} batch - Batch dimension (usually 1)
 * @returns {Float32Array} Shape: (batch, 3, height, width)
 */
function rgbaToNCHW(rgba, width, height, batch) {
  const pixels = width * height;
  const out = new Float32Array(batch * 3 * pixels);

  for (let i = 0; i < pixels; i++) {
    const r = rgba[i * 4];
    const g = rgba[i * 4 + 1];
    const b = rgba[i * 4 + 2];
    // Normalize to [-1, 1]: pixel / 127.5 - 1.0
    out[0 * pixels + i] = r / 127.5 - 1.0;  // R plane
    out[1 * pixels + i] = g / 127.5 - 1.0;  // G plane
    out[2 * pixels + i] = b / 127.5 - 1.0;  // B plane
  }

  return out;
}

/**
 * Preprocess a video frame for both content and distortion networks.
 *
 * @param {ImageBitmap|HTMLCanvasElement|OffscreenCanvas} source
 * @returns {{content: Float32Array, patches: Float32Array}}
 *   content: (1, 3, 256, 256), patches: (9, 3, 360, 640)
 */
export function preprocessFrame(source) {
  return {
    content: resizeAndNormalizeForContent(source),
    patches: extractDistortionPatches(source),
  };
}
