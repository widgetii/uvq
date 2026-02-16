/**
 * UVQ 1.5 ONNX inference engine.
 *
 * Framework-agnostic: accepts an `ort` module so it works with both
 * onnxruntime-web (browser/WebGPU) and onnxruntime-node (Node.js/CPU).
 *
 * Data flow (all NCHW):
 *   ContentNet:     (1, 3, 256, 256)  -> (1, 128, 8, 8)
 *   DistortionNet:  (9, 3, 360, 640)  -> (9, 128, 8, 8) -> reassemble -> (1, 128, 24, 24)
 *   AggregationNet: (1, 128, 8, 8) + (1, 128, 24, 24) -> (1, 1) score in [1, 5]
 */

export class UVQ {
  /**
   * @param {object} ort - The ONNX Runtime module (onnxruntime-web or onnxruntime-node).
   */
  constructor(ort) {
    this.ort = ort;
    this.contentSession = null;
    this.distortionSession = null;
    this.aggSession = null;
  }

  /**
   * Load all three ONNX model sessions.
   *
   * @param {string|ArrayBuffer} contentPath  - Path or buffer for content_net.onnx
   * @param {string|ArrayBuffer} distortionPath - Path or buffer for distortion_net.onnx
   * @param {string|ArrayBuffer} aggPath      - Path or buffer for aggregation_net.onnx
   * @param {object} [options]                - ort.InferenceSession.SessionOptions
   * @returns {Promise<void>}
   */
  async load(contentPath, distortionPath, aggPath, options) {
    const opts = options || {};
    const [c, d, a] = await Promise.all([
      this.ort.InferenceSession.create(contentPath, opts),
      this.ort.InferenceSession.create(distortionPath, opts),
      this.ort.InferenceSession.create(aggPath, opts),
    ]);
    this.contentSession = c;
    this.distortionSession = d;
    this.aggSession = a;
  }

  /**
   * Run a dummy inference to warm up WebGPU pipelines (no-op for CPU).
   */
  async warmup() {
    const Tensor = this.ort.Tensor;
    const dummyContent = new Tensor("float32", new Float32Array(1 * 3 * 256 * 256), [1, 3, 256, 256]);
    const dummyPatch = new Tensor("float32", new Float32Array(1 * 3 * 360 * 640), [1, 3, 360, 640]);
    const dummyCF = new Tensor("float32", new Float32Array(1 * 128 * 8 * 8), [1, 128, 8, 8]);
    const dummyDF = new Tensor("float32", new Float32Array(1 * 128 * 24 * 24), [1, 128, 24, 24]);

    await this.contentSession.run({ input: dummyContent });
    await this.distortionSession.run({ input: dummyPatch });
    await this.aggSession.run({ content: dummyCF, distortion: dummyDF });
  }

  /**
   * Extract content features from a pre-processed 256x256 frame.
   *
   * @param {Float32Array} inputData - NCHW data of shape (1, 3, 256, 256)
   * @returns {Promise<{data: Float32Array, dims: number[]}>}
   */
  async inferContentFeatures(inputData) {
    const Tensor = this.ort.Tensor;
    const tensor = new Tensor("float32", inputData, [1, 3, 256, 256]);
    const result = await this.contentSession.run({ input: tensor });
    const output = result.output;
    return { data: output.data, dims: output.dims };
  }

  /**
   * Extract distortion features from 9 patches and reassemble into a spatial grid.
   *
   * @param {Float32Array} patchesData - NCHW data of shape (9, 3, 360, 640)
   * @returns {Promise<{data: Float32Array, dims: number[]}>}
   */
  async inferDistortionFeatures(patchesData) {
    const Tensor = this.ort.Tensor;
    const tensor = new Tensor("float32", patchesData, [9, 3, 360, 640]);
    const result = await this.distortionSession.run({ input: tensor });
    const patchFeatures = result.output; // (9, 128, 8, 8)
    return this.reassemblePatches(patchFeatures.data);
  }

  /**
   * Compute quality score from content and distortion features.
   *
   * @param {{data: Float32Array, dims: number[]}} contentFeat  - (1, 128, 8, 8)
   * @param {{data: Float32Array, dims: number[]}} distortionFeat - (1, 128, 24, 24)
   * @returns {Promise<number>} Quality score in [1, 5]
   */
  async inferScore(contentFeat, distortionFeat) {
    const Tensor = this.ort.Tensor;
    const cTensor = new Tensor("float32", contentFeat.data, contentFeat.dims);
    const dTensor = new Tensor("float32", distortionFeat.data, distortionFeat.dims);
    const result = await this.aggSession.run({ content: cTensor, distortion: dTensor });
    return result.output.data[0];
  }

  /**
   * Full pipeline: content input + distortion patches -> quality score.
   *
   * @param {Float32Array} contentInput - (1, 3, 256, 256)
   * @param {Float32Array} patchesInput - (9, 3, 360, 640)
   * @returns {Promise<number>} Quality score in [1, 5]
   */
  async infer(contentInput, patchesInput) {
    const contentFeat = await this.inferContentFeatures(contentInput);
    const distortionFeat = await this.inferDistortionFeatures(patchesInput);
    return this.inferScore(contentFeat, distortionFeat);
  }

  /**
   * Reassemble 9 patch features (3x3 grid) into a single spatial feature map.
   *
   * Input:  (9, 128, 8, 8)  — patches in row-major order (py*3 + px)
   * Output: (1, 128, 24, 24) — assembled spatial grid
   *
   * @param {Float32Array} patchData - Raw data from (9, 128, 8, 8) tensor
   * @returns {{data: Float32Array, dims: number[]}}
   */
  reassemblePatches(patchData) {
    const C = 128, PH = 8, PW = 8;
    const gridY = 3, gridX = 3;
    const outH = gridY * PH; // 24
    const outW = gridX * PW; // 24
    const out = new Float32Array(1 * C * outH * outW);

    for (let p = 0; p < 9; p++) {
      const py = Math.floor(p / gridX);
      const px = p % gridX;
      const patchOffset = p * C * PH * PW;

      for (let c = 0; c < C; c++) {
        for (let fh = 0; fh < PH; fh++) {
          for (let fw = 0; fw < PW; fw++) {
            const srcIdx = patchOffset + c * PH * PW + fh * PW + fw;
            const outY = py * PH + fh;
            const outX = px * PW + fw;
            const dstIdx = c * outH * outW + outY * outW + outX;
            out[dstIdx] = patchData[srcIdx];
          }
        }
      }
    }

    return { data: out, dims: [1, C, outH, outW] };
  }
}
