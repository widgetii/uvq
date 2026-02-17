/**
 * Video decoder abstraction for benchmarking decode methods.
 *
 * Two implementations:
 *  - VideoElementDecoder: <video> element seeking + canvas drawImage (current approach)
 *  - WebCodecsDecoder: mp4box.js demuxing + VideoDecoder API
 *
 * Both implement:
 *  - init(file) -> { width, height, duration }
 *  - decodeFrame(t) -> OffscreenCanvas with frame at t+0.5s
 *  - dispose()
 */

import MP4Box from "mp4box";

// ---------------------------------------------------------------------------
// VideoElementDecoder — wraps the existing <video> seek + drawImage approach
// ---------------------------------------------------------------------------

export class VideoElementDecoder {
  constructor() {
    this._video = document.querySelector("#video-el");
    this._canvas = null;
    this._ctx = null;
    this._url = null;
    this.initTime = 0;
  }

  async init(file) {
    const t0 = performance.now();
    this._url = URL.createObjectURL(file);
    this._video.src = this._url;

    await new Promise((resolve, reject) => {
      this._video.onloadedmetadata = resolve;
      this._video.onerror = () => reject(new Error("Failed to load video"));
    });

    const width = this._video.videoWidth;
    const height = this._video.videoHeight;
    const duration = Math.floor(this._video.duration);

    this._canvas = new OffscreenCanvas(width, height);
    this._ctx = this._canvas.getContext("2d", { willReadFrequently: true });

    this.initTime = performance.now() - t0;
    return { width, height, duration };
  }

  async decodeFrame(t) {
    this._video.currentTime = t + 0.5;
    await new Promise((resolve) => { this._video.onseeked = resolve; });
    this._ctx.drawImage(this._video, 0, 0);
    return this._canvas;
  }

  dispose() {
    if (this._url) {
      URL.revokeObjectURL(this._url);
      this._url = null;
    }
  }
}

// ---------------------------------------------------------------------------
// WebCodecsDecoder — mp4box.js demuxing + VideoDecoder API
// ---------------------------------------------------------------------------

export class WebCodecsDecoder {
  constructor() {
    this._config = null;    // VideoDecoderConfig
    this._canvas = null;
    this._ctx = null;
    this._frameCache = null; // Map<second, OffscreenCanvas> — pre-decoded frames
    this.initTime = 0;       // total init (demux + decode)
    this.demuxTime = 0;      // demux only
    this.decodeTime = 0;     // batch decode only
  }

  /**
   * @param {File} file
   * @param {object} [opts]
   * @param {function(number):void} [opts.onProgress] - called with 0..1 during batch decode
   */
  async init(file, opts) {
    const onProgress = opts?.onProgress;
    const t0 = performance.now();
    const buffer = await file.arrayBuffer();

    const demuxStart = performance.now();
    const { config, samples, width, height, duration } = await this._demux(buffer);
    this.demuxTime = performance.now() - demuxStart;
    this._config = config;

    this._canvas = new OffscreenCanvas(width, height);
    this._ctx = this._canvas.getContext("2d", { willReadFrequently: true });

    // Build CTS-sorted index for target-frame lookup
    const ctsSorted = samples
      .map((s, i) => ({ cts: s.cts, sampleIdx: i }))
      .sort((a, b) => a.cts - b.cts);

    // Determine target CTS values (one per second at t+0.5)
    const targetCtsSet = new Set();
    for (let t = 0; t < duration; t++) {
      const targetUs = (t + 0.5) * 1e6;
      const idx = bsearchCts(ctsSorted, targetUs);
      targetCtsSet.add(samples[ctsSorted[idx].sampleIdx].cts);
    }

    // Single-pass batch decode: feed all samples, capture target frames
    const decodeStart = performance.now();
    this._frameCache = await this._batchDecode(
      samples, targetCtsSet, duration, ctsSorted, width, height, onProgress,
    );
    this.decodeTime = performance.now() - decodeStart;

    this.initTime = performance.now() - t0;
    return { width, height, duration };
  }

  /**
   * Demux an MP4 buffer using mp4box.js to extract samples and decoder config.
   */
  _demux(buffer) {
    return new Promise((resolve, reject) => {
      const mp4 = MP4Box.createFile();
      let videoTrack = null;

      mp4.onReady = (info) => {
        videoTrack = info.videoTracks[0];
        if (!videoTrack) {
          reject(new Error("No video track found in file"));
          return;
        }
        mp4.setExtractionOptions(videoTrack.id);
        mp4.start();
      };

      const allSamples = [];

      mp4.onSamples = (_id, _user, samples) => {
        for (const s of samples) {
          allSamples.push({
            cts: (s.cts / s.timescale) * 1e6,       // microseconds (presentation time)
            dts: (s.dts / s.timescale) * 1e6,       // microseconds (decode time)
            duration: (s.duration / s.timescale) * 1e6,
            is_sync: s.is_sync,
            data: new Uint8Array(s.data),            // copy — mp4box may reuse buffers
          });
        }
      };

      mp4.onError = (e) => reject(new Error(`MP4 demux error: ${e}`));

      // Feed entire buffer to mp4box
      buffer.fileStart = 0;
      mp4.appendBuffer(buffer);
      mp4.flush();

      // Build config once all samples extracted
      if (!videoTrack) {
        reject(new Error("No video track found"));
        return;
      }

      const codec = videoTrack.codec;
      const width = videoTrack.video.width;
      const height = videoTrack.video.height;
      const durationSec = Math.floor(videoTrack.duration / videoTrack.timescale);

      // Extract codec-specific description (avcC / hvcC / vpcC / av1C box)
      const trak = mp4.getTrackById(videoTrack.id);
      const entry = trak.mdia.minf.stbl.stsd.entries[0];
      const descBox = entry.avcC || entry.hvcC || entry.vpcC || entry.av1C;

      let description;
      if (descBox) {
        const stream = new MP4Box.DataStream(undefined, 0, MP4Box.DataStream.BIG_ENDIAN);
        descBox.write(stream);
        // stream.buffer may be pre-allocated larger; use stream.position for actual length
        description = new Uint8Array(stream.buffer, 8, stream.position - 8);
      }

      const config = { codec, codedWidth: width, codedHeight: height };
      if (description) config.description = description;

      // Samples stay in decode order (DTS) — do NOT sort by CTS
      resolve({
        config,
        samples: allSamples,
        width,
        height,
        duration: durationSec,
      });
    });
  }

  /**
   * Single-pass batch decode: feed all samples through one VideoDecoder,
   * capture the frame closest to each target CTS, store as OffscreenCanvas.
   */
  async _batchDecode(samples, targetCtsSet, duration, ctsSorted, width, height, onProgress) {
    // Map target CTS -> second index for fast lookup
    const ctsToSecond = new Map();
    for (let t = 0; t < duration; t++) {
      const targetUs = (t + 0.5) * 1e6;
      const idx = bsearchCts(ctsSorted, targetUs);
      const cts = samples[ctsSorted[idx].sampleIdx].cts;
      ctsToSecond.set(cts, t);
    }

    const cache = new Map();   // second -> OffscreenCanvas
    const pending = new Map(); // cts -> { second, canvas }

    for (const [cts, t] of ctsToSecond) {
      pending.set(cts, { second: t, canvas: null });
    }

    return new Promise((resolve, reject) => {
      let samplesQueued = 0;
      const totalSamples = samples.length;

      const decoder = new VideoDecoder({
        output: (frame) => {
          for (const [cts, entry] of pending) {
            const diff = Math.abs(frame.timestamp - cts);
            if (diff < 500_000) {
              if (!entry.canvas) {
                entry.canvas = new OffscreenCanvas(width, height);
              }
              entry.canvas.getContext("2d").drawImage(frame, 0, 0);
            }
          }
          frame.close();
        },
        error: (e) => reject(new Error(`VideoDecoder error: ${e.message}`)),
      });

      decoder.configure(this._config);

      // Feed samples in batches, yielding to the event loop for UI updates
      const BATCH_SIZE = 64;
      const feedBatch = () => {
        const end = Math.min(samplesQueued + BATCH_SIZE, totalSamples);
        for (let i = samplesQueued; i < end; i++) {
          const s = samples[i];
          decoder.decode(new EncodedVideoChunk({
            type: s.is_sync ? "key" : "delta",
            timestamp: s.cts,
            duration: s.duration,
            data: s.data,
          }));
        }
        samplesQueued = end;
        if (onProgress) onProgress(samplesQueued / totalSamples);

        if (samplesQueued < totalSamples) {
          setTimeout(feedBatch, 0);
        } else {
          decoder.flush().then(() => {
            decoder.close();
            for (const [, entry] of pending) {
              if (entry.canvas) {
                cache.set(entry.second, entry.canvas);
              }
            }
            resolve(cache);
          }).catch(reject);
        }
      };
      feedBatch();
    });
  }

  async decodeFrame(t) {
    const cached = this._frameCache.get(t);
    if (cached) {
      this._ctx.drawImage(cached, 0, 0);
    }
    return this._canvas;
  }

  dispose() {
    this._frameCache = null;
    this._config = null;
    this._canvas = null;
    this._ctx = null;
  }
}

/**
 * Binary search on a CTS-sorted index array for the entry closest to targetUs.
 * Returns the index into the sorted array.
 */
function bsearchCts(sorted, targetUs) {
  let lo = 0, hi = sorted.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (sorted[mid].cts < targetUs) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo > 0) {
    const diffLo = Math.abs(sorted[lo].cts - targetUs);
    const diffPrev = Math.abs(sorted[lo - 1].cts - targetUs);
    if (diffPrev < diffLo) lo = lo - 1;
  }
  return lo;
}
