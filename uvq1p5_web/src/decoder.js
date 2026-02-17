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
    // _samples: array in decode order (DTS), each with { cts, dts, duration, is_sync, data }
    this._samples = null;
    // _ctsSorted: array of { cts, sampleIdx } sorted by cts for presentation-time lookup
    this._ctsSorted = null;
    this._config = null;    // VideoDecoderConfig
    this._canvas = null;
    this._ctx = null;
    this.initTime = 0;
  }

  async init(file) {
    const t0 = performance.now();
    const buffer = await file.arrayBuffer();

    const { config, samples, width, height, duration } = await this._demux(buffer);
    this._config = config;
    this._samples = samples;

    // Build CTS-sorted index for presentation-time seeking
    this._ctsSorted = samples
      .map((s, i) => ({ cts: s.cts, sampleIdx: i }))
      .sort((a, b) => a.cts - b.cts);

    this._canvas = new OffscreenCanvas(width, height);
    this._ctx = this._canvas.getContext("2d", { willReadFrequently: true });

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

  async decodeFrame(t) {
    const targetUs = (t + 0.5) * 1e6; // target timestamp in microseconds

    // Find sample with closest CTS (presentation time) to target
    const targetIdx = this._findByCts(targetUs);

    // Walk back in decode order to nearest sync (keyframe)
    let syncIdx = targetIdx;
    while (syncIdx > 0 && !this._samples[syncIdx].is_sync) {
      syncIdx--;
    }

    // Decode from keyframe through target, pick frame closest to target CTS
    const targetCts = this._samples[targetIdx].cts;
    const frame = await this._decodeRange(syncIdx, targetIdx, targetCts);
    this._ctx.drawImage(frame, 0, 0);
    frame.close();

    return this._canvas;
  }

  /**
   * Find the sample index (in decode-order array) whose CTS is closest to targetUs.
   * Uses the CTS-sorted index for binary search, then maps back to decode-order index.
   */
  _findByCts(targetUs) {
    const sorted = this._ctsSorted;
    let lo = 0, hi = sorted.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (sorted[mid].cts < targetUs) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    // lo is the first entry >= targetUs; check if lo-1 is closer
    if (lo > 0) {
      const diffLo = Math.abs(sorted[lo].cts - targetUs);
      const diffPrev = Math.abs(sorted[lo - 1].cts - targetUs);
      if (diffPrev < diffLo) lo = lo - 1;
    }
    return sorted[lo].sampleIdx;
  }

  /**
   * Create a fresh VideoDecoder, feed samples syncIdx..targetIdx in decode order,
   * flush, collect all output frames, and return the one closest to targetCts.
   */
  _decodeRange(syncIdx, targetIdx, targetCts) {
    return new Promise((resolve, reject) => {
      const frames = [];

      const decoder = new VideoDecoder({
        output: (frame) => { frames.push(frame); },
        error: (e) => {
          // Close any already-collected frames to avoid GC warnings
          for (const f of frames) f.close();
          frames.length = 0;
          reject(new Error(`VideoDecoder error: ${e.message}`));
        },
      });

      decoder.configure(this._config);

      for (let i = syncIdx; i <= targetIdx; i++) {
        const s = this._samples[i];
        decoder.decode(new EncodedVideoChunk({
          type: s.is_sync ? "key" : "delta",
          timestamp: s.cts,
          duration: s.duration,
          data: s.data,
        }));
      }

      decoder.flush().then(() => {
        decoder.close();
        if (frames.length === 0) {
          reject(new Error("No frame decoded"));
          return;
        }
        // Pick the frame whose timestamp is closest to targetCts
        let best = 0;
        let bestDiff = Math.abs(frames[0].timestamp - targetCts);
        for (let i = 1; i < frames.length; i++) {
          const diff = Math.abs(frames[i].timestamp - targetCts);
          if (diff < bestDiff) {
            bestDiff = diff;
            best = i;
          }
        }
        // Close all frames except the best one
        for (let i = 0; i < frames.length; i++) {
          if (i !== best) frames[i].close();
        }
        resolve(frames[best]);
      }).catch((e) => {
        for (const f of frames) f.close();
        reject(e);
      });
    });
  }

  dispose() {
    this._samples = null;
    this._ctsSorted = null;
    this._config = null;
    this._canvas = null;
    this._ctx = null;
  }
}
