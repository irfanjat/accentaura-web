(function (global, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    global.AccentFeatures = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  const SR = 22050;
  const N_FFT = 2048;
  const HOP = 512;
  const N_BINS = N_FFT / 2 + 1;
  const N_MFCC = 20;
  const FRAME_LEN = 2048;
  const WINDOW_SEC = 5.0;
  const HOP_SEC = 2.5;
  const MIN_DURATION = 2.5;
  const MIN_RMS = 0.005;
  const PAD = N_FFT >> 1;

  const bitRev = new Uint32Array(N_FFT);
  const bits = Math.log2(N_FFT);
  for (let i = 0; i < N_FFT; i++) {
    let x = i;
    let r = 0;
    for (let j = 0; j < bits; j++) {
      r = (r << 1) | (x & 1);
      x >>= 1;
    }
    bitRev[i] = r;
  }
  const cosT = new Float64Array(N_FFT / 2);
  const sinT = new Float64Array(N_FFT / 2);
  for (let i = 0; i < N_FFT / 2; i++) {
    cosT[i] = Math.cos((2 * Math.PI * i) / N_FFT);
    sinT[i] = Math.sin((2 * Math.PI * i) / N_FFT);
  }

  const hann = new Float64Array(N_FFT);
  for (let n = 0; n < N_FFT; n++) {
    hann[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N_FFT);
  }

  let FB = null;

  function fft(re, im) {
    for (let i = 0; i < N_FFT; i++) {
      const j = bitRev[i];
      if (j > i) {
        let t = re[i];
        re[i] = re[j];
        re[j] = t;
        t = im[i];
        im[i] = im[j];
        im[j] = t;
      }
    }
    for (let size = 2; size <= N_FFT; size <<= 1) {
      const half = size >> 1;
      const step = N_FFT / size;
      for (let i = 0; i < N_FFT; i += size) {
        for (let j = 0; j < half; j++) {
          const k = j * step;
          const wr = cosT[k];
          const wi = -sinT[k];
          const a = i + j;
          const b = a + half;
          const tr = re[b] * wr - im[b] * wi;
          const ti = re[b] * wi + im[b] * wr;
          re[b] = re[a] - tr;
          im[b] = im[a] - ti;
          re[a] += tr;
          im[a] += ti;
        }
      }
    }
  }

  function mean(a) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i];
    return a.length ? s / a.length : 0;
  }

  function rms(a) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * a[i];
    return a.length ? Math.sqrt(s / a.length) : 0;
  }

  function quantile(sorted, q) {
    if (!sorted.length) return 0;
    const pos = (sorted.length - 1) * q;
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    if (lo === hi) return sorted[lo];
    return sorted[lo] * (hi - pos) + sorted[hi] * (pos - lo);
  }

  function frameRms(y, frameLen, hop, center) {
    const pad = center ? frameLen >> 1 : 0;
    const total = y.length + pad * 2;
    const n = Math.max(1, 1 + Math.floor((total - frameLen) / hop));
    const out = new Float64Array(n);
    for (let t = 0; t < n; t++) {
      let s = 0;
      const start = t * hop;
      for (let i = 0; i < frameLen; i++) {
        const p = start + i - pad;
        const v = p >= 0 && p < y.length ? y[p] : 0;
        s += v * v;
      }
      out[t] = Math.sqrt(s / frameLen);
    }
    return out;
  }

  function preprocess(y) {
    const duration = y.length / SR;
    if (duration < MIN_DURATION) {
      return {
        ok: false,
        error:
          "Audio is too short (" +
          duration.toFixed(1) +
          "s). Please record at least " +
          MIN_DURATION +
          "s (recommended: 10-25s) for a reliable acoustic reading.",
        duration: duration
      };
    }

    const rawRms = rms(y);
    if (rawRms < MIN_RMS) {
      return {
        ok: false,
        error:
          "The recording appears to be silent or extremely faint. Please check your microphone input volume and speak clearly.",
        duration: duration,
        rms: rawRms
      };
    }

    const fr = frameRms(y, FRAME_LEN, HOP, true);
    let maxFrame = 0;
    for (let i = 0; i < fr.length; i++) if (fr[i] > maxFrame) maxFrame = fr[i];
    const threshold = maxFrame * Math.pow(10, -25 / 20);
    let first = 0;
    let last = fr.length - 1;
    while (first < fr.length && fr[first] < threshold) first++;
    while (last >= 0 && fr[last] < threshold) last--;
    if (first > last) {
      first = 0;
      last = fr.length - 1;
    }
    let start = Math.max(0, first * HOP);
    let end = Math.min(y.length, last * HOP + FRAME_LEN);
    if ((end - start) / SR < MIN_DURATION) {
      start = 0;
      end = y.length;
    }

    let peak = 0;
    for (let i = start; i < end; i++) {
      const v = Math.abs(y[i]);
      if (v > peak) peak = v;
    }
    const scale = peak > 0 ? 0.95 / peak : 1;
    const out = new Float32Array(end - start);
    for (let i = 0; i < out.length; i++) out[i] = y[start + i] * scale;

    const abs = new Float64Array(out.length);
    for (let i = 0; i < out.length; i++) abs[i] = Math.abs(out[i]);
    abs.sort();
    const floor = quantile(abs, 0.1) + 1e-8;
    const sig = quantile(abs, 0.95) + 1e-8;
    const snr = 20 * Math.log10(sig / floor);
    const finalDuration = out.length / SR;

    let rating;
    if (snr > 22 && finalDuration >= 8) {
      rating = "Excellent (High SNR, clear articulation)";
    } else if (snr > 14) {
      rating = "Good (Sufficient acoustic signal)";
    } else {
      rating = "Acceptable (Elevated background noise detected)";
    }

    return {
      ok: true,
      y: out,
      duration: finalDuration,
      rawDuration: duration,
      rms: rms(out),
      snr: snr,
      rating: rating
    };
  }

  function splitIntoWindows(y) {
    const win = Math.round(WINDOW_SEC * SR);
    const hop = Math.round(HOP_SEC * SR);
    if (y.length <= win) return [y];
    const windows = [];
    for (let s = 0; s + win <= y.length; s += hop) {
      const chunk = y.subarray(s, s + win);
      if (rms(chunk) >= MIN_RMS) windows.push(chunk);
    }
    if (!windows.length) windows.push(y.subarray(0, Math.min(win, y.length)));
    return windows;
  }

  function setFilterbanks(fb) {
    FB = fb;
  }

  function ensureFilterbanks(fb) {
    if (fb) FB = fb;
    if (!FB) throw new Error("Accent filterbanks are not loaded");
    return FB;
  }

  function applySparse(fbRows, power) {
    const out = new Float64Array(fbRows.length);
    for (let m = 0; m < fbRows.length; m++) {
      const row = fbRows[m];
      const w = row.w;
      const s = row.s;
      let sum = 0;
      for (let i = 0; i < w.length; i++) sum += w[i] * power[s + i];
      out[m] = sum;
    }
    return out;
  }

  function applyDense(fbRows, power, rows) {
    const out = new Float64Array(rows);
    for (let m = 0; m < rows; m++) {
      const row = fbRows[m];
      let sum = 0;
      for (let i = 0; i < row.length; i++) sum += row[i] * power[i];
      out[m] = sum;
    }
    return out;
  }

  function powerToDb(vals, refValue, amin, topDb) {
    const out = new Float64Array(vals.length);
    const r = Math.max(amin, refValue);
    const offset = 10 * Math.log10(r);
    let max = -Infinity;
    for (let i = 0; i < vals.length; i++) {
      const v = 10 * Math.log10(Math.max(amin, vals[i])) - offset;
      out[i] = v;
      if (v > max) max = v;
    }
    if (topDb != null) {
      const floor = max - topDb;
      for (let i = 0; i < out.length; i++) if (out[i] < floor) out[i] = floor;
    }
    return out;
  }

  function stats(series, n) {
    const T = series.length / n;
    const meanArr = new Float64Array(n);
    const stdArr = new Float64Array(n);
    const skewArr = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      let s = 0;
      for (let t = 0; t < T; t++) s += series[t * n + k];
      const mu = T ? s / T : 0;
      let m2 = 0;
      let m3 = 0;
      for (let t = 0; t < T; t++) {
        const d = series[t * n + k] - mu;
        const d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
      }
      m2 /= T || 1;
      m3 /= T || 1;
      const sd = Math.sqrt(Math.max(0, m2));
      meanArr[k] = mu;
      stdArr[k] = sd;
      skewArr[k] = sd > 1e-12 ? m3 / (sd * sd * sd) : 0;
    }
    return { mean: meanArr, std: stdArr, skew: skewArr };
  }

  function sgDeriv(series, n, kernel) {
    const T = series.length / n;
    const half = (kernel.length - 1) >> 1;
    const out = new Float64Array(series.length);
    for (let t = 0; t < T; t++) {
      for (let k = 0; k < n; k++) {
        let sum = 0;
        for (let j = 0; j < kernel.length; j++) {
          const tt = t + half - j;
          const idx = (tt < 0 ? 0 : tt >= T ? T - 1 : tt) * n + k;
          sum += kernel[j] * series[idx];
        }
        out[t * n + k] = sum;
      }
    }
    return out;
  }

  function extractFeatures(y) {
    const fb = ensureFilterbanks();
    const L = y.length;
    const T = 1 + Math.floor(L / HOP);
    const padded = new Float32Array(PAD + L + PAD);
    padded.set(y, PAD);

    const re = new Float64Array(N_FFT);
    const im = new Float64Array(N_FFT);
    const powerFlat = new Float64Array(T * N_BINS);
    const magFlat = new Float64Array(T * N_BINS);
    const rmsArr = new Float64Array(T);
    const zcrArr = new Float64Array(T);
    const centroid = new Float64Array(T);
    const bandwidth = new Float64Array(T);
    const rolloff = new Float64Array(T);

    for (let t = 0; t < T; t++) {
      const off = t * HOP;
      let energy = 0;
      let crossings = 0;
      let prevPositive = false;
      for (let n = 0; n < N_FFT; n++) {
        const v = padded[off + n];
        energy += v * v;
        const pos = v >= 0;
        if (n > 0 && pos !== prevPositive) crossings++;
        prevPositive = pos;
        re[n] = v * hann[n];
        im[n] = 0;
      }
      rmsArr[t] = Math.sqrt(energy / N_FFT);
      zcrArr[t] = crossings / N_FFT;
      fft(re, im);

      let totalMag = 0;
      let weighted = 0;
      let weighted2 = 0;
      let frameMax = 0;
      for (let k = 0; k < N_BINS; k++) {
        const p = re[k] * re[k] + im[k] * im[k];
        const m = Math.sqrt(p);
        powerFlat[t * N_BINS + k] = p;
        magFlat[t * N_BINS + k] = m;
        totalMag += m;
        const f = fb.freqs[k];
        weighted += f * m;
        weighted2 += f * f * m;
        if (m > frameMax) frameMax = m;
      }
      const c = totalMag > 1e-12 ? weighted / totalMag : 0;
      centroid[t] = c;
      const bw = totalMag > 1e-12 ? Math.sqrt(Math.max(0, weighted2 / totalMag - c * c)) : 0;
      bandwidth[t] = bw;

      const target = 0.85 * totalMag;
      let acc = 0;
      let roll = fb.freqs[N_BINS - 1];
      for (let k = 0; k < N_BINS; k++) {
        acc += magFlat[t * N_BINS + k];
        if (acc >= target) {
          roll = fb.freqs[k];
          break;
        }
      }
      rolloff[t] = roll;
    }

    const mel128 = new Float64Array(T * 128);
    const mel16 = new Float64Array(T * 16);
    const chroma = new Float64Array(T * 12);
    for (let t = 0; t < T; t++) {
      const p = powerFlat.subarray(t * N_BINS, (t + 1) * N_BINS);
      const m128 = applySparse(fb.mel128, p);
      const m16 = applySparse(fb.mel16, p);
      const ch = applyDense(fb.chroma, p, 12);
      let chMax = 0;
      for (let i = 0; i < 12; i++) if (ch[i] > chMax) chMax = ch[i];
      for (let i = 0; i < 12; i++) chroma[t * 12 + i] = chMax > 1e-12 ? ch[i] / chMax : 0;
      for (let i = 0; i < 128; i++) mel128[t * 128 + i] = m128[i];
      for (let i = 0; i < 16; i++) mel16[t * 16 + i] = m16[i];
    }

    const logmel = new Float64Array(T * 128);
    for (let t = 0; t < T; t++) {
      const db = powerToDb(
        mel128.subarray(t * 128, (t + 1) * 128),
        1.0,
        1e-10,
        80.0
      );
      for (let i = 0; i < 128; i++) logmel[t * 128 + i] = db[i];
    }

    const mfcc = new Float64Array(T * N_MFCC);
    for (let t = 0; t < T; t++) {
      for (let k = 0; k < N_MFCC; k++) {
        const row = fb.dct[k];
        let s = 0;
        for (let n = 0; n < 128; n++) s += row[n] * logmel[t * 128 + n];
        mfcc[t * N_MFCC + k] = s;
      }
    }

    const mStats = stats(mfcc, N_MFCC);
    const d1 = sgDeriv(mfcc, N_MFCC, fb.sg1);
    const d2 = sgDeriv(mfcc, N_MFCC, fb.sg2);
    const d1Stats = stats(d1, N_MFCC);
    const d2Stats = stats(d2, N_MFCC);

    const cStats = stats(centroid, 1);
    const bwStats = stats(bandwidth, 1);
    const rollStats = stats(rolloff, 1);
    const zcrStats = stats(zcrArr, 1);
    const rmsStats = stats(rmsArr, 1);
    const chromaStats = stats(chroma, 12);

    let globalMax = 0;
    for (let i = 0; i < mel16.length; i++) if (mel16[i] > globalMax) globalMax = mel16[i];
    const logmel16 = new Float64Array(T * 16);
    for (let t = 0; t < T; t++) {
      const db = powerToDb(
        mel16.subarray(t * 16, (t + 1) * 16),
        globalMax,
        1e-10,
        80.0
      );
      for (let i = 0; i < 16; i++) logmel16[t * 16 + i] = db[i];
    }
    const mel16Stats = stats(logmel16, 16);

    const contrast = new Float64Array(T * 7);
    for (let b = 0; b < 7; b++) {
      const band = fb.contrast_bands[b];
      const vals = new Float64Array(band.length);
      for (let t = 0; t < T; t++) {
        for (let i = 0; i < band.length; i++) {
          vals[i] = magFlat[t * N_BINS + band[i]];
        }
        vals.sort();
        const idx = Math.max(1, Math.round(0.02 * band.length));
        let valley = 0;
        let peak = 0;
        for (let i = 0; i < idx; i++) valley += vals[i];
        for (let i = band.length - idx; i < band.length; i++) peak += vals[i];
        valley /= idx;
        peak /= idx;
        contrast[t * 7 + b] =
          10 * Math.log10(Math.max(1e-10, peak)) -
          10 * Math.log10(Math.max(1e-10, valley));
      }
    }
    const contrastStats = stats(contrast, 7);

    const out = new Float32Array(220);
    let o = 0;
    for (let i = 0; i < 20; i++) out[o++] = mStats.mean[i];
    for (let i = 0; i < 20; i++) out[o++] = mStats.std[i];
    for (let i = 0; i < 20; i++) out[o++] = mStats.skew[i];
    for (let i = 0; i < 20; i++) out[o++] = d1Stats.mean[i];
    for (let i = 0; i < 20; i++) out[o++] = d1Stats.std[i];
    for (let i = 0; i < 20; i++) out[o++] = d2Stats.mean[i];
    for (let i = 0; i < 20; i++) out[o++] = d2Stats.std[i];
    out[o++] = cStats.mean[0];
    out[o++] = cStats.std[0];
    out[o++] = bwStats.mean[0];
    out[o++] = bwStats.std[0];
    out[o++] = rollStats.mean[0];
    out[o++] = rollStats.std[0];
    for (let i = 0; i < 7; i++) out[o++] = contrastStats.mean[i];
    for (let i = 0; i < 7; i++) out[o++] = contrastStats.std[i];
    out[o++] = zcrStats.mean[0];
    out[o++] = zcrStats.std[0];
    out[o++] = rmsStats.mean[0];
    out[o++] = rmsStats.std[0];
    for (let i = 0; i < 12; i++) out[o++] = chromaStats.mean[i];
    for (let i = 0; i < 12; i++) out[o++] = chromaStats.std[i];
    for (let i = 0; i < 16; i++) out[o++] = mel16Stats.mean[i];
    for (let i = 0; i < 16; i++) out[o++] = mel16Stats.std[i];

    for (let i = 0; i < out.length; i++) {
      if (!Number.isFinite(out[i])) out[i] = 0;
    }
    return out;
  }

  function extractMatrix(windows) {
    const rows = windows.length;
    const data = new Float32Array(rows * 220);
    for (let i = 0; i < rows; i++) {
      const f = extractFeatures(windows[i]);
      data.set(f, i * 220);
    }
    return data;
  }

  return {
    SR: SR,
    N_FFT: N_FFT,
    HOP: HOP,
    N_FEATURES: 220,
    setFilterbanks: setFilterbanks,
    preprocess: preprocess,
    splitIntoWindows: splitIntoWindows,
    extractFeatures: extractFeatures,
    extractMatrix: extractMatrix
  };
});
