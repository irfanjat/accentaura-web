(function () {
  const SR = 22050;
  const MIN_CONFIDENCE_THRESHOLD = 0.28;
  const DATA_BASE = "data/";
  const ASSET_BASE = "";

  let filterbanks = null;
  let model = null;
  let accents = null;
  let modelPromise = null;
  let accentsPromise = null;

  function assetUrl(path) {
    return ASSET_BASE + path;
  }

  async function fetchJson(name) {
    const base = new URL(DATA_BASE, document.baseURI).href;
    const res = await fetch(new URL(name, base).href, { cache: "force-cache" });
    if (!res.ok) throw new Error("Failed to load " + name + " (" + res.status + ")");
    return res.json();
  }

  function loadAccents() {
    if (accents) return Promise.resolve(accents);
    if (!accentsPromise) {
      accentsPromise = fetchJson("accents.json")
        .then(function (data) {
          accents = data;
          return data;
        })
        .catch(function (err) {
          accentsPromise = null;
          throw err;
        });
    }
    return accentsPromise;
  }

  function init() {
    if (model) return Promise.resolve();
    if (!modelPromise) {
      modelPromise = Promise.all([fetchJson("filterbanks.json"), fetchJson("model.json"), loadAccents()])
        .then(function (results) {
          filterbanks = results[0];
          model = results[1];
          window.AccentFeatures.setFilterbanks(filterbanks);
        })
        .catch(function (err) {
          modelPromise = null;
          throw err;
        });
    }
    return modelPromise;
  }

  function resampleLinear(input, srcRate, dstRate) {
    if (srcRate === dstRate) return input;
    const ratio = dstRate / srcRate;
    const outLen = Math.max(1, Math.round(input.length * ratio));
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const pos = i / ratio;
      const i0 = Math.floor(pos);
      const i1 = Math.min(input.length - 1, i0 + 1);
      const frac = pos - i0;
      out[i] = input[i0] * (1 - frac) + input[i1] * frac;
    }
    return out;
  }

  async function decodeBlob(blob) {
    const arrayBuffer = await blob.arrayBuffer();
    let audioBuffer = null;
    const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    if (OfflineCtx) {
      try {
        const ctx = new OfflineCtx(1, 1, SR);
        audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
      } catch (err) {
        audioBuffer = null;
      }
    }
    if (!audioBuffer) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) throw new Error("This browser does not support audio decoding.");
      const ctx = new AudioCtx();
      try {
        audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
      } finally {
        if (ctx.close) ctx.close();
      }
    }

    const channels = audioBuffer.numberOfChannels;
    const length = audioBuffer.length;
    let mono = new Float32Array(length);
    for (let c = 0; c < channels; c++) {
      const data = audioBuffer.getChannelData(c);
      for (let i = 0; i < length; i++) mono[i] += data[i] / channels;
    }
    if (audioBuffer.sampleRate !== SR) {
      mono = resampleLinear(mono, audioBuffer.sampleRate, SR);
    }
    return mono;
  }

  function predictProbabilities(features) {
    const classes = model.classes;
    const totals = new Float64Array(classes.length);
    const trees = model.trees;
    for (let i = 0; i < trees.length; i++) {
      const tree = trees[i];
      const feat = tree.feature;
      const thr = tree.threshold;
      const left = tree.left;
      const right = tree.right;
      const leafIndex = tree.leaf_index;
      const leafProbs = tree.leaf_probs;
      let node = 0;
      while (feat[node] !== -2) {
        const value = features[feat[node]];
        node = value <= thr[node] ? left[node] : right[node];
      }
      const li = leafIndex[node];
      if (li >= 0) {
        const base = li * classes.length;
        for (let c = 0; c < classes.length; c++) totals[c] += leafProbs[base + c];
      }
    }
    for (let c = 0; c < totals.length; c++) totals[c] /= trees.length;
    return totals;
  }

  function rankPredictions(probs) {
    const classes = model.classes;
    const order = classes.map(function (_, i) {
      return i;
    });
    order.sort(function (a, b) {
      return probs[b] - probs[a];
    });
    return order.map(function (idx) {
      const cls = classes[idx];
      const meta = (accents && accents.accents && accents.accents[cls]) || {};
      const prob = probs[idx];
      return {
        accent_id: String(cls).toLowerCase(),
        name: meta.name || cls + " English",
        variant: meta.variant || cls,
        flag: meta.flag || "🌐",
        region: meta.region || "Global",
        key_features: meta.key_features || "Regional phonetic characteristics",
        color: meta.color || "#38bdf8",
        probability: Math.round(prob * 10000) / 10000,
        confidence_percent: Math.round(prob * 1000) / 10
      };
    });
  }

  function timestampNow() {
    const d = new Date();
    const pad = function (n) {
      return String(n).padStart(2, "0");
    };
    return (
      d.getFullYear() +
      "-" +
      pad(d.getMonth() + 1) +
      "-" +
      pad(d.getDate()) +
      " " +
      pad(d.getHours()) +
      ":" +
      pad(d.getMinutes()) +
      ":" +
      pad(d.getSeconds())
    );
  }

  async function predictFromBlob(blob) {
    await init();
    if (!window.AccentFeatures) throw new Error("Acoustic feature engine unavailable.");
    const y = await decodeBlob(blob);
    const pre = window.AccentFeatures.preprocess(y);
    if (!pre.ok) {
      return {
        success: false,
        error: pre.error,
        quality_info: { duration: pre.duration || 0, is_valid: false }
      };
    }

    const windows = window.AccentFeatures.splitIntoWindows(pre.y);
    const matrix = window.AccentFeatures.extractMatrix(windows);
    const nFeatures = window.AccentFeatures.N_FEATURES;
    const probs = new Float64Array(model.classes.length);
    for (let w = 0; w < windows.length; w++) {
      const f = matrix.subarray(w * nFeatures, (w + 1) * nFeatures);
      const p = predictProbabilities(f);
      for (let c = 0; c < probs.length; c++) probs[c] += p[c];
    }
    for (let c = 0; c < probs.length; c++) probs[c] /= windows.length;

    const ranked = rankPredictions(probs);
    const top = ranked[0];
    const uncertain = top.probability < MIN_CONFIDENCE_THRESHOLD;
    const metrics = model.metrics || {};

    return {
      success: true,
      detected_accent: top.name,
      detected_accent_id: top.accent_id,
      detected_flag: top.flag,
      detected_variant: top.variant,
      detected_region: top.region,
      confidence_score: top.confidence_percent,
      is_uncertain: uncertain,
      uncertainty_reason: uncertain
        ? "Low confidence score: Acoustic signals overlap with multiple dialect boundaries. Speak 15-30 seconds for clearer phonological separation."
        : null,
      educational_description: top.key_features,
      top_candidates: ranked.slice(0, 3),
      all_candidates: ranked,
      audio_quality: {
        duration_seconds: Math.round(pre.duration * 100) / 100,
        sample_rate: SR,
        rms_energy: Math.round(pre.rms * 10000) / 10000,
        estimated_snr_db: Math.round(pre.snr * 10) / 10,
        rating: pre.rating,
        segments_analyzed: windows.length
      },
      timestamp: timestampNow(),
      model_metadata: {
        model_name: model.model_name || "ExtraTreesClassifier",
        dataset_source: "Speech Accent Archive (George Mason University)",
        dataset_license: "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC-BY-NC-SA 4.0)",
        methodology:
          "Client-side acoustic ML classification using 220-dimensional MFCC, spectral contrast, chroma, and mel-filterbank features. No speech-to-text transcription or generative AI.",
        dataset_accuracy_disclosure:
          metrics.accuracy != null
            ? (metrics.accuracy * 100).toFixed(1) +
              "% cross-validation accuracy on unseen speakers"
            : "Evaluated via StratifiedGroupKFold by speaker"
      }
    };
  }

  window.AccentPipeline = {
    init: init,
    loadAccents: loadAccents,
    predictFromBlob: predictFromBlob,
    assetUrl: assetUrl,
    decodeBlob: decodeBlob
  };
})();
