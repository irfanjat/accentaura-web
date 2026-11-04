/**
 * AccentAura – English Accent Identifier
 * Minimalist Commercial Frontend Controller
 */

class AccentAuraApp {
  constructor() {
    // Audio Context & Recording
    this.audioCtx = null;
    this.mediaStream = null;
    this.mediaRecorder = null;
    this.analyser = null;
    this.recordedChunks = [];
    this.recordedBlob = null;
    this.isRecording = false;

    // Timer & Animation
    this.timerInterval = null;
    this.secondsRecorded = 0;
    this.animFrameId = null;
    this.ambientAnimId = null;

    // DOM Elements - Navigation & Stages
    this.sections = {
      home: document.getElementById("sec-home"),
      detect: document.getElementById("sec-detect"),
      explore: document.getElementById("sec-explore"),
      about: document.getElementById("sec-about")
    };

    this.stages = {
      detection: document.getElementById("detection-stage"),
      loading: document.getElementById("loading-stage"),
      results: document.getElementById("results-stage")
    };

    // DOM Elements - Studio
    this.liveCanvas = document.getElementById("live-canvas");
    this.liveCtx = this.liveCanvas ? this.liveCanvas.getContext("2d") : null;
    this.ambientCanvas = document.getElementById("ambient-canvas");
    this.ambientCtx = this.ambientCanvas ? this.ambientCanvas.getContext("2d") : null;

    this.statusBadge = document.getElementById("recording-status");
    this.timerDisplay = document.getElementById("recording-timer");
    this.btnRecord = document.getElementById("btn-record");
    this.recordLabel = document.getElementById("record-label");
    this.btnStop = document.getElementById("btn-stop");
    this.btnPlayback = document.getElementById("btn-playback");
    this.btnRerecord = document.getElementById("btn-rerecord");
    this.btnAnalyze = document.getElementById("btn-analyze");
    this.btnTryAgain = document.getElementById("btn-try-again");

    // Audio Elements
    this.globalAudio = document.getElementById("global-audio-player");
    this.currentPlayingSampleBtn = null;
  }

  init() {
    this.bindNavigation();
    this.bindStudioEvents();
    this.initCanvasScaling();
    this.startAmbientWave();
    this.loadExploreAccents();
    if (window.AccentPipeline) {
      setTimeout(() => {
        window.AccentPipeline.init().catch(() => {});
      }, 1500);
    }

    // Default route from hash
    const initialHash = window.location.hash.replace("#", "") || "home";
    this.navigateTo(initialHash);
  }

  /* ========================================================
     NAVIGATION
     ======================================================== */
  bindNavigation() {
    document.querySelectorAll("[data-nav]").forEach(el => {
      el.addEventListener("click", (e) => {
        e.preventDefault();
        const target = el.getAttribute("data-nav");
        this.navigateTo(target);
      });
    });

    window.addEventListener("hashchange", () => {
      const hash = window.location.hash.replace("#", "") || "home";
      this.navigateTo(hash);
    });
  }

  navigateTo(sectionKey) {
    if (!this.sections[sectionKey]) {
      sectionKey = "home";
    }

    // Update section visibility
    Object.keys(this.sections).forEach(k => {
      if (k === sectionKey) {
        this.sections[k].classList.add("active");
      } else {
        this.sections[k].classList.remove("active");
      }
    });

    // Update nav links
    document.querySelectorAll(".nav-item").forEach(item => {
      if (item.getAttribute("data-nav") === sectionKey) {
        item.classList.add("active");
      } else {
        item.classList.remove("active");
      }
    });

    // If navigating to detect, reset to detection stage
    if (sectionKey === "detect") {
      this.showDetectionStage("detection");
      this.resizeCanvas();
    }

    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  showDetectionStage(stageKey) {
    Object.keys(this.stages).forEach(k => {
      if (k === stageKey) {
        this.stages[k].classList.add("active");
      } else {
        this.stages[k].classList.remove("active");
      }
    });
  }

  /* ========================================================
     CANVAS SIZING & AMBIENT VISUALIZER
     ======================================================== */
  initCanvasScaling() {
    const handleResize = () => {
      this.resizeCanvas();
    };
    window.addEventListener("resize", handleResize);
    this.resizeCanvas();
  }

  resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;

    // Resize Live Studio Canvas
    if (this.liveCanvas && this.liveCtx) {
      const rect = this.liveCanvas.getBoundingClientRect();
      this.liveCanvas.width = rect.width * dpr;
      this.liveCanvas.height = rect.height * dpr;
      this.liveCtx.scale(dpr, dpr);
      if (!this.isRecording) {
        this.drawIdleLiveWave();
      }
    }

    // Resize Ambient Home Canvas
    if (this.ambientCanvas && this.ambientCtx) {
      const rect = this.ambientCanvas.getBoundingClientRect();
      this.ambientCanvas.width = rect.width * dpr;
      this.ambientCanvas.height = rect.height * dpr;
      this.ambientCtx.scale(dpr, dpr);
    }
  }

  startAmbientWave() {
    let step = 0;
    const render = () => {
      if (!this.ambientCanvas || !this.ambientCtx) return;
      const rect = this.ambientCanvas.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;

      this.ambientCtx.clearRect(0, 0, w, h);

      // Subtle cyan/blue dual harmonic wave
      this.ambientCtx.beginPath();
      this.ambientCtx.strokeStyle = "rgba(34, 211, 238, 0.4)";
      this.ambientCtx.lineWidth = 1.8;

      for (let x = 0; x < w; x++) {
        const y = h / 2 + Math.sin(x * 0.02 + step) * 12 + Math.sin(x * 0.05 - step) * 6;
        if (x === 0) this.ambientCtx.moveTo(x, y);
        else this.ambientCtx.lineTo(x, y);
      }
      this.ambientCtx.stroke();

      this.ambientCtx.beginPath();
      this.ambientCtx.strokeStyle = "rgba(59, 130, 246, 0.25)";
      this.ambientCtx.lineWidth = 1.4;

      for (let x = 0; x < w; x++) {
        const y = h / 2 + Math.cos(x * 0.015 - step * 0.7) * 16;
        if (x === 0) this.ambientCtx.moveTo(x, y);
        else this.ambientCtx.lineTo(x, y);
      }
      this.ambientCtx.stroke();

      step += 0.035;
      this.ambientAnimId = requestAnimationFrame(render);
    };

    render();
  }

  drawIdleLiveWave() {
    if (!this.liveCanvas || !this.liveCtx) return;
    const rect = this.liveCanvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    this.liveCtx.clearRect(0, 0, w, h);
    this.liveCtx.fillStyle = "#080B16";
    this.liveCtx.fillRect(0, 0, w, h);

    // Subtle resting guideline
    this.liveCtx.beginPath();
    this.liveCtx.strokeStyle = "rgba(59, 130, 246, 0.25)";
    this.liveCtx.lineWidth = 1.5;

    for (let x = 0; x < w; x++) {
      const y = h / 2 + Math.sin(x * 0.03) * 4;
      if (x === 0) this.liveCtx.moveTo(x, y);
      else this.liveCtx.lineTo(x, y);
    }
    this.liveCtx.stroke();
  }

  /* ========================================================
     STUDIO RECORDING ENGINE
     ======================================================== */
  bindStudioEvents() {
    if (this.btnRecord) {
      this.btnRecord.addEventListener("click", () => {
        if (!this.isRecording) {
          this.startRecording();
        }
      });
    }

    if (this.btnStop) {
      this.btnStop.addEventListener("click", () => this.stopRecording());
    }

    if (this.btnPlayback) {
      this.btnPlayback.addEventListener("click", () => this.togglePlayback());
    }

    if (this.btnRerecord) {
      this.btnRerecord.addEventListener("click", () => this.resetStudio());
    }

    if (this.btnAnalyze) {
      this.btnAnalyze.addEventListener("click", () => this.analyzeRecordedAccent());
    }

    if (this.btnTryAgain) {
      this.btnTryAgain.addEventListener("click", () => {
        this.resetStudio();
        this.showDetectionStage("detection");
      });
    }
  }

  async startRecording() {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Microphone capture is not supported in this browser environment.");
        return;
      }

      const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
      if (!this.audioCtx || this.audioCtx.state === "closed") {
        this.audioCtx = new AudioCtxClass();
      }
      if (this.audioCtx.state === "suspended") {
        await this.audioCtx.resume();
      }

      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
      });

      if (window.AccentPipeline) {
        window.AccentPipeline.init().catch(() => {});
      }

      this.analyser = this.audioCtx.createAnalyser();
      this.analyser.fftSize = 2048;
      this.analyser.smoothingTimeConstant = 0.85;

      const source = this.audioCtx.createMediaStreamSource(this.mediaStream);
      source.connect(this.analyser);

      this.recordedChunks = [];
      const mime = this.getSupportedMime();
      this.mediaRecorder = new MediaRecorder(this.mediaStream, mime ? { mimeType: mime } : undefined);

      this.mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          this.recordedChunks.push(e.data);
        }
      };

      this.mediaRecorder.onstop = () => {
        this.onRecordingStopped();
      };

      this.mediaRecorder.start(100);
      this.isRecording = true;

      this.updateStudioUI("recording");

      this.secondsRecorded = 0;
      this.updateTimerDisplay();
      this.timerInterval = setInterval(() => {
        this.secondsRecorded++;
        this.updateTimerDisplay();
        if (this.secondsRecorded >= 45) {
          this.stopRecording();
        }
      }, 1000);

      this.renderLiveWaveformLoop();

    } catch (err) {
      console.error("Recording error:", err);
      alert("Microphone access was denied or no device was detected.");
      this.resetStudio();
    }
  }

  getSupportedMime() {
    const types = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg", "audio/mp4", "audio/wav"];
    for (const t of types) {
      if (MediaRecorder.isTypeSupported(t)) return t;
    }
    return "";
  }

  stopRecording() {
    if (!this.isRecording) return;
    this.isRecording = false;

    if (this.timerInterval) {
      clearInterval(this.timerInterval);
      this.timerInterval = null;
    }

    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }

    if (this.mediaRecorder && this.mediaRecorder.state !== "inactive") {
      this.mediaRecorder.stop();
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(t => t.stop());
      this.mediaStream = null;
    }

    this.updateStudioUI("stopped");
  }

  onRecordingStopped() {
    const mime = this.mediaRecorder ? this.mediaRecorder.mimeType : "audio/webm";
    this.recordedBlob = new Blob(this.recordedChunks, { type: mime });

    this.drawRecordedEnvelope();
  }

  async convertBlobToWav(blob) {
    try {
      if (!this.audioCtx) {
        const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
        this.audioCtx = new AudioCtxClass();
      }
      const arrayBuffer = await blob.arrayBuffer();
      const audioBuffer = await this.audioCtx.decodeAudioData(arrayBuffer);

      const numChannels = 1;
      const sampleRate = audioBuffer.sampleRate;
      const channelData = audioBuffer.getChannelData(0);
      const dataLength = channelData.length * 2;
      const buffer = new ArrayBuffer(44 + dataLength);
      const view = new DataView(buffer);

      const writeString = (offset, str) => {
        for (let i = 0; i < str.length; i++) {
          view.setUint8(offset + i, str.charCodeAt(i));
        }
      };

      writeString(0, 'RIFF');
      view.setUint32(4, 36 + dataLength, true);
      writeString(8, 'WAVE');
      writeString(12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true); // PCM
      view.setUint16(22, numChannels, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      writeString(36, 'data');
      view.setUint32(40, dataLength, true);

      let offset = 44;
      for (let i = 0; i < channelData.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, channelData[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      }

      return new Blob([view], { type: 'audio/wav' });
    } catch (e) {
      console.warn("WAV conversion notice:", e);
      return blob;
    }
  }

  togglePlayback() {
    if (!this.recordedBlob || !this.globalAudio) return;

    if (this.globalAudio.paused) {
      const url = URL.createObjectURL(this.recordedBlob);
      this.globalAudio.src = url;
      this.globalAudio.play().then(() => {
        if (this.btnPlayback) {
          this.btnPlayback.textContent = "⏸ Pause";
        }
      });
      this.globalAudio.onended = () => {
        if (this.btnPlayback) {
          this.btnPlayback.textContent = "▶ Listen";
        }
      };
    } else {
      this.globalAudio.pause();
      if (this.btnPlayback) {
        this.btnPlayback.textContent = "▶ Listen";
      }
    }
  }

  resetStudio() {
    if (this.isRecording) this.stopRecording();
    if (this.globalAudio) {
      this.globalAudio.pause();
      this.globalAudio.src = "";
    }

    this.recordedChunks = [];
    this.recordedBlob = null;
    this.secondsRecorded = 0;
    this.updateTimerDisplay();
    this.updateStudioUI("idle");
    this.drawIdleLiveWave();
  }

  updateStudioUI(state) {
    if (!this.statusBadge) return;

    if (state === "recording") {
      this.statusBadge.textContent = "Recording";
      this.statusBadge.className = "status-indicator recording";
      if (this.btnRecord) this.btnRecord.disabled = true;
      if (this.recordLabel) this.recordLabel.textContent = "Recording...";
      if (this.btnStop) this.btnStop.disabled = false;
      if (this.btnPlayback) this.btnPlayback.disabled = true;
      if (this.btnRerecord) this.btnRerecord.disabled = true;
      if (this.btnAnalyze) this.btnAnalyze.disabled = true;
    } else if (state === "stopped") {
      this.statusBadge.textContent = "Recorded";
      this.statusBadge.className = "status-indicator ready";
      if (this.btnRecord) this.btnRecord.disabled = true;
      if (this.recordLabel) this.recordLabel.textContent = "Start Recording";
      if (this.btnStop) this.btnStop.disabled = true;
      if (this.btnPlayback) {
        this.btnPlayback.disabled = false;
        this.btnPlayback.textContent = "▶ Listen";
      }
      if (this.btnRerecord) this.btnRerecord.disabled = false;
      if (this.btnAnalyze) this.btnAnalyze.disabled = false;
    } else {
      this.statusBadge.textContent = "Ready";
      this.statusBadge.className = "status-indicator idle";
      if (this.btnRecord) this.btnRecord.disabled = false;
      if (this.recordLabel) this.recordLabel.textContent = "Start Recording";
      if (this.btnStop) this.btnStop.disabled = true;
      if (this.btnPlayback) {
        this.btnPlayback.disabled = true;
        this.btnPlayback.textContent = "▶ Listen";
      }
      if (this.btnRerecord) this.btnRerecord.disabled = true;
      if (this.btnAnalyze) this.btnAnalyze.disabled = true;
    }
  }

  updateTimerDisplay() {
    if (!this.timerDisplay) return;
    const m = String(Math.floor(this.secondsRecorded / 60)).padStart(2, "0");
    const s = String(this.secondsRecorded % 60).padStart(2, "0");
    this.timerDisplay.textContent = `${m}:${s}`;
  }

  renderLiveWaveformLoop() {
    if (!this.isRecording || !this.analyser || !this.liveCanvas || !this.liveCtx) return;

    const bufferLength = this.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    this.analyser.getByteTimeDomainData(dataArray);

    const rect = this.liveCanvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    this.liveCtx.clearRect(0, 0, w, h);
    this.liveCtx.fillStyle = "#080B16";
    this.liveCtx.fillRect(0, 0, w, h);

    // Center guideline
    this.liveCtx.beginPath();
    this.liveCtx.strokeStyle = "rgba(34, 211, 238, 0.15)";
    this.liveCtx.moveTo(0, h / 2);
    this.liveCtx.lineTo(w, h / 2);
    this.liveCtx.stroke();

    // Clean glowing wave line
    this.liveCtx.beginPath();
    const sliceWidth = w / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * h) / 2;
      if (i === 0) this.liveCtx.moveTo(x, y);
      else this.liveCtx.lineTo(x, y);
      x += sliceWidth;
    }

    this.liveCtx.strokeStyle = "#22D3EE";
    this.liveCtx.lineWidth = 2.2;
    this.liveCtx.stroke();

    this.animFrameId = requestAnimationFrame(() => this.renderLiveWaveformLoop());
  }

  drawRecordedEnvelope() {
    if (!this.liveCanvas || !this.liveCtx) return;
    const rect = this.liveCanvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    this.liveCtx.clearRect(0, 0, w, h);
    this.liveCtx.fillStyle = "#080B16";
    this.liveCtx.fillRect(0, 0, w, h);

    const bars = 40;
    const barWidth = w / bars - 3;
    this.liveCtx.fillStyle = "#3B82F6";

    for (let i = 0; i < bars; i++) {
      const factor = Math.sin((i / bars) * Math.PI);
      const barH = Math.max(6, factor * (h * 0.65) * (0.4 + 0.6 * Math.sin(i * 1.5)));
      const x = i * (barWidth + 3);
      const y = (h - barH) / 2;
      this.liveCtx.fillRect(x, y, barWidth, barH);
    }
  }

  /* ========================================================
     ANALYSIS & RESULTS VIEW
     ======================================================== */
  async analyzeRecordedAccent() {
    if (!this.recordedBlob) {
      alert("No audio recording found. Please record your speech first.");
      return;
    }

    this.showDetectionStage("loading");

    try {
      if (!window.AccentPipeline) {
        throw new Error("Analysis engine is unavailable in this browser.");
      }

      const data = await window.AccentPipeline.predictFromBlob(this.recordedBlob);

      if (!data.success) {
        alert(data.error || "Unable to analyze audio. Please record at least 10–20 seconds of speech.");
        this.showDetectionStage("detection");
        return;
      }

      this.renderResults(data);
      this.showDetectionStage("results");

    } catch (err) {
      console.error("Analysis error:", err);
      alert("Unable to analyze the recording in this browser. Please try again.");
      this.showDetectionStage("detection");
    }
  }

  renderResults(data) {
    document.getElementById("res-flag").textContent = data.detected_flag;
    document.getElementById("res-name").textContent = data.detected_accent;
    document.getElementById("res-variant").textContent = `${data.detected_variant} • ${data.detected_region}`;
    document.getElementById("res-score").textContent = `${data.confidence_score}%`;
    document.getElementById("res-meter").style.width = `${data.confidence_score}%`;
    document.getElementById("res-desc").textContent = data.educational_description;

    // Top 3 Candidates
    const candidatesEl = document.getElementById("res-candidates");
    if (candidatesEl) {
      candidatesEl.innerHTML = data.top_candidates.map(c => `
        <div class="candidate-item">
          <div class="candidate-name">${c.flag} ${c.name}</div>
          <div class="candidate-bar-wrap">
            <div class="candidate-bar-value" style="width: ${c.confidence_percent}%; background-color: ${c.color};"></div>
          </div>
          <div class="candidate-pct">${c.confidence_percent}%</div>
        </div>
      `).join("");
    }

    // Audio Metadata
    const q = data.audio_quality;
    document.getElementById("res-duration").textContent = `${q.duration_seconds}s`;
    document.getElementById("res-quality").textContent = q.rating.split(" ")[0];
    document.getElementById("res-time").textContent = data.timestamp.split(" ")[1];
  }

  /* ========================================================
     EXPLORE ACCENTS
     ======================================================== */
  async loadExploreAccents() {
    const container = document.getElementById("explore-accents-container");
    if (!container) return;

    try {
      if (!window.AccentPipeline) return;
      const data = await window.AccentPipeline.loadAccents();
      if (!data.success) return;

      const accents = data.accents;
      container.innerHTML = Object.keys(accents).map(k => {
        const acc = accents[k];
        const wordsList = acc.example_words ? acc.example_words.join(", ") : "water, speech, vowel";
        const sampleId = acc.sample_id || "english1";

        return `
          <div class="accent-card">
            <div class="card-top">
              <span class="card-flag">${acc.flag}</span>
              <div>
                <h3 class="card-title">${acc.name}</h3>
                <div class="card-variant">${acc.variant}</div>
              </div>
            </div>
            <p class="card-features">${acc.key_features}</p>
            <div class="card-words-title">Example Words</div>
            <div class="card-words">${wordsList}</div>
            <button class="btn-sample" data-sample="${sampleId}">
              🔊 Listen to Speaker
            </button>
          </div>
        `;
      }).join("");

      // Bind audio sample triggers
      container.querySelectorAll(".btn-sample").forEach(btn => {
        btn.addEventListener("click", () => {
          const sampleId = btn.getAttribute("data-sample");
          this.playReferenceSample(sampleId, btn);
        });
      });

    } catch (e) {
      console.warn("Failed to load accents list:", e);
    }
  }

  playReferenceSample(sampleId, btnElement) {
    if (!this.globalAudio) return;

    if (this.currentPlayingSampleBtn === btnElement && !this.globalAudio.paused) {
      this.globalAudio.pause();
      btnElement.classList.remove("playing");
      btnElement.textContent = "🔊 Listen to Speaker";
      this.currentPlayingSampleBtn = null;
      return;
    }

    if (this.currentPlayingSampleBtn) {
      this.currentPlayingSampleBtn.classList.remove("playing");
      this.currentPlayingSampleBtn.textContent = "🔊 Listen to Speaker";
    }

    this.currentPlayingSampleBtn = btnElement;
    btnElement.classList.add("playing");
    btnElement.textContent = "⏹ Stop Playback";

    this.globalAudio.src = `dataset/raw_audio/${sampleId}.mp3`;
    this.globalAudio.play().catch(e => console.warn("Audio playback notice:", e));

    this.globalAudio.onended = () => {
      btnElement.classList.remove("playing");
      btnElement.textContent = "🔊 Listen to Speaker";
      this.currentPlayingSampleBtn = null;
    };
  }
}

// Startup
document.addEventListener("DOMContentLoaded", () => {
  window.accentAura = new AccentAuraApp();
  window.accentAura.init();
});
