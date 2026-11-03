"""
English Accent Identifier - Audio Preprocessor
Acoustic signal validation, silence trimming, and normalization
"""
import os
import shutil
import subprocess
import tempfile
import numpy as np
import librosa
import soundfile as sf
from scipy.stats import skew, kurtosis
try:
    from backend.config import (
        SAMPLE_RATE, MIN_AUDIO_DURATION_SEC, MIN_RMS_ENERGY,
        RECOMMENDED_MIN_SEC, RECOMMENDED_MAX_SEC
    )
except ImportError:
    from config import (
        SAMPLE_RATE, MIN_AUDIO_DURATION_SEC, MIN_RMS_ENERGY,
        RECOMMENDED_MIN_SEC, RECOMMENDED_MAX_SEC
    )

class AudioProcessor:
    def __init__(self, target_sr=SAMPLE_RATE):
        self.target_sr = target_sr

    def get_ffmpeg_path(self):
        """Locates ffmpeg executable from imageio-ffmpeg or system PATH."""
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return shutil.which("ffmpeg")

    def load_audio(self, audio_source):
        """
        Loads an audio file path into a 1D float32 numpy array.
        Resamples automatically to target_sr and forces mono.
        Automatically converts WebM, Opus, Ogg, and AAC via ffmpeg if librosa/soundfile cannot natively open it.
        """
        try:
            y, sr = librosa.load(audio_source, sr=self.target_sr, mono=True)
            return y, sr
        except Exception as load_err:
            ffmpeg_exe = self.get_ffmpeg_path()
            if not ffmpeg_exe:
                raise RuntimeError(f"Audio format error: {load_err}. FFmpeg binary not found.")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                tmp_wav_path = tmp_wav.name

            try:
                cmd = [
                    ffmpeg_exe, "-y",
                    "-i", str(audio_source),
                    "-ar", str(self.target_sr),
                    "-ac", "1",
                    "-c:a", "pcm_s16le",
                    tmp_wav_path
                ]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if res.returncode != 0:
                    err_msg = res.stderr.decode("utf-8", errors="ignore")
                    raise RuntimeError(f"Audio conversion error: {err_msg}")

                y, sr = sf.read(tmp_wav_path, dtype="float32")
                return y, sr
            finally:
                if os.path.exists(tmp_wav_path):
                    try:
                        os.remove(tmp_wav_path)
                    except OSError:
                        pass

    def validate_and_preprocess(self, audio_source):
        """
        Loads audio, checks duration and energy quality, trims silence,
        normalizes peak amplitude, and returns preprocessed waveform + metrics.
        """
        y, sr = self.load_audio(audio_source)
        raw_duration = len(y) / sr

        # 1. Check basic duration
        if raw_duration < MIN_AUDIO_DURATION_SEC:
            return {
                "is_valid": False,
                "error": f"Audio is too short ({raw_duration:.1f}s). Please record at least {MIN_AUDIO_DURATION_SEC}s (recommended: 10–25s) for a reliable acoustic reading.",
                "duration": raw_duration
            }

        # 2. Check energy (silence / empty mic detection)
        rms = float(np.sqrt(np.mean(y**2)))
        if rms < MIN_RMS_ENERGY:
            return {
                "is_valid": False,
                "error": "The recording appears to be silent or extremely faint. Please check your microphone input volume and speak clearly.",
                "duration": raw_duration,
                "rms": rms
            }

        # 3. Trim non-speech margins (top_db=25 threshold)
        y_trimmed, trim_indices = librosa.effects.trim(y, top_db=25)
        trimmed_duration = len(y_trimmed) / sr

        if trimmed_duration < MIN_AUDIO_DURATION_SEC:
            # Fall back to raw if over-trimmed
            y_proc = y
            final_duration = raw_duration
        else:
            y_proc = y_trimmed
            final_duration = trimmed_duration

        # 4. Peak amplitude normalization
        max_val = np.max(np.abs(y_proc))
        if max_val > 0:
            y_proc = y_proc / max_val * 0.95

        # 5. Compute acoustic quality metrics
        rms_final = float(np.sqrt(np.mean(y_proc**2)))
        noise_floor = np.percentile(np.abs(y_proc), 10) + 1e-8
        signal_peak = np.percentile(np.abs(y_proc), 95) + 1e-8
        estimated_snr_db = float(20 * np.log10(signal_peak / noise_floor))

        if estimated_snr_db > 22 and final_duration >= RECOMMENDED_MIN_SEC:
            quality_rating = "Excellent (High SNR, clear articulation)"
        elif estimated_snr_db > 14:
            quality_rating = "Good (Sufficient acoustic signal)"
        else:
            quality_rating = "Acceptable (Elevated background noise detected)"

        return {
            "is_valid": True,
            "y": y_proc,
            "sr": sr,
            "duration": round(final_duration, 2),
            "raw_duration": round(raw_duration, 2),
            "rms_energy": round(rms_final, 4),
            "snr_db": round(estimated_snr_db, 1),
            "quality_rating": quality_rating,
            "trim_start": round(trim_indices[0] / sr, 2),
            "trim_end": round(trim_indices[1] / sr, 2)
        }

    def split_into_windows(self, y, sr, window_sec=5.0, hop_sec=2.5):
        """
        Splits a single speaker's recording into overlapping acoustic segments.
        Used for dataset feature extraction to avoid speaker leakage.
        """
        window_samples = int(window_sec * sr)
        hop_samples = int(hop_sec * sr)
        windows = []

        if len(y) <= window_samples:
            return [y]

        for start in range(0, len(y) - window_samples + 1, hop_samples):
            chunk = y[start : start + window_samples]
            # Ensure chunk has adequate speech energy
            chunk_rms = np.sqrt(np.mean(chunk**2))
            if chunk_rms >= MIN_RMS_ENERGY:
                windows.append(chunk)

        # Catch tail if not empty
        if len(windows) == 0:
            windows.append(y[:window_samples])

        return windows

    def save_temp_wav(self, y, sr, out_path):
        """Saves a numpy waveform to a 16-bit PCM WAV file."""
        sf.write(out_path, y, sr, subtype='PCM_16')
        return out_path
