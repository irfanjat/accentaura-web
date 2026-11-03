"""
English Accent Identifier - Acoustic Feature Extractor
Comprehensive extraction of MFCCs, Spectral features, and Mel-Spectrograms
"""
import numpy as np
import librosa
from scipy.stats import skew, kurtosis
try:
    from backend.config import SAMPLE_RATE, N_MFCC, N_MELS, HOP_LENGTH, N_FFT
except ImportError:
    from config import SAMPLE_RATE, N_MFCC, N_MELS, HOP_LENGTH, N_FFT

class FeatureExtractor:
    def __init__(self, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft

    def extract_features(self, y):
        """
        Extracts a comprehensive 1D acoustic feature vector from an audio waveform y.
        Returns: numpy array of shape (D,) and list of feature names.
        """
        features = []
        feature_names = []

        # 1. Mel-Frequency Cepstral Coefficients (MFCCs)
        mfcc = librosa.feature.mfcc(
            y=y, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length
        )
        # Static MFCC stats
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_skew = skew(mfcc, axis=1)

        features.extend(mfcc_mean)
        feature_names.extend([f"mfcc_{i}_mean" for i in range(self.n_mfcc)])
        features.extend(mfcc_std)
        feature_names.extend([f"mfcc_{i}_std" for i in range(self.n_mfcc)])
        features.extend(mfcc_skew)
        feature_names.extend([f"mfcc_{i}_skew" for i in range(self.n_mfcc)])

        # 2. Delta MFCCs (Velocity / Temporal dynamics)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta_mean = np.mean(delta_mfcc, axis=1)
        delta_std = np.std(delta_mfcc, axis=1)
        features.extend(delta_mean)
        feature_names.extend([f"delta_mfcc_{i}_mean" for i in range(self.n_mfcc)])
        features.extend(delta_std)
        feature_names.extend([f"delta_mfcc_{i}_std" for i in range(self.n_mfcc)])

        # 3. Delta-Delta MFCCs (Acceleration)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        delta2_mean = np.mean(delta2_mfcc, axis=1)
        delta2_std = np.std(delta2_mfcc, axis=1)
        features.extend(delta2_mean)
        feature_names.extend([f"delta2_mfcc_{i}_mean" for i in range(self.n_mfcc)])
        features.extend(delta2_std)
        feature_names.extend([f"delta2_mfcc_{i}_std" for i in range(self.n_mfcc)])

        # 4. Spectral Centroid (Timbre brightness)
        cent = librosa.feature.spectral_centroid(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        features.append(float(np.mean(cent)))
        features.append(float(np.std(cent)))
        feature_names.extend(["spectral_centroid_mean", "spectral_centroid_std"])

        # 5. Spectral Bandwidth (Spectral spread)
        bw = librosa.feature.spectral_bandwidth(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        features.append(float(np.mean(bw)))
        features.append(float(np.std(bw)))
        feature_names.extend(["spectral_bandwidth_mean", "spectral_bandwidth_std"])

        # 6. Spectral Rolloff (High-frequency energy skew)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, roll_percent=0.85)
        features.append(float(np.mean(rolloff)))
        features.append(float(np.std(rolloff)))
        feature_names.extend(["spectral_rolloff_mean", "spectral_rolloff_std"])

        # 7. Spectral Contrast (Valley-to-peak ratio across 7 frequency sub-bands)
        contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        features.extend(np.mean(contrast, axis=1))
        feature_names.extend([f"spectral_contrast_{i}_mean" for i in range(contrast.shape[0])])
        features.extend(np.std(contrast, axis=1))
        feature_names.extend([f"spectral_contrast_{i}_std" for i in range(contrast.shape[0])])

        # 8. Zero-Crossing Rate (Fricative / Consonantal noisiness proxy)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)
        features.append(float(np.mean(zcr)))
        features.append(float(np.std(zcr)))
        feature_names.extend(["zero_crossing_rate_mean", "zero_crossing_rate_std"])

        # 9. RMS Energy (Vocal dynamics)
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)
        features.append(float(np.mean(rms)))
        features.append(float(np.std(rms)))
        feature_names.extend(["rms_energy_mean", "rms_energy_std"])

        # 10. Chroma STFT (Harmonic pitch distribution)
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        features.extend(np.mean(chroma, axis=1))
        feature_names.extend([f"chroma_{i}_mean" for i in range(12)])
        features.extend(np.std(chroma, axis=1))
        feature_names.extend([f"chroma_{i}_std" for i in range(12)])

        # 11. Mel Spectrogram Summary (first 16 frequency bands)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=16, n_fft=self.n_fft, hop_length=self.hop_length
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        features.extend(np.mean(log_mel, axis=1))
        feature_names.extend([f"log_mel_{i}_mean" for i in range(16)])
        features.extend(np.std(log_mel, axis=1))
        feature_names.extend([f"log_mel_{i}_std" for i in range(16)])

        feature_vector = np.nan_to_num(np.array(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return feature_vector, feature_names
