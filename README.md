# English Accent Identifier 🎙️
**Acoustic Machine Learning & Speech Signal Processing System**  
*Department of English Language & Literature • University Exhibition 2026*

A complete, production-grade English Accent Identifier that listens to a visitor's spoken English and predicts which regional or national English accent variety they most closely match using traditional machine learning and audio signal processing.

> **Zero Generative AI & Scientific Integrity:**  
> This system contains **no generative AI, no ChatGPT, no paid cloud APIs, and no speech-to-text transcriptions**. All predictions are derived from acoustic signal processing (MFCCs, spectral roll-off, formant bandwidths, and spectral contrast) evaluated by a trained machine learning classifier on real human speakers.

---

## 🏛️ Exhibition Showcase Features

1. **Acoustic Voice Recording Studio:**
   - Captures microphone input via HTML5 `MediaRecorder` & Web Audio API.
   - Real-time time-domain oscilloscope canvas with glowing electric cyan waveform.
   - Recording timer, digital scrubber, playback, and re-record controls.
   - Elicitation reading passage: Standardized phonetic paragraph (*"Please call Stella..."*).

2. **Acoustic Accent Identification Pipeline:**
   - **Quality Validation:** Inspects duration, RMS energy, and estimated Signal-to-Noise Ratio (SNR in dB).
   - **Pre-processing:** Trims leading/trailing non-speech margins (top_db=25), normalizes peak amplitude, and segments into 5-second acoustic speech windows.
   - **Feature Extraction:** Extracts 188 spectral and cepstral features per window.
   - **Probabilistic Classification:** Ensemble averages probability distributions across speech windows.
   - **Confidence & Uncertainty:** Discloses Top 3 candidate categories with confidence bars and marks ambiguous recordings as uncertain if below statistical thresholds.

3. **Explore Accents (Educational Section):**
   - Sociolinguistic background, regional origins, and phonological characteristics for 7 English varieties:
     1. British English (SSBE / RP)
     2. American English (GenAm)
     3. Australian English
     4. Pakistani English (PakE)
     5. Indian English
     6. Irish English (Hiberno-English)
     7. Scottish English (SSE)
   - Genuine audio sample clips from the Speech Accent Archive.

4. **Transatlantic UK vs. US Comparison:**
   - Side-by-side contrast cards covering alveolar flapping [ɾ] vs aspiration (*water*), yod-dropping, TRAP-BATH split, and vocabulary shifts (*lift/elevator*, *flat/apartment*, *boot/trunk*, *bonnet/hood*).

5. **Exhibition Kiosk Mode:**
   - Welcome banner for booth visitors.
   - One-click **"Start Speaking Now"** and **"Reset Session for Next Visitor"**.
   - Automatic deletion of temporary voice recordings from disk on session reset to protect visitor privacy.

6. **Verified Model Transparency:**
   - Displays real cross-validation metrics on unseen speakers.
   - Interactive 7x7 Confusion Matrix.

---

## 💻 Technology Stack

| Layer | Component | Description |
|---|---|---|
| **Frontend** | HTML5, CSS3, JavaScript, Bootstrap 5 | Dark navy theme, electric blue/cyan glow, responsive cards |
| **Backend** | Python 3, Flask | REST API endpoints for recording ingestion, prediction, and audio playback |
| **Signal Processing** | Librosa, SoundFile, SciPy | 22.05 kHz audio resampling, silence trimming, RMS energy, SNR estimation |
| **Feature Extraction** | NumPy, Librosa | 188 acoustic dimensions (MFCCs, Deltas, Spectral Centroid, Contrast, Mel bands) |
| **Machine Learning** | Scikit-learn, Joblib | Random Forest & Support Vector Machine (SVM) trained with probability calibration |

---

## 🔬 Dataset & Machine Learning Methodology

### 1. Dataset Provenance
- Source: **George Mason University Speech Accent Archive**
- License: **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC-BY-NC-SA 4.0)**
- Speakers: 55 independent human speakers reading the exact same standardized phonetic passage (*"Please call Stella..."*):
  - 8 American English speakers
  - 8 British English speakers
  - 8 Australian English speakers
  - 8 Pakistani English speakers
  - 8 Indian English speakers
  - 8 Irish English speakers
  - 7 Scottish English speakers

### 2. Zero Speaker Leakage Cross-Validation
- All training and validation splits were conducted using **5-Fold Stratified Group K-Fold Cross-Validation grouped by speaker ID**.
- Speakers in the test fold never appeared in the training fold, ensuring true out-of-sample generalization.

### 3. Feature Extraction (188 Dimensions)
- **MFCCs (1-20):** Mean, Standard Deviation, Skewness (60 features)
- **Delta MFCCs:** Mean, Standard Deviation (40 features)
- **Delta-Delta MFCCs:** Mean, Standard Deviation (40 features)
- **Spectral Centroid:** Mean, Standard Deviation (2 features)
- **Spectral Bandwidth:** Mean, Standard Deviation (2 features)
- **Spectral Rolloff (85%):** Mean, Standard Deviation (2 features)
- **Spectral Contrast (7 sub-bands):** Mean, Standard Deviation (14 features)
- **Zero Crossing Rate (ZCR):** Mean, Standard Deviation (2 features)
- **RMS Energy:** Mean, Standard Deviation (2 features)
- **Chroma STFT (12 bins):** Mean, Standard Deviation (24 features)
- **Log Mel Spectrogram (16 bins):** Mean, Standard Deviation (32 features)

---

## 📂 Project Structure

```
english-accent-identifier/
├── frontend/
│   ├── index.html              # Modern exhibition dashboard & recording studio
│   ├── css/
│   │   └── style.css           # Dark navy, cyan neon glow, oscilloscope visualizer
│   └── js/
│       └── app.js              # Web Audio API recording & API integration
│
├── backend/
│   ├── app.py                  # Flask REST API server
│   ├── audio_processor.py      # Audio validation, silence trimming, and SNR estimation
│   ├── feature_extractor.py    # 188-dimensional acoustic feature extraction
│   ├── accent_classifier.py    # Probabilistic inference & educational description engine
│   ├── model_training.py       # Dataset builder, 5-fold GroupKFold training, and evaluation
│   └── config.py               # Audio parameters, paths, and accent metadata
│
├── models/
│   ├── accent_classifier.joblib# Trained ML classifier model
│   ├── scaler.joblib           # StandardScaler model
│   ├── label_encoder.joblib    # LabelEncoder model
│   └── metrics.json            # Verified accuracy, precision, recall, and confusion matrix
│
├── dataset/
│   ├── raw_audio/              # 55 cached Speech Accent Archive MP3 recordings
│   ├── speakers_catalog.json   # Speaker metadata catalog
│   └── extracted_features.csv  # 430 segmented acoustic feature vectors
│
├── temp_recordings/            # Temporary user voice recordings (auto-cleaned)
├── requirements.txt            # Python dependencies
└── README.md                   # Full documentation
```

---

## 🚀 Running the Application

### 1. Start the Flask Backend Server
```bash
cd /home/irfanali/Desktop/english-accent-identifier
python3 backend/app.py
```
The server will start on:
**[http://localhost:5000](http://localhost:5000)**

### 2. Open in Your Web Browser
Open your browser to:
```
http://localhost:5000
```

### 3. Retraining the Model (Optional)
To retrain or re-evaluate the ML classifier:
```bash
python3 backend/model_training.py
```

---

## 🖥️ Desktop Kiosk Packaging

To run this application as a standalone desktop kiosk application without a terminal:
```bash
pip install pyinstaller
pyinstaller --name "EnglishAccentIdentifier" --add-data "frontend:frontend" --add-data "models:models" backend/app.py
```
Or wrap with **Electron** pointing to `http://localhost:5000`.
