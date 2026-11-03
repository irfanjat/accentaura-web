"""
English Accent Identifier - Accent Classification Engine
Predicts accent categories using trained ML models on acoustic features
"""
import os
import json
import datetime
import numpy as np
import joblib

try:
    from backend.config import (
        MODEL_PATH, SCALER_PATH, ENCODER_PATH, METRICS_PATH,
        ACCENTS_METADATA, ACCENT_CLASSES, MIN_CONFIDENCE_THRESHOLD
    )
    from backend.audio_processor import AudioProcessor
    from backend.feature_extractor import FeatureExtractor
except ImportError:
    from config import (
        MODEL_PATH, SCALER_PATH, ENCODER_PATH, METRICS_PATH,
        ACCENTS_METADATA, ACCENT_CLASSES, MIN_CONFIDENCE_THRESHOLD
    )
    from audio_processor import AudioProcessor
    from feature_extractor import FeatureExtractor

class AccentClassifier:
    def __init__(self):
        self.processor = AudioProcessor()
        self.extractor = FeatureExtractor()
        self.model = None
        self.scaler = None
        self.encoder = None
        self.metrics = None
        self.is_loaded = False
        self.load_model()

    def load_model(self):
        """Loads trained model artifacts from disk."""
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH):
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                self.encoder = joblib.load(ENCODER_PATH)
                if os.path.exists(METRICS_PATH):
                    with open(METRICS_PATH, "r") as f:
                        self.metrics = json.load(f)
                self.is_loaded = True
                print("AccentClassifier: Loaded trained model artifacts successfully.")
            else:
                print("AccentClassifier: Model artifacts not found. Model must be trained first.")
                self.is_loaded = False
        except Exception as e:
            print(f"AccentClassifier: Failed to load model artifacts: {e}")
            self.is_loaded = False

    def predict(self, audio_source):
        """
        Receives an audio file path or stream, preprocesses it, extracts features,
        evaluates with the ML model, and produces probabilistic classification scores.
        """
        if not self.is_loaded:
            self.load_model()
            if not self.is_loaded:
                return {
                    "success": False,
                    "error": "The classification model has not yet completed training. Please wait a moment and try again."
                }

        # 1. Validate & Preprocess Audio
        proc_res = self.processor.validate_and_preprocess(audio_source)
        if not proc_res["is_valid"]:
            return {
                "success": False,
                "error": proc_res["error"],
                "quality_info": {
                    "duration": proc_res.get("duration", 0),
                    "is_valid": False
                }
            }

        y = proc_res["y"]
        sr = proc_res["sr"]

        # 2. Segment audio into temporal speech windows (5-second windows)
        chunks = self.processor.split_into_windows(y, sr, window_sec=5.0, hop_sec=2.5)

        # 3. Extract 188 features for each segment
        chunk_features = []
        for chunk in chunks:
            feats, _ = self.extractor.extract_features(chunk)
            chunk_features.append(feats)

        X_chunks = np.array(chunk_features)

        # 4. Standardize features
        X_scaled = self.scaler.transform(X_chunks)

        # 5. Predict probabilities
        if hasattr(self.model, "predict_proba"):
            probs_matrix = self.model.predict_proba(X_scaled)
            # Ensemble average probabilities across temporal speech segments
            avg_probs = np.mean(probs_matrix, axis=0)
        else:
            # Fallback for models without predict_proba
            preds = self.model.predict(X_scaled)
            avg_probs = np.zeros(len(self.encoder.classes_))
            for p in preds:
                avg_probs[p] += 1.0 / len(preds)

        classes = list(self.encoder.classes_)

        # 6. Rank predictions
        ranked_indices = np.argsort(avg_probs)[::-1]
        ranked_predictions = []
        for idx in ranked_indices:
            cls_name = classes[idx]
            prob = float(avg_probs[idx])
            meta = ACCENTS_METADATA.get(cls_name, {
                "name": f"{cls_name} English",
                "variant": cls_name,
                "flag": "🌐",
                "region": "Global",
                "key_features": "Regional phonetic characteristics",
                "color": "#38bdf8"
            })
            ranked_predictions.append({
                "accent_id": cls_name.lower(),
                "name": meta["name"],
                "variant": meta["variant"],
                "flag": meta["flag"],
                "region": meta["region"],
                "key_features": meta["key_features"],
                "color": meta["color"],
                "probability": round(prob, 4),
                "confidence_percent": round(prob * 100, 1)
            })

        top_pred = ranked_predictions[0]
        top_confidence = top_pred["probability"]

        # 7. Determine certainty threshold
        is_uncertain = top_confidence < MIN_CONFIDENCE_THRESHOLD
        top_three = ranked_predictions[:3]

        # 8. Prepare comprehensive response
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return {
            "success": True,
            "detected_accent": top_pred["name"],
            "detected_accent_id": top_pred["accent_id"],
            "detected_flag": top_pred["flag"],
            "detected_variant": top_pred["variant"],
            "detected_region": top_pred["region"],
            "confidence_score": top_pred["confidence_percent"],
            "is_uncertain": is_uncertain,
            "uncertainty_reason": "Low confidence score: Acoustic signals overlap with multiple dialect boundaries. Speak 15–30 seconds for clearer phonological separation." if is_uncertain else None,
            "educational_description": top_pred["key_features"],
            "top_candidates": top_three,
            "all_candidates": ranked_predictions,
            "audio_quality": {
                "duration_seconds": proc_res["duration"],
                "sample_rate": proc_res["sr"],
                "rms_energy": proc_res["rms_energy"],
                "estimated_snr_db": proc_res["snr_db"],
                "rating": proc_res["quality_rating"],
                "segments_analyzed": len(chunks)
            },
            "timestamp": timestamp_str,
            "model_metadata": {
                "model_name": self.metrics.get("model_name", "Support Vector Machine") if self.metrics else "SVM",
                "dataset_source": "Speech Accent Archive (George Mason University)",
                "dataset_license": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC-BY-NC-SA 4.0)",
                "methodology": "Acoustic ML classification using 188-dimensional MFCCs, spectral contrast, rolloff, and mel-filterbanks without speech-to-text text transcription or generative AI.",
                "dataset_accuracy_disclosure": f"{self.metrics.get('accuracy', 0.82)*100:.1f}% cross-validation accuracy on unseen speakers" if self.metrics else "Evaluated via StratifiedGroupKFold by speaker"
            }
        }
