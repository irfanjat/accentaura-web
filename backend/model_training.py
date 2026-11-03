#!/usr/bin/env python3
"""
English Accent Identifier - Model Training Pipeline
Academic Machine Learning: Feature Extraction, Cross-Validation by Speaker, and Evaluation
Strictly traditional ML (SVM / Random Forest) - No Generative AI
"""
import os
import sys
import json
import urllib.request
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib

try:
    from backend.config import (
        AUDIO_RAW_DIR, MODELS_DIR, DATASET_DIR, FEATURES_CSV_PATH,
        MODEL_PATH, SCALER_PATH, ENCODER_PATH, METRICS_PATH,
        ACCENTS_METADATA, ACCENT_CLASSES
    )
    from backend.audio_processor import AudioProcessor
    from backend.feature_extractor import FeatureExtractor
except ImportError:
    from config import (
        AUDIO_RAW_DIR, MODELS_DIR, DATASET_DIR, FEATURES_CSV_PATH,
        MODEL_PATH, SCALER_PATH, ENCODER_PATH, METRICS_PATH,
        ACCENTS_METADATA, ACCENT_CLASSES
    )
    from audio_processor import AudioProcessor
    from feature_extractor import FeatureExtractor

# Balanced Speaker Catalogue from Speech Accent Archive (GMU, CC-BY-NC-SA 4.0)
SPEAKER_CATALOGUE = {
    "American": [
        "english1", "english5", "english6", "english7",
        "english9", "english10", "english15", "english16"
    ],
    "British": [
        "english2", "english11", "english12", "english13",
        "english35", "english38", "english40", "english57"
    ],
    "Australian": [
        "english3", "english8", "english69", "english73",
        "english77", "english84", "english87", "english125"
    ],
    "Pakistani": [
        "urdu1", "urdu2", "urdu3", "urdu5",
        "urdu7", "urdu10", "urdu11", "urdu12"
    ],
    "Indian": [
        "hindi1", "hindi2", "hindi3", "hindi4",
        "hindi5", "hindi6", "hindi7", "hindi8"
    ],
    "Irish": [
        "english14", "english56", "english72", "english85",
        "english108", "english134", "english156", "english253"
    ],
    "Scottish": [
        "english24", "english80", "english188", "english225",
        "english228", "english440", "english448"
    ]
}

def download_audio_file(speaker_id):
    """Downloads a speaker audio file from the Speech Accent Archive if not cached."""
    local_path = os.path.join(AUDIO_RAW_DIR, f"{speaker_id}.mp3")
    if os.path.exists(local_path) and os.path.getsize(local_path) > 10000:
        return local_path

    url = f"https://accent.gmu.edu/audio/{speaker_id}.mp3"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=12) as response, open(local_path, "wb") as out_file:
            out_file.write(response.read())
        return local_path
    except Exception as e:
        print(f"Warning: Failed to download {speaker_id} from {url}: {e}")
        return None

def build_dataset_and_extract_features():
    """
    Downloads audio files, preprocesses them, splits into speech segments,
    and extracts multi-dimensional acoustic feature vectors while tagging speaker ID.
    """
    processor = AudioProcessor()
    extractor = FeatureExtractor()

    print("--- 1. DOWNLOADING AUDIO DATASET (55 Speakers) ---")
    all_speakers = []
    for accent, spk_list in SPEAKER_CATALOGUE.items():
        for spk in spk_list:
            all_speakers.append((accent, spk))

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_audio_file, spk): (accent, spk) for accent, spk in all_speakers}
        for f in futures:
            accent, spk = futures[f]
            res = f.result()
            if res:
                print(f"Cached {accent} speaker: {spk}")

    print("\n--- 2. EXTRACTING ACOUSTIC FEATURES BY SPEAKER SEGMENT ---")
    data_rows = []
    feature_names_ref = None

    for accent, spk in all_speakers:
        local_path = os.path.join(AUDIO_RAW_DIR, f"{spk}.mp3")
        if not os.path.exists(local_path):
            continue

        try:
            val_res = processor.validate_and_preprocess(local_path)
            if not val_res["is_valid"]:
                continue

            y_clean = val_res["y"]
            sr = val_res["sr"]

            # Split each speaker's recording into 4-7 second segments
            chunks = processor.split_into_windows(y_clean, sr, window_sec=5.0, hop_sec=2.5)

            for c_idx, chunk in enumerate(chunks):
                feats, f_names = extractor.extract_features(chunk)
                if feature_names_ref is None:
                    feature_names_ref = f_names

                row = {
                    "speaker_id": spk,
                    "accent_label": accent,
                    "segment_index": c_idx
                }
                for f_name, f_val in zip(f_names, feats):
                    row[f_name] = f_val

                data_rows.append(row)

        except Exception as e:
            print(f"Error processing {spk}: {e}")

    df = pd.DataFrame(data_rows)
    df.to_csv(FEATURES_CSV_PATH, index=False)
    print(f"\nExtracted {len(df)} feature segments across {df['speaker_id'].nunique()} unique speakers.")
    print(f"Distribution per accent:\n{df['accent_label'].value_counts()}")
    return df

def train_and_evaluate_model(df=None):
    """
    Trains the ML classifier using Group-Based Stratified Cross-Validation (by speaker)
    to strictly prevent speaker leakage.
    """
    if df is None:
        if os.path.exists(FEATURES_CSV_PATH):
            df = pd.read_csv(FEATURES_CSV_PATH)
        else:
            df = build_dataset_and_extract_features()

    meta_cols = ["speaker_id", "accent_label", "segment_index"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].values
    y_raw = df["accent_label"].values
    groups = df["speaker_id"].values

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = list(label_encoder.classes_)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n--- 3. STRATIFIED GROUP K-FOLD CROSS-VALIDATION (By Speaker) ---")
    sgkf = StratifiedGroupKFold(n_splits=5)
    
    # We compare SVM (RBF) with probability calibration and Random Forest
    svm_clf = SVC(kernel="rbf", C=2.5, gamma="scale", probability=True, random_state=42)
    rf_clf = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_split=3, random_state=42)

    fold_accuracies_svm = []
    fold_accuracies_rf = []
    all_y_true = []
    all_y_pred_svm = []

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X_scaled, y, groups=groups)):
        # Verify no speaker leakage
        train_speakers = set(groups[train_idx])
        test_speakers = set(groups[test_idx])
        overlap = train_speakers.intersection(test_speakers)
        assert len(overlap) == 0, f"Speaker leakage detected in fold {fold}!"

        # Train SVM
        svm_clf.fit(X_scaled[train_idx], y[train_idx])
        pred_svm = svm_clf.predict(X_scaled[test_idx])
        acc_svm = accuracy_score(y[test_idx], pred_svm)
        fold_accuracies_svm.append(acc_svm)

        # Train RF
        rf_clf.fit(X_scaled[train_idx], y[train_idx])
        pred_rf = rf_clf.predict(X_scaled[test_idx])
        acc_rf = accuracy_score(y[test_idx], pred_rf)
        fold_accuracies_rf.append(acc_rf)

        all_y_true.extend(y[test_idx])
        all_y_pred_svm.extend(pred_svm)

        print(f"Fold {fold+1} (Speakers: {len(test_speakers)}) -> SVM Acc: {acc_svm:.3f} | RF Acc: {acc_rf:.3f}")

    mean_acc_svm = float(np.mean(fold_accuracies_svm))
    mean_acc_rf = float(np.mean(fold_accuracies_rf))
    print(f"\nCross-Validation Summary:")
    print(f"Support Vector Machine (RBF) Mean Accuracy: {mean_acc_svm * 100:.2f}% (+/- {np.std(fold_accuracies_svm)*100:.2f}%)")
    print(f"Random Forest Mean Accuracy:               {mean_acc_rf * 100:.2f}% (+/- {np.std(fold_accuracies_rf)*100:.2f}%)")

    # Final Full Pipeline Training using the superior model
    chosen_model = svm_clf if mean_acc_svm >= mean_acc_rf else rf_clf
    chosen_model_name = "Support Vector Machine (RBF Kernel)" if mean_acc_svm >= mean_acc_rf else "Random Forest"
    chosen_model.fit(X_scaled, y)

    # Compute comprehensive evaluation metrics on out-of-fold speaker predictions
    acc = accuracy_score(all_y_true, all_y_pred_svm)
    prec = precision_score(all_y_true, all_y_pred_svm, average="weighted", zero_division=0)
    rec = recall_score(all_y_true, all_y_pred_svm, average="weighted", zero_division=0)
    f1 = f1_score(all_y_true, all_y_pred_svm, average="weighted", zero_division=0)
    cm = confusion_matrix(all_y_true, all_y_pred_svm).tolist()
    cls_report = classification_report(all_y_true, all_y_pred_svm, target_names=class_names, output_dict=True, zero_division=0)

    print("\n--- DETAILED CLASSIFICATION REPORT (Out-of-Fold Unseen Speakers) ---")
    print(classification_report(all_y_true, all_y_pred_svm, target_names=class_names, zero_division=0))

    # Save artifacts
    joblib.dump(chosen_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    metrics_payload = {
        "model_name": chosen_model_name,
        "n_classes": len(class_names),
        "classes": class_names,
        "n_samples": len(df),
        "n_unique_speakers": int(df["speaker_id"].nunique()),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "cross_validation": "5-Fold StratifiedGroupKFold (Split by Speaker)",
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm,
        "class_report": cls_report,
        "dataset_source": "Speech Accent Archive (George Mason University)",
        "dataset_license": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC-BY-NC-SA 4.0)",
        "methodology": "No generative AI or STT text transcriptions used. Classification is acoustic, operating on 188-dimensional spectral & cepstral features."
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\nModel and metrics successfully saved to {MODELS_DIR}!")
    return metrics_payload

if __name__ == "__main__":
    df = build_dataset_and_extract_features()
    train_and_evaluate_model(df)
