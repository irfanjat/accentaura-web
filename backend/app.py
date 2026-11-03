#!/usr/bin/env python3
"""
English Accent Identifier - Flask Application Server
REST API for Audio Signal Processing, Acoustic Feature Extraction, and ML Classification
"""
import os
import sys
import glob
import time
import uuid
import json
from flask import Flask, request, jsonify, send_from_directory, render_template

# Ensure backend root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.config import (
    FRONTEND_DIR, TEMP_UPLOAD_DIR, AUDIO_RAW_DIR, METRICS_PATH,
    ACCENTS_METADATA, ACCENT_CLASSES
)
from backend.accent_classifier import AccentClassifier

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    template_folder=FRONTEND_DIR
)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload

# Initialize classifier
classifier = AccentClassifier()

@app.route("/")
def index():
    """Serves the master frontend UI."""
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:path>")
def static_proxy(path):
    """Serves static files (CSS, JS, Assets) from frontend."""
    return send_from_directory(FRONTEND_DIR, path)

@app.route("/api/accents", methods=["GET"])
def get_accents():
    """Returns the 7 codified accent profiles and educational characteristics."""
    return jsonify({
        "success": True,
        "accents": ACCENTS_METADATA,
        "classes": ACCENT_CLASSES
    })

@app.route("/api/model-info", methods=["GET"])
def get_model_info():
    """Returns verified model architecture, cross-validation metrics, and dataset disclosures."""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
        return jsonify({
            "success": True,
            "metrics": metrics
        })
    else:
        return jsonify({
            "success": False,
            "error": "Model metrics not available."
        }), 404

@app.route("/api/predict-accent", methods=["POST"])
def predict_accent():
    """
    Receives voice recording from the user's microphone.
    Preprocesses the audio signal, extracts acoustic features,
    and returns top candidate accents with confidence scores.
    """
    if "audio" not in request.files:
        return jsonify({
            "success": False,
            "error": "No audio file was received in the request."
        }), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({
            "success": False,
            "error": "Empty audio filename received."
        }), 400

    # Generate unique session filename
    ext = os.path.splitext(audio_file.filename)[1].lower()
    if not ext or ext not in [".wav", ".webm", ".mp3", ".ogg", ".m4a"]:
        ext = ".webm"

    unique_id = f"rec_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    save_filename = f"{unique_id}{ext}"
    saved_path = os.path.join(TEMP_UPLOAD_DIR, save_filename)

    try:
        audio_file.save(saved_path)

        # Run acoustic classification pipeline
        result = classifier.predict(saved_path)

        if not result["success"]:
            # Clean up invalid recording immediately
            if os.path.exists(saved_path):
                os.remove(saved_path)
            return jsonify(result), 400

        # Attach playback URL
        result["playback_url"] = f"/api/playback/{save_filename}"
        return jsonify(result)

    except Exception as e:
        if os.path.exists(saved_path):
            os.remove(saved_path)
        return jsonify({
            "success": False,
            "error": f"Signal processing error: {str(e)}"
        }), 500

@app.route("/api/playback/<filename>", methods=["GET"])
def playback_audio(filename):
    """Serves the temporary user recording for playback in the dashboard."""
    return send_from_directory(TEMP_UPLOAD_DIR, filename)

@app.route("/api/sample-audio/<speaker_id>", methods=["GET"])
def sample_audio(speaker_id):
    """Serves Speech Accent Archive model recordings."""
    filename = f"{speaker_id}.mp3"
    return send_from_directory(AUDIO_RAW_DIR, filename)

@app.route("/api/reset-session", methods=["POST"])
def reset_session():
    """
    Exhibition Mode: Cleans up temporary user voice recordings
    to maintain visitor privacy and free disk resources.
    """
    try:
        deleted_count = 0
        for f in glob.glob(os.path.join(TEMP_UPLOAD_DIR, "*.*")):
            try:
                os.remove(f)
                deleted_count += 1
            except OSError:
                pass

        return jsonify({
            "success": True,
            "message": f"Exhibition session reset. {deleted_count} temporary audio files deleted.",
            "deleted_count": deleted_count
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "app": "AccentAura"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting English Accent Identifier on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
