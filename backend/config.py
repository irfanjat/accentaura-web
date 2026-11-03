"""
English Accent Identifier - Configuration
Academic Machine Learning & Signal Processing Settings
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
AUDIO_RAW_DIR = os.path.join(DATASET_DIR, "raw_audio")
TEMP_UPLOAD_DIR = os.path.join(BASE_DIR, "temp_recordings")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(AUDIO_RAW_DIR, exist_ok=True)
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# Audio Processing Parameters
SAMPLE_RATE = 22050
N_MFCC = 20
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

# Recording Validation Requirements
MIN_AUDIO_DURATION_SEC = 2.5
RECOMMENDED_MIN_SEC = 8.0
RECOMMENDED_MAX_SEC = 35.0
MIN_RMS_ENERGY = 0.005  # Silence detection threshold

# Accent Categories & Metadata
ACCENTS_METADATA = {
    "British": {
        "name": "British English",
        "variant": "Standard Southern British (SSBE) / RP",
        "flag": "🇬🇧",
        "region": "United Kingdom",
        "key_features": "Non-rhotic, broad TRAP-BATH split (/ɑː/), LOT-CLOTH distinction, coda T-glottaling.",
        "example_words": ["Water /ˈwɔːtə/", "Dance /dɑːns/", "Schedule /ˈʃɛdjuːl/", "Car /kɑː/"],
        "sample_id": "english2",
        "color": "#3B82F6"
    },
    "American": {
        "name": "American English",
        "variant": "General American (GenAm)",
        "flag": "🇺🇸",
        "region": "United States",
        "key_features": "Fully rhotic post-vocalic /r/, voiced alveolar tapping [ɾ] in 'water', LOT-PALM merger.",
        "example_words": ["Water /ˈwɑːɾɚ/", "Center /ˈsɛnɚ/", "Schedule /ˈskɛdʒuːl/", "Herb /ɝb/"],
        "sample_id": "english1",
        "color": "#22D3EE"
    },
    "Australian": {
        "name": "Australian English",
        "variant": "General Australian",
        "flag": "🇦🇺",
        "region": "Commonwealth of Australia",
        "key_features": "Australian vowel shift (diphthongized FLEECE and wide FACE), backed PRICE, non-rhotic.",
        "example_words": ["Today /təˈdɑɪ/", "Night /nɒɪt/", "Car /kɐː/", "Water /ˈwoːtə/"],
        "sample_id": "english3",
        "color": "#10B981"
    },
    "Pakistani": {
        "name": "Pakistani English",
        "variant": "Pakistani Standard English (PakE)",
        "flag": "🇵🇰",
        "region": "Pakistan",
        "key_features": "Retroflex plosives [ʈ, ɖ], dental stops for fricatives [t̪, d̪], syllable-timed cadence, prosthetic s-clusters.",
        "example_words": ["Station /ɪsˈʈeːʃən/", "School /ɪsˈkuːl/", "Water /ˈʋɑːʈəɾ/", "Police /poːˈliːs/"],
        "sample_id": "urdu1",
        "color": "#059669"
    },
    "Indian": {
        "name": "Indian English",
        "variant": "Standard Indian English",
        "flag": "🇮🇳",
        "region": "India",
        "key_features": "Retroflex consonants, monophthongal /eː/ and /oː/, syllable-timed isochronous rhythm, [ʋ] approximation.",
        "example_words": ["Doctor /ˈɖɒkʈəɾ/", "Train /ʈɾeːn/", "Water /ˈʋɑːʈəɾ/", "Time /ʈaɪm/"],
        "sample_id": "hindi1",
        "color": "#F59E0B"
    },
    "Irish": {
        "name": "Irish English",
        "variant": "Hiberno-English",
        "flag": "🇮🇪",
        "region": "Ireland",
        "key_features": "Rhotic with alveolar tap [ɾ], T-slitting [t̞], clear /l/ in all environments, lilting musical intonation.",
        "example_words": ["Three /t̪ɹiː/", "Water /ˈwɑt̞əɹ/", "Right /ɹaɪt̞/", "Film /ˈfɪləm/"],
        "sample_id": "english14",
        "color": "#14B8A6"
    },
    "Scottish": {
        "name": "Scottish English",
        "variant": "Scottish Standard English (SSE)",
        "flag": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
        "region": "Scotland",
        "key_features": "Tapped/trilled /r/ [ɾ], voiceless velar fricative /x/ in 'loch', Aitken's Vowel Length Rule, POOL-PULL merger.",
        "example_words": ["Loch /lɒx/", "Bird /bɪɾd/", "Water /ˈwɔʔəɾ/", "Which /ʍɪtʃ/"],
        "sample_id": "english24",
        "color": "#A78BFA"
    }
}

ACCENT_CLASSES = list(ACCENTS_METADATA.keys())

# Model Paths
MODEL_PATH = os.path.join(MODELS_DIR, "accent_classifier.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
FEATURES_CSV_PATH = os.path.join(DATASET_DIR, "extracted_features.csv")

# Confidence thresholds
MIN_CONFIDENCE_THRESHOLD = 0.28  # Probability threshold to avoid over-confident false assignments
