"""Fixed project settings shared by every stage."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_AUDIO_DIR = PROJECT_ROOT / "data" / "raw" / "genres_original"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# Fixed label map (folder name -> class index). Never change the order.
LABEL_MAP = {"classical": 0, "country": 1, "disco": 2, "hiphop": 3}
GENRES = list(LABEL_MAP.keys())

# Audio -> 500-D features (Stage 5 decision: resample to 20 kHz)
SAMPLE_RATE = 20_000            # samples per second after resampling
CLIP_SAMPLES = SAMPLE_RATE      # 1-second clip = 20,000 samples
POOL_SIZE = 40                  # average-pooling window
FEATURE_DIM = CLIP_SAMPLES // POOL_SIZE   # 500
