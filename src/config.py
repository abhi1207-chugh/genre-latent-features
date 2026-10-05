"""Fixed project settings shared by every stage."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_AUDIO_DIR = PROJECT_ROOT / "data" / "raw" / "genres_original"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# Fixed label map (folder name -> class index). Never change the order.
LABEL_MAP = {"classical": 0, "country": 1, "disco": 2, "hiphop": 3}
GENRES = list(LABEL_MAP.keys())
