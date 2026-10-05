"""Load every audio file of the 4 genres once and record basic facts about it."""
import librosa
import pandas as pd

from src.config import GENRES, LABEL_MAP, RAW_AUDIO_DIR


def inspect_track(path):
    """Load one file at its native sample rate and describe it.

    Returns a dict. If the file cannot be loaded, 'error' holds the message.
    """
    try:
        waveform, sample_rate = librosa.load(path, sr=None, mono=True)
    except Exception as error:
        return {"loaded": False, "error": str(error)}

    n_samples = len(waveform)
    return {
        "loaded": True,
        "error": "",
        "sample_rate": sample_rate,
        "n_samples": n_samples,
        "duration_s": n_samples / sample_rate,
        "n_full_1s_clips": n_samples // sample_rate,  # partial last second dropped
        "min_amplitude": float(waveform.min()),
        "max_amplitude": float(waveform.max()),
    }


def inspect_dataset():
    """Inspect all tracks of the 4 selected genres. Returns one row per track."""
    rows = []
    for genre in GENRES:
        for path in sorted((RAW_AUDIO_DIR / genre).glob("*.wav")):
            row = {
                "track_id": path.stem,  # e.g. "classical.00000"
                "genre": genre,
                "label": LABEL_MAP[genre],
                "path": str(path.relative_to(RAW_AUDIO_DIR)),
            }
            row.update(inspect_track(path))
            rows.append(row)
    return pd.DataFrame(rows)
