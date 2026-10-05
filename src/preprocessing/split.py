"""Track-level train / validation / test split, stratified by genre.

The split is made on TRACKS, before any clips exist, so all clips of one
song always end up in the same split (no leakage between splits).
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import GENRES, LABEL_MAP, RAW_AUDIO_DIR


def list_tracks():
    """One row per audio file of the 4 genres: track_id, genre, label, path."""
    rows = []
    for genre in GENRES:
        for path in sorted((RAW_AUDIO_DIR / genre).glob("*.wav")):
            rows.append({
                "track_id": path.stem,
                "genre": genre,
                "label": LABEL_MAP[genre],
                "path": str(path.relative_to(RAW_AUDIO_DIR)),
            })
    return pd.DataFrame(rows)


def make_track_split(tracks, val_fraction=0.15, test_fraction=0.15, seed=42):
    """Add a 'split' column ('train' / 'val' / 'test') to the tracks table.

    Two steps, both stratified by genre label:
      1. all tracks        -> test  + rest
      2. rest              -> val   + train
    """
    n_test = round(len(tracks) * test_fraction)
    n_val = round(len(tracks) * val_fraction)

    rest, test = train_test_split(
        tracks, test_size=n_test, stratify=tracks["label"], random_state=seed
    )
    train, val = train_test_split(
        rest, test_size=n_val, stratify=rest["label"], random_state=seed
    )

    split = pd.concat([
        train.assign(split="train"),
        val.assign(split="val"),
        test.assign(split="test"),
    ])
    return split.sort_values("track_id").reset_index(drop=True)
