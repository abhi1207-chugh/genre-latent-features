"""Clip-level and track-level classification metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(y_true, y_pred):
    """Accuracy and macro-averaged precision / recall / F1 (all 4 genres weighted equally)."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def track_level_predictions(clip_probs, track_ids, clip_labels):
    """Average the clip probability vectors of each track, then take argmax.

    Returns (track_true, track_pred), one entry per track.
    """
    probs = pd.DataFrame(clip_probs)
    probs["track_id"] = track_ids
    mean_probs = probs.groupby("track_id").mean()          # one row per track

    labels = pd.Series(clip_labels, index=track_ids)
    track_true = labels.groupby(level=0).first().loc[mean_probs.index].to_numpy()
    track_pred = mean_probs.to_numpy().argmax(axis=1)
    return track_true, track_pred


def clip_and_track_metrics(clip_probs, y, track_ids):
    """Metrics at both levels from clip probabilities. Keys prefixed clip_ / track_."""
    clip = classification_metrics(y, np.argmax(clip_probs, axis=1))
    track = classification_metrics(*track_level_predictions(clip_probs, track_ids, y))
    return {**{f"clip_{k}": v for k, v in clip.items()},
            **{f"track_{k}": v for k, v in track.items()}}
