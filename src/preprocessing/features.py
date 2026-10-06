"""Raw audio track -> 1-second clips -> average pooling -> 500-D vectors.

Also the global standardizer (one mean, one std, fit on train clips only).
"""
import librosa
import numpy as np

from src.config import CLIP_SAMPLES, FEATURE_DIM, POOL_SIZE, RAW_AUDIO_DIR, SAMPLE_RATE


def track_to_clips(path):
    """Load one track and return its pooled clips, shape (n_clips, 500)."""
    # librosa resamples 22,050 Hz -> 20,000 Hz while loading
    waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return waveform_to_clips(waveform)


def waveform_to_clips(waveform):
    """Mono waveform at 20 kHz -> pooled 1-second clips, shape (n_clips, 500)."""
    # Cut into full 1-second clips; the partial last second is dropped
    n_clips = len(waveform) // CLIP_SAMPLES
    clips = waveform[: n_clips * CLIP_SAMPLES].reshape(n_clips, CLIP_SAMPLES)

    # Average pooling: group every 40 consecutive samples and take their mean.
    # (n_clips, 20000) -> (n_clips, 500, 40) -> mean over last axis -> (n_clips, 500)
    pooled = clips.reshape(n_clips, FEATURE_DIM, POOL_SIZE).mean(axis=2)
    return pooled.astype(np.float32)


def build_clip_dataset(split_table):
    """Turn every track in the split table into clips.

    Returns a dict of arrays, one entry per clip:
      X (n, 500) float32, label, track_id, split, clip_index (position in its track)
    """
    features, labels, track_ids, splits, clip_indices = [], [], [], [], []

    for track in split_table.itertuples():
        pooled = track_to_clips(RAW_AUDIO_DIR / track.path)
        n_clips = len(pooled)

        features.append(pooled)
        labels += [track.label] * n_clips
        track_ids += [track.track_id] * n_clips
        splits += [track.split] * n_clips
        clip_indices += list(range(n_clips))

    return {
        "X": np.concatenate(features),
        "label": np.array(labels, dtype=np.int64),
        "track_id": np.array(track_ids),
        "split": np.array(splits),
        "clip_index": np.array(clip_indices, dtype=np.int64),
    }


def fit_global_scaler(X_train):
    """One mean and one std over ALL values of the training clips."""
    return float(X_train.mean()), float(X_train.std())


def apply_scaler(X, mean, std):
    return ((X - mean) / std).astype(np.float32)
