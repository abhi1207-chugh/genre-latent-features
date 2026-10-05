"""Load the saved clips, standardize them, and wrap them for PyTorch."""
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import PROCESSED_DIR
from src.preprocessing.features import apply_scaler


def load_split(split_name):
    """Standardized arrays for one split ('train', 'val' or 'test').

    Returns X (n, 500) float32, y (n,) int64, track_id (n,) str.
    The scaler is the one fit on the train clips in Stage 5.
    """
    data = np.load(PROCESSED_DIR / "clips_unscaled.npz")
    with open(PROCESSED_DIR / "scaler.json") as f:
        scaler = json.load(f)

    keep = data["split"] == split_name
    X = apply_scaler(data["X"][keep], scaler["mean"], scaler["std"])
    return X, data["label"][keep], data["track_id"][keep]


class ClipDataset(Dataset):
    """One item = (500-D clip as float tensor, genre label as int tensor)."""

    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def make_loader(X, y, batch_size=512, shuffle=False, seed=42):
    """DataLoader with a seeded shuffle, so batch order is reproducible."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ClipDataset(X, y), batch_size=batch_size,
                      shuffle=shuffle, generator=generator)
