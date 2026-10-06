"""Load the saved clips, standardize them, and wrap them for PyTorch."""
import json
import math

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


class GPUBatches:
    """All clips stored once on the device; yields (x, y) batches of it.

    Replaces the DataLoader for training: no per-item Python calls, no copying
    each batch to the GPU. Shuffling uses a seeded CPU generator, so the batch
    order is reproducible, and its state can be saved for resuming.
    """

    def __init__(self, X, y, device, batch_size=512, shuffle=False, seed=42):
        self.X = torch.from_numpy(X).to(device)      # 8,396 x 500 floats = 17 MB
        self.y = torch.from_numpy(y).to(device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return math.ceil(len(self.y) / self.batch_size)

    def __iter__(self):
        n = len(self.y)
        if self.shuffle:
            order = torch.randperm(n, generator=self.generator).to(self.X.device)
        else:
            order = torch.arange(n, device=self.X.device)
        for start in range(0, n, self.batch_size):
            batch = order[start:start + self.batch_size]
            yield self.X[batch], self.y[batch]


def make_loader(X, y, batch_size=512, shuffle=False, seed=42):
    """DataLoader with a seeded shuffle, so batch order is reproducible."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ClipDataset(X, y), batch_size=batch_size,
                      shuffle=shuffle, generator=generator)
