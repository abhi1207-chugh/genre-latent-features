"""Stage 6: check the Dataset / DataLoader on the train and val splits.

The test split is not loaded here; it stays locked until Stage 20.

Run from the project root:
    .venv/bin/python experiments/stage06_check_dataloader.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.preprocessing.dataset import load_split, make_loader

X_train, y_train, tracks_train = load_split("train")
X_val, y_val, tracks_val = load_split("val")

print(f"train: X {X_train.shape}, y {y_train.shape}, {len(set(tracks_train))} tracks")
print(f"val:   X {X_val.shape}, y {y_val.shape}, {len(set(tracks_val))} tracks")
print(f"train after scaling: mean = {X_train.mean():+.4f}, std = {X_train.std():.4f}")

train_loader = make_loader(X_train, y_train, batch_size=512, shuffle=True, seed=42)
val_loader = make_loader(X_val, y_val, batch_size=512, shuffle=False)

print(f"\nBatches per epoch: train = {len(train_loader)}, val = {len(val_loader)}")

x_batch, y_batch = next(iter(train_loader))
print(f"One train batch: x {tuple(x_batch.shape)} {x_batch.dtype}, "
      f"y {tuple(y_batch.shape)} {y_batch.dtype}")
print(f"Labels in that batch (counts of 0/1/2/3): {torch.bincount(y_batch, minlength=4).tolist()}")

# Last batch is smaller: 8396 = 16 * 512 + 204
sizes = [len(y) for _, y in train_loader]
print(f"Train batch sizes: first = {sizes[0]}, last = {sizes[-1]}, total = {sum(sizes)}")

# Reproducibility: a new loader with the same seed gives the same first batch
_, y_again = next(iter(make_loader(X_train, y_train, shuffle=True, seed=42)))
print(f"\nSame seed -> same first batch: {torch.equal(y_batch, y_again)}")

# Shuffling really happens: epoch 1 and epoch 2 differ
_, y_epoch2 = next(iter(train_loader))
print(f"Epoch 2 first batch differs from epoch 1: {not torch.equal(y_batch, y_epoch2)}")
