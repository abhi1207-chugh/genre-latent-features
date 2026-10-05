"""Stage 8: train the paper's dense vanilla autoencoder (reconstruction only).

Run from the project root:
    .venv/bin/python experiments/stage08_train_dense_ae.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.decomposition import PCA

from src.config import PROJECT_ROOT, RESULTS_DIR
from src.models.dense_autoencoder import DenseAutoencoder
from src.preprocessing.dataset import load_split, make_loader
from src.training.train_autoencoder import train_autoencoder
from src.utils import get_device, set_seed

set_seed(42)
device = get_device()

X_train, y_train, _ = load_split("train")
X_val, y_val, _ = load_split("val")
train_loader = make_loader(X_train, y_train, batch_size=512, shuffle=True, seed=42)
val_loader = make_loader(X_val, y_val, batch_size=512)

model = DenseAutoencoder(input_dim=500, latent_dim=64, dropout=0.1)
n_params = sum(p.numel() for p in model.parameters())
print(f"Dense AE: {n_params:,} parameters, device = {device}")

start = time.time()
checkpoint = PROJECT_ROOT / "checkpoints" / "dense_ae.pt"
history, best_epoch, best_val = train_autoencoder(
    model, train_loader, val_loader, device, checkpoint, lr=1e-4, max_epochs=500, patience=20
)
print(f"Trained {len(history)} epochs in {time.time() - start:.0f} s, best epoch = {best_epoch}")

print("\nEvery 25th epoch:")
print(history.iloc[::25].round(4).to_string(index=False))

# Reference points for judging the number
zero_error = float((X_val ** 2).mean())                     # always predict 0 (the train mean)
pca = PCA(n_components=64, random_state=42).fit(X_train)    # best LINEAR 64-D compression
pca_error = float(((pca.inverse_transform(pca.transform(X_val)) - X_val) ** 2).mean())

print(f"\nValidation reconstruction error (MSE per value):")
print(f"  predict all zeros   : {zero_error:.4f}")
print(f"  PCA 64 (linear)     : {pca_error:.4f}")
print(f"  dense AE (best ep.) : {best_val:.4f}")

history.to_csv(RESULTS_DIR / "dense_ae_history.csv", index=False)
print(f"\nSaved checkpoints/dense_ae.pt and results/dense_ae_history.csv")
