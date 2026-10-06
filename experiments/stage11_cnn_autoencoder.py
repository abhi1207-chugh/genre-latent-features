"""Stage 11: build the CNN autoencoder, check every shape, then train it as a
plain autoencoder. Its validation reconstruction error = AE_reference_error.

Run from the project root:
    .venv/bin/python experiments/stage11_cnn_autoencoder.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import PROJECT_ROOT, RESULTS_DIR
from src.models.cnn_autoencoder import CNNAutoencoder
from src.preprocessing.dataset import load_split, make_loader
from src.training.train_autoencoder import train_autoencoder
from src.utils import get_device, set_seed

set_seed(42)
device = get_device()
model = CNNAutoencoder(latent_dim=64, dropout=0.1)

# ---- Part 1: shapes, layer by layer, on a dummy batch of 8 clips ----
x = torch.randn(8, 500)
print("Encoder:")
h = x.unsqueeze(1)
print(f"  input (unsqueezed)   {tuple(h.shape)}")
for i, block in enumerate(model.encoder.blocks, start=1):
    h = block(h)
    print(f"  after conv block {i}   {tuple(h.shape)}")
z = model.encoder.head(h)
print(f"  z                    {tuple(z.shape)}")

print("Decoder:")
h = model.decoder.expand(z).view(-1, 64, 31)
print(f"  after Linear+reshape {tuple(h.shape)}")
decoder_names = ["upsample block 1", "upsample block 2", "upsample block 3",
                 "Upsample to 500", "final Conv 16->1"]
for name, layer in zip(decoder_names, model.decoder.blocks):
    h = layer(h)
    print(f"  after {name:15s}{tuple(h.shape)}")
x_hat, _ = model(x)
print(f"  x_hat                {tuple(x_hat.shape)}")

count = lambda m: sum(p.numel() for p in m.parameters())
print(f"\nParameters: encoder {count(model.encoder):,}, decoder {count(model.decoder):,}, "
      f"total {count(model):,}")

# ---- Part 2: train as a plain autoencoder (same loop and settings as the dense AE) ----
X_train, y_train, _ = load_split("train")
X_val, y_val, _ = load_split("val")
train_loader = make_loader(X_train, y_train, batch_size=512, shuffle=True, seed=42)
val_loader = make_loader(X_val, y_val, batch_size=512)

set_seed(42)
model = CNNAutoencoder(latent_dim=64, dropout=0.1)
start = time.time()
checkpoint = PROJECT_ROOT / "checkpoints" / "cnn_ae.pt"
history, best_epoch, best_val = train_autoencoder(
    model, train_loader, val_loader, device, checkpoint, lr=1e-4, max_epochs=500, patience=20
)
print(f"\nTrained {len(history)} epochs in {time.time() - start:.0f} s on {device}, best epoch = {best_epoch}")
print(history.iloc[::25].round(4).to_string(index=False))

zero_error = float((X_val ** 2).mean())
print(f"\nValidation reconstruction error (MSE per value):")
print(f"  predict all zeros : {zero_error:.4f}")
print(f"  PCA 64 (Stage 8)  : 0.7495")
print(f"  dense AE (Stage 8): 1.0058")
print(f"  CNN AE            : {best_val:.4f}   <- AE_reference_error")
print(f"  1.5 x reference   : {1.5 * best_val:.4f}")

history.to_csv(RESULTS_DIR / "cnn_ae_history.csv", index=False)
with open(RESULTS_DIR / "ae_reference.json", "w") as f:
    json.dump({"AE_reference_error": best_val, "model": "CNN AE (reconstruction only)",
               "best_epoch": best_epoch, "split": "val"}, f, indent=2)
print("\nSaved checkpoints/cnn_ae.pt, results/cnn_ae_history.csv, results/ae_reference.json")
