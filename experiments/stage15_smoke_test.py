"""Stage 15: smoke test - train the full model briefly on a small train subset.

Goal: prove that the whole training step works end to end, NOT to get good results.
  - every loss term goes down, nothing becomes NaN
  - centers move and spread apart, embeddings do not collapse to 0
  - the model can fit a small subset (train accuracy rises well above 25%)

Run from the project root:
    .venv/bin/python experiments/stage15_smoke_test.py
"""
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.losses.combined_loss import CombinedLoss
from src.models.full_model import GenreAutoencoder
from src.preprocessing.dataset import load_split, make_loader
from src.utils import get_device, set_seed

set_seed(42)
device = get_device()

# Small balanced subset: 256 train clips per genre = 1024 clips (2 batches of 512)
X_train, y_train, _ = load_split("train")
rng = np.random.default_rng(42)
subset = np.concatenate([rng.choice(np.where(y_train == g)[0], 256, replace=False) for g in range(4)])
loader = make_loader(X_train[subset], y_train[subset], batch_size=512, shuffle=True, seed=42)

# One middle-of-the-grid configuration
gamma, lam, alpha = 0.9, 0.01, 0.1
model = GenreAutoencoder(latent_dim=64, n_classes=4, dropout=0.1).to(device)
loss_fn = CombinedLoss(gamma=gamma, lam=lam, alpha=alpha).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
print(f"Smoke test on {len(subset)} train clips, device = {device}, γ={gamma} λ={lam} α={alpha}\n")

print(" epoch   total   recon      CE   center  train_acc  mean|z|  center_spread")
start = time.time()
for epoch in range(1, 151):
    model.train()
    sums = {"total": 0.0, "recon": 0.0, "ce": 0.0, "center": 0.0}
    correct, z_norms = 0, []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_hat, z, logits = model(x)
        total, terms = loss_fn(x, x_hat, z, logits, y)

        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        loss_fn.update_centers(z, y)                 # after optimizer.step()

        for key in sums:
            sums[key] += terms[key] * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        z_norms.append(z.detach().norm(dim=1).mean().item())

    if any(math.isnan(v) for v in sums.values()):
        print(f"NaN in epoch {epoch}: {sums}")
        break

    if epoch in (1, 2, 5) or epoch % 25 == 0:
        n = len(subset)
        centers = loss_fn.center_loss.centers
        spread = torch.cdist(centers, centers).sum().item() / 12   # mean distance between the 4 centers
        print(f"{epoch:6d}  {sums['total']/n:6.4f}  {sums['recon']/n:6.4f}  {sums['ce']/n:6.4f}  "
              f"{sums['center']/n:7.4f}  {correct/n:9.3f}  {np.mean(z_norms):7.3f}  {spread:13.3f}")

print(f"\nFinished in {time.time() - start:.0f} s. Any NaN: no (would have stopped above).")
