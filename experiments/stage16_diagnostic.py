"""Stage 16 diagnostic: continue the single run (γ=0.9, λ=0.01, α=0.1) from its
epoch-103 checkpoint to epoch 300 WITHOUT early stopping.

Question: does reconstruction leave the plateau, when, and what happens to
val total loss and 5-NN accuracy? The original run's files are only read.

Run from the project root:
    .venv/bin/python -u experiments/stage16_diagnostic.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from src.config import PROJECT_ROOT, RESULTS_DIR
from src.evaluation.embeddings import extract_embeddings, knn_probe
from src.losses.combined_loss import CombinedLoss
from src.models.full_model import GenreAutoencoder
from src.preprocessing.dataset import GPUBatches, load_split
from src.training.train_model import evaluate, train_epoch
from src.utils import get_device

device = get_device()
X_train, y_train, _ = load_split("train")
X_val, y_val, _ = load_split("val")
train_batches = GPUBatches(X_train, y_train, device, 512, shuffle=True, seed=42)
val_batches = GPUBatches(X_val, y_val, device, 512)

model = GenreAutoencoder().to(device)
loss_fn = CombinedLoss(gamma=0.9, lam=0.01, alpha=0.1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

state = torch.load(PROJECT_ROOT / "checkpoints/runs/stage16_minepochs100_g0.9_l0.01_a0.1/last.pt", weights_only=False)
model.load_state_dict(state["model"])
loss_fn.load_state_dict(state["loss_fn"])
optimizer.load_state_dict(state["optimizer"])
train_batches.generator.set_state(state["batch_rng"])
torch.set_rng_state(state["torch_rng"])
rows = [{k: r[k] for k in ["epoch", "train_recon", "val_recon", "val_ce", "val_total", "val_accuracy"]}
        for r in state["history"]]
print(f"Resumed after epoch {state['epoch']}\n")


def knn_now():
    return knn_probe(extract_embeddings(model.encoder, X_train, device), y_train,
                     extract_embeddings(model.encoder, X_val, device), y_val)


print(" epoch  train_recon  val_recon  val_CE  val_total  val_acc  val_5NN")
for epoch in range(state["epoch"] + 1, 301):
    train = train_epoch(model, loss_fn, optimizer, train_batches)
    val = evaluate(model, loss_fn, val_batches)
    row = {"epoch": epoch, "train_recon": train["recon"], "val_recon": val["recon"],
           "val_ce": val["ce"], "val_total": val["total"], "val_accuracy": val["accuracy"]}
    if epoch % 25 == 0:
        row["val_knn5"] = knn_now()
        print(f"{epoch:6d}  {train['recon']:11.4f}  {val['recon']:9.4f}  {val['ce']:6.4f}  "
              f"{val['total']:9.4f}  {val['accuracy']:7.3f}  {row['val_knn5']:7.3f}", flush=True)
    rows.append(row)

history = pd.DataFrame(rows)
best = history.loc[history["val_total"].idxmin()]
print(f"\nLowest val total over epochs 1-300: epoch {int(best['epoch'])} "
      f"(val total {best['val_total']:.4f}, val recon {best['val_recon']:.4f}, val CE {best['val_ce']:.4f})")
below = history[history["val_recon"] < 1.10]
print("First epoch with val recon < 1.10:", int(below["epoch"].iloc[0]) if len(below) else "never")
history.to_csv(RESULTS_DIR / "runs" / "stage16_diagnostic_to300.csv", index=False)
print("Saved results/runs/stage16_diagnostic_to300.csv")
