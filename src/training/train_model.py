"""Reusable training function for the full model (one configuration of γ, λ, α).

Used by the grid (Stage 17), the ablation (Stage 19) and the CV folds (Stage 21).

Per run:
  checkpoints/runs/<name>/last.pt   saved every 5 epochs -> an interrupted run resumes
  checkpoints/runs/<name>/best.pt   epoch with the lowest validation total loss
  results/runs/<name>/history.csv   every loss term, train and val, per epoch
  results/runs/<name>/summary.json  final validation metrics of the best epoch
A run whose summary.json exists is finished and is skipped.

Early stopping: stop when the val total loss has not improved for `patience`
epochs, but never before `min_epochs` = 300. Reason (Stage 16 diagnostic): in the
joint model the reconstruction loss sits on a plateau until ~epoch 124, while the
classifier already overfits; stopping on val total before that ends the run
before the decoder has learned anything.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from src.config import PROJECT_ROOT, RESULTS_DIR
from src.evaluation.embeddings import extract_embeddings, knn_probe
from src.losses.combined_loss import CombinedLoss
from src.models.full_model import GenreAutoencoder
from src.preprocessing.dataset import GPUBatches
from src.utils import set_seed

TERMS = ["total", "recon", "ce", "center"]
SAVE_LAST_EVERY = 5


def save_atomic(obj, path):
    """Write to a temporary file, then rename. An interruption during saving
    can then never leave a half-written (corrupted) checkpoint behind."""
    tmp_path = path.with_suffix(".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def train_epoch(model, loss_fn, optimizer, batches):
    """One pass over the training batches. Returns the running-average loss terms.

    Sums stay on the GPU; they are converted to Python floats once, at the end.
    """
    model.train()
    sums = {key: 0.0 for key in TERMS}
    for x, y in batches:
        x_hat, z, logits = model(x)
        total, terms = loss_fn(x, x_hat, z, logits, y)

        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        loss_fn.update_centers(z, y)                         # after optimizer.step()

        for key in TERMS:
            sums[key] = sums[key] + terms[key] * len(y)
    n = len(batches.y)
    return {key: float(value) / n for key, value in sums.items()}


@torch.no_grad()
def evaluate(model, loss_fn, batches):
    """Average loss terms and classifier accuracy over all batches, in eval mode."""
    model.eval()
    sums, correct = {key: 0.0 for key in TERMS}, 0
    for x, y in batches:
        x_hat, z, logits = model(x)
        _, terms = loss_fn(x, x_hat, z, logits, y)
        for key in TERMS:
            sums[key] = sums[key] + terms[key] * len(y)
        correct = correct + (logits.argmax(dim=1) == y).sum()
    n = len(batches.y)
    return {**{key: float(value) / n for key, value in sums.items()}, "accuracy": float(correct) / n}


def train_one_config(name, gamma, lam, alpha, X_train, y_train, X_val, y_val, device,
                     seed=42, lr=1e-4, batch_size=512, dropout=0.1,
                     max_epochs=500, min_epochs=300, patience=20, verbose=True):
    """Train one configuration (or resume / skip it). Returns the summary dict."""
    run_dir = PROJECT_ROOT / "checkpoints" / "runs" / name
    result_dir = RESULTS_DIR / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    if (result_dir / "summary.json").exists():
        if verbose:
            print(f"[{name}] already finished - skipped")
        return json.loads((result_dir / "summary.json").read_text())

    set_seed(seed)
    model = GenreAutoencoder(latent_dim=64, n_classes=4, dropout=dropout).to(device)
    loss_fn = CombinedLoss(gamma=gamma, lam=lam, alpha=alpha).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_batches = GPUBatches(X_train, y_train, device, batch_size, shuffle=True, seed=seed)
    val_batches = GPUBatches(X_val, y_val, device, batch_size)

    history, start_epoch, best_val, best_epoch = [], 1, float("inf"), 0
    if (run_dir / "last.pt").exists():                            # resume
        state = torch.load(run_dir / "last.pt", weights_only=False)
        model.load_state_dict(state["model"])
        loss_fn.load_state_dict(state["loss_fn"])                 # contains the centers
        optimizer.load_state_dict(state["optimizer"])
        train_batches.generator.set_state(state["batch_rng"])
        torch.set_rng_state(state["torch_rng"])
        history, best_val, best_epoch = state["history"], state["best_val"], state["best_epoch"]
        start_epoch = state["epoch"] + 1
        if verbose:
            print(f"[{name}] resuming from epoch {start_epoch} (best so far: epoch {best_epoch})")

    def save_last(epoch):
        save_atomic({"model": model.state_dict(), "loss_fn": loss_fn.state_dict(),
                     "optimizer": optimizer.state_dict(), "epoch": epoch,
                     "batch_rng": train_batches.generator.get_state(),
                     "torch_rng": torch.get_rng_state(),
                     "history": history, "best_val": best_val, "best_epoch": best_epoch},
                    run_dir / "last.pt")

    start_time = time.time()
    epoch = start_epoch - 1
    for epoch in range(start_epoch, max_epochs + 1):
        train = train_epoch(model, loss_fn, optimizer, train_batches)
        val = evaluate(model, loss_fn, val_batches)
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train.items()},
                        **{f"val_{k}": v for k, v in val.items()}})

        if val["total"] < best_val:
            best_val, best_epoch = val["total"], epoch
            save_atomic({"model": model.state_dict(), "loss_fn": loss_fn.state_dict()},
                        run_dir / "best.pt")

        stop = epoch >= min_epochs and epoch - best_epoch >= patience
        if epoch % SAVE_LAST_EVERY == 0 or stop or epoch == max_epochs:
            save_last(epoch)

        if verbose and (epoch % 25 == 0 or epoch == start_epoch or stop):
            print(f"[{name}] epoch {epoch:3d}  train total {train['total']:.4f}  "
                  f"val total {val['total']:.4f}  val recon {val['recon']:.4f}  "
                  f"val CE {val['ce']:.4f}  val acc {val['accuracy']:.3f}  (best epoch {best_epoch})",
                  flush=True)
        if stop:
            break

    pd.DataFrame(history).to_csv(result_dir / "history.csv", index=False)

    # Final validation metrics of the BEST epoch
    best = torch.load(run_dir / "best.pt", weights_only=False)
    model.load_state_dict(best["model"])
    loss_fn.load_state_dict(best["loss_fn"])
    val = evaluate(model, loss_fn, val_batches)
    Z_train = extract_embeddings(model.encoder, X_train, device)
    Z_val = extract_embeddings(model.encoder, X_val, device)

    summary = {
        "name": name, "gamma": gamma, "lambda": lam, "alpha": alpha, "seed": seed,
        "best_epoch": best_epoch, "epochs_run": len(history),
        "val_total": val["total"], "val_recon": val["recon"], "val_ce": val["ce"],
        "val_center": val["center"], "val_classifier_accuracy": val["accuracy"],
        "val_knn5_accuracy": knn_probe(Z_train, y_train, Z_val, y_val, k=5),
        "mean_z_norm_val": float(np.linalg.norm(Z_val, axis=1).mean()),
        "train_minutes_this_session": (time.time() - start_time) / 60,
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def train_with_fallback(*args, device, **kwargs):
    """Train on the chosen device; if MPS fails, continue on CPU from the last checkpoint."""
    try:
        return train_one_config(*args, device=device, **kwargs)
    except RuntimeError as error:
        if device.type != "mps":
            raise
        print(f"MPS error ({error}); continuing on CPU from the last checkpoint")
        return train_one_config(*args, device=torch.device("cpu"), **kwargs)
