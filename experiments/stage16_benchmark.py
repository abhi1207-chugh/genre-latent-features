"""Stage 16 (pre-step): check the runtime optimizations are correct, and measure them.

  Part 1  equivalence: vectorized center update == old loop version
  Part 2  speed per epoch (train + val): old pipeline vs new pipeline
          (mixed precision bf16 was also measured: 2.669 vs 2.355 s/epoch, i.e.
           slower, so it was removed from the code)
  Part 3  two training processes at the same time on the GPU (throughput)

Short runs only (15 epochs each). Run from the project root:
    .venv/bin/python -u experiments/stage16_benchmark.py
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.losses.combined_loss import CombinedLoss
from src.models.full_model import GenreAutoencoder
from src.preprocessing.dataset import GPUBatches, load_split, make_loader
from src.training.train_model import evaluate, train_epoch
from src.utils import get_device, set_seed

device = get_device()
N_EPOCHS = 15              # epoch 1 = warm-up, epochs 2..15 are timed
SCRATCH = Path("/private/tmp/claude-501/-Users-abhinav-Projects-genre-latent-features/"
               "dc70153e-c3ef-4caa-bbfe-ca88b1607b46/scratchpad/benchmark_ckpt.pt")


def old_update_centers(centers, alpha, z, y):
    """The previous loop version (Stage 13), kept here only for comparison."""
    for j in range(len(centers)):
        in_class = y == j
        n_j = int(in_class.sum())
        if n_j == 0:
            continue
        centers[j] -= alpha * (centers[j] - z[in_class]).sum(dim=0) / (1 + n_j)


def old_epoch(model, loss_fn, optimizer, train_loader, val_loader):
    """Previous pipeline: DataLoader, .item() every batch, loop center update, save every epoch."""
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x_hat, z, logits = model(x)
        total, terms = loss_fn(x, x_hat, z, logits, y)
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        with torch.no_grad():
            old_update_centers(loss_fn.center_loss.centers, loss_fn.center_loss.alpha, z.detach(), y)
        _ = {k: v.item() for k, v in terms.items()}          # old: 4 syncs per batch
    model.eval()
    val_total = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            x_hat, z, logits = model(x)
            _, terms = loss_fn(x, x_hat, z, logits, y)
            val_total += terms["total"].item() * len(y)
    return val_total / len(val_loader.dataset)


def time_run(mode, X_train, y_train, X_val, y_val):
    """Seconds per epoch (epochs 2..N) and val total loss after N epochs."""
    set_seed(42)
    model = GenreAutoencoder().to(device)
    loss_fn = CombinedLoss(gamma=0.9, lam=0.01, alpha=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if mode == "old":
        train_data = make_loader(X_train, y_train, 512, shuffle=True, seed=42)
        val_data = make_loader(X_val, y_val, 512)
    else:
        train_data = GPUBatches(X_train, y_train, device, 512, shuffle=True, seed=42)
        val_data = GPUBatches(X_val, y_val, device, 512)

    for epoch in range(1, N_EPOCHS + 1):
        if epoch == 2:
            start = time.time()
        if mode == "old":
            val_total = old_epoch(model, loss_fn, optimizer, train_data, val_data)
            torch.save({"model": model.state_dict(), "opt": optimizer.state_dict()}, SCRATCH)
        else:
            train_epoch(model, loss_fn, optimizer, train_data)
            val_total = evaluate(model, loss_fn, val_data)["total"]
            if epoch % 5 == 0:
                torch.save({"model": model.state_dict(), "opt": optimizer.state_dict()}, SCRATCH)
    seconds = (time.time() - start) / (N_EPOCHS - 1)
    return seconds, val_total


X_train, y_train, _ = load_split("train")
X_val, y_val, _ = load_split("val")

if "--worker" in sys.argv:                      # Part 3 child process
    seconds, _ = time_run("new fp32", X_train, y_train, X_val, y_val)
    print(f"{seconds:.3f}")
    sys.exit()

# ---- Part 1: equivalence of the vectorized center update ----
set_seed(0)
z = torch.randn(512, 64, device=device)
y = torch.randint(0, 3, (512,), device=device)          # genre 3 absent on purpose
new = CombinedLoss(gamma=0.9, lam=0.01, alpha=0.5).to(device)
new.center_loss.centers[:] = torch.randn(4, 64, device=device)
old_centers = new.center_loss.centers.clone()
new.update_centers(z, y)
old_update_centers(old_centers, 0.5, z, y)
print("Part 1 - equivalence")
print(f"  vectorized update == loop update: {torch.allclose(new.center_loss.centers, old_centers, atol=1e-6)}"
      f"  (max difference {(new.center_loss.centers - old_centers).abs().max().item():.2e})")
x = torch.randn(32, 500, device=device)
yb = torch.randint(0, 4, (32,), device=device)
x_hat, zb, logits = GenreAutoencoder().to(device).eval()(x)
total, terms = new(x, x_hat, zb, logits, yb)
by_hand = 0.9 * float(terms["recon"]) + 0.1 * float(terms["ce"]) + 0.01 * float(terms["center"])
print(f"  combined loss still equals formula: {abs(float(total.detach()) - by_hand) < 1e-5}")

# ---- Part 2: speed per epoch ----
print(f"\nPart 2 - seconds per epoch (train + val), {N_EPOCHS} epochs each, device = {device}")
results = {}
for mode in ["old", "new fp32"]:
    results[mode] = time_run(mode, X_train, y_train, X_val, y_val)
    print(f"  {mode:9s}: {results[mode][0]:.3f} s/epoch   val total after {N_EPOCHS} epochs = {results[mode][1]:.4f}",
          flush=True)
old_s, fp32_s = results["old"][0], results["new fp32"][0]
print(f"  speed-up new vs old: {old_s / fp32_s:.2f}x")

# ---- Part 3: two processes at once ----
print("\nPart 3 - two training processes at the same time (new fp32)")
workers = [subprocess.Popen([sys.executable, __file__, "--worker"], stdout=subprocess.PIPE, text=True)
           for _ in range(2)]
parallel = [float(w.communicate()[0].strip().splitlines()[-1]) for w in workers]
print(f"  each process: {parallel[0]:.3f} and {parallel[1]:.3f} s/epoch (alone: {fp32_s:.3f})")
print(f"  throughput gain with 2 processes: {2 * fp32_s / max(parallel):.2f}x")
