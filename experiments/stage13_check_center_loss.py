"""Stage 13: test the center loss module on small dummy data.

Uses 2-D embeddings so every number can be checked by hand.

Run from the project root:
    .venv/bin/python experiments/stage13_check_center_loss.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.losses.center_loss import CenterLoss
from src.utils import set_seed

set_seed(42)

# --- 1. Loss value by hand ---
center_loss = CenterLoss(n_classes=2, latent_dim=2, alpha=0.5)
center_loss.centers[:] = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
z = torch.tensor([[1.0, 0.0], [0.0, 2.0], [10.0, 13.0]], requires_grad=True)
y = torch.tensor([0, 0, 1])
loss = center_loss(z, y)
# squared distances: 1, 4, 9  ->  0.5 * mean(1, 4, 9) = 0.5 * 14/3
print(f"1. Loss = {loss.item():.4f}   by hand 0.5 * (1 + 4 + 9) / 3 = {0.5 * 14 / 3:.4f}")

# --- 2. Gradient w.r.t. z is (z - c) / batch_size ---
loss.backward()
expected = (z.detach() - center_loss.centers[y]) / 3
print(f"2. Gradient on z matches (z - c) / 3: {torch.allclose(z.grad, expected)}")
print(f"   gradient pulls z towards its center: {z.grad.numpy().round(3).tolist()}")

# --- 3. Centers are not trainable parameters (Adam must not update them) ---
print(f"3. Trainable parameters in CenterLoss: {len(list(center_loss.parameters()))}   "
      f"buffers: {[name for name, _ in center_loss.named_buffers()]}")

# --- 4. Update rule by hand (alpha = 0.5) ---
center_loss.update_centers(z, y)
# class 0: c=(0,0), z=(1,0),(0,2): delta = ((0-1)+(0-0), (0-0)+(0-2)) / (1+2) = (-1/3, -2/3)
#          c <- c - 0.5 * delta = (1/6, 1/3)
# class 1: c=(10,10), z=(10,13): delta = (0, -3) / 2 = (0, -1.5)  ->  c = (10, 10.75)
print(f"4. Centers after one update: {center_loss.centers.numpy().round(4).tolist()}")
print(f"   by hand:                  [[{1/6:.4f}, {1/3:.4f}], [10.0, 10.75]]")

# --- 5. A genre missing from the batch keeps its center ---
before = center_loss.centers[1].clone()
center_loss.update_centers(torch.tensor([[5.0, 5.0]]), torch.tensor([0]))
print(f"5. Genre 1 absent from batch -> center unchanged: {torch.equal(before, center_loss.centers[1])}")

# --- 6. Effect of alpha: how fast centers reach the true class mean ---
# 4 genres, 64-D, fixed embeddings around different means; centers start at 0.
print("6. Distance from center to true genre mean after k batches (64-D, 4 genres):")
true_means = torch.randn(4, 64) * 3
labels = torch.arange(4).repeat_interleave(128)                     # 128 clips per genre
for alpha in [0.01, 0.1, 0.5]:
    set_seed(0)
    cl = CenterLoss(n_classes=4, latent_dim=64, alpha=alpha)
    distances = []
    for step in range(1, 101):
        batch = torch.randint(0, len(labels), (512,))
        z_batch = true_means[labels[batch]] + torch.randn(512, 64)  # noisy embeddings
        cl.update_centers(z_batch, labels[batch])
        if step in (1, 10, 100):
            distances.append((cl.centers - true_means).norm(dim=1).mean().item())
    print(f"   alpha = {alpha:<4}  after 1 / 10 / 100 batches: "
          + " / ".join(f"{d:6.3f}" for d in distances))
print(f"   (starting distance = {true_means.norm(dim=1).mean():.3f})")

# --- 7. Works with the real sizes and on the device ---
from src.utils import get_device
device = get_device()
cl = CenterLoss(n_classes=4, latent_dim=64, alpha=0.5).to(device)
z = torch.randn(512, 64, device=device)
y = torch.randint(0, 4, (512,), device=device)
print(f"7. Real sizes on {device}: loss = {cl(z, y).item():.3f}, centers shape {tuple(cl.centers.shape)}, "
      f"centers device {cl.centers.device}")
