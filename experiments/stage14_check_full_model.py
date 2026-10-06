"""Stage 14: check the full model and the combined loss on one dummy batch (no training).

Run from the project root:
    .venv/bin/python experiments/stage14_check_full_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.losses.combined_loss import CombinedLoss
from src.models.full_model import GenreAutoencoder
from src.utils import set_seed

set_seed(42)
model = GenreAutoencoder(latent_dim=64, n_classes=4, dropout=0.1)
count = lambda m: sum(p.numel() for p in m.parameters())
print(f"Parameters: encoder {count(model.encoder):,} + decoder {count(model.decoder):,} "
      f"+ classifier {count(model.classifier):,} = {count(model):,}")

x = torch.randn(16, 500)
y = torch.arange(4).repeat(4)                     # 4 clips of each genre

# 1. Forward shapes
x_hat, z, logits = model(x)
print(f"\n1. x {tuple(x.shape)} -> z {tuple(z.shape)} -> x_hat {tuple(x_hat.shape)}, logits {tuple(logits.shape)}")

# 2. Total = formula, each term logged
gamma, lam, alpha = 0.9, 0.01, 0.1
loss_fn = CombinedLoss(gamma=gamma, lam=lam, alpha=alpha)
total, terms = loss_fn(x, x_hat, z, logits, y)
by_hand = gamma * terms["recon"] + (1 - gamma) * terms["ce"] + lam * terms["center"]
print(f"\n2. Terms: recon {terms['recon']:.4f}, CE {terms['ce']:.4f}, center {terms['center']:.4f}")
print(f"   total {terms['total']:.4f} = 0.9·recon + 0.1·CE + 0.01·center = {by_hand:.4f}")

# 3. One backward pass reaches all three parts; centers are not optimizer parameters
total.backward()
has_grad = lambda m: all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
print(f"\n3. Gradients reach encoder: {has_grad(model.encoder)}, decoder: {has_grad(model.decoder)}, "
      f"classifier: {has_grad(model.classifier)}")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
n_optimized = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
print(f"   Adam optimizes {n_optimized:,} values (= model parameters, centers excluded: "
      f"{n_optimized == count(model)})")


def encoder_gradient(gamma, lam, use_center_term=True):
    """Gradient on the first encoder conv weights for one loss setting."""
    model.zero_grad()
    x_hat, z, logits = model(x)
    fn = CombinedLoss(gamma=gamma, lam=lam, alpha=0.1)
    fn.center_loss.centers[:] = 1.0               # non-zero centers so the term matters
    total, terms = fn(x, x_hat, z, logits, y)
    if not use_center_term:
        total = total - lam * fn.center_loss(z, y)
    total.backward()
    return model.encoder.blocks[0][0].weight.grad.clone()


model.eval()                                      # dropout off, so repeated passes are identical
# 4. λ = 0 (Model A): center term has no effect on training
g_lambda0 = encoder_gradient(gamma=0.9, lam=0.0)
g_no_center = encoder_gradient(gamma=0.9, lam=0.0, use_center_term=False)
print(f"\n4. λ = 0 gives the same gradients as having no center term: {torch.allclose(g_lambda0, g_no_center)}")

# 5. λ > 0 changes the encoder gradient (the center loss really pushes z)
g_lambda01 = encoder_gradient(gamma=0.9, lam=0.1)
print(f"5. λ = 0.1 changes the encoder gradient: {not torch.allclose(g_lambda0, g_lambda01)}")

# 6. γ = 1, λ = 0 is exactly the plain autoencoder loss
x_hat, z, logits = model(x)
total, _ = CombinedLoss(gamma=1.0, lam=0.0, alpha=0.1)(x, x_hat, z, logits, y)
print(f"6. γ = 1, λ = 0 equals plain MSE: {torch.isclose(total, torch.nn.functional.mse_loss(x_hat, x)).item()}")

# 7. Centers move only through update_centers (α), not through the loss
before = loss_fn.center_loss.centers.clone()
loss_fn.update_centers(z.detach(), y)
print(f"7. update_centers moved the centers: {not torch.equal(before, loss_fn.center_loss.centers)}")
