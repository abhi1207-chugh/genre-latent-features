"""Combined loss:  L = γ·L_recon + (1−γ)·L_CE + λ·L_center

  γ (gamma)  : reconstruction / classification trade-off
  λ (lambda) : weight of the center loss in the total (how hard z is pulled to its center)
  α (alpha)  : center learning rate - lives inside CenterLoss.update_centers()

λ = 0 gives Model A of the ablation (recon + CE only). The center loss is still
computed and logged then, but it has no effect on the gradients.
"""
import torch.nn as nn

from src.losses.center_loss import CenterLoss


class CombinedLoss(nn.Module):
    def __init__(self, gamma, lam, alpha, n_classes=4, latent_dim=64):
        super().__init__()
        self.gamma = gamma
        self.lam = lam                                    # "lambda" is a Python keyword
        self.mse = nn.MSELoss()                           # mean over batch and 500 values
        self.cross_entropy = nn.CrossEntropyLoss()        # takes RAW logits
        self.center_loss = CenterLoss(n_classes, latent_dim, alpha)

    def forward(self, x, x_hat, z, logits, y):
        recon = self.mse(x_hat, x)
        ce = self.cross_entropy(logits, y)
        center = self.center_loss(z, y)
        total = self.gamma * recon + (1 - self.gamma) * ce + self.lam * center

        # Each term logged separately (plain floats, so no graph is kept)
        terms = {"total": total.item(), "recon": recon.item(),
                 "ce": ce.item(), "center": center.item()}
        return total, terms

    def update_centers(self, z, y):
        """Call once per batch, AFTER optimizer.step()."""
        self.center_loss.update_centers(z, y)
