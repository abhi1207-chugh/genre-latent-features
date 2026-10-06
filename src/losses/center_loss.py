"""Center loss (Wen et al., 2016): pull each embedding towards the center of its genre.

    L_center = mean over batch of  1/2 * ||z_i - c_{y_i}||^2

Two different hyperparameters - never mix them up:
  lambda (λ) = how much L_center counts in the total loss.  Used in the COMBINED
               loss (Stage 14), not in this file. It controls how hard the
               ENCODER is pushed towards the centers.
  alpha  (α) = center learning rate. Used here in update_centers(). It controls
               how fast the CENTERS move towards the embeddings of their genre.

The centers are NOT trained by the optimizer (Adam). They are a buffer, updated
by the rule from the paper:
    delta_c_j = sum_{i in batch, y_i = j} (c_j - z_i) / (1 + n_j)
    c_j      <- c_j - alpha * delta_c_j
where n_j = number of clips of genre j in the batch.
"""
import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    def __init__(self, n_classes=4, latent_dim=64, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        # register_buffer: saved in the checkpoint and moved with .to(device),
        # but NOT returned by model.parameters(), so Adam never touches it.
        self.register_buffer("centers", torch.zeros(n_classes, latent_dim))

    def forward(self, z, y):
        """Center loss for a batch. Gradient flows into z only (centers are constants here)."""
        centers_of_batch = self.centers[y]                    # (batch, latent_dim)
        squared_distance = ((z - centers_of_batch) ** 2).sum(dim=1)
        return 0.5 * squared_distance.mean()

    @torch.no_grad()
    def update_centers(self, z, y):
        """Move each genre's center towards that genre's embeddings in this batch."""
        z = z.detach()
        for j in range(len(self.centers)):
            in_class = y == j
            n_j = int(in_class.sum())
            if n_j == 0:
                continue                                      # genre absent: center unchanged
            delta = (self.centers[j] - z[in_class]).sum(dim=0) / (1 + n_j)
            self.centers[j] -= self.alpha * delta
