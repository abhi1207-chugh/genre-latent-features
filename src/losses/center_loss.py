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
        """Move each genre's center towards that genre's embeddings in this batch.

        All genres at once (no Python loop, so the GPU never has to wait for the CPU):
            sum_j (c_j - z_i) = n_j * c_j - (sum of z_i of genre j)
        A genre absent from the batch has n_j = 0 and sum 0, so delta = 0: unchanged.
        """
        z = z.detach()
        n_classes = len(self.centers)
        counts = torch.bincount(y, minlength=n_classes).float().unsqueeze(1)   # (K, 1) = n_j
        z_sums = torch.zeros_like(self.centers).index_add_(0, y, z)            # (K, D)
        delta = (counts * self.centers - z_sums) / (1 + counts)
        self.centers -= self.alpha * delta
