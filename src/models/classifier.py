"""Genre classifier head on the 64-D embedding: 64 -> 32 -> 16 -> 4 logits (paper's sizes)."""
import torch.nn as nn


class GenreClassifier(nn.Module):
    def __init__(self, latent_dim=64, n_classes=4, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, n_classes),        # raw logits: NO softmax here
        )

    def forward(self, z):                    # z: (batch, 64)
        return self.layers(z)                # logits: (batch, 4)
