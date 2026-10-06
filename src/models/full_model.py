"""Full model: CNN encoder -> 64-D z -> (decoder -> 500-D reconstruction, classifier -> 4 logits).

The center loss is not part of the model; it lives in the combined loss (src/losses/combined_loss.py).
"""
import torch.nn as nn

from src.models.classifier import GenreClassifier
from src.models.cnn_autoencoder import CNNDecoder, CNNEncoder


class GenreAutoencoder(nn.Module):
    def __init__(self, latent_dim=64, n_classes=4, dropout=0.1):
        super().__init__()
        self.encoder = CNNEncoder(latent_dim, dropout)
        self.decoder = CNNDecoder(latent_dim)
        self.classifier = GenreClassifier(latent_dim, n_classes, dropout)

    def forward(self, x):                    # x: (batch, 500)
        z = self.encoder(x)                  # (batch, 64)   the genre embedding
        x_hat = self.decoder(z)              # (batch, 500)  reconstruction
        logits = self.classifier(z)          # (batch, 4)    raw logits
        return x_hat, z, logits
