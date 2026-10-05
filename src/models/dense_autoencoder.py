"""The paper's dense (fully connected) autoencoder: 500 -> 256 -> 192 -> 128 -> 64 and back."""
import torch.nn as nn


def dense_block(n_in, n_out, dropout):
    """Linear layer -> ReLU -> Dropout."""
    return nn.Sequential(nn.Linear(n_in, n_out), nn.ReLU(), nn.Dropout(dropout))


class DenseEncoder(nn.Module):
    def __init__(self, input_dim=500, latent_dim=64, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            dense_block(input_dim, 256, dropout),
            dense_block(256, 192, dropout),
            dense_block(192, 128, dropout),
            nn.Linear(128, latent_dim),          # no activation: z can be any real number
        )

    def forward(self, x):                        # x: (batch, 500)
        return self.layers(x)                    # z: (batch, 64)


class DenseDecoder(nn.Module):
    def __init__(self, latent_dim=64, output_dim=500, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            dense_block(latent_dim, 128, dropout),
            dense_block(128, 192, dropout),
            dense_block(192, 256, dropout),
            nn.Linear(256, output_dim),          # no activation: standardized inputs can be negative
        )

    def forward(self, z):                        # z: (batch, 64)
        return self.layers(z)                    # x_hat: (batch, 500)


class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim=500, latent_dim=64, dropout=0.1):
        super().__init__()
        self.encoder = DenseEncoder(input_dim, latent_dim, dropout)
        self.decoder = DenseDecoder(latent_dim, input_dim, dropout)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
