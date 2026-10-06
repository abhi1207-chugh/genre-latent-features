"""1D CNN encoder and decoder (design approved in Stage 10).

Encoder: (batch, 500) -> 4 conv blocks -> (batch, 64, 31) -> flatten -> Linear -> z (batch, 64)
Decoder: z -> Linear -> (batch, 64, 31) -> upsample + conv back to (batch, 500)
"""
import torch.nn as nn

# Lengths after each pooling step: 500 -> 250 -> 125 -> 62 -> 31
FINAL_CHANNELS, FINAL_LENGTH = 64, 31


def conv_block(c_in, c_out, kernel):
    """Conv1d ('same' padding) -> BatchNorm -> ReLU -> MaxPool(2): length is halved."""
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel_size=kernel, padding=kernel // 2),
        nn.BatchNorm1d(c_out),
        nn.ReLU(),
        nn.MaxPool1d(2),
    )


def upsample_block(c_in, c_out, kernel, out_length):
    """Upsample to an exact length -> Conv1d -> BatchNorm -> ReLU."""
    return nn.Sequential(
        nn.Upsample(size=out_length, mode="linear"),
        nn.Conv1d(c_in, c_out, kernel_size=kernel, padding=kernel // 2),
        nn.BatchNorm1d(c_out),
        nn.ReLU(),
    )


class CNNEncoder(nn.Module):
    def __init__(self, latent_dim=64, dropout=0.1):
        super().__init__()
        self.blocks = nn.Sequential(
            conv_block(1, 16, kernel=9),     # (16, 250)
            conv_block(16, 32, kernel=7),    # (32, 125)
            conv_block(32, 64, kernel=5),    # (64, 62)
            conv_block(64, 64, kernel=3),    # (64, 31)
        )
        self.head = nn.Sequential(
            nn.Flatten(),                                     # (1984,)
            nn.Dropout(dropout),
            nn.Linear(FINAL_CHANNELS * FINAL_LENGTH, latent_dim),  # z, no activation
        )

    def forward(self, x):                    # x: (batch, 500)
        x = x.unsqueeze(1)                   # (batch, 1, 500): one input channel
        return self.head(self.blocks(x))     # (batch, 64)


class CNNDecoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.expand = nn.Linear(latent_dim, FINAL_CHANNELS * FINAL_LENGTH)
        self.blocks = nn.Sequential(
            upsample_block(64, 64, kernel=3, out_length=62),
            upsample_block(64, 32, kernel=5, out_length=125),
            upsample_block(32, 16, kernel=7, out_length=250),
            nn.Upsample(size=500, mode="linear"),
            nn.Conv1d(16, 1, kernel_size=9, padding=4),      # no activation
        )

    def forward(self, z):                    # z: (batch, 64)
        x = self.expand(z).view(-1, FINAL_CHANNELS, FINAL_LENGTH)   # (batch, 64, 31)
        return self.blocks(x).squeeze(1)     # (batch, 500)


class CNNAutoencoder(nn.Module):
    def __init__(self, latent_dim=64, dropout=0.1):
        super().__init__()
        self.encoder = CNNEncoder(latent_dim, dropout)
        self.decoder = CNNDecoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
