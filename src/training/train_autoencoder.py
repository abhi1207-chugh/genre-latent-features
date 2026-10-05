"""Train a plain autoencoder (reconstruction loss only) with early stopping.

Works for any model whose forward(x) returns (x_hat, z): dense AE now, CNN AE later.
"""
import pandas as pd
import torch
import torch.nn as nn


@torch.no_grad()
def reconstruction_error(model, loader, device):
    """Mean squared error per input value, averaged over all clips in the loader."""
    model.eval()                                   # dropout off
    total_error, total_values = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        x_hat, _ = model(x)
        total_error += ((x_hat - x) ** 2).sum().item()
        total_values += x.numel()
    return total_error / total_values


def train_autoencoder(model, train_loader, val_loader, device, checkpoint_path,
                      lr=1e-4, max_epochs=500, patience=20):
    """Adam on MSE. Keeps the epoch with the lowest validation error.

    Saves the best weights to checkpoint_path and returns the history table.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()                             # mean over batch and 500 values

    history = []
    best_val, best_epoch = float("inf"), 0
    for epoch in range(1, max_epochs + 1):
        model.train()                              # dropout on
        for x, _ in train_loader:                  # labels are ignored
            x = x.to(device)
            x_hat, _ = model(x)
            loss = mse(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_error = reconstruction_error(model, train_loader, device)
        val_error = reconstruction_error(model, val_loader, device)
        history.append({"epoch": epoch, "train_recon": train_error, "val_recon": val_error})

        if val_error < best_val:
            best_val, best_epoch = val_error, epoch
            torch.save(model.state_dict(), checkpoint_path)
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(torch.load(checkpoint_path))   # restore the best epoch
    return pd.DataFrame(history), best_epoch, best_val
