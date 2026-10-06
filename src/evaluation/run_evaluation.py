"""Evaluate one trained run (its best.pt) on an evaluation split.

Used with the validation split in Stage 19 and with the test split in Stage 20.
"""
import numpy as np
import torch
from sklearn.metrics import silhouette_score

from src.config import PROJECT_ROOT
from src.evaluation.embeddings import extract_embeddings, knn_probe, linear_probe
from src.evaluation.metrics import clip_and_track_metrics
from src.models.full_model import GenreAutoencoder


def load_best_model(run_name, device):
    model = GenreAutoencoder().to(device)
    state = torch.load(PROJECT_ROOT / "checkpoints" / "runs" / run_name / "best.pt", weights_only=False)
    model.load_state_dict(state["model"])
    return model.eval()


@torch.no_grad()
def predict(model, X, device, batch_size=512):
    """Clip probabilities (softmax only here, at inference), embeddings and reconstructions."""
    probs, embeddings, recon_sq_error = [], [], 0.0
    for start in range(0, len(X), batch_size):
        x = torch.from_numpy(X[start:start + batch_size]).to(device)
        x_hat, z, logits = model(x)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        embeddings.append(z.cpu().numpy())
        recon_sq_error += ((x_hat - x) ** 2).sum().item()
    return np.concatenate(probs), np.concatenate(embeddings), recon_sq_error / X.size


def evaluate_run(run_name, X_train, y_train, X_eval, y_eval, tracks_eval, device):
    """All ablation / final metrics for one run on one evaluation split."""
    model = load_best_model(run_name, device)
    eval_probs, Z_eval, recon = predict(model, X_eval, device)
    Z_train = extract_embeddings(model.encoder, X_train, device)

    return {
        "run": run_name,
        "recon": recon,
        **clip_and_track_metrics(eval_probs, y_eval, tracks_eval),     # classifier head
        "knn5_accuracy": knn_probe(Z_train, y_train, Z_eval, y_eval),
        "linear_probe_accuracy": linear_probe(Z_train, y_train, Z_eval, y_eval),
        "silhouette": float(silhouette_score(Z_eval, y_eval)),
        "mean_z_norm": float(np.linalg.norm(Z_eval, axis=1).mean()),
    }, Z_eval
