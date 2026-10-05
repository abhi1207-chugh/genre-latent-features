"""Stage 9: evaluate the frozen 64-D embeddings of the dense vanilla AE (validation only).

Compared with the raw 500-D input and PCA-64 using the same three metrics.

Run from the project root:
    .venv/bin/python experiments/stage09_eval_dense_ae_embeddings.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
from sklearn.decomposition import PCA

from src.config import PROJECT_ROOT, RESULTS_DIR
from src.evaluation.embeddings import evaluate_embeddings, extract_embeddings
from src.models.dense_autoencoder import DenseAutoencoder
from src.preprocessing.dataset import load_split
from src.utils import get_device, set_seed
from src.visualization.plots import plot_pca_panels

set_seed(42)
device = get_device()

X_train, y_train, _ = load_split("train")
X_val, y_val, _ = load_split("val")

# Load the trained dense AE and keep only its encoder
model = DenseAutoencoder(input_dim=500, latent_dim=64, dropout=0.1)
model.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints" / "dense_ae.pt"))
Z_train = extract_embeddings(model.encoder, X_train, device)
Z_val = extract_embeddings(model.encoder, X_val, device)
print(f"Dense AE embeddings: train {Z_train.shape}, val {Z_val.shape}")

pca = PCA(n_components=64, random_state=42).fit(X_train)

representations = {
    "raw 500-D": (X_train, X_val),
    "PCA 64-D": (pca.transform(X_train), pca.transform(X_val)),
    "dense AE 64-D": (Z_train, Z_val),
}

rows = []
for name, (train_features, val_features) in representations.items():
    metrics = evaluate_embeddings(train_features, y_train, val_features, y_val)
    rows.append({"representation": name, **metrics})

table = pd.DataFrame(rows)
print("\nValidation embedding metrics (chance accuracy = 0.25):")
print(table.round(3).to_string(index=False))

table.round(4).to_csv(RESULTS_DIR / "dense_ae_embeddings_val.csv", index=False)

figure_path = RESULTS_DIR / "figures" / "pca_raw_vs_dense_ae_val.png"
plot_pca_panels([("Raw 500-D input", X_val), ("Dense AE 64-D embedding", Z_val)],
                y_val, figure_path, "Validation clips, 2-D PCA view")
print(f"\nSaved results/dense_ae_embeddings_val.csv and {figure_path.relative_to(PROJECT_ROOT)}")
