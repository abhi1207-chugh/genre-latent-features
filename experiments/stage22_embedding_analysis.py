"""Stage 22: final embedding analysis on the TEST split (no training, nothing tuned).

Six representations, all judged the same way (fit probes on train, score on test):
  raw 500-D input | PCA 64-D | dense AE 64-D | CNN AE 64-D | Model A 64-D (λ=0) | Model B 64-D (λ=0.1, selected)
Metrics: 5-NN accuracy, linear-probe accuracy, silhouette (overall and per genre).
Figures: 2-D PCA and t-SNE views of raw, CNN AE, Model A, Model B.

Run from the project root:
    .venv/bin/python experiments/stage22_embedding_analysis.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

from src.config import PROJECT_ROOT, RESULTS_DIR
from src.evaluation.embeddings import evaluate_embeddings, extract_embeddings
from src.evaluation.run_evaluation import load_best_model
from src.models.cnn_autoencoder import CNNAutoencoder
from src.models.dense_autoencoder import DenseAutoencoder
from src.preprocessing.dataset import load_split
from src.utils import get_device, set_seed
from src.visualization.plots import GENRE_NAMES, plot_2d_panels

set_seed(42)
device = get_device()
X_train, y_train, _ = load_split("train")
X_test, y_test, _ = load_split("test")

representations = {"raw 500-D": (X_train, X_test)}
pca = PCA(n_components=64, random_state=42).fit(X_train)
representations["PCA 64-D"] = (pca.transform(X_train), pca.transform(X_test))

for name, model_class, checkpoint in [("dense AE 64-D", DenseAutoencoder, "dense_ae.pt"),
                                      ("CNN AE 64-D", CNNAutoencoder, "cnn_ae.pt")]:
    ae = model_class()
    ae.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints" / checkpoint))
    representations[name] = (extract_embeddings(ae.encoder, X_train, device),
                             extract_embeddings(ae.encoder, X_test, device))

for name, run in [("Model A 64-D (λ=0)", "ablation_A_g0.9_l0.0_a0.1_s42"),
                  ("Model B 64-D (λ=0.1, selected)", "ablation_B_g0.9_l0.1_a0.1_s42")]:
    encoder = load_best_model(run, device).encoder
    representations[name] = (extract_embeddings(encoder, X_train, device),
                             extract_embeddings(encoder, X_test, device))

rows = []
for name, (Z_train, Z_test) in representations.items():
    row = {"representation": name, **evaluate_embeddings(Z_train, y_train, Z_test, y_test)}
    per_clip = silhouette_samples(Z_test, y_test)
    for label, genre in enumerate(GENRE_NAMES):
        row[f"silhouette {genre}"] = per_clip[y_test == label].mean()
    rows.append(row)
table = pd.DataFrame(rows)

print("Embedding quality on TEST (1797 clips; probes fit on train; seed 42 for A and B):")
print(table[["representation", "knn5_accuracy", "linear_probe_accuracy", "silhouette"]]
      .round(4).to_string(index=False))
print("\nSilhouette per genre (TEST) - how tight each genre is vs. its nearest other genre:")
print(table[["representation"] + [f"silhouette {g}" for g in GENRE_NAMES]].round(4).to_string(index=False))
table.round(4).to_csv(RESULTS_DIR / "embeddings_test.csv", index=False)

shown = ["raw 500-D", "CNN AE 64-D", "Model A 64-D (λ=0)", "Model B 64-D (λ=0.1, selected)"]
panels = [(name, representations[name][1]) for name in shown]
for method, label in [("pca", "PCA"), ("tsne", "t-SNE")]:
    path = RESULTS_DIR / "figures" / f"{method}_embeddings_test.png"
    plot_2d_panels(panels, y_test, path, f"TEST clips, 2-D {label} view", method=method, ncols=2)
    print(f"Saved {path.relative_to(PROJECT_ROOT)}")
print("Saved results/embeddings_test.csv")
