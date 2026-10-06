"""Stage 23: create the remaining report figures (no training, nothing tuned).

  figures/architecture.png           model diagram with shapes and parameter counts
  figures/loss_curves_selected.png   total / recon / CE / center, train vs val, selected Model B
  figures/reconstructions_test.png   one TEST clip per genre: input vs Model B vs CNN AE
  figures/model_comparison_test.png  clip and track accuracy of all models on TEST

Run from the project root:
    .venv/bin/python experiments/stage23_report_figures.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from src.config import PROJECT_ROOT, RESULTS_DIR
from src.evaluation.run_evaluation import load_best_model
from src.models.cnn_autoencoder import CNNAutoencoder
from src.preprocessing.dataset import load_split
from src.visualization.report_figures import (plot_architecture, plot_loss_curves,
                                              plot_model_comparison, plot_reconstructions)

FIGURES = RESULTS_DIR / "figures"
selected = json.loads((RESULTS_DIR / "selected_model.json").read_text())

# 1. Architecture
plot_architecture(FIGURES / "architecture.png")

# 2. Loss curves of the selected model
history = pd.read_csv(RESULTS_DIR / "runs" / selected["run"] / "history.csv")
plot_loss_curves(history, selected["best_epoch"], FIGURES / "loss_curves_selected.png",
                 f"Selected Model B (γ {selected['gamma']}, λ {selected['lambda']}, α {selected['alpha']}): "
                 f"loss per epoch")

# 3. Reconstruction examples: clip 10 of the first TEST track of each genre
X_test, y_test, tracks_test = load_split("test")
model_b = load_best_model(selected["run"], "cpu")
cnn_ae = CNNAutoencoder()
cnn_ae.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints" / "cnn_ae.pt"))
cnn_ae.eval()
examples = []
with torch.no_grad():
    for label in range(4):
        first_track = tracks_test[y_test == label][0]
        index = (tracks_test == first_track).nonzero()[0][10]
        x = torch.from_numpy(X_test[index:index + 1])
        examples.append((label, first_track, X_test[index],
                         {"Model B (selected)": model_b(x)[0][0].numpy(),
                          "CNN AE (reconstruction only)": cnn_ae(x)[0][0].numpy()}))
plot_reconstructions(examples, FIGURES / "reconstructions_test.png",
                     "Reconstruction of one TEST clip per genre (standardized waveform after pooling)")

# 4. Model comparison on TEST (from Stage 20)
comparison = pd.read_csv(RESULTS_DIR / "test_comparison.csv")
plot_model_comparison(comparison, FIGURES / "model_comparison_test.png",
                      "All models on the TEST split (A/B: mean ± std over 3 seeds)")

for name in ["architecture", "loss_curves_selected", "reconstructions_test", "model_comparison_test"]:
    print(f"Saved results/figures/{name}.png")
