"""Stage 7: raw-feature baselines, evaluated on the VALIDATION split only.

  raw 500-D + 5-NN      raw 500-D + SVM
  PCA 64-D  + 5-NN      PCA 64-D  + SVM
  raw 500-D + 2-layer NN (paper's baseline)

Run from the project root:
    .venv/bin/python experiments/stage07_raw_baselines.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.decomposition import PCA

from src.config import RESULTS_DIR
from src.evaluation.baselines import knn_baseline, svm_baseline, two_layer_nn_baseline
from src.evaluation.metrics import clip_and_track_metrics
from src.preprocessing.dataset import load_split
from src.utils import set_seed

set_seed(42)

X_train, y_train, _ = load_split("train")
X_val, y_val, tracks_val = load_split("val")

# PCA fit on train only, 64 components (same size as our embedding)
pca = PCA(n_components=64, random_state=42).fit(X_train)
P_train, P_val = pca.transform(X_train), pca.transform(X_val)
print(f"PCA 64 components explain {pca.explained_variance_ratio_.sum():.1%} of train variance\n")

runs = [
    ("raw 500-D + 5-NN", lambda: knn_baseline(X_train, y_train, X_val)),
    ("raw 500-D + SVM", lambda: svm_baseline(X_train, y_train, X_val, y_val)),
    ("PCA 64-D + 5-NN", lambda: knn_baseline(P_train, y_train, P_val)),
    ("PCA 64-D + SVM", lambda: svm_baseline(P_train, y_train, P_val, y_val)),
    ("raw 500-D + 2-layer NN", lambda: two_layer_nn_baseline(X_train, y_train, X_val, y_val)),
]

rows = []
for name, run in runs:
    start = time.time()
    _, val_probs, info = run()
    metrics = clip_and_track_metrics(val_probs, y_val, tracks_val)
    rows.append({"baseline": name, **metrics, "seconds": time.time() - start, "info": str(info)})
    print(f"{name:24s} done in {time.time() - start:5.1f} s   {info}")

table = pd.DataFrame(rows)
columns = ["baseline", "clip_accuracy", "clip_f1", "track_accuracy", "track_f1"]
print("\nValidation results (1799 clips / 60 tracks; chance = 0.25):")
print(table[columns].round(3).to_string(index=False))

table.round(4).to_csv(RESULTS_DIR / "baselines_val.csv", index=False)
print("\nSaved results/baselines_val.csv")
