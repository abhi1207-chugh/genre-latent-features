"""Stage 20: final evaluation on the held-out TEST split (first and only use of test so far).

Nothing is tuned here: every model and hyperparameter was fixed on validation earlier.
  Part 1  Model A (λ = 0) and Model B (λ = 0.1, the selected config), 3 seeds each
  Part 2  baselines with their validation-chosen settings (refit on train only)
  Part 3  confusion matrices (clip and track level) for B seed 42 (selected) and A seed 42

Run from the project root:
    .venv/bin/python experiments/stage20_final_evaluation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import PROJECT_ROOT, RESULTS_DIR
from src.evaluation.baselines import two_layer_nn_baseline
from src.evaluation.embeddings import extract_embeddings
from src.evaluation.metrics import clip_and_track_metrics, track_level_predictions
from src.evaluation.run_evaluation import evaluate_run, load_best_model, predict
from src.models.cnn_autoencoder import CNNAutoencoder
from src.models.dense_autoencoder import DenseAutoencoder
from src.preprocessing.dataset import load_split
from src.utils import get_device, set_seed
from src.visualization.plots import plot_confusion_panels

set_seed(42)
device = get_device()
X_train, y_train, _ = load_split("train")
X_val, y_val, _ = load_split("val")          # only for the 2-layer NN's early stopping
X_test, y_test, tracks_test = load_split("test")
print(f"TEST split: {len(X_test)} clips from {len(set(tracks_test))} tracks "
      f"(used for the first time in this project)\n")

# ---------------- Part 1: Model A vs Model B, 3 seeds each ----------------
run = lambda model, lam, seed: f"ablation_{model}_g0.9_l{lam}_a0.1_s{seed}"
rows = []
for model, lam in [("A", 0.0), ("B", 0.1)]:
    for seed in [42, 43, 44]:
        metrics, _ = evaluate_run(run(model, lam, seed), X_train, y_train,
                                  X_test, y_test, tracks_test, device)
        rows.append({"model": model, "seed": seed, **metrics})
per_run = pd.DataFrame(rows)
metrics = ["recon", "clip_accuracy", "clip_precision", "clip_recall", "clip_f1",
           "track_accuracy", "track_precision", "track_recall", "track_f1",
           "knn5_accuracy", "linear_probe_accuracy", "silhouette"]
print("Part 1 - per run (TEST):")
print(per_run[["model", "seed", "recon", "clip_accuracy", "clip_f1", "track_accuracy", "track_f1",
               "knn5_accuracy", "linear_probe_accuracy", "silhouette"]].round(4).to_string(index=False))

summary = per_run.groupby("model")[metrics].agg(["mean", "std"]).T.unstack()
summary.columns = [f"{m} {s}" for m, s in summary.columns]
summary["B - A"] = summary["B mean"] - summary["A mean"]
print("\nPart 1 - mean ± std over 3 seeds (TEST):")
print(summary.round(4).to_string())

# ---------------- Part 2: baselines (settings fixed on validation in Stages 7 and 9) ----------------
def clip_track(probs):
    return clip_and_track_metrics(probs, y_test, tracks_test)


baseline_rows = []
def add(name, probs):
    baseline_rows.append({"model": name, **clip_track(probs)})


pca = PCA(n_components=64, random_state=42).fit(X_train)
P_train, P_test = pca.transform(X_train), pca.transform(X_test)
knn = lambda A, B: KNeighborsClassifier(n_neighbors=5).fit(A, y_train).predict_proba(B)
svm = lambda A, B: CalibratedClassifierCV(SVC(kernel="rbf", C=10, gamma="scale"),
                                          ensemble=False).fit(A, y_train).predict_proba(B)
logreg = lambda A, B: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)
                                    ).fit(A, y_train).predict_proba(B)

add("raw 500-D + 5-NN", knn(X_train, X_test))
add("raw 500-D + SVM (C=10)", svm(X_train, X_test))
add("PCA 64-D + 5-NN", knn(P_train, P_test))
add("PCA 64-D + SVM (C=10)", svm(P_train, P_test))
nn_model, _, _ = two_layer_nn_baseline(X_train, y_train, X_val, y_val)
add("raw 500-D + 2-layer NN (paper)", nn_model.predict_proba(X_test))

for name, model_class, checkpoint in [("dense AE", DenseAutoencoder, "dense_ae.pt"),
                                      ("CNN AE", CNNAutoencoder, "cnn_ae.pt")]:
    ae = model_class()
    ae.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints" / checkpoint))
    Z_train = extract_embeddings(ae.encoder, X_train, device)
    Z_test = extract_embeddings(ae.encoder, X_test, device)
    add(f"{name} 64-D + 5-NN", knn(Z_train, Z_test))
    add(f"{name} 64-D + linear probe", logreg(Z_train, Z_test))

cols = ["model", "clip_accuracy", "clip_f1", "track_accuracy", "track_f1"]

# Proposed models in the same table: classifier head, mean ± std over seeds, plus selected B seed 42
for model, label in [("A", "Model A (recon+CE) classifier"), ("B", "Model B (+center) classifier")]:
    part = per_run[per_run["model"] == model]
    baseline_rows.append({"model": f"{label}, mean of 3 seeds",
                          **{c: part[c].mean() for c in cols[1:]},
                          **{f"{c}_std": part[c].std() for c in cols[1:]}})
selected = per_run[(per_run["model"] == "B") & (per_run["seed"] == 42)].iloc[0]
baseline_rows.append({"model": "SELECTED: Model B seed 42 classifier", **{c: selected[c] for c in cols[1:]}})
comparison = pd.DataFrame(baseline_rows)
print("\nPart 2 - all models on TEST (clip = 1-s clips, track = averaged clip probabilities):")
print(comparison[cols].round(4).to_string(index=False))

# ---------------- Part 3: confusion matrices ----------------
panels = []
for model, lam, label in [("B", 0.1, "Model B seed 42 (selected)"), ("A", 0.0, "Model A seed 42")]:
    net = load_best_model(run(model, lam, 42), device)
    probs, _, _ = predict(net, X_test, device)
    clip_cm = confusion_matrix(y_test, probs.argmax(axis=1), labels=range(4))
    track_true, track_pred = track_level_predictions(probs, tracks_test, y_test)
    track_cm = confusion_matrix(track_true, track_pred, labels=range(4))
    panels += [(f"{label}\nclip level ({len(y_test)} clips)", clip_cm),
               (f"{label}\ntrack level ({len(track_true)} tracks)", track_cm)]
    print(f"\n{label} confusion (rows = true Classical/Country/Disco/Hip-Hop):")
    print(f"  clip:\n{clip_cm}\n  track:\n{track_cm}")

plot_confusion_panels(panels[:2], RESULTS_DIR / "figures" / "confusion_selected_B_test.png",
                      "TEST confusion matrices — selected Model B (γ 0.9, λ 0.1, α 0.1, seed 42)")
plot_confusion_panels(panels[2:], RESULTS_DIR / "figures" / "confusion_A_test.png",
                      "TEST confusion matrices — Model A (γ 0.9, λ 0, seed 42)")

per_run.round(4).to_csv(RESULTS_DIR / "test_ablation_per_run.csv", index=False)
summary.round(4).to_csv(RESULTS_DIR / "test_ablation_summary.csv")
comparison.round(4).to_csv(RESULTS_DIR / "test_comparison.csv", index=False)
print("\nSaved results/test_ablation_per_run.csv, test_ablation_summary.csv, test_comparison.csv, "
      "figures/confusion_selected_B_test.png, figures/confusion_A_test.png")
