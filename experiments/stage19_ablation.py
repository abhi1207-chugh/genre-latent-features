"""Stage 19: center-loss ablation, 3 seeds each, VALIDATION data only.

  Model A: γ = 0.9, λ = 0    (recon + CE)            seeds 42, 43, 44
  Model B: γ = 0.9, λ = 0.1  (recon + CE + center)   seeds 42, 43, 44   (= selected config 6)
  α = 0.1 for both (it has no effect on training when λ = 0). Everything else identical.

Steps, from the project root:
    .venv/bin/python experiments/stage19_ablation.py --prepare      # reuse config 6 as B, seed 42
    .venv/bin/python -u experiments/stage19_ablation.py --worker 0 > results/logs/ablation_worker0.log 2>&1 &
    .venv/bin/python -u experiments/stage19_ablation.py --worker 1 > results/logs/ablation_worker1.log 2>&1 &
    .venv/bin/python experiments/stage19_ablation.py --evaluate     # after both workers finish
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import PROJECT_ROOT, RESULTS_DIR

GAMMA, ALPHA = 0.9, 0.1
MODELS = {"A": 0.0, "B": 0.1}            # model -> λ
SEEDS = [42, 43, 44]


def run_name(model, seed):
    return f"ablation_{model}_g{GAMMA}_l{MODELS[model]}_a{ALPHA}_s{seed}"


# B seed 42 is grid config 6 (same γ, λ, α, seed and code) -> reused.
REUSED = {run_name("B", 42): "grid_g0.9_l0.1_a0.1"}
# Model B runs trained longest in the grid (~440 epochs), so they go first on each worker.
WORKER_RUNS = {0: [("B", 43), ("A", 42), ("A", 44)], 1: [("B", 44), ("A", 43)]}


def prepare():
    for target, source in REUSED.items():
        for base in (PROJECT_ROOT / "checkpoints" / "runs", RESULTS_DIR / "runs"):
            if not (base / target).exists():
                shutil.copytree(base / source, base / target)
        print(f"reused {source} as {target}")


def work(worker):
    from src.preprocessing.dataset import load_split
    from src.training.train_model import train_with_fallback
    from src.utils import get_device

    device = get_device()
    X_train, y_train, _ = load_split("train")
    X_val, y_val, _ = load_split("val")
    for model, seed in WORKER_RUNS[worker]:
        name = run_name(model, seed)
        print(f"=== worker {worker}: {name} ===", flush=True)
        s = train_with_fallback(name, gamma=GAMMA, lam=MODELS[model], alpha=ALPHA, seed=seed,
                                X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                                device=device)
        print(f"=== done {name}: best epoch {s['best_epoch']}, val recon {s['val_recon']:.4f}, "
              f"val 5-NN {s['val_knn5_accuracy']:.4f} ===", flush=True)


def evaluate():
    from src.evaluation.run_evaluation import evaluate_run
    from src.preprocessing.dataset import load_split
    from src.utils import get_device
    from src.visualization.plots import plot_pca_panels

    device = get_device()
    X_train, y_train, _ = load_split("train")
    X_val, y_val, tracks_val = load_split("val")

    rows, embeddings = [], {}
    for model in MODELS:
        for seed in SEEDS:
            metrics, Z_val = evaluate_run(run_name(model, seed), X_train, y_train,
                                          X_val, y_val, tracks_val, device)
            rows.append({"model": model, "lambda": MODELS[model], "seed": seed, **metrics})
            embeddings[(model, seed)] = Z_val
    per_run = pd.DataFrame(rows)

    metrics = ["recon", "clip_accuracy", "clip_f1", "track_accuracy", "track_f1",
               "knn5_accuracy", "linear_probe_accuracy", "silhouette", "mean_z_norm"]
    print("Per run (validation):")
    print(per_run[["model", "seed"] + metrics].round(4).to_string(index=False))

    summary = per_run.groupby("model")[metrics].agg(["mean", "std"])
    lines = []
    for metric in metrics:
        a_mean, a_std = summary.loc["A", (metric, "mean")], summary.loc["A", (metric, "std")]
        b_mean, b_std = summary.loc["B", (metric, "mean")], summary.loc["B", (metric, "std")]
        lines.append({"metric": metric, "A (λ=0) mean": a_mean, "A std": a_std,
                      "B (λ=0.1) mean": b_mean, "B std": b_std, "B − A": b_mean - a_mean})
    table = pd.DataFrame(lines)
    print("\nMean ± std over 3 seeds (validation):")
    print(table.round(4).to_string(index=False))

    per_run.round(4).to_csv(RESULTS_DIR / "ablation_val_per_run.csv", index=False)
    table.round(4).to_csv(RESULTS_DIR / "ablation_val_summary.csv", index=False)

    figure = RESULTS_DIR / "figures" / "pca_ablation_A_vs_B_val.png"
    plot_pca_panels([("Model A: recon + CE (λ = 0), seed 42", embeddings[("A", 42)]),
                     ("Model B: + center loss (λ = 0.1), seed 42", embeddings[("B", 42)])],
                    y_val, figure, "Validation embeddings, 2-D PCA view")
    print(f"\nSaved results/ablation_val_per_run.csv, results/ablation_val_summary.csv, "
          f"{figure.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    if "--prepare" in sys.argv:
        prepare()
    elif "--worker" in sys.argv:
        work(int(sys.argv[sys.argv.index("--worker") + 1]))
    elif "--evaluate" in sys.argv:
        evaluate()
    else:
        print(__doc__)
