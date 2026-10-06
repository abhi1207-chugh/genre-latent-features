"""Stage 17: 9-run grid, γ ∈ {0.5, 0.9, 0.99} × λ ∈ {0.001, 0.01, 0.1}, α = 0.1 fixed.

(The write-up's 27-run grid also varied α ∈ {0.01, 0.1, 0.5}; reduced to 9 runs
by user decision for runtime. The effect of α is NOT studied.)

Three steps, run from the project root:
    .venv/bin/python experiments/stage17_grid.py --prepare          # reuse the Stage 16 run as config 5
    .venv/bin/python -u experiments/stage17_grid.py --worker 0 > results/logs/grid_worker0.log 2>&1 &
    .venv/bin/python -u experiments/stage17_grid.py --worker 1 > results/logs/grid_worker1.log 2>&1 &
    .venv/bin/python experiments/stage17_grid.py --collect          # after both workers finish

Interrupted? Start the two workers again: finished runs are skipped, unfinished ones resume.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import PROJECT_ROOT, RESULTS_DIR

ALPHA = 0.1
CONFIGS = [(g, l) for g in (0.5, 0.9, 0.99) for l in (0.001, 0.01, 0.1)]   # configs 1..9


def run_name(gamma, lam):
    return f"grid_g{gamma}_l{lam}_a{ALPHA}"


# Fixed split of the 8 new runs over the two processes (config numbers 1..9).
# The γ = 0.99 runs (expected to train longest) are spread over both workers.
# Config 5 (γ = 0.9, λ = 0.01) is the Stage 16 run, reused by --prepare.
WORKER_CONFIGS = {0: [7, 9, 1, 3], 1: [8, 4, 6, 2]}
REUSED = {5: "stage16_g0.9_l0.01_a0.1"}

E_TRIVIAL = 1.1190    # val recon of always outputting 0 (Stage 8)


def prepare():
    """Copy the Stage 16 run's files, unchanged, under the grid name of config 5."""
    for number, source in REUSED.items():
        target = run_name(*CONFIGS[number - 1])
        for base in (PROJECT_ROOT / "checkpoints" / "runs", RESULTS_DIR / "runs"):
            if not (base / target).exists():
                shutil.copytree(base / source, base / target)
        print(f"config {number}: reused {source} as {target}")


def work(worker):
    from src.preprocessing.dataset import load_split
    from src.training.train_model import train_with_fallback
    from src.utils import get_device

    device = get_device()
    X_train, y_train, _ = load_split("train")
    X_val, y_val, _ = load_split("val")
    for number in WORKER_CONFIGS[worker]:
        gamma, lam = CONFIGS[number - 1]
        print(f"=== worker {worker}: config {number}  γ={gamma}  λ={lam}  α={ALPHA} ===", flush=True)
        summary = train_with_fallback(run_name(gamma, lam), gamma=gamma, lam=lam, alpha=ALPHA,
                                      X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                                      device=device)
        print(f"=== done config {number}: best epoch {summary['best_epoch']}, "
              f"val recon {summary['val_recon']:.4f}, val 5-NN {summary['val_knn5_accuracy']:.4f} ===",
              flush=True)


def collect():
    ae_reference = json.loads((RESULTS_DIR / "ae_reference.json").read_text())["AE_reference_error"]
    rows = []
    for number, (gamma, lam) in enumerate(CONFIGS, start=1):
        path = RESULTS_DIR / "runs" / run_name(gamma, lam) / "summary.json"
        if not path.exists():
            print(f"config {number} ({run_name(gamma, lam)}): NOT RUN")
            continue
        s = json.loads(path.read_text())
        rows.append({"config": number, "gamma": gamma, "lambda": lam, "alpha": ALPHA,
                     "best_epoch": s["best_epoch"], "epochs_run": s["epochs_run"],
                     "val_recon": s["val_recon"], "recon_ratio_to_AE": s["val_recon"] / ae_reference,
                     "valid_1.5x_rule": s["val_recon"] <= 1.5 * ae_reference,
                     "beats_trivial": s["val_recon"] < E_TRIVIAL,
                     "val_knn5_accuracy": s["val_knn5_accuracy"],
                     "val_classifier_accuracy": s["val_classifier_accuracy"],
                     "val_ce": s["val_ce"], "val_center": s["val_center"],
                     "mean_z_norm_val": s["mean_z_norm_val"], "run": run_name(gamma, lam)})
    table = pd.DataFrame(rows)
    print(f"AE_reference_error = {ae_reference:.4f}   1.5x threshold = {1.5 * ae_reference:.4f}   "
          f"trivial (output zeros) = {E_TRIVIAL:.4f}\n")
    print(table.drop(columns=["run", "alpha"]).round(4).to_string(index=False))
    table.to_csv(RESULTS_DIR / "grid_9runs.csv", index=False)
    print("\nSaved results/grid_9runs.csv")


if __name__ == "__main__":
    if "--prepare" in sys.argv:
        prepare()
    elif "--worker" in sys.argv:
        work(int(sys.argv[sys.argv.index("--worker") + 1]))
    elif "--collect" in sys.argv:
        collect()
    else:
        print(__doc__)
