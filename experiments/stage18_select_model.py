"""Stage 18: automatic model selection on the 9 grid runs + Pareto plot (validation only).

Run from the project root:
    .venv/bin/python experiments/stage18_select_model.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import PROJECT_ROOT, RESULTS_DIR
from src.evaluation.selection import pareto_front, select_model
from src.visualization.plots import plot_pareto

E_TRIVIAL = 1.1190
grid = pd.read_csv(RESULTS_DIR / "grid_9runs.csv")
ae_reference = json.loads((RESULTS_DIR / "ae_reference.json").read_text())["AE_reference_error"]

selected, table = select_model(grid, ae_reference, factor=1.5, tie_points=0.01)
table["pareto"] = pareto_front(table)

best_knn = table.loc[table["valid"], "val_knn5_accuracy"].max()
print(f"AE_reference_error = {ae_reference:.4f}  ->  valid if val recon <= {1.5 * ae_reference:.4f}")
print(f"Valid runs: {table['valid'].sum()} of {len(table)}")
print(f"Best val 5-NN among valid runs: {best_knn:.4f}  ->  tie candidates have 5-NN >= {best_knn - 0.01:.4f}\n")

columns = ["config", "gamma", "lambda", "val_recon", "val_knn5_accuracy", "valid",
           "beats_trivial", "tie_candidate", "pareto", "selected"]
print(table[columns].round(4).to_string(index=False))

print(f"\nSELECTED: config {int(selected['config'])}  γ = {selected['gamma']}  λ = {selected['lambda']}  "
      f"α = {selected['alpha']}")
print(f"  val recon {selected['val_recon']:.4f} (ratio to AE {selected['val_recon'] / ae_reference:.3f}), "
      f"val 5-NN {selected['val_knn5_accuracy']:.4f}, classifier acc {selected['val_classifier_accuracy']:.4f}, "
      f"best epoch {int(selected['best_epoch'])}")
print(f"  beats trivial 'output zeros' error ({E_TRIVIAL}): {bool(selected['beats_trivial'])}")

table.to_csv(RESULTS_DIR / "grid_selection.csv", index=False)
(RESULTS_DIR / "selected_model.json").write_text(json.dumps({
    "config": int(selected["config"]), "run": selected["run"],
    "gamma": float(selected["gamma"]), "lambda": float(selected["lambda"]), "alpha": float(selected["alpha"]),
    "best_epoch": int(selected["best_epoch"]), "val_recon": float(selected["val_recon"]),
    "val_knn5_accuracy": float(selected["val_knn5_accuracy"]),
    "rule": "R0: val recon <= 1.5 x AE_reference; max val 5-NN; within 1 pp -> lowest val recon",
}, indent=2))

figure = RESULTS_DIR / "figures" / "pareto_grid_val.png"
plot_pareto(table, ae_reference, E_TRIVIAL, figure)
print(f"\nSaved results/grid_selection.csv, results/selected_model.json, {figure.relative_to(PROJECT_ROOT)}")
