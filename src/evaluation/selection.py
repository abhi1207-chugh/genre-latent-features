"""Automatic model selection on validation results (rule R0, as written in the write-up).

  1. valid  <=> val recon error <= 1.5 x AE_reference_error
  2. among valid runs: highest val 5-NN accuracy
  3. tie rule: every valid run within 1 percentage point of that best 5-NN
     is a candidate; the candidate with the LOWEST val recon error is selected
"""
import pandas as pd


def select_model(grid, ae_reference, factor=1.5, tie_points=0.01):
    """Return (selected row, annotated table). Uses only the validation columns."""
    table = grid.copy()
    table["valid"] = table["val_recon"] <= factor * ae_reference

    valid = table[table["valid"]]
    if valid.empty:
        raise ValueError("No run satisfies the reconstruction constraint")

    best_knn = valid["val_knn5_accuracy"].max()
    table["tie_candidate"] = table["valid"] & (table["val_knn5_accuracy"] >= best_knn - tie_points)

    candidates = table[table["tie_candidate"]]
    selected = candidates.loc[candidates["val_recon"].idxmin()]
    table["selected"] = table["config"] == selected["config"]
    return selected, table


def pareto_front(table):
    """Runs not beaten on BOTH axes by another run (lower recon AND higher 5-NN)."""
    on_front = []
    for _, run in table.iterrows():
        dominated = ((table["val_recon"] <= run["val_recon"]) &
                     (table["val_knn5_accuracy"] >= run["val_knn5_accuracy"]) &
                     ((table["val_recon"] < run["val_recon"]) |
                      (table["val_knn5_accuracy"] > run["val_knn5_accuracy"]))).any()
        on_front.append(not dominated)
    return pd.Series(on_front, index=table.index)
