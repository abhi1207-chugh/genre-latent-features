# Results index

Every table and figure listed in the write-up (§20), with the file that holds it and
the script that produced it. All numbers come from **one track-level split**
(280 / 60 / 60 tracks); the ablation uses 3 seeds. Model selection used validation
data only; the test split was used once, in Stage 20 (and re-read in Stages 22–23
for analysis, nothing tuned).

Deviations from the write-up (user decisions, see CLAUDE.md):
- Grid reduced from 27 to **9 runs** — α fixed at 0.1; the effect of α was **not studied**.
- **5-fold StratifiedGroupKFold: NOT RUN** (Stage 21 skipped) — no CV mean ± std exists.
- Selection rule kept as written (val recon ≤ 1.5 × AE_reference). Known weakness:
  1.5 × 0.802 = 1.203 is above the "output zeros" error 1.119, so non-reconstructing
  runs also pass (they lost here only through the 1-pp tie rule).

## Tables
| Write-up table | File | Script |
|---|---|---|
| 1. Dataset distribution | `dataset_distribution.csv`, `split_distribution.csv` | `experiments/stage03_inspect_dataset.py`, `stage04_make_split.py` |
| 2. Baseline comparison | `baselines_val.csv` (val), `test_comparison.csv` (test, with Models A/B) | `stage07_raw_baselines.py`, `stage20_final_evaluation.py` |
| 3. Hyperparameter experiments (9 runs) | `grid_9runs.csv`, `grid_selection.csv`, `selected_model.json` | `stage17_grid.py`, `stage18_select_model.py` |
| 4. Final model metrics (test, clip + track) | `test_ablation_per_run.csv` (row B, seed 42), `test_comparison.csv` | `stage20_final_evaluation.py` |
| 5. Ablation (center loss) | `ablation_val_summary.csv`, `ablation_val_per_run.csv` (val); `test_ablation_summary.csv` (test) | `stage19_ablation.py`, `stage20_final_evaluation.py` |
| 6. Cross-validation results | **NOT RUN** (Stage 21 skipped) | — |
| extra: embedding quality (test) | `embeddings_test.csv`; val: `dense_ae_embeddings_val.csv` | `stage22_embedding_analysis.py`, `stage09_eval_dense_ae_embeddings.py` |
| extra: AE reference error | `ae_reference.json` (CNN AE val recon 0.8020) | `stage11_cnn_autoencoder.py` |

## Figures (`figures/`)
| Write-up figure | File |
|---|---|
| 1. Model architecture | `architecture.png` |
| 2. Training / validation loss curves | `loss_curves_selected.png` (total, recon, CE, center) |
| 3. Reconstruction examples | `reconstructions_test.png` |
| 4. Hyperparameter Pareto plot | `pareto_grid_val.png` |
| 5. Confusion matrix | `confusion_selected_B_test.png`, `confusion_A_test.png` |
| 6. Latent-space visualization | `pca_embeddings_test.png`, `tsne_embeddings_test.png`, `pca_ablation_A_vs_B_val.png`, `pca_raw_vs_dense_ae_val.png` |
| 7. Baseline vs proposed model | `model_comparison_test.png` |

## Key numbers (test, 60 held-out tracks / 1,797 clips)
| Model | Clip acc | Track acc |
|---|---|---|
| Best baseline: raw 500-D + RBF-SVM (C = 10) | 0.611 | 0.683 |
| CNN AE 64-D + 5-NN (unsupervised) | 0.554 | 0.683 |
| Model A: recon + CE (3 seeds) | 0.769 ± 0.009 | 0.878 ± 0.026 |
| Model B: + center loss (3 seeds) | 0.730 ± 0.006 | 0.894 ± 0.035 |
| **Selected: Model B, seed 42** (γ 0.9, λ 0.1, α 0.1) | **0.732** | **0.867** |

Center loss did not improve the embedding: Model A beats Model B on clip accuracy,
5-NN and linear probe for every seed (val and test); silhouette is unchanged; Model B
only reconstructs better.

## Other folders
- `runs/<run>/` — `history.csv` (every loss term per epoch) and `summary.json` for every trained run.
- `logs/` — console logs of the long runs (grid, ablation, Stage 16).
- Model weights: `checkpoints/` (not in git); the selected model for inference is in `export/`.
