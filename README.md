# Latent Feature Extraction for Musical Genres from Raw Audio Using a CNN-Based Autoencoder

UE24CS352A Machine Learning mini-project.

> Work in progress — full setup / run instructions will be added in Stage 24.

**Status**
- Stages 0–20 done: data, track-level split, preprocessing, baselines, autoencoders,
  9-run γ × λ grid (α fixed at 0.1), model selection, center-loss ablation (3 seeds),
  final test evaluation. Results in `results/`, figures in `results/figures/`.
- Stage 21 (grouped 5-fold cross-validation) was **skipped** (not run) — all reported
  numbers are from the single track-level train/val/test split (plus 3 seeds for the ablation).
- **Using the trained model / live demo:** see [`export/README.md`](export/README.md).
