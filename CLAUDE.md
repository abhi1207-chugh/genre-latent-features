# CLAUDE.md — Project instructions for Claude Code

## Project
**Latent Feature Extraction for Musical Genres from Raw Audio Using a CNN-Based Autoencoder**
University ML mini-project (UE24CS352A), team of two. Deadline: Sat Oct 10 2026, 11:59 PM IST.

Source documents are in `docs/`:
- `docs/ML_Project_Final_Writeup.md` — **MAIN SPECIFICATION. Highest priority.**
- `docs/paper.pdf` — original paper (Sawhney, Vasavada, Wang): original methodology.
- `docs/poster.pdf` — original poster.
- `docs/guidelines.pdf` — university deliverables (private GitHub repo + README, PDF write-up, slides, live demo).

Read the write-up before any stage. If the write-up and anything else conflict, STOP and ask.

## How to work with me (MOST IMPORTANT)
I am implementing this project step by step and must understand every line for my viva.
- Do ONE stage at a time. Never generate the whole project or many files at once.
- Never continue to the next stage automatically. After each stage, STOP and wait for "done" or my output/error.
- Never assume earlier code worked. Run it and show me the real output.
- Keep each stage small (well under 500 lines). Beginner-readable code, meaningful names, comments only where useful.
- If a design decision is ambiguous, explain the options and ASK before implementing.
- If I hit an error, debug only that stage.

Start every stage with this format:
```
## Stage X — Name
### Goal
### Why
### Files
### Code
### How to run
### Expected output
### Check
```

## Scientific honesty (non-negotiable)
- Never fabricate accuracy, F1, precision, recall, reconstruction error, best hyperparameters, or ablation results.
- Anything not yet run is written as `NOT RUN` or `PENDING EXPERIMENT`.
- Do not claim center loss helps unless the ablation shows it.
- Do not claim the CNN helps unless compared against the equivalent dense encoder.
- Do not claim we beat the paper (different genres, split, architecture).
- Never use the test set for hyperparameter selection.

## Do NOT introduce
Transformers, LSTMs, GRUs, attention, MFCCs or Mel-spectrograms in the main model, β-TCVAE, pretrained models, ensembles, or any unnecessarily complex architecture. Do not redesign the approach.

## Environment
- MacBook Air/Pro M1, 8 GB RAM, 256 GB SSD, no NVIDIA GPU. Everything runs locally (no Colab).
- Python virtual environment in `.venv/`. Install with `pip`.
- Device selection: `cuda` if available, else `mps` (Apple GPU), else `cpu`. Fall back to CPU if MPS errors.
- Keep memory modest: the model is tiny; the full dataset is ~12k clips × 500 floats.

## Specification summary (from the write-up)
- Dataset: GTZAN (Kaggle), genres: classical, country, disco, hiphop (~400 tracks).
- Label map (fixed): 0 = Classical, 1 = Country, 2 = Disco, 3 = Hip-Hop.
- Split at TRACK level, stratified by genre, BEFORE making clips. Keep `track_id` for every clip.
- Preprocessing: track → 1-second clips → raw waveform → average pooling → 500-D → standardization (fit on train only). LibROSA for loading.
- Model: 1D CNN encoder → 64-D embedding z → (a) decoder → 500-D reconstruction, (b) classifier 64→32→16→4 logits, (c) center loss with one 64-D center per genre.
- Loss: `L = γ·L_recon + (1−γ)·L_CE + λ·L_center`. Log each term separately.
  - γ = reconstruction/classification trade-off. λ = center-loss weight. α = center learning rate (how fast centers move). Never confuse λ and α.
  - CrossEntropyLoss on raw logits (no softmax during training); softmax only at inference.
- Training: PyTorch, Adam, lr 1e-4, batch 512, latent 64, dropout ~0.1, early stopping on validation, fixed seeds, checkpoints, resumable experiments.
- Vanilla autoencoder (reconstruction only) → its validation reconstruction error = `AE_reference_error`.
- Grid: γ ∈ {0.5, 0.9, 0.99}, λ ∈ {0.001, 0.01, 0.1}, α ∈ {0.01, 0.1, 0.5} → 27 runs. Validation data only.
- Selection (automatic): valid if val recon error ≤ 1.5 × AE_reference_error; among valid, highest val 5-NN accuracy on frozen 64-D embeddings; if within ~1 percentage point, prefer lower recon error.
- Baselines: raw 500-D + kNN, raw 500-D + SVM, PCA + classifier, simple NN classifier, vanilla AE embeddings.
- Embedding evaluation: 5-NN probe, linear probe, silhouette score, PCA (mandatory), t-SNE/UMAP optional.
- Metrics: accuracy, precision, recall, F1, confusion matrix — at clip level AND track level (track = average clip probability vectors, then argmax).
- Ablation: Model A (recon + CE) vs Model B (recon + CE + center), same everything else.
- Final: 5-fold StratifiedGroupKFold, group = track_id, report mean ± std.
- Plots: loss curves (total, recon, CE, center), confusion matrix, PCA of embeddings, Pareto plot (x = val recon error, y = val 5-NN acc, line at 1.5 × AE_ref, selected model highlighted).

## Open decisions (ask me when we reach the stage — do not decide silently)
- Stage 5: sample rate / pooling. At 22,050 Hz, 1 s / 40 = 551, not 500. Options: load at 20 kHz (1 s = 20,000 → 500) or keep 22,050 Hz and crop to 20,000 samples. Also global vs per-feature standardization. **DECIDED: resample to 20 kHz on load; global standardization (1 mean, 1 std) fit on train clips only.**
- Stage 8: vanilla AE encoder — paper's dense encoder or the same CNN encoder as our model (fairness of the 1.5× constraint).
- Stage 21: how the held-out test set and 5-fold CV coexist, and the early-stopping validation set inside each fold.

## Project structure
```
data/raw/            GTZAN audio (gitignored)
data/processed/      pooled features, split.csv, scaler (gitignored)
src/preprocessing/   inspection, split, waveform → 500-D
src/models/          encoders, decoder, classifier, full model
src/losses/          center loss, combined loss
src/training/        training loop, early stopping, checkpoints
src/evaluation/      metrics, probes, silhouette, track voting, CV
src/visualization/   plots
experiments/         one runnable script per stage
checkpoints/         model weights (gitignored)
results/             CSV tables + figures/
notebooks/           optional exploration
report/              PDF write-up + slides
docs/                source documents
```

## Stage order
0 Understand project · 1 Project setup · 2 Environment · 3 Dataset inspection · 4 Track-level split · 5 Audio preprocessing · 6 Dataset/DataLoader · 7 Raw-feature baselines · 8 Vanilla AE · 9 Vanilla embeddings evaluation · 10 Design CNN encoder (show shapes, wait for approval) · 11 CNN encoder + decoder · 12 Add classifier · 13 Center loss module (test on dummy data) · 14 Full model + combined loss · 15 Smoke test · 16 Reusable training function (test ONE config) · 17 27-run grid · 18 Model selection + Pareto plot · 19 Center-loss ablation · 20 Final evaluation (clip + track) · 21 Grouped 5-fold CV · 22 Final embedding analysis · 23 Results organization · 24 README · 25 Viva prep (one question at a time)

## Progress log
Update this list as stages finish.
- Stage 0 — done (understanding confirmed).
- Stage 1 — done (structure, .gitignore, git repo pushed to github.com/abhi1207-chugh/genre-latent-features).
- Stage 2 — done (.venv, requirements.txt pinned, src/utils.py: get_device + set_seed; device = mps).
- Stage 3 — done (Kaggle download script; 400/400 tracks load, all 22050 Hz, ~30 s; 11,992 full 1-s clips; src/config.py holds LABEL_MAP).
- Stage 4 — done (70/15/15 track split, stratified, seed 42: 280/60/60 tracks = 70/15/15 per genre; data/processed/split.csv; test set locked until Stage 20).
- Stage 5 — done (20 kHz, 1-s clips, avg-pool 40 → 11,992 × 500; train/val/test clips 8396/1799/1797; unscaled features in clips_unscaled.npz + scaler.json so the scaler can be refit per CV fold).
- Stage 6 — done (src/preprocessing/dataset.py: load_split → standardized X, y, track_id; ClipDataset; make_loader with seeded shuffle; batch 512 → 17 train / 4 val batches).
- Stage 7 — PENDING
