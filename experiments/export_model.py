"""Finalization: export the selected model for inference / the live demo.

Creates (committed to git, unlike checkpoints/ and data/processed/):
  export/genre_model.pt     model weights + train scaler + preprocessing settings + label map
  export/test_tracks.csv    the 60 held-out TEST tracks (never seen in training) for the demo
Then verifies the export: predicting every test track through the export must
reproduce the Stage 20 test results of the selected model.

Run from the project root:
    .venv/bin/python experiments/export_model.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from src.config import (FEATURE_DIM, GENRES, POOL_SIZE, PROCESSED_DIR, PROJECT_ROOT,
                        RAW_AUDIO_DIR, RESULTS_DIR, SAMPLE_RATE)
from src.inference.predict import GenrePredictor

EXPORT_DIR = PROJECT_ROOT / "export"
EXPORT_DIR.mkdir(exist_ok=True)

selected = json.loads((RESULTS_DIR / "selected_model.json").read_text())
scaler = json.loads((PROCESSED_DIR / "scaler.json").read_text())
best = torch.load(PROJECT_ROOT / "checkpoints" / "runs" / selected["run"] / "best.pt", weights_only=False)

test_results = pd.read_csv(RESULTS_DIR / "test_ablation_per_run.csv")
stage20 = test_results[(test_results["model"] == "B") & (test_results["seed"] == 42)].iloc[0]

package = {
    "model_state": best["model"],          # encoder + decoder + classifier (centers not needed)
    "info": {
        "description": "Selected Model B: CNN autoencoder + classifier + center loss",
        "source_run": selected["run"], "gamma": selected["gamma"], "lambda": selected["lambda"],
        "alpha": selected["alpha"], "seed": 42, "best_epoch": selected["best_epoch"],
        "genres": ["Classical", "Country", "Disco", "Hip-Hop"],     # label 0..3
        "sample_rate": SAMPLE_RATE, "clip_seconds": 1, "pool_size": POOL_SIZE,
        "feature_dim": FEATURE_DIM, "latent_dim": 64,
        "scaler_mean": scaler["mean"], "scaler_std": scaler["std"],
        "test_clip_accuracy": float(stage20["clip_accuracy"]),
        "test_track_accuracy": float(stage20["track_accuracy"]),
    },
}
torch.save(package, EXPORT_DIR / "genre_model.pt")
print(f"Saved export/genre_model.pt ({(EXPORT_DIR / 'genre_model.pt').stat().st_size / 1e6:.2f} MB)")

split = pd.read_csv(PROCESSED_DIR / "split.csv")
test_tracks = split[split["split"] == "test"][["track_id", "genre", "path"]]
test_tracks.to_csv(EXPORT_DIR / "test_tracks.csv", index=False)
print(f"Saved export/test_tracks.csv ({len(test_tracks)} held-out tracks)\n")

# ---- Verification: run the exported predictor on every test track ----
predictor = GenrePredictor(EXPORT_DIR / "genre_model.pt", device="cpu")
names = package["info"]["genres"]
correct_tracks, correct_clips, n_clips = 0, 0, 0
for track in test_tracks.itertuples():
    result = predictor.predict_file(RAW_AUDIO_DIR / track.path)
    true_name = names[GENRES.index(track.genre)]
    correct_tracks += result["genre"] == true_name
    correct_clips += sum(g == true_name for g in result["clip_genres"])
    n_clips += result["n_clips"]

print("Export check on the 60 TEST tracks (CPU, through the exported file):")
print(f"  clip accuracy  {correct_clips / n_clips:.4f}   (Stage 20: {stage20['clip_accuracy']:.4f})")
print(f"  track accuracy {correct_tracks / len(test_tracks):.4f}   (Stage 20: {stage20['track_accuracy']:.4f})")
print(f"  clips {n_clips} (Stage 5 test clips: 1797)")
