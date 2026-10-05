"""Stage 5: turn all 400 tracks into standardized 500-D clip vectors.

Run from the project root:
    .venv/bin/python experiments/stage05_preprocess_audio.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.preprocessing.features import apply_scaler, build_clip_dataset, fit_global_scaler

split_table = pd.read_csv(PROCESSED_DIR / "split.csv")

# 1. Audio -> pooled clips (unscaled)
start = time.time()
data = build_clip_dataset(split_table)
print(f"Processed {len(split_table)} tracks in {time.time() - start:.1f} s")
print(f"X shape: {data['X'].shape}  dtype: {data['X'].dtype}")

# 2. Clips per split, and leakage check at clip level
clips = pd.DataFrame({"split": data["split"], "track_id": data["track_id"], "label": data["label"]})
print("\nClips per split and label (0=classical 1=country 2=disco 3=hiphop):")
print(pd.crosstab(clips["label"], clips["split"], margins=True, margins_name="total")
      [["train", "val", "test", "total"]].to_string())

tracks_per_split = clips.groupby("split")["track_id"].apply(set)
overlap = (tracks_per_split["train"] & tracks_per_split["val"]) | \
          (tracks_per_split["train"] & tracks_per_split["test"]) | \
          (tracks_per_split["val"] & tracks_per_split["test"])
print(f"\nTracks shared between splits: {len(overlap)}")

# 3. Global standardization, fit on TRAIN clips only
is_train = data["split"] == "train"
mean, std = fit_global_scaler(data["X"][is_train])
print(f"\nScaler fit on train clips only: mean = {mean:.6f}, std = {std:.6f}")

X_scaled = apply_scaler(data["X"], mean, std)
for name in ["train", "val", "test"]:
    part = X_scaled[data["split"] == name]
    print(f"  {name:5s} after scaling: mean = {part.mean():+.4f}, std = {part.std():.4f}")

# 4. Save unscaled features + scaler separately.
#    Unscaled is kept so the scaler can be refit per fold in the 5-fold CV (Stage 21).
np.savez(PROCESSED_DIR / "clips_unscaled.npz", **data)
with open(PROCESSED_DIR / "scaler.json", "w") as f:
    json.dump({"mean": mean, "std": std, "fit_on": "train clips of split.csv"}, f, indent=2)
print("\nSaved data/processed/clips_unscaled.npz and data/processed/scaler.json")
