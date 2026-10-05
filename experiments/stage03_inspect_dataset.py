"""Stage 3: inspect the 400 GTZAN tracks of our 4 genres.

Run from the project root:
    .venv/bin/python experiments/stage03_inspect_dataset.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS_DIR
from src.preprocessing.inspect_dataset import inspect_dataset

tracks = inspect_dataset()

# 1. Files that failed to load
failed = tracks[~tracks["loaded"]]
print(f"Tracks found: {len(tracks)}   loaded OK: {tracks['loaded'].sum()}   failed: {len(failed)}")
for _, row in failed.iterrows():
    print(f"  FAILED {row['path']}: {row['error']}")

ok = tracks[tracks["loaded"]]

# 2. Sample rates (we need to know this for the Stage 5 decision)
print(f"\nSample rates found: {sorted(int(sr) for sr in ok['sample_rate'].unique())}")

# 3. Per-genre summary (this is the 'dataset distribution' table)
summary = ok.groupby("genre").agg(
    tracks=("track_id", "count"),
    min_duration_s=("duration_s", "min"),
    mean_duration_s=("duration_s", "mean"),
    max_duration_s=("duration_s", "max"),
    total_1s_clips=("n_full_1s_clips", "sum"),
)
print("\nPer-genre summary:")
print(summary.round(3).to_string())
print(f"\nTotal 1-second clips: {ok['n_full_1s_clips'].sum()}")

# 4. Unusually short tracks (fewer than 29 full seconds)
short = ok[ok["n_full_1s_clips"] < 29]
print(f"\nTracks shorter than 29 s: {len(short)}")
if len(short) > 0:
    print(short[["track_id", "duration_s"]].to_string(index=False))

# 5. Amplitude range (librosa returns floats in [-1, 1])
print(f"\nAmplitude range over all tracks: "
      f"[{ok['min_amplitude'].min():.3f}, {ok['max_amplitude'].max():.3f}]")

# Save tables
RESULTS_DIR.mkdir(exist_ok=True)
tracks.to_csv(RESULTS_DIR / "dataset_inspection.csv", index=False)
summary.round(3).to_csv(RESULTS_DIR / "dataset_distribution.csv")
print(f"\nSaved results/dataset_inspection.csv and results/dataset_distribution.csv")
