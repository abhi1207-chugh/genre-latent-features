"""Stage 4: create the track-level 70/15/15 split and save it.

Run from the project root:
    .venv/bin/python experiments/stage04_make_split.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import PROCESSED_DIR, RESULTS_DIR
from src.preprocessing.split import list_tracks, make_track_split

tracks = list_tracks()
split = make_track_split(tracks, val_fraction=0.15, test_fraction=0.15, seed=42)

# 1. Tracks per split and genre
table = pd.crosstab(split["genre"], split["split"], margins=True, margins_name="total")
table = table[["train", "val", "test", "total"]]
print("Tracks per genre and split:")
print(table.to_string())

# 2. Leakage check: every track appears exactly once, in exactly one split
each_track_once = split["track_id"].is_unique and len(split) == len(tracks)
print(f"\nEvery track in exactly one split: {each_track_once}")

# 3. Reproducibility check: same seed -> identical split
again = make_track_split(tracks, val_fraction=0.15, test_fraction=0.15, seed=42)
print(f"Same seed gives identical split: {split.equals(again)}")

# 4. A few example rows
print("\nFirst 5 rows of split.csv:")
print(split.head().to_string(index=False))

# Save: data/processed/split.csv is used by later stages,
# results/split_distribution.csv goes into the report.
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
split.to_csv(PROCESSED_DIR / "split.csv", index=False)
table.to_csv(RESULTS_DIR / "split_distribution.csv")
print("\nSaved data/processed/split.csv and results/split_distribution.csv")
