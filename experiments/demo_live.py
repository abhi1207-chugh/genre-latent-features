"""Live-style demo: stream a song to the model one second at a time.

Simulates a microphone / player feeding audio in 1-second chunks: after every second the
model predicts that second, and the running answer is the average of all seconds so far.

Run from the project root (needs the GTZAN files in data/raw/):
    .venv/bin/python experiments/demo_live.py                       # one held-out test song per genre
    .venv/bin/python experiments/demo_live.py path/to/song.wav      # any audio file
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, RAW_AUDIO_DIR
from src.inference.predict import GenrePredictor

SECONDS_SHOWN = 10
predictor = GenrePredictor()                       # export/genre_model.pt on CPU
names = predictor.genres


def stream(path, true_genre=None):
    waveform, sr = librosa.load(path, sr=None, mono=True)            # as a player would deliver it
    print(f"\n♪ {Path(path).name}" + (f"   (true genre: {true_genre})" if true_genre else ""))
    print("   sec | this second          | running answer (average so far)")
    seconds = len(waveform) // sr
    probs_so_far = []
    for second in range(seconds):
        chunk = waveform[second * sr:(second + 1) * sr]
        start = time.perf_counter()
        clip = predictor.predict_waveform(chunk, sr)                # resample + pool + model
        latency = (time.perf_counter() - start) * 1000
        probs_so_far.append(clip["clip_probabilities"][0])
        running = np.mean(probs_so_far, axis=0)
        now = names[int(np.argmax(clip["clip_probabilities"][0]))]
        if second < SECONDS_SHOWN:
            print(f"   {second + 1:3d} | {now:9s} {clip['probabilities'][now]:5.0%}      | "
                  f"{names[int(running.argmax())]:9s} {running.max():5.0%}   ({latency:.0f} ms)")
    final = names[int(running.argmax())]
    verdict = "" if true_genre is None else ("  ✔ correct" if final == true_genre else "  ✘ wrong")
    print(f"   ... {seconds} seconds total → FINAL: {final} "
          f"({', '.join(f'{g} {p:.0%}' for g, p in zip(names, running))}){verdict}")


if len(sys.argv) > 1:
    stream(sys.argv[1])
else:
    tracks = pd.read_csv(PROJECT_ROOT / "export" / "test_tracks.csv")
    label = {"classical": "Classical", "country": "Country", "disco": "Disco", "hiphop": "Hip-Hop"}
    for genre in ["classical", "country", "disco", "hiphop"]:
        track = tracks[tracks["genre"] == genre].iloc[1]
        stream(RAW_AUDIO_DIR / track["path"], label[genre])
