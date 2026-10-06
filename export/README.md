# Inference package — live genre prediction demo

This folder holds everything needed to run the trained model on new audio.
No training, no dataset and no GPU needed (runs on CPU in well under a second per song).

| File | What it is |
|---|---|
| `genre_model.pt` | Selected model (Model B: γ = 0.9, λ = 0.1, α = 0.1, seed 42, epoch 419) **+ the training scaler + preprocessing settings + label map** in one file |
| `test_tracks.csv` | The 60 held-out **test** tracks (15 per genre). The model never saw them during training or model selection — use these for an honest demo. |

Code: `src/inference/predict.py` (class `GenrePredictor`).

## Setup (once)
```bash
git clone <repo> && cd genre-latent-features
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Use it
Command line, any audio file librosa can read (wav, flac, ogg, mp3):
```bash
.venv/bin/python -m src.inference.predict path/to/song.wav
```
```
Predicted genre: Classical   (30 one-second clips)
  Classical   84.4%
  Country     13.0%
  Hip-Hop      1.6%
  Disco        1.0%
```

From Python (e.g. inside the demo app):
```python
from src.inference.predict import GenrePredictor

predictor = GenrePredictor()                         # loads export/genre_model.pt, CPU
result = predictor.predict_file("song.wav")
# or, for raw samples such as a microphone buffer (any sample rate, mono or stereo):
result = predictor.predict_waveform(samples, sample_rate)

result["genre"]               # "Disco"
result["probabilities"]       # {"Classical": 0.01, "Country": 0.05, "Disco": 0.80, "Hip-Hop": 0.14}
result["n_clips"]             # number of full 1-second clips used
result["clip_genres"]         # prediction for every 1-second clip (good for a live timeline)
result["clip_probabilities"]  # array (n_clips, 4)
```

What happens inside (identical to training): mono → resample to 20 kHz → full
1-second clips (leftover < 1 s dropped) → average pooling by 40 → 500 values per clip
→ standardize with the **training** mean/std → model → softmax per clip →
average the clip probabilities → genre with the highest average.

## Demo audio
Download GTZAN once (Kaggle token needed, see `experiments/stage03_download_gtzan.sh`),
then play tracks listed in `test_tracks.csv`, e.g.
`data/raw/genres_original/disco/disco.00018.wav`.

## Read before the demo (honest limitations)
- **Only 4 genres**: Classical, Country, Disco, Hip-Hop. Any other music is forced into one of these.
- **Expected accuracy** on held-out GTZAN test tracks: **86.7 % per track** (52/60) and
  **73.2 % per 1-second clip**. Classical is easiest; **Disco ↔ Hip-Hop** is the most common confusion.
- **Loudness matters.** The input is the raw waveform, standardized with the GTZAN training
  statistics, and the genres differ strongly in loudness (classical clips are ~10× quieter than
  hip-hop). A quiet microphone recording, or music played softly, may be pushed towards
  *Classical*. Playing GTZAN files directly (not via speaker → microphone) avoids this.
- Give it **at least ~5–10 seconds** of audio; the song-level answer averages over clips.
- Need ≥ 1 full second of audio, otherwise a `ValueError` is raised.
- Not tested on recordings outside GTZAN; treat microphone results as a demonstration, not a measurement.
