"""Genre prediction for new audio with the exported model (export/genre_model.pt).

Same pipeline as training: mono -> 20 kHz -> 1-second clips -> average pooling 40
-> 500-D -> standardize with the TRAIN scaler -> model -> softmax per clip
-> average the clip probabilities -> genre of the whole recording.

Python:
    from src.inference.predict import GenrePredictor
    predictor = GenrePredictor()                    # loads export/genre_model.pt
    result = predictor.predict_file("song.wav")
    result = predictor.predict_waveform(samples, sample_rate)   # e.g. microphone buffer

Command line (from the project root):
    .venv/bin/python -m src.inference.predict path/to/song.wav
"""
import sys

import librosa
import numpy as np
import torch

from src.config import PROJECT_ROOT
from src.models.full_model import GenreAutoencoder
from src.preprocessing.features import apply_scaler, waveform_to_clips

DEFAULT_MODEL = PROJECT_ROOT / "export" / "genre_model.pt"


class GenrePredictor:
    def __init__(self, model_path=DEFAULT_MODEL, device="cpu"):
        package = torch.load(model_path, map_location=device, weights_only=False)
        self.info = package["info"]
        self.genres = self.info["genres"]                     # index -> name
        self.sample_rate = self.info["sample_rate"]           # 20,000
        self.scaler_mean = self.info["scaler_mean"]
        self.scaler_std = self.info["scaler_std"]
        self.device = torch.device(device)
        self.model = GenreAutoencoder(latent_dim=64, n_classes=4).to(self.device)
        self.model.load_state_dict(package["model_state"])
        self.model.eval()                                     # dropout off

    def predict_file(self, path):
        waveform, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        return self._predict(waveform)

    def predict_waveform(self, waveform, sample_rate):
        """waveform: 1-D (or channels x samples) float array in [-1, 1]."""
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 2:                                # stereo -> mono
            waveform = waveform.mean(axis=0)
        if sample_rate != self.sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=self.sample_rate)
        return self._predict(waveform)

    @torch.no_grad()
    def _predict(self, waveform):
        clips = waveform_to_clips(waveform)
        if len(clips) == 0:
            raise ValueError("Audio is shorter than 1 second; need at least one full second.")
        x = torch.from_numpy(apply_scaler(clips, self.scaler_mean, self.scaler_std)).to(self.device)
        _, _, logits = self.model(x)
        clip_probs = torch.softmax(logits, dim=1).cpu().numpy()    # softmax only at inference
        track_probs = clip_probs.mean(axis=0)                     # average over clips
        return {
            "genre": self.genres[int(track_probs.argmax())],
            "probabilities": {g: float(p) for g, p in zip(self.genres, track_probs)},
            "n_clips": len(clips),
            "clip_genres": [self.genres[i] for i in clip_probs.argmax(axis=1)],
            "clip_probabilities": clip_probs,
        }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    result = GenrePredictor().predict_file(sys.argv[1])
    print(f"Predicted genre: {result['genre']}   ({result['n_clips']} one-second clips)")
    for genre, p in sorted(result["probabilities"].items(), key=lambda item: -item[1]):
        print(f"  {genre:10s} {p:6.1%}")
