"""Stage 16: train ONE configuration with the reusable training function.

Run from the project root (about 12-20 min on MPS):
    .venv/bin/python -u experiments/stage16_train_one_config.py

If interrupted (Ctrl+C, laptop sleep), run the same command again: it resumes.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS_DIR
from src.preprocessing.dataset import load_split
from src.training.train_model import train_with_fallback
from src.utils import get_device

device = get_device()
X_train, y_train, _ = load_split("train")
X_val, y_val, _ = load_split("val")

summary = train_with_fallback(
    "stage16_g0.9_l0.01_a0.1", gamma=0.9, lam=0.01, alpha=0.1,
    X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, device=device,
)

ae_reference = json.loads((RESULTS_DIR / "ae_reference.json").read_text())["AE_reference_error"]
print("\nSummary (validation, best epoch):")
for key, value in summary.items():
    print(f"  {key:28s} {value:.4f}" if isinstance(value, float) else f"  {key:28s} {value}")
print(f"\n  AE_reference_error           {ae_reference:.4f}   (val recon / reference = "
      f"{summary['val_recon'] / ae_reference:.3f})")
