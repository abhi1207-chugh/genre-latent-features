"""Stage 2: check that all libraries import and that the device works.

Run from the project root:
    .venv/bin/python experiments/stage02_check_environment.py
"""
import sys
from pathlib import Path

# Make "src" importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import soundfile
import torch

from src.utils import get_device, set_seed

print(f"Python       {sys.version.split()[0]}")
print(f"torch        {torch.__version__}")
print(f"numpy        {np.__version__}")
print(f"pandas       {pd.__version__}")
print(f"scikit-learn {sklearn.__version__}")
print(f"matplotlib   {matplotlib.__version__}")
print(f"librosa      {librosa.__version__}")
print(f"soundfile    {soundfile.__version__}")

device = get_device()
print(f"\nSelected device: {device}")

# A tiny computation on the device: a batch of 512 inputs of size 500
# through one linear layer to 64 values (the shapes of our real model).
set_seed(42)
x = torch.randn(512, 500, device=device)
layer = torch.nn.Linear(500, 64).to(device)
z = layer(x)
print(f"Test forward pass on {device}: {tuple(x.shape)} -> {tuple(z.shape)}")

# Reproducibility check: same seed must give the same random numbers.
set_seed(42)
a = torch.randn(3)
set_seed(42)
b = torch.randn(3)
print(f"Seeds reproducible: {torch.equal(a, b)}")
