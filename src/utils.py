"""Small helpers shared by every stage: device selection and fixed seeds."""
import random

import numpy as np
import torch


def get_device():
    """Return cuda if available, else mps (Apple GPU), else cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed=42):
    """Fix all random number generators so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds cuda and mps
