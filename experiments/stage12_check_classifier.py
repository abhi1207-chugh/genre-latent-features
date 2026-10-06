"""Stage 12: check the classifier head on dummy data (no training yet).

Run from the project root:
    .venv/bin/python experiments/stage12_check_classifier.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from src.models.classifier import GenreClassifier
from src.models.cnn_autoencoder import CNNEncoder
from src.utils import set_seed

set_seed(42)
encoder = CNNEncoder(latent_dim=64, dropout=0.1)
classifier = GenreClassifier(latent_dim=64, n_classes=4, dropout=0.1)
print(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")

# 1. Shapes: clip -> encoder -> z -> classifier -> logits
x = torch.randn(6, 500)
y = torch.tensor([0, 1, 2, 3, 0, 1])
z = encoder(x)
logits = classifier(z)
print(f"\nx {tuple(x.shape)} -> z {tuple(z.shape)} -> logits {tuple(logits.shape)}")
print(f"Logits of first clip (raw, any real numbers): {logits[0].detach().numpy().round(3)}")

# 2. CrossEntropyLoss takes RAW logits; it applies log-softmax internally
ce = nn.CrossEntropyLoss()
loss = ce(logits, y)
manual = -torch.log_softmax(logits, dim=1)[torch.arange(6), y].mean()
print(f"\nCrossEntropyLoss(logits, y)      = {loss.item():.4f}")
print(f"manual -mean(log softmax[true]) = {manual.item():.4f}")
print(f"Untrained model ~ uniform guess: ln(4) = {math.log(4):.4f}")

# 3. The mistake we avoid: softmax before CrossEntropyLoss (softmax applied twice).
#    Shown on a confident, CORRECT prediction: the loss should be close to 0.
confident = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
true_class = torch.tensor([0])
print(f"\nConfident correct prediction, logits {confident[0].tolist()}:")
print(f"  right: CE(logits, y)          = {ce(confident, true_class).item():.4f}")
print(f"  WRONG: CE(softmax(logits), y) = {ce(torch.softmax(confident, dim=1), true_class).item():.4f}"
      f"  (can never go below ln(1 + 3/e) = {math.log(1 + 3 / math.e):.4f})")

# 4. At inference only: softmax -> probabilities, argmax -> predicted genre
classifier.eval()
with torch.no_grad():
    probs = torch.softmax(classifier(encoder.eval()(x)), dim=1)
print(f"\nInference probabilities, first clip: {probs[0].numpy().round(3)}  sum = {probs[0].sum():.4f}")
print(f"Predicted genres: {probs.argmax(dim=1).tolist()}")

# 5. Gradients reach the encoder through z (needed for joint training)
classifier.train(); encoder.train()
ce(classifier(encoder(x)), y).backward()
first_conv = encoder.blocks[0][0].weight
print(f"\nGradient reaches first encoder conv layer: {first_conv.grad is not None and first_conv.grad.abs().sum() > 0}")
