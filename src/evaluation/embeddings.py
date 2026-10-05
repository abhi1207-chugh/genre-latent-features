"""Evaluate frozen embeddings: 5-NN probe, linear probe, silhouette score."""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@torch.no_grad()
def extract_embeddings(encoder, X, device, batch_size=512):
    """Run the (frozen) encoder on X in batches. Returns a numpy array (n, latent_dim)."""
    encoder.to(device).eval()                     # dropout off, weights not updated
    chunks = []
    for start in range(0, len(X), batch_size):
        x = torch.from_numpy(X[start:start + batch_size]).to(device)
        chunks.append(encoder(x).cpu().numpy())
    return np.concatenate(chunks)


def knn_probe(Z_train, y_train, Z_val, y_val, k=5):
    """Accuracy of a k-NN classifier on the embeddings as they are (no rescaling)."""
    model = KNeighborsClassifier(n_neighbors=k).fit(Z_train, y_train)
    return accuracy_score(y_val, model.predict(Z_val))


def linear_probe(Z_train, y_train, Z_val, y_val, seed=42):
    """Accuracy of logistic regression on the embeddings.

    The embeddings are standardized first (fit on train) only to help the optimizer;
    a linear model on rescaled features is still a linear model.
    """
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, random_state=seed))
    model.fit(Z_train, y_train)
    return accuracy_score(y_val, model.predict(Z_val))


def evaluate_embeddings(Z_train, y_train, Z_val, y_val):
    """All three embedding metrics on the validation split."""
    return {
        "knn5_accuracy": knn_probe(Z_train, y_train, Z_val, y_val),
        "linear_probe_accuracy": linear_probe(Z_train, y_train, Z_val, y_val),
        "silhouette": float(silhouette_score(Z_val, y_val)),   # -1 .. 1, higher = tighter genre clusters
    }
