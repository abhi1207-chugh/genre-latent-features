"""Classical baselines on fixed features: 5-NN, RBF-SVM, and the paper's 2-layer NN.

Every model is fit on train and judged on validation only.
Each function returns (fitted_model, validation clip probabilities, info dict).
"""
import copy

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

CLASSES = np.array([0, 1, 2, 3])


def knn_baseline(X_train, y_train, X_val, k=5):
    model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    return model, model.predict_proba(X_val), {"k": k}


def svm_baseline(X_train, y_train, X_val, y_val, C_values=(0.1, 1, 10, 100)):
    """RBF SVM. C is chosen by validation accuracy; then the best C is refit with
    probability outputs (needed for track-level averaging).

    SVMs output scores, not probabilities. CalibratedClassifierCV turns the
    scores into probabilities (Platt scaling) using cross-validation on train.
    """
    val_acc = {}
    for C in C_values:
        model = SVC(kernel="rbf", C=C, gamma="scale").fit(X_train, y_train)
        val_acc[C] = accuracy_score(y_val, model.predict(X_val))

    best_C = max(val_acc, key=val_acc.get)
    model = CalibratedClassifierCV(SVC(kernel="rbf", C=best_C, gamma="scale"),
                                   ensemble=False).fit(X_train, y_train)
    return model, model.predict_proba(X_val), {"best_C": best_C, "val_acc_per_C": val_acc}


def two_layer_nn_baseline(X_train, y_train, X_val, y_val,
                          max_epochs=500, patience=20, seed=42):
    """Paper's baseline: 500 -> 128 (tanh) -> 4, Adam lr 1e-4, batch 512.

    Trained one epoch at a time; stops when validation loss has not improved
    for `patience` epochs and returns the best epoch's model.
    """
    model = MLPClassifier(hidden_layer_sizes=(128,), activation="tanh", solver="adam",
                          learning_rate_init=1e-4, batch_size=512, random_state=seed)

    best_loss, best_model, best_epoch = np.inf, None, 0
    for epoch in range(1, max_epochs + 1):
        model.partial_fit(X_train, y_train, classes=CLASSES)   # one pass over train
        val_loss = log_loss(y_val, model.predict_proba(X_val), labels=CLASSES)

        if val_loss < best_loss:
            best_loss, best_model, best_epoch = val_loss, copy.deepcopy(model), epoch
        elif epoch - best_epoch >= patience:
            break

    info = {"best_epoch": best_epoch, "epochs_run": epoch, "best_val_loss": best_loss}
    return best_model, best_model.predict_proba(X_val), info
