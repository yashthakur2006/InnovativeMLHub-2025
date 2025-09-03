import os, numpy as np, pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

def synthetic_regression(n=2000, d=16, seed=42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=(d, 1))
    y = (X @ w)[:, 0] + 0.25 * rng.normal(size=n)
    return X, y

def synthetic_classification(n=2000, d=16, seed=42, n_classes=2) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    W = rng.normal(size=(d, n_classes))
    logits = X @ W
    y = logits.argmax(axis=1)
    return X, y

def as_torch(X, y=None):
    import torch
    Xt = torch.tensor(X, dtype=torch.float32)
    if y is None:
        return Xt, None
    yt = torch.tensor(y, dtype=torch.long) if y.dtype.kind in ("i", "u") else torch.tensor(y, dtype=torch.float32)
    return Xt, yt

def toy_classification_split(seed=42):
    X, y = synthetic_classification(seed=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    return as_torch(Xtr, ytr) + as_torch(Xte, yte)

def toy_regression_split(seed=42):
    X, y = synthetic_regression(seed=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    return as_torch(Xtr, ytr) + as_torch(Xte, yte)
