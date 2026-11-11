"""Neural and tabular baselines. All receive exactly the same windows and features as AirCross."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression


class MLPBaseline(nn.Module):
    """Flattened window -> 2 hidden layers -> log1p(NOx)."""

    def __init__(self, n_features: int, seq_len: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features * seq_len, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, x):
        return self.net(x.flatten(1)).squeeze(-1)


class CNNBaseline(nn.Module):
    """1D convolution over the time axis -> global average pool -> log1p(NOx)."""

    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=2, padding=1), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=2), nn.GELU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.head(self.conv(x.transpose(1, 2)).squeeze(-1)).squeeze(-1)


# ---------------- tabular / trivial baselines (predict NOx in kg/h) ----------------

class GlobalMean:
    name = "Global mean"

    def fit(self, X, y, fac):
        self.m = float(np.mean(y)); return self

    def predict(self, X, fac):
        return np.full(len(X), self.m)


class FacilityMean:
    """Per-facility training mean. Undefined for unseen facilities (returns NaN)."""
    name = "Per-facility mean"

    def fit(self, X, y, fac):
        self.m = {f: float(np.mean(y[fac == f])) for f in np.unique(fac)}; return self

    def predict(self, X, fac):
        return np.array([self.m.get(f, np.nan) for f in fac])


class LoadRegression:
    """Emission-factor baseline: NOx = a + b * load (last slot), fitted in kg/h."""
    name = "Load regression"

    def __init__(self, load_idx: int):
        self.load_idx = load_idx

    def fit(self, X, y, fac):
        self.lr = LinearRegression().fit(X[:, -1, [self.load_idx]], y); return self

    def predict(self, X, fac):
        return np.maximum(self.lr.predict(X[:, -1, [self.load_idx]]), 0.0)


class GradientBoosting:
    """HistGradientBoosting on the flattened window, log1p target."""
    name = "Gradient boosting"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X, y, fac):
        self.gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                                early_stopping=False, random_state=self.seed)
        self.gb.fit(X.reshape(len(X), -1), np.log1p(y)); return self

    def predict(self, X, fac):
        return np.maximum(np.expm1(self.gb.predict(X.reshape(len(X), -1))), 0.0)
