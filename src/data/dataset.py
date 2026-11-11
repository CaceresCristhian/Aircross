"""
Feature engineering, window construction and data splits for the Phase A benchmark.

A sample is a window of `seq_len` CONSECUTIVE overpass slots at one facility.
With overpass hours [13, 14] LST and seq_len = 4, one window covers 13:00 and 14:00 LST
on two consecutive calendar days. Windows with any missing slot are discarded.
The target is NOx (kg/h) at the last slot; it must be an operating hour with complete NOx.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WEATHER_FEATURES = ["wind_u_ms", "wind_v_ms", "wind_speed_10m", "surface_pressure",
                    "temperature_2m", "boundary_layer_height", "month_sin", "month_cos", "hour_flag"]
LOAD_FEATURES = ["load_mw"]
FEATURE_SETS = {
    "weather+load": WEATHER_FEATURES + LOAD_FEATURES,
    "weather": WEATHER_FEATURES,
}


def add_calendar_features(df: pd.DataFrame, hours: list[int]) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["Date"])
    df["month_sin"] = np.sin(2 * np.pi * (dt.dt.month - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (dt.dt.month - 1) / 12.0)
    df["hour_flag"] = (df["Hour"] == max(hours)).astype(float)
    # Integer slot index: consecutive overpass hours have consecutive slots.
    day = (dt - pd.Timestamp("2000-01-01")).dt.days.astype(np.int64)
    pos = df["Hour"].map({h: i for i, h in enumerate(sorted(hours))}).astype(np.int64)
    df["slot"] = day * len(hours) + pos
    df["year"] = dt.dt.year
    return df


@dataclass
class Windows:
    X: np.ndarray            # (N, T, F) float32, raw (unnormalized) features
    y: np.ndarray            # (N,) NOx kg/h at last slot
    facility: np.ndarray     # (N,) facility id
    year: np.ndarray         # (N,) year of the target slot
    date: np.ndarray         # (N,) date string of the target slot
    hour: np.ndarray         # (N,) hour (LST) of the target slot
    features: list[str]


def build_windows(df: pd.DataFrame, features: list[str], seq_len: int) -> Windows:
    """Build windows of `seq_len` consecutive slots per facility. `df` must contain `slot`."""
    Xs, ys, fs, yrs, dts, hrs = [], [], [], [], [], []
    for fid, g in df.sort_values(["Facility ID", "slot"]).groupby("Facility ID", sort=True):
        slot = g["slot"].to_numpy()
        feats = g[features].to_numpy(dtype=np.float64)
        target_ok = (g["operating"] & g["nox_complete"]).to_numpy()
        finite = np.isfinite(feats).all(axis=1)
        n = len(g)
        if n < seq_len:
            continue
        ends = np.arange(seq_len - 1, n)
        starts = ends - (seq_len - 1)
        consecutive = (slot[ends] - slot[starts]) == (seq_len - 1)
        # all rows in window finite
        fin_cum = np.concatenate([[0], np.cumsum(~finite)])
        window_finite = (fin_cum[ends + 1] - fin_cum[starts]) == 0
        valid = consecutive & window_finite & target_ok[ends]
        ends = ends[valid]
        if len(ends) == 0:
            continue
        idx = ends[:, None] - np.arange(seq_len - 1, -1, -1)[None, :]
        Xs.append(feats[idx].astype(np.float32))
        ys.append(g["nox_kgh"].to_numpy()[ends])
        fs.append(np.full(len(ends), fid))
        yrs.append(g["year"].to_numpy()[ends])
        dts.append(g["Date"].to_numpy()[ends])
        hrs.append(g["Hour"].to_numpy()[ends])
    return Windows(np.concatenate(Xs), np.concatenate(ys).astype(np.float64), np.concatenate(fs),
                   np.concatenate(yrs), np.concatenate(dts), np.concatenate(hrs), list(features))


def subset(w: Windows, mask: np.ndarray) -> Windows:
    return Windows(w.X[mask], w.y[mask], w.facility[mask], w.year[mask], w.date[mask], w.hour[mask], w.features)


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Normalizer":
        flat = X.reshape(-1, X.shape[-1])
        return cls(flat.mean(axis=0), flat.std(axis=0) + 1e-6)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    def to_dict(self, features) -> dict:
        return {f: {"mean": float(m), "std": float(s)} for f, m, s in zip(features, self.mean, self.std)}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * R * np.arcsin(np.sqrt(a)))


def cluster_facilities(coords: dict[int, tuple[float, float]], threshold_km: float = 50.0) -> list[list[int]]:
    """Group facilities within threshold_km into connected clusters to prevent geographic leakage."""
    fids = sorted(coords.keys())
    adj = {f: set() for f in fids}
    for i in range(len(fids)):
        for j in range(i + 1, len(fids)):
            lat1, lon1 = coords[fids[i]]
            lat2, lon2 = coords[fids[j]]
            if haversine_km(lat1, lon1, lat2, lon2) < threshold_km:
                adj[fids[i]].add(fids[j])
                adj[fids[j]].add(fids[i])
    visited = set()
    clusters = []
    for f in fids:
        if f not in visited:
            comp = []
            q = [f]
            visited.add(f)
            while q:
                curr = q.pop()
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
            clusters.append(sorted(comp))
    return sorted(clusters, key=lambda c: c[0])


def spatial_folds(facilities: np.ndarray, k: int, seed: int,
                  coords: dict[int, tuple[float, float]] | None = None) -> list[np.ndarray]:
    """Group facilities by 50km clusters and partition clusters into k disjoint test groups."""
    fids = sorted(np.unique(facilities).tolist())
    if coords is not None and all(f in coords for f in fids):
        clusters = cluster_facilities({f: coords[f] for f in fids}, threshold_km=50.0)
    else:
        clusters = [[f] for f in fids]

    rng = np.random.default_rng(seed)
    cluster_order = list(range(len(clusters)))
    rng.shuffle(cluster_order)
    splits = np.array_split(cluster_order, k)

    folds = []
    for c_idxs in splits:
        fold_fids = sorted([f for c_idx in c_idxs for f in clusters[c_idx]])
        folds.append(np.array(fold_fids, dtype=np.int64))
    return folds


def make_splits(w: Windows, cfg_split: dict, coords: dict[int, tuple[float, float]] | None = None) -> list[dict]:
    """Return list of split dicts with boolean masks fit/val/test over windows."""
    splits = []
    yr = w.year
    splits.append({
        "split": "temporal", "fold": 0,
        "fit": np.isin(yr, cfg_split["temporal_train_years"]),
        "val": np.isin(yr, cfg_split["temporal_val_years"]),
        "test": np.isin(yr, cfg_split["temporal_test_years"]),
        "test_facilities": sorted(np.unique(w.facility).tolist()),
    })
    folds = spatial_folds(w.facility, cfg_split["spatial_folds"], cfg_split["spatial_split_seed"], coords)
    for k, test_fids in enumerate(folds):
        is_test_fac = np.isin(w.facility, test_fids)
        splits.append({
            "split": "spatial", "fold": k,
            "fit": ~is_test_fac & np.isin(yr, cfg_split["spatial_fit_years"]),
            "val": ~is_test_fac & np.isin(yr, cfg_split["spatial_val_years"]),
            "test": is_test_fac,
            "test_facilities": test_fids.tolist(),
        })
    return splits
