"""Regression metrics used for every table. All inputs in kg/h."""

import numpy as np


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def log_mae(y, p):
    """Mean absolute error of log1p values (scale-free; ~ relative error for large values)."""
    return float(np.mean(np.abs(np.log1p(y) - np.log1p(np.maximum(p, 0.0)))))


def bias(y, p):
    """Mean prediction error (p - y) in kg/h. Measures systematic over- or under-prediction."""
    return float(np.mean(p - y))


def mape(y, p, floor=10.0):
    """Kept only for backwards comparison. Not reported in main tables: unstable at low load."""
    return float(np.mean(np.abs(y - p) / np.maximum(np.abs(y), floor)) * 100.0)


METRICS = {"MAE": mae, "RMSE": rmse, "R2": r2, "logMAE": log_mae, "Bias": bias}


def all_metrics(y, p) -> dict:
    return {k: f(y, p) for k, f in METRICS.items()}


def facility_bootstrap_ci(y, P, fac, metric=mae, n_boot=2000, seed=0, alpha=0.05, clusters=None):
    """95% CI of `metric` by resampling facilities or clusters with replacement.

    y: (n,), P: (n_seeds, n) predictions, fac: (n,) facility ids.
    clusters: optional mapping or array of cluster ids. If provided, clusters
              are resampled with replacement, including all facilities within.
    For each replicate the metric is averaged over seeds.
    """
    rng = np.random.default_rng(seed)
    if clusters is not None:
        if isinstance(clusters, dict):
            group_ids = np.array([clusters[f] for f in fac])
        else:
            group_ids = np.asarray(clusters)
    else:
        group_ids = fac

    unique_groups = np.unique(group_ids)
    rows = {g: np.flatnonzero(group_ids == g) for g in unique_groups}
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([rows[g] for g in pick])
        stats[b] = np.mean([metric(y[idx], P[s, idx]) for s in range(P.shape[0])])
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def paired_facility_bootstrap_diff_ci(y, P_model, P_ref, fac, metric=mae, n_boot=2000, seed=0, alpha=0.05, clusters=None):
    """95% CI of Delta = Metric(model) - Metric(ref) by resampling facilities or clusters with replacement.

    If 0 is within [ci_lo, ci_hi], the difference is not statistically significant.
    """
    rng = np.random.default_rng(seed)
    if clusters is not None:
        if isinstance(clusters, dict):
            group_ids = np.array([clusters[f] for f in fac])
        else:
            group_ids = np.asarray(clusters)
    else:
        group_ids = fac

    unique_groups = np.unique(group_ids)
    rows = {g: np.flatnonzero(group_ids == g) for g in unique_groups}
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([rows[g] for g in pick])
        m_val = np.mean([metric(y[idx], P_model[s, idx]) for s in range(P_model.shape[0])])
        r_val = np.mean([metric(y[idx], P_ref[s, idx]) for s in range(P_ref.shape[0])])
        diffs[b] = m_val - r_val
    return float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


cluster_bootstrap_ci = facility_bootstrap_ci
paired_cluster_bootstrap_diff_ci = paired_facility_bootstrap_diff_ci
