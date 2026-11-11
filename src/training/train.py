"""
AirCross Phase A training and benchmarking.

For every split (1 temporal + K spatial folds) and every seed:
- fit normalization on the fit set only,
- train neural models with validation-based early stopping and LR schedule,
- fit tabular baselines on the same windows,
- save predictions on the test set.

Output: experiments/<run_id>/
    config.yaml        exact config used
    splits.json        facilities per fold, window counts
    normalizers.json   per split normalization statistics
    predictions.parquet  one row per (model, feature_set, split, fold, seed, test window)
    training_log.csv   epochs, best val loss per neural run
    checkpoints/       best.pt per neural run
Tables are produced separately by src/evaluation/generate_latex_tables.py.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import CONFIG_PATH, EXPERIMENTS_DIR, PROCESSED_DIR, load_config  # noqa: E402
from src.data.dataset import (FEATURE_SETS, Normalizer, add_calendar_features,  # noqa: E402
                              build_windows, make_splits, subset)
from src.models.aircross import AirCross  # noqa: E402
from src.models.baselines import (CNNBaseline, FacilityMean, GlobalMean,  # noqa: E402
                                  GradientBoosting, LoadRegression, MLPBaseline)

MET_FEATURES = ["wind_u_ms", "wind_v_ms", "wind_speed_10m", "surface_pressure",
                "temperature_2m", "boundary_layer_height"]

# (model name, feature set) combinations evaluated
NEURAL_RUNS = [
    ("AirCross", "weather+load"),
    ("AirCross (no cross-attention)", "weather+load"),
    ("MLP", "weather+load"),
    ("1D-CNN", "weather+load"),
    ("AirCross", "weather"),
]
TABULAR_RUNS = [
    ("Global mean", "weather+load"),
    ("Per-facility mean", "weather+load"),
    ("Load regression", "weather+load"),
    ("Gradient boosting", "weather+load"),
    ("Gradient boosting", "weather"),
]


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_neural(name: str, features: list[str], cfg: dict) -> nn.Module:
    m, T, F = cfg["model"], cfg["model"]["seq_len"], len(features)
    if name.startswith("AirCross"):
        return AirCross(F, [features.index(f) for f in MET_FEATURES], T, m["embed_dim"], m["num_heads"],
                        m["num_layers"], m["dropout"], use_cross_attention="no cross" not in name)
    if name == "MLP":
        return MLPBaseline(F, T, dropout=m["dropout"])
    if name == "1D-CNN":
        return CNNBaseline(F, dropout=m["dropout"])
    raise ValueError(name)


def build_tabular(name: str, features: list[str], seed: int):
    if name == "Global mean":
        return GlobalMean()
    if name == "Per-facility mean":
        return FacilityMean()
    if name == "Load regression":
        return LoadRegression(features.index("load_mw"))
    if name == "Gradient boosting":
        return GradientBoosting(seed)
    raise ValueError(name)


def train_neural(model, Xf, yf, Xv, yv, tcfg, device, ckpt: Path):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    loss_fn = nn.SmoothL1Loss()
    Xf, yf = torch.from_numpy(Xf).to(device), torch.from_numpy(np.log1p(yf).astype(np.float32)).to(device)
    Xv, yv = torch.from_numpy(Xv).to(device), torch.from_numpy(np.log1p(yv).astype(np.float32)).to(device)
    bs, best, bad, best_epoch = tcfg["batch_size"], float("inf"), 0, 0
    for epoch in range(1, tcfg["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(len(Xf), device=device)
        for i in range(0, len(Xf), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xf[b]), yf[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val = evaluate_loss(model, Xv, yv, loss_fn, bs)
        sched.step(val)
        if val < best - 1e-4:
            best, bad, best_epoch = val, 0, epoch
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= tcfg["patience"]:
                break
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return best, best_epoch, epoch


@torch.no_grad()
def evaluate_loss(model, X, y, loss_fn, bs):
    model.eval()
    tot = 0.0
    for i in range(0, len(X), bs * 8):
        tot += loss_fn(model(X[i:i + bs * 8]), y[i:i + bs * 8]).item() * len(X[i:i + bs * 8])
    return tot / len(X)


@torch.no_grad()
def predict_neural(model, X, device, bs=4096) -> np.ndarray:
    model.eval()
    X = torch.from_numpy(X).to(device)
    out = [model(X[i:i + bs]).cpu().numpy() for i in range(0, len(X), bs)]
    return np.maximum(np.expm1(np.concatenate(out)), 0.0)


def load_windows(cfg) -> dict:
    df = pd.read_parquet(PROCESSED_DIR / "cems_era5_synchronized.parquet")
    df = add_calendar_features(df, cfg["data"]["overpass_hours_lst"])
    full = FEATURE_SETS["weather+load"]
    w = build_windows(df, full, cfg["model"]["seq_len"])
    return w


def main(config_path: str | Path = CONFIG_PATH, quick: bool = False) -> Path:
    cfg = load_config(config_path)
    tcfg = dict(cfg["train"])
    seeds = tcfg["seeds"][:1] if quick else tcfg["seeds"]
    if quick:
        tcfg["max_epochs"] = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + ("_quick" if quick else "")
    run_dir = EXPERIMENTS_DIR / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, run_dir / "config.yaml")

    w = load_windows(cfg)
    print(f"Windows: {len(w.y):,} targets from {len(np.unique(w.facility))} facilities; device={device}")
    reg_path = PROCESSED_DIR / "facility_registry.csv"
    coords = None
    if reg_path.exists():
        rdf = pd.read_csv(reg_path)
        coords = {int(r["Facility ID"]): (float(r["latitude"]), float(r["longitude"])) for _, r in rdf.iterrows()}
    splits = make_splits(w, cfg["split"], coords=coords)
    json.dump([{"split": s["split"], "fold": s["fold"], "test_facilities": [int(f) for f in s["test_facilities"]],
                "n_fit": int(s["fit"].sum()), "n_val": int(s["val"].sum()), "n_test": int(s["test"].sum())}
               for s in splits], open(run_dir / "splits.json", "w"), indent=2)

    full_feats = w.features
    preds, logs, norms = [], [], {}
    t0 = time.time()
    for sp in splits:
        tag = f"{sp['split']}{sp['fold']}"
        fit_w, val_w, test_w = subset(w, sp["fit"]), subset(w, sp["val"]), subset(w, sp["test"])
        for fs_name in ("weather+load", "weather"):
            feats = FEATURE_SETS[fs_name]
            cols = [full_feats.index(f) for f in feats]
            Xf_raw, Xv_raw, Xt_raw = fit_w.X[..., cols], val_w.X[..., cols], test_w.X[..., cols]
            norm = Normalizer.fit(Xf_raw)
            norms[f"{tag}|{fs_name}"] = norm.to_dict(feats)
            Xf, Xv, Xt = norm.transform(Xf_raw), norm.transform(Xv_raw), norm.transform(Xt_raw)

            def record(model_name, seed, p):
                preds.append(pd.DataFrame({
                    "model": model_name, "feature_set": fs_name, "split": sp["split"], "fold": sp["fold"],
                    "seed": seed, "facility": test_w.facility, "year": test_w.year, "date": test_w.date,
                    "hour": test_w.hour, "y_true": test_w.y, "y_pred": p}))

            for name, fsn in TABULAR_RUNS:
                if fsn != fs_name:
                    continue
                if name == "Per-facility mean" and sp["split"] == "spatial":
                    continue  # undefined for unseen facilities
                m = build_tabular(name, feats, 0).fit(Xf_raw, fit_w.y, fit_w.facility)
                record(name, 0, m.predict(Xt_raw, test_w.facility))

            for name, fsn in NEURAL_RUNS:
                if fsn != fs_name:
                    continue
                for seed in seeds:
                    set_seed(seed)
                    model = build_neural(name, feats, cfg)
                    ckpt = run_dir / "checkpoints" / f"{tag}_{name.replace(' ', '_').replace('(', '').replace(')', '')}_{fs_name}_s{seed}.pt"
                    best, best_ep, last_ep = train_neural(model, Xf, fit_w.y, Xv, val_w.y, tcfg, device, ckpt)
                    record(name, seed, predict_neural(model, Xt, device))
                    logs.append({"split": sp["split"], "fold": sp["fold"], "model": name, "feature_set": fs_name,
                                 "seed": seed, "best_val_loss": best, "best_epoch": best_ep, "last_epoch": last_ep})
                    print(f"[{time.time() - t0:6.0f}s] {tag:9s} {name:30s} {fs_name:13s} seed={seed} "
                          f"val={best:.4f} best_ep={best_ep}/{last_ep}", flush=True)

    pd.concat(preds, ignore_index=True).to_parquet(run_dir / "predictions.parquet", index=False)
    pd.DataFrame(logs).to_csv(run_dir / "training_log.csv", index=False)
    json.dump(norms, open(run_dir / "normalizers.json", "w"), indent=2)
    (EXPERIMENTS_DIR / "LATEST").write_text(run_id)
    print(f"Done in {time.time() - t0:.0f}s -> {run_dir}")
    return run_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--quick", action="store_true", help="1 seed, 3 epochs smoke test")
    args = ap.parse_args()
    main(args.config, args.quick)
