"""
Generate publication-quality paper figures (300 DPI) from real processed data and experiment outputs.

fig_facilities_map.png      : selected facilities (EIA-860 coordinates) coloured by mean NOx
fig_pred_vs_obs.png         : temporal test, observed vs predicted (AirCross vs gradient boosting)
fig_spatial_per_facility.png: spatial hold-out MAE per facility for the main models
fig_regime_breakdown.png    : wind speed regime and regional climate regime breakdowns
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import EXPERIMENTS_DIR, FIGURES_DIR, RAW_DIR, PROCESSED_DIR
from src.data.registry import load_registry

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def facilities_map(reg):
    states = gpd.read_file(RAW_DIR / "us_states.geojson")
    target_states = ["Texas", "Ohio", "Pennsylvania"]
    
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    states.plot(ax=ax, color="#f5f7fa", edgecolor="#cbd5e1", linewidth=0.6)
    states[states["name"].isin(target_states)].plot(
        ax=ax, color="#e2e8f0", edgecolor="#64748b", linewidth=0.9
    )
    
    sc = ax.scatter(
        reg["longitude"], reg["latitude"],
        c=reg["mean_nox_kgh"], s=55, cmap="YlOrRd",
        edgecolor="#1e293b", linewidth=0.7, zorder=5, alpha=0.95
    )
    
    ax.text(-99.0, 31.5, "Texas\n(13 facilities)", fontsize=8, color="#334155", fontweight="bold", ha="center")
    ax.text(-82.5, 40.2, "Ohio (4)", fontsize=7.5, color="#334155", fontweight="bold", ha="right")
    ax.text(-77.2, 41.2, "PA (2)", fontsize=7.5, color="#334155", fontweight="bold", ha="left")
    
    cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Mean Operating Afternoon NO$_x$ (kg/h)", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)
    
    ax.set_xlim(-108, -72)
    ax.set_ylim(25, 44.5)
    ax.set_xlabel(r"Longitude ($^\circ$W)", fontsize=8.5)
    ax.set_ylabel(r"Latitude ($^\circ$N)", fontsize=8.5)
    ax.set_title("Geographic Distribution of the 19 EPA CEMS Baseload Stations (OH, PA, TX)", fontsize=9, pad=6)
    
    fig.savefig(FIGURES_DIR / "fig_facilities_map.png")
    plt.close(fig)


def pred_vs_obs(df):
    t = df[(df["split"] == "temporal") & (df["feature_set"] == "weather+load") & (df["seed"] == 0)]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2), sharex=True, sharey=True)
    lim = float(np.quantile(t["y_true"], 0.995))
    
    for ax, m, title in zip(axes, ["Gradient boosting", "AirCross"], ["(a) Histogram Gradient Boosting", "(b) AirCross (Cross-Attention)"]):
        g = t[t["model"] == m]
        y_t = g["y_true"].to_numpy()
        y_p = g["y_pred"].to_numpy()
        mae_val = float(np.mean(np.abs(y_t - y_p)))
        r_val = float(np.corrcoef(y_t, y_p)[0, 1])
        r2_val = 1.0 - float(np.sum((y_t - y_p)**2) / np.sum((y_t - np.mean(y_t))**2))
        
        hb = ax.hexbin(y_t, y_p, gridsize=45, extent=(0, lim, 0, lim), bins="log", cmap="Blues", mincnt=1, edgecolors="none")
        ax.plot([0, lim], [0, lim], color="#dc2626", linestyle="--", lw=1.0, label="1:1 Ideal")
        
        ax.set_title(title, fontsize=8.5, pad=4)
        ax.set_xlabel("Observed Stack NO$_x$ (kg/h)", fontsize=8)
        
        stats_text = f"MAE: {mae_val:.1f} kg/h\n$R^2$: {r2_val:.3f}\n$r$: {r_val:.3f}"
        ax.text(0.05, 0.92, stats_text, transform=ax.transAxes, fontsize=7.5,
                verticalalignment="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cbd5e1", alpha=0.9))
        ax.grid(True, linestyle=":", alpha=0.4)
        
    axes[0].set_ylabel("Predicted Stack NO$_x$ (kg/h)", fontsize=8)
    axes[1].legend(loc="lower right", fontsize=7.5)
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_pred_vs_obs.png")
    plt.close(fig)


def spatial_per_facility(run_dir, reg):
    pf = pd.read_csv(run_dir / "summary_spatial_per_facility.csv")
    models = ["Global mean", "Load regression", "Gradient boosting", "AirCross"]
    colors = ["#94a3b8", "#f59e0b", "#3b82f6", "#10b981"]
    
    piv = pf.pivot(index="facility", columns="model", values="MAE")[models]
    names = reg.set_index("Facility ID")["cems_name"]
    piv = piv.loc[piv["AirCross"].sort_values().index]
    
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    y = np.arange(len(piv))
    height = 0.20
    
    for i, (m, c) in enumerate(zip(models, colors)):
        ax.barh(y + (i - 1.5) * height, piv[m], height=height, label=m, color=c, edgecolor="#1e293b", linewidth=0.3)
        
    ax.set_yticks(y)
    ax.set_yticklabels([f"{names.get(f, f)} ({f})" for f in piv.index], fontsize=6.8)
    ax.set_xlabel("50 km Cluster-Independent Spatial Hold-Out MAE (kg/h)", fontsize=8)
    ax.set_title("Per-Facility Out-of-Sample Performance on Unseen Facilities", fontsize=9, pad=6)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.95)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_spatial_per_facility.png")
    plt.close(fig)


def regime_breakdown(run_dir):
    preds = pd.read_parquet(run_dir / "predictions.parquet")
    cems = pd.read_parquet(PROCESSED_DIR / "cems_era5_synchronized.parquet")
    
    cems_cols = cems[["Facility ID", "Date", "Hour", "wind_speed_10m", "state"]].rename(
        columns={"Facility ID": "facility", "Date": "date", "Hour": "hour"}
    )
    cems_cols["facility"] = cems_cols["facility"].astype(str)
    preds["facility"] = preds["facility"].astype(str)
    cems_cols["date"] = cems_cols["date"].astype(str)
    preds["date"] = preds["date"].astype(str)
    
    t = preds[(preds["split"] == "temporal") & (preds["feature_set"] == "weather+load")].copy()
    t_avg = t.groupby(["facility", "date", "hour", "model"], as_index=False).agg({"y_true": "first", "y_pred": "mean"})
    m = t_avg.merge(cems_cols, on=["facility", "date", "hour"], how="left")
    
    models = ["Per-facility mean", "Load regression", "Gradient boosting", "AirCross"]
    labels = ["Facility mean", "Load reg.", "Grad. boost.", "AirCross"]
    calm_maes, adv_maes, tx_maes, mw_maes = [], [], [], []
    
    for mdl in models:
        sub = m[m["model"] == mdl]
        calm_maes.append(float(np.mean(np.abs(sub[sub["wind_speed_10m"] < 2.0]["y_true"] - sub[sub["wind_speed_10m"] < 2.0]["y_pred"]))))
        adv_maes.append(float(np.mean(np.abs(sub[sub["wind_speed_10m"] >= 2.0]["y_true"] - sub[sub["wind_speed_10m"] >= 2.0]["y_pred"]))))
        tx_maes.append(float(np.mean(np.abs(sub[sub["state"] == "TX"]["y_true"] - sub[sub["state"] == "TX"]["y_pred"]))))
        mw_maes.append(float(np.mean(np.abs(sub[sub["state"].isin(["OH", "PA"])]["y_true"] - sub[sub["state"].isin(["OH", "PA"])]["y_pred"]))))
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    x = np.arange(len(models))
    width = 0.35
    
    ax1.bar(x - width/2, calm_maes, width, label=r"Calm ($w < 2\,\mathrm{m/s}$, $n=3{,}877$)", color="#fb923c", edgecolor="#1e293b", linewidth=0.4)
    ax1.bar(x + width/2, adv_maes, width, label=r"Advective ($w \geq 2\,\mathrm{m/s}$, $n=14{,}421$)", color="#38bdf8", edgecolor="#1e293b", linewidth=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7.0)
    ax1.set_ylabel("Temporal Test MAE (kg/h)", fontsize=8)
    ax1.set_title("(a) Atmospheric Wind Regime", fontsize=8.5)
    ax1.legend(fontsize=7.0)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)
    
    ax2.bar(x - width/2, tx_maes, width, label=r"Texas (13 plants, $n=12{,}681$)", color="#f472b6", edgecolor="#1e293b", linewidth=0.4)
    ax2.bar(x + width/2, mw_maes, width, label=r"OH \& PA (6 plants, $n=5{,}617$)", color="#818cf8", edgecolor="#1e293b", linewidth=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7.0)
    ax2.set_title("(b) Regional Fleet Subgroup", fontsize=8.5)
    ax2.legend(fontsize=7.0)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_regime_breakdown.png")
    plt.close(fig)


def main(run=None):
    run_id = run or (EXPERIMENTS_DIR / "LATEST").read_text().strip()
    run_dir = EXPERIMENTS_DIR / run_id
    reg = load_registry()
    df = pd.read_parquet(run_dir / "predictions.parquet")
    facilities_map(reg)
    pred_vs_obs(df)
    spatial_per_facility(run_dir, reg)
    regime_breakdown(run_dir)
    print(f"Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    main(ap.parse_args().run)
