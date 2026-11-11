"""
Generate every number used in the paper from experiment outputs and processed data.

Reads : experiments/<run_id>/predictions.parquet, splits.json (run_id from experiments/LATEST
        or --run), data/02_processed/facility_registry.csv, cems_era5_synchronized.parquet
Writes: paper/tables/*.tex      (tables included with \\input)
        paper/tables/macros.tex (\\newcommand for every number quoted in the text)
        experiments/<run_id>/summary_*.csv, results_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import EXPERIMENTS_DIR, PROCESSED_DIR, TABLES_DIR, load_config  # noqa: E402
from src.data.dataset import cluster_facilities, haversine_km  # noqa: E402
from src.data.registry import load_registry  # noqa: E402
from src.evaluation.metrics import (METRICS, all_metrics, facility_bootstrap_ci,  # noqa: E402
                                    paired_facility_bootstrap_diff_ci, mae)

MODEL_ORDER = ["Global mean", "Per-facility mean", "Load regression", "Gradient boosting",
               "MLP", "1D-CNN", "AirCross (no cross-attention)", "AirCross"]
MACROS: dict[str, str] = {}


def macro(name: str, value) -> None:
    assert name.isalpha(), name
    MACROS[name] = str(value)


def seed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Metrics per seed (pooled over folds), then mean/std over seeds."""
    rows = []
    for (model, fs), g in df.groupby(["model", "feature_set"]):
        per_seed = pd.DataFrame([all_metrics(s["y_true"].to_numpy(), s["y_pred"].to_numpy())
                                 for _, s in g.groupby("seed")])
        row = {"model": model, "feature_set": fs, "n_seeds": len(per_seed), "n_test": int(len(g) / len(per_seed))}
        for k in METRICS:
            row[f"{k}_mean"] = per_seed[k].mean()
            row[f"{k}_std"] = per_seed[k].std(ddof=1) if len(per_seed) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    out["order"] = out["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    return out.sort_values(["feature_set", "order"], ascending=[False, True]).drop(columns="order")


def add_bootstrap(summary: pd.DataFrame, df: pd.DataFrame, cfg, cluster_map: dict[int, int] | None = None) -> pd.DataFrame:
    ref_df = df[(df["model"] == "Load regression") & (df["feature_set"] == "weather+load")]
    piv_ref = ref_df.pivot_table(index=["facility", "date", "hour"], columns="seed", values="y_pred")
    
    lo, hi = [], []
    d_lo, d_hi = [], []
    for r in summary.itertuples():
        g = df[(df["model"] == r.model) & (df["feature_set"] == r.feature_set)]
        piv = g.pivot_table(index=["facility", "date", "hour"], columns="seed", values="y_pred")
        common_idx = piv.index.intersection(piv_ref.index) if len(piv_ref) else piv.index
        piv_sub = piv.loc[common_idx]
        y = g.groupby(["facility", "date", "hour"])["y_true"].first().loc[common_idx].to_numpy()
        fac = common_idx.get_level_values("facility").to_numpy()
        
        a, b = facility_bootstrap_ci(y, piv_sub.to_numpy().T, fac, mae,
                                     cfg["evaluation"]["bootstrap_samples"],
                                     cfg["evaluation"]["bootstrap_seed"],
                                     clusters=cluster_map)
        lo.append(a); hi.append(b)
        
        if r.model == "Load regression" and r.feature_set == "weather+load":
            d_lo.append(0.0); d_hi.append(0.0)
        elif len(piv_ref):
            piv_ref_sub = piv_ref.loc[common_idx]
            da, db = paired_facility_bootstrap_diff_ci(y, piv_sub.to_numpy().T, piv_ref_sub.to_numpy().T, fac, mae,
                                                       cfg["evaluation"]["bootstrap_samples"],
                                                       cfg["evaluation"]["bootstrap_seed"],
                                                       clusters=cluster_map)
            d_lo.append(da); d_hi.append(db)
        else:
            d_lo.append(float("nan")); d_hi.append(float("nan"))
            
    summary = summary.copy()
    summary["MAE_ci_lo"], summary["MAE_ci_hi"] = lo, hi
    summary["dMAE_ci_lo"], summary["dMAE_ci_hi"] = d_lo, d_hi
    return summary


def fmt(m, s, n_seeds, digits=1):
    return f"{m:.{digits}f}" if n_seeds <= 1 else f"{m:.{digits}f} $\\pm$ {s:.{digits}f}"


def write_results_table(summary, fname, caption, label, feature_set="weather+load", ci=False, extra=None):
    s = summary[summary["feature_set"] == feature_set]
    best = s["MAE_mean"].min()
    
    if ci:
        env = "table*"
        rbox = r"\resizebox{\textwidth}{!}{"
        cols = "lcccccc" + ("c" if extra is not None else "")
        head = r"\textbf{Model} & \textbf{MAE} & \textbf{RMSE} & \textbf{$R^2$} & \textbf{Bias} & \textbf{MAE 95\% CI} & \textbf{$\Delta$MAE vs LoadReg}"
        if extra is not None:
            head += r" & \textbf{MAE$_{24\text{--}25}$}"
    else:
        env = "table"
        rbox = r"\resizebox{\columnwidth}{!}{"
        cols = "lccccc"
        head = r"\textbf{Model} & \textbf{MAE} & \textbf{RMSE} & \textbf{$R^2$} & \textbf{logMAE} & \textbf{Bias}"
        
    lines = [r"\begin{" + env + r"}[htbp]", r"\caption{" + caption + "}", r"\label{" + label + "}", r"\centering",
             r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}", rbox, r"\begin{tabular}{" + cols + "}", r"\toprule",
             head + r" \\", r"\midrule"]
             
    for r in s.itertuples():
        name = r.model
        mae_s = fmt(r.MAE_mean, r.MAE_std, r.n_seeds)
        if abs(r.MAE_mean - best) < 1e-9:
            mae_s = r"\textbf{" + mae_s + "}"
            
        if ci:
            row = (f"{name} & {mae_s} & {fmt(r.RMSE_mean, r.RMSE_std, r.n_seeds)} & "
                   f"{fmt(r.R2_mean, r.R2_std, r.n_seeds, 3)} & {fmt(r.Bias_mean, r.Bias_std, r.n_seeds)} & "
                   f"[{r.MAE_ci_lo:.1f}, {r.MAE_ci_hi:.1f}]")
            if r.model == "Load regression":
                row += " & 0.0 (ref)"
            else:
                row += f" & [{r.dMAE_ci_lo:+.1f}, {r.dMAE_ci_hi:+.1f}]"
            if extra is not None:
                e = extra[(extra["model"] == r.model) & (extra["feature_set"] == r.feature_set)]
                row += f" & {fmt(e['MAE_mean'].iloc[0], e['MAE_std'].iloc[0], e['n_seeds'].iloc[0])}" if len(e) else " & --"
        else:
            row = (f"{name} & {mae_s} & {fmt(r.RMSE_mean, r.RMSE_std, r.n_seeds)} & "
                   f"{fmt(r.R2_mean, r.R2_std, r.n_seeds, 3)} & {fmt(r.logMAE_mean, r.logMAE_std, r.n_seeds, 3)} & "
                   f"{fmt(r.Bias_mean, r.Bias_std, r.n_seeds)}")
                   
        lines.append(row + r" \\")
        
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{" + env + r"}"]
    (TABLES_DIR / fname).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ablation_table(temp, spat, fname):
    lines = [r"\begin{table}[htbp]", r"\caption{Ablations: input features and cross-attention pooling (MAE, kg/h, mean $\pm$ s.d. over seeds).}",
             r"\label{tab:ablation}", r"\centering", r"\footnotesize", r"\resizebox{\columnwidth}{!}{", r"\begin{tabular}{llcc}", r"\toprule",
             r"\textbf{Model} & \textbf{Inputs} & \textbf{Temporal} & \textbf{Spatial} \\", r"\midrule"]
    combos = [("Gradient boosting", "weather"), ("Gradient boosting", "weather+load"),
              ("AirCross", "weather"), ("AirCross (no cross-attention)", "weather+load"), ("AirCross", "weather+load")]
    for m, fs in combos:
        vals = []
        for s in (temp, spat):
            r = s[(s["model"] == m) & (s["feature_set"] == fs)]
            vals.append(fmt(r["MAE_mean"].iloc[0], r["MAE_std"].iloc[0], r["n_seeds"].iloc[0]) if len(r) else "--")
        lines.append(f"{m} & {fs.replace('+', ' + ')} & {vals[0]} & {vals[1]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    (TABLES_DIR / fname).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset_tables(cfg, reg, splits, run_dir, coords=None, clusters=None):
    sync = pd.read_parquet(PROCESSED_DIR / "cems_era5_synchronized.parquet")
    n_windows = int(json.load(open(run_dir / "splits.json"))[0]["n_fit"] + json.load(open(run_dir / "splits.json"))[0]["n_val"]
                    + json.load(open(run_dir / "splits.json"))[0]["n_test"])
    usable = sync[sync["operating"] & sync["nox_complete"]]
    macro("NFacilities", len(reg))
    if clusters is not None:
        macro("NClusters", len(clusters))
        max_intra_dist = 0.0
        if coords is not None:
            for members in clusters:
                if len(members) > 1:
                    for i in range(len(members)):
                        for j in range(i + 1, len(members)):
                            d = haversine_km(coords[members[i]][0], coords[members[i]][1], coords[members[j]][0], coords[members[j]][1])
                            if d > max_intra_dist:
                                max_intra_dist = d
        macro("MaxClusterDistKm", f"{max_intra_dist:.1f}")
    for st in cfg["data"]["states"]:
        macro("NFac" + st.capitalize(), int((reg["state"] == st).sum()))
    macro("NFacilityHours", f"{len(sync):,}")
    macro("NOperatingHours", f"{len(usable):,}")
    macro("NNonOperatingHours", f"{int((~sync['operating']).sum()):,}")
    macro("NMissingNOxHours", f"{int((sync['operating'] & ~sync['nox_complete']).sum()):,}")
    macro("NWindows", f"{n_windows:,}")
    macro("MeanWindSpeed", f"{sync['wind_speed_10m'].mean():.2f}")
    macro("MeanNOx", f"{usable['nox_kgh'].mean():.1f}")
    macro("MedianNOx", f"{usable['nox_kgh'].median():.1f}")
    macro("NSeeds", len(cfg["train"]["seeds"]))
    macro("NFolds", cfg["split"]["spatial_folds"])
    macro("YearFirst", min(cfg["data"]["years"])); macro("YearLast", max(cfg["data"]["years"]))
    sel = cfg["data"]["selection"]
    macro("SelMinNOx", f"{sel['min_mean_nox_kgh']:.0f}")
    macro("SelMinHours", sel["min_operating_hours_per_year"])

    fold_of = {f: s["fold"] for s in splits if s["split"] == "spatial" for f in s["test_facilities"]}
    lines = [r"\begin{table}[htbp]", r"\caption{Facilities selected by the rule in Section~\ref{sec:data} (EIA-860 coordinates). Fold = spatial test fold (clustered within 50~km).}",
             r"\label{tab:facilities}", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
             r"\resizebox{\columnwidth}{!}{", r"\begin{tabular}{rlcccc}", r"\toprule",
             r"\textbf{ID} & \textbf{Facility (CEMS name)} & \textbf{St.} & \textbf{Lat, Lon} & \textbf{Mean NO$_x$ (kg/h)} & \textbf{Fold} \\",
             r"\midrule"]
    for r in reg.sort_values(["state", "Facility ID"]).itertuples():
        fid = int(getattr(r, "_1"))
        name = str(r.cems_name).replace("&", r"\&")
        lines.append(f"{fid} & {name} & {r.state} & {r.latitude:.3f}, {r.longitude:.3f} & {r.mean_nox_kgh:.0f} & {fold_of.get(fid, '--')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    (TABLES_DIR / "table_facilities.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_macros_for_results(temp, spat, spat_new):
    def get(s, m, fs, k="MAE_mean"):
        r = s[(s["model"] == m) & (s["feature_set"] == fs)]
        return float(r[k].iloc[0]) if len(r) else float("nan")

    short = {"Global mean": "GlobalMean", "Per-facility mean": "FacMean", "Load regression": "LoadReg",
             "Gradient boosting": "GB", "MLP": "MLP", "1D-CNN": "CNN", "AirCross (no cross-attention)": "AirCrossNoX",
             "AirCross": "AirCross"}
    for tag, s in (("Temp", temp), ("Spat", spat), ("SpatNew", spat_new)):
        for m, code in short.items():
            for fs, fcode in (("weather+load", ""), ("weather", "W")):
                for k, kc, d in (("MAE_mean", "MAE", 1), ("MAE_std", "MAEsd", 1), ("RMSE_mean", "RMSE", 1),
                                 ("R2_mean", "R", 3), ("logMAE_mean", "logMAE", 3), ("Bias_mean", "Bias", 1)):
                    v = get(s, m, fs, k)
                    if np.isfinite(v):
                        macro(f"{tag}{code}{fcode}{kc}", f"{v:.{d}f}")
                if "MAE_ci_lo" in s.columns:
                    lo, hi = get(s, m, fs, "MAE_ci_lo"), get(s, m, fs, "MAE_ci_hi")
                    if np.isfinite(lo):
                        macro(f"{tag}{code}{fcode}CIlo", f"{lo:.1f}"); macro(f"{tag}{code}{fcode}CIhi", f"{hi:.1f}")
                if "dMAE_ci_lo" in s.columns:
                    dlo, dhi = get(s, m, fs, "dMAE_ci_lo"), get(s, m, fs, "dMAE_ci_hi")
                    if np.isfinite(dlo):
                        macro(f"{tag}{code}{fcode}dCIlo", f"{dlo:+.1f}"); macro(f"{tag}{code}{fcode}dCIhi", f"{dhi:+.1f}")
                        
    # relative improvements quoted in the text
    for tag, s in (("Temp", temp), ("Spat", spat)):
        a = get(s, "AirCross", "weather+load"); gb = get(s, "Gradient boosting", "weather+load")
        gm = get(s, "Global mean", "weather+load"); lr = get(s, "Load regression", "weather+load")
        aw = get(s, "AirCross", "weather")
        macro(f"{tag}AirCrossVsGlobalPct", f"{100 * (gm - a) / gm:.1f}")
        macro(f"{tag}AirCrossVsLoadRegPct", f"{100 * (lr - a) / lr:.1f}")
        macro(f"{tag}GBVsAirCrossPct", f"{100 * (a - gb) / a:.1f}")
        macro(f"{tag}LoadGainPct", f"{100 * (aw - a) / aw:.1f}")
        macro(f"{tag}WeatherOnlyVsGlobalPct", f"{100 * (gm - aw) / gm:.1f}")


def write_facility_results(df_spat, reg, run_dir):
    rows = []
    for (fac, model), g in df_spat[df_spat["feature_set"] == "weather+load"].groupby(["facility", "model"]):
        rows.append({"facility": fac, "model": model,
                     "MAE": np.mean([mae(s["y_true"].to_numpy(), s["y_pred"].to_numpy()) for _, s in g.groupby("seed")]),
                     "mean_true": g["y_true"].mean()})
    out = pd.DataFrame(rows)
    out.to_csv(run_dir / "summary_spatial_per_facility.csv", index=False)
    piv = out.pivot(index="facility", columns="model", values="MAE")
    best = piv.idxmin(axis=1)
    macro("NFacAirCrossBest", int((best == "AirCross").sum()))
    macro("NFacGBBest", int((best == "Gradient boosting").sum()))
    macro("NFacGlobalBest", int((best == "Global mean").sum()))
    macro("NFacLoadRegBest", int((best == "Load regression").sum()))
    return out


def compute_regime_macros(df: pd.DataFrame, sync: pd.DataFrame) -> None:
    cems_cols = sync[["Facility ID", "Date", "Hour", "wind_speed_10m", "state"]].rename(
        columns={"Facility ID": "facility", "Date": "date", "Hour": "hour"}
    )
    cems_cols["facility"] = cems_cols["facility"].astype(str)
    cems_cols["date"] = cems_cols["date"].astype(str)
    
    t = df[(df["split"] == "temporal") & (df["feature_set"] == "weather+load")].copy()
    t["facility"] = t["facility"].astype(str)
    t["date"] = t["date"].astype(str)
    
    # Average across seeds for multi-seed models
    t_avg = t.groupby(["facility", "date", "hour", "model"], as_index=False).agg({"y_true": "first", "y_pred": "mean"})
    m = t_avg.merge(cems_cols, on=["facility", "date", "hour"], how="left")
    
    calm = m[m["wind_speed_10m"] < 2.0]
    adv = m[m["wind_speed_10m"] >= 2.0]
    tx = m[m["state"] == "TX"]
    mw = m[m["state"].isin(["OH", "PA"])]
    
    macro("NCalmTestWindows", f"{len(calm[calm['model'] == 'AirCross']):,}")
    macro("NAdvTestWindows", f"{len(adv[adv['model'] == 'AirCross']):,}")
    macro("NTxTestWindows", f"{len(tx[tx['model'] == 'AirCross']):,}")
    macro("NMidwestTestWindows", f"{len(mw[mw['model'] == 'AirCross']):,}")
    macro("NTxPlants", str(tx[tx["model"] == "AirCross"]["facility"].nunique()))
    macro("NMidwestPlants", str(mw[mw["model"] == "AirCross"]["facility"].nunique()))
    
    for mdl, code in [
        ("Per-facility mean", "FacMean"),
        ("Load regression", "LoadReg"),
        ("Gradient boosting", "GB"),
        ("AirCross", "AirCross"),
    ]:
        sub = m[m["model"] == mdl]
        c_mae = np.mean(np.abs(sub[sub["wind_speed_10m"] < 2.0]["y_true"] - sub[sub["wind_speed_10m"] < 2.0]["y_pred"]))
        a_mae = np.mean(np.abs(sub[sub["wind_speed_10m"] >= 2.0]["y_true"] - sub[sub["wind_speed_10m"] >= 2.0]["y_pred"]))
        tx_mae = np.mean(np.abs(sub[sub["state"] == "TX"]["y_true"] - sub[sub["state"] == "TX"]["y_pred"]))
        mw_mae = np.mean(np.abs(sub[sub["state"].isin(["OH", "PA"])]["y_true"] - sub[sub["state"].isin(["OH", "PA"])]["y_pred"]))
        macro(f"Regime{code}CalmMAE", f"{c_mae:.1f}")
        macro(f"Regime{code}AdvMAE", f"{a_mae:.1f}")
        macro(f"Regime{code}TxMAE", f"{tx_mae:.1f}")
        macro(f"Regime{code}MidwestMAE", f"{mw_mae:.1f}")
        
    ac_adv = float(MACROS["RegimeAirCrossAdvMAE"])
    ac_calm = float(MACROS["RegimeAirCrossCalmMAE"])
    macro("RegimeAirCrossAdvGainMAE", f"{ac_calm - ac_adv:.1f}")

    lr_adv = float(MACROS["RegimeLoadRegAdvMAE"])
    lr_calm = float(MACROS["RegimeLoadRegCalmMAE"])
    macro("RegimeLoadRegAdvGainMAE", f"{lr_calm - lr_adv:.1f}")
    
    lr_mw = float(MACROS["RegimeLoadRegMidwestMAE"])
    ac_mw = float(MACROS["RegimeAirCrossMidwestMAE"])
    gb_mw = float(MACROS["RegimeGBMidwestMAE"])
    macro("RegimeAirCrossMidwestGainMAE", f"{lr_mw - ac_mw:.1f}")
    macro("RegimeAirCrossMidwestVsLoadRegPct", f"{100 * (lr_mw - ac_mw) / lr_mw:.1f}")
    macro("RegimeGBMidwestGainMAE", f"{lr_mw - gb_mw:.1f}")
    macro("RegimeGBMidwestVsLoadRegPct", f"{100 * (lr_mw - gb_mw) / lr_mw:.1f}")

    # Bootstrap CIs for regional fleet differences
    rng = np.random.default_rng(42)
    def fast_boot_diff(sub_m, m1, m2, B=1000):
        p1 = sub_m[sub_m["model"] == m1].set_index(["facility", "date", "hour"])["y_pred"]
        p2 = sub_m[sub_m["model"] == m2].set_index(["facility", "date", "hour"])["y_pred"]
        y = sub_m[sub_m["model"] == m1].set_index(["facility", "date", "hour"])["y_true"]
        idx = p1.index.intersection(p2.index)
        p1, p2, y = p1.loc[idx], p2.loc[idx], y.loc[idx]
        e1 = np.abs(y.to_numpy() - p1.to_numpy())
        e2 = np.abs(y.to_numpy() - p2.to_numpy())
        fac_arr = idx.get_level_values("facility").to_numpy()
        unique_f = np.unique(fac_arr)
        fac_e1 = {f: e1[fac_arr == f] for f in unique_f}
        fac_e2 = {f: e2[fac_arr == f] for f in unique_f}
        diffs = []
        for _ in range(B):
            sampled = rng.choice(unique_f, size=len(unique_f), replace=True)
            tot_e1 = np.concatenate([fac_e1[f] for f in sampled])
            tot_e2 = np.concatenate([fac_e2[f] for f in sampled])
            diffs.append(np.mean(tot_e1) - np.mean(tot_e2))
        return np.percentile(diffs, [2.5, 97.5])

    ci_mw = fast_boot_diff(mw, "AirCross", "Gradient boosting")
    ci_tx = fast_boot_diff(tx, "AirCross", "Gradient boosting")
    macro("RegimeMWDiffAirCrossGBlo", f"{ci_mw[0]:.1f}")
    macro("RegimeMWDiffAirCrossGBhi", f"{ci_mw[1]:.1f}")
    macro("RegimeTXDiffAirCrossGBlo", f"{ci_tx[0]:.1f}")
    macro("RegimeTXDiffAirCrossGBhi", f"{ci_tx[1]:.1f}")


def main(run: str | None = None) -> None:
    cfg = load_config()
    run_id = run or (EXPERIMENTS_DIR / "LATEST").read_text().strip()
    run_dir = EXPERIMENTS_DIR / run_id
    df = pd.read_parquet(run_dir / "predictions.parquet")
    splits = json.load(open(run_dir / "splits.json"))
    reg = load_registry()
    coords = {int(r["Facility ID"]): (float(r["latitude"]), float(r["longitude"])) for _, r in reg.iterrows()}
    clusters = cluster_facilities(coords, threshold_km=50.0)
    cluster_map = {fid: cid for cid, members in enumerate(clusters) for fid in members}

    df_temp = df[df["split"] == "temporal"]
    temp = add_bootstrap(seed_metrics(df_temp), df_temp, cfg, cluster_map=cluster_map)
    
    df_spat = df[df["split"] == "spatial"]
    spat = add_bootstrap(seed_metrics(df_spat), df_spat, cfg, cluster_map=cluster_map)
    
    df_spat_new = df_spat[df_spat["year"].isin(cfg["split"]["temporal_test_years"])]
    spat_new = add_bootstrap(seed_metrics(df_spat_new), df_spat_new, cfg, cluster_map=cluster_map)
    
    for name, s in (("temporal", temp), ("spatial", spat), ("spatial_new_period", spat_new)):
        s.to_csv(run_dir / f"summary_{name}.csv", index=False)

    write_results_table(temp, "table_temporal.tex",
                        r"Temporal hold-out: train 2020--2022, early stopping on 2023, test 2024--2025 (all facilities). "
                        r"Errors in kg/h NO$_x$; mean $\pm$ s.d. over seeds where applicable. "
                        r"Bias = mean($\hat{y} - y$).", "tab:temporal")
    write_results_table(spat, "table_spatial.tex",
                        r"Spatial hold-out: grouped " + str(cfg["split"]["spatial_folds"]) + r"-fold cross-validation over 50~km spatial clusters "
                        r"(every cluster is unseen in exactly one fold; all years). 95\% CI by geographic cluster bootstrap. "
                        r"$\Delta$MAE: paired bootstrap difference vs.\ Load Regression. "
                        r"MAE$_{24\text{--}25}$: unseen facilities in 2024--2025 only.", "tab:spatial", ci=True, extra=spat_new)
    write_ablation_table(temp, spat, "table_ablation.tex")
    write_dataset_tables(cfg, reg, splits, run_dir, coords=coords, clusters=clusters)
    write_macros_for_results(temp, spat, spat_new)
    write_facility_results(df_spat, reg, run_dir)
    sync = pd.read_parquet(PROCESSED_DIR / "cems_era5_synchronized.parquet")
    compute_regime_macros(df, sync)
    macro("RunID", run_id.replace("_", r"\_"))
    learned = ["AirCross", "Gradient boosting", "MLP", "1D-CNN"]
    s_learned = spat[spat.model.isin(learned) & (spat.feature_set == "weather+load")]
    half_widths = (s_learned["dMAE_ci_hi"] - s_learned["dMAE_ci_lo"]) / 2
    mde = float(np.round(half_widths.mean(), 1))
    load_reg_mae = spat.loc[(spat.model == "Load regression") & (spat.feature_set == "weather+load"), "MAE_mean"].values[0]
    macro("MinDetectableEffectKgH", f"{mde:.1f}")
    macro("MinDetectableEffectPct", f"{100 * mde / load_reg_mae:.1f}")

    lines = ["% AUTO-GENERATED by src/evaluation/generate_latex_tables.py. Do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())]
    (TABLES_DIR / "macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    json.dump(MACROS, open(run_dir / "results_summary.json", "w"), indent=2)
    print(f"Wrote tables and {len(MACROS)} macros from run {run_id}")
    print("\n--- Temporal Hold-Out ---")
    print(temp[["model", "feature_set", "MAE_mean", "MAE_std", "RMSE_mean", "R2_mean", "Bias_mean"]].to_string(index=False))
    print("\n--- Spatial Hold-Out ---")
    print(spat[["model", "feature_set", "MAE_mean", "MAE_std", "MAE_ci_lo", "MAE_ci_hi", "dMAE_ci_lo", "dMAE_ci_hi", "Bias_mean"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    main(ap.parse_args().run)
