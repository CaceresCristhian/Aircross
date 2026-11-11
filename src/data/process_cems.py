"""
AirCross: process raw EPA CEMS unit-hours into facility-hours and select facilities.

Input : data/01_raw/cems/cems_<year>_<st>.parquet   (from download_cems.py)
Output: data/02_processed/cems_hourly.parquet        (all selected facility-hours, flagged)
        data/02_processed/facility_registry.csv      (selected facilities + EIA-860 coordinates)

Rules
- Hours are EPA Part 75 hour-beginning, local standard time (LST, no DST).
- Missing NOx is NOT treated as zero. A facility-hour is `nox_complete` only if every
  operating unit reported NOx mass. Incomplete hours are kept but flagged.
- A facility-hour is `operating` when summed gross load > `operating_min_load_mw`.
- Facilities are selected by the rule in config.yaml (`data.selection`), not by hand.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import PROCESSED_DIR, RAW_CEMS_DIR, load_config  # noqa: E402
from src.data.registry import REGISTRY_PATH, load_eia860_plants  # noqa: E402

LB_TO_KG = 0.45359237
SHORT_TON_TO_T = 0.90718474
NUM_COLS = ["Gross Load (MW)", "Heat Input (mmBtu)", "NOx Mass (lbs)", "SO2 Mass (lbs)",
            "CO2 Mass (short tons)", "Operating Time"]


def load_unit_hours(cfg) -> pd.DataFrame:
    frames = []
    for year in cfg["years"]:
        for st in cfg["states"]:
            p = RAW_CEMS_DIR / f"cems_{year}_{st.lower()}.parquet"
            if not p.exists():
                raise FileNotFoundError(f"{p} missing. Run src/data/download_cems.py first.")
            frames.append(pd.read_parquet(p))
    df = pd.concat(frames, ignore_index=True)
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def aggregate_facility_hours(units: pd.DataFrame, min_load: float) -> pd.DataFrame:
    units = units.copy()
    load = units["Gross Load (MW)"].fillna(0.0)
    units["unit_operating"] = load > 0.0
    # An operating unit with missing NOx makes the facility-hour incomplete.
    units["unit_nox_missing"] = units["unit_operating"] & units["NOx Mass (lbs)"].isna()

    g = units.groupby(["Facility ID", "Date", "Hour"], sort=False)
    fac = g.agg(
        state=("State", "first"),
        cems_name=("Facility Name", "first"),
        load_mw=("Gross Load (MW)", lambda s: s.sum(min_count=1)),
        heat_input_mmbtu=("Heat Input (mmBtu)", lambda s: s.sum(min_count=1)),
        nox_lbs=("NOx Mass (lbs)", lambda s: s.sum(min_count=1)),
        so2_lbs=("SO2 Mass (lbs)", lambda s: s.sum(min_count=1)),
        co2_short_tons=("CO2 Mass (short tons)", lambda s: s.sum(min_count=1)),
        n_units=("Unit ID", "nunique"),
        n_units_operating=("unit_operating", "sum"),
        n_units_nox_missing=("unit_nox_missing", "sum"),
    ).reset_index()

    fac["load_mw"] = fac["load_mw"].fillna(0.0)
    fac["operating"] = fac["load_mw"] > min_load
    fac["nox_complete"] = (fac["n_units_nox_missing"] == 0) & fac["nox_lbs"].notna()
    fac["nox_kgh"] = fac["nox_lbs"] * LB_TO_KG
    fac["so2_kgh"] = fac["so2_lbs"] * LB_TO_KG
    fac["co2_th"] = fac["co2_short_tons"] * SHORT_TON_TO_T
    fac["year"] = pd.to_datetime(fac["Date"]).dt.year
    return fac.drop(columns=["nox_lbs", "so2_lbs", "co2_short_tons"])


def select_facilities(fac: pd.DataFrame, cfg) -> pd.DataFrame:
    sel = cfg["selection"]
    usable = fac[fac["operating"] & fac["nox_complete"]]
    mean_nox = usable.groupby("Facility ID")["nox_kgh"].mean()
    hours_per_year = usable.groupby(["Facility ID", "year"]).size().unstack(fill_value=0)
    years_ok = (hours_per_year >= sel["min_operating_hours_per_year"]).sum(axis=1)
    keep = mean_nox.index[(mean_nox >= sel["min_mean_nox_kgh"]).values
                          & (years_ok.reindex(mean_nox.index).fillna(0) >= sel["min_years_present"]).values]
    stats = pd.DataFrame({
        "mean_nox_kgh": mean_nox.reindex(keep),
        "operating_hours": usable.groupby("Facility ID").size().reindex(keep),
        "years_present": years_ok.reindex(keep),
    })
    return stats.reset_index()


def main() -> None:
    cfg = load_config()["data"]
    units = load_unit_hours(cfg)
    print(f"Loaded {len(units):,} unit-hours ({cfg['years'][0]}-{cfg['years'][-1]}, {cfg['states']}, "
          f"hours {cfg['overpass_hours_lst']} LST).")

    fac = aggregate_facility_hours(units, cfg["operating_min_load_mw"])
    print(f"Aggregated to {len(fac):,} facility-hours across {fac['Facility ID'].nunique()} facilities.")

    stats = select_facilities(fac, cfg)
    names = fac.groupby("Facility ID")[["cems_name", "state"]].last().reset_index()
    eia = load_eia860_plants()
    reg = (stats.merge(names, on="Facility ID", how="left")
                .merge(eia, left_on="Facility ID", right_on="plant_code", how="left"))
    missing = reg[reg["latitude"].isna() | reg["longitude"].isna()]
    if len(missing):
        print(f"Dropping {len(missing)} facilities without EIA-860 coordinates: {missing['Facility ID'].tolist()}")
        reg = reg.drop(missing.index)
    reg = reg.drop(columns=["plant_code"]).sort_values("Facility ID").reset_index(drop=True)
    reg.to_csv(REGISTRY_PATH, index=False)
    print(f"Selected {len(reg)} facilities -> {REGISTRY_PATH.name}")
    print(reg[["Facility ID", "cems_name", "state", "mean_nox_kgh", "operating_hours", "latitude", "longitude"]]
          .to_string(index=False))

    out = fac[fac["Facility ID"].isin(reg["Facility ID"])].copy()
    out = out.merge(reg[["Facility ID", "latitude", "longitude"]], on="Facility ID", how="left")
    out.to_parquet(PROCESSED_DIR / "cems_hourly.parquet", index=False)
    print(f"cems_hourly.parquet: {len(out):,} facility-hours "
          f"({int((out['operating'] & out['nox_complete']).sum()):,} operating with complete NOx, "
          f"{int((~out['operating']).sum()):,} non-operating, "
          f"{int((out['operating'] & ~out['nox_complete']).sum()):,} operating with missing NOx).")


if __name__ == "__main__":
    main()
