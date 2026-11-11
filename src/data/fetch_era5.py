"""
AirCross: fetch ERA5 hourly reanalysis (Open-Meteo archive API) for each selected facility
and synchronize it with CEMS facility-hours.

Units and time are requested explicitly:
- wind_speed_unit=ms        (Open-Meteo default is km/h)
- timezone=GMT              (timestamps in UTC)
- models=era5               (pure ERA5, no ERA5-Land blending)
ERA5 UTC timestamps are shifted to local standard time with a fixed per-state offset
(config.data.utc_offset_lst_h), matching EPA Part 75 hours (LST, no DST).
The ERA5 value at the start of the CEMS hour (hour-beginning) is used.

Input : data/02_processed/cems_hourly.parquet, facility_registry.csv
Output: data/01_raw/meteo/era5_<fid>_<lat>_<lon>.parquet (cache, UTC + LST columns)
        data/02_processed/era5_hourly.parquet
        data/02_processed/cems_era5_synchronized.parquet
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import PROCESSED_DIR, RAW_METEO_DIR, load_config  # noqa: E402
from src.data.registry import load_registry  # noqa: E402

API = "https://archive-api.open-meteo.com/v1/archive"


def fetch_facility(fid, lat, lon, cfg) -> pd.DataFrame:
    cache = RAW_METEO_DIR / f"era5_{fid}_{lat:.4f}_{lon:.4f}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    m = cfg["meteo"]
    params = {
        "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
        "start_date": f"{min(cfg['years'])}-01-01", "end_date": f"{max(cfg['years'])}-12-31",
        "hourly": ",".join(m["variables"]),
        "wind_speed_unit": m["wind_speed_unit"], "timezone": m["timezone"], "models": "era5",
    }
    for attempt in range(8):
        r = requests.get(API, params=params, timeout=60)
        if r.status_code == 200:
            break
        wait = 10 * (attempt + 1) if r.status_code == 429 else 3
        print(f"    HTTP {r.status_code}; retry in {wait}s")
        time.sleep(wait)
    r.raise_for_status()
    js = r.json()
    assert js["hourly_units"]["wind_speed_10m"] == "m/s", js["hourly_units"]
    assert js.get("timezone") in ("GMT", "UTC"), js.get("timezone")
    df = pd.DataFrame(js["hourly"])
    df["time_utc"] = pd.to_datetime(df.pop("time"))
    df.to_parquet(cache, index=False)
    time.sleep(1.0)
    return df


def main() -> None:
    cfg = load_config()["data"]
    reg = load_registry()
    frames = []
    for row in reg.itertuples(index=False):
        fid, st = int(row[reg.columns.get_loc("Facility ID")]), row.state
        print(f"ERA5 {fid} {row.cems_name} ({row.latitude:.4f}, {row.longitude:.4f})")
        df = fetch_facility(fid, row.latitude, row.longitude, cfg)
        offset = cfg["utc_offset_lst_h"][st]
        df = df.copy()
        df["time_lst"] = df["time_utc"] + pd.to_timedelta(offset, unit="h")
        df["Date"] = df["time_lst"].dt.strftime("%Y-%m-%d")
        df["Hour"] = df["time_lst"].dt.hour
        df = df[df["Hour"].isin(cfg["overpass_hours_lst"])]
        rad = np.radians(df["wind_direction_10m"])  # direction the wind blows FROM
        df["wind_u_ms"] = -df["wind_speed_10m"] * np.sin(rad)
        df["wind_v_ms"] = -df["wind_speed_10m"] * np.cos(rad)
        df["Facility ID"] = fid
        frames.append(df)

    meteo = pd.concat(frames, ignore_index=True)
    meteo.to_parquet(PROCESSED_DIR / "era5_hourly.parquet", index=False)
    print(f"ERA5: {len(meteo):,} facility-hours; mean 10 m wind speed = {meteo['wind_speed_10m'].mean():.2f} m/s")

    cems = pd.read_parquet(PROCESSED_DIR / "cems_hourly.parquet")
    keep = ["Facility ID", "Date", "Hour", "time_utc", "wind_speed_10m", "wind_direction_10m", "wind_u_ms",
            "wind_v_ms", "surface_pressure", "temperature_2m", "boundary_layer_height"]
    sync = cems.merge(meteo[keep], on=["Facility ID", "Date", "Hour"], how="inner", validate="one_to_one")
    lost = len(cems) - len(sync)
    sync.to_parquet(PROCESSED_DIR / "cems_era5_synchronized.parquet", index=False)
    print(f"Synchronized {len(sync):,} facility-hours ({lost} CEMS rows without ERA5 match).")


if __name__ == "__main__":
    main()
