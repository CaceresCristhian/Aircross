"""Facility registry helpers.

Facility names and coordinates are never hand-typed. They come from:
- EPA CEMS (`Facility ID` = EIA/ORIS plant code, `Facility Name`), and
- EIA Form 860, Schedule 2 (Plant file) for latitude/longitude.

`process_cems.py` applies the selection rule in config.yaml and writes the final
registry to data/02_processed/facility_registry.csv, which every other script reads.
"""

import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import PROCESSED_DIR, RAW_EIA_DIR, load_config  # noqa: E402

REGISTRY_PATH = PROCESSED_DIR / "facility_registry.csv"


def load_eia860_plants() -> pd.DataFrame:
    """Return EIA-860 plant table: plant_code, eia_plant_name, state, latitude, longitude."""
    url = load_config()["data"]["sources"]["eia860_url"]
    zpath = RAW_EIA_DIR / Path(url).name
    if not zpath.exists():
        r = requests.get(url, timeout=180, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        if not r.content.startswith(b"PK"):
            # EIA moves older releases to /archive/
            url = url.replace("/eia860/xls/", "/eia860/archive/xls/")
            r = requests.get(url, timeout=180, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
        zpath.write_bytes(r.content)
    with zipfile.ZipFile(zpath) as zf:
        name = next(n for n in zf.namelist() if n.startswith("2___Plant"))
        df = pd.read_excel(zf.open(name), header=1)
    df = df.rename(columns={"Plant Code": "plant_code", "Plant Name": "eia_plant_name",
                            "State": "eia_state", "Latitude": "latitude", "Longitude": "longitude"})
    df = df[["plant_code", "eia_plant_name", "eia_state", "latitude", "longitude"]].dropna(subset=["plant_code"])
    df["plant_code"] = df["plant_code"].astype(int)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return df


def load_registry() -> pd.DataFrame:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"{REGISTRY_PATH} missing. Run src/data/process_cems.py first.")
    return pd.read_csv(REGISTRY_PATH)
