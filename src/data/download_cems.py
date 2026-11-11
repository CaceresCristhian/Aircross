"""
AirCross: download raw EPA CEMS hourly data (PUDL raw archives on Zenodo).

- 2020-2022: state-year zips from Zenodo record `cems_state_record` (epacems-<year>-<st>.zip).
  If an extracted CSV (epacems-<year>-<st>.csv) already exists locally it is reused.
- 2023-2025: national-year zips from Zenodo record `cems_national_record`
  (epacems-<year>.zip, quarterly CSVs inside), filtered to the configured states.

Only the configured overpass hours (local standard time) and the columns needed
downstream are kept. Output: data/01_raw/cems/cems_<year>_<st>.parquet

No values are modified here; this is a lossless subset of the raw files.
"""

import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import RAW_CEMS_DIR, load_config  # noqa: E402

KEEP_COLS = [
    "State", "Facility Name", "Facility ID", "Unit ID", "Date", "Hour",
    "Operating Time", "Gross Load (MW)", "Heat Input (mmBtu)",
    "NOx Mass (lbs)", "SO2 Mass (lbs)", "CO2 Mass (short tons)", "Primary Fuel Type",
]
ZENODO = "https://zenodo.org/api/records/{rec}/files/{name}/content"


def _download(url: str, dest: Path) -> None:
    print(f"  downloading {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)


def _filter_csv(fh, states, hours) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(fh, usecols=KEEP_COLS, chunksize=500_000, low_memory=False):
        chunk = chunk[chunk["State"].isin(states) & chunk["Hour"].isin(hours)]
        if not chunk.empty:
            parts.append(chunk)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KEEP_COLS)


def _save_by_state(df: pd.DataFrame, year: int, states) -> None:
    for st in states:
        out = RAW_CEMS_DIR / f"cems_{year}_{st.lower()}.parquet"
        sub = df[df["State"] == st]
        sub.to_parquet(out, index=False)
        print(f"  saved {out.name}: {len(sub):,} unit-hours")


def main() -> None:
    cfg = load_config()["data"]
    states, hours = cfg["states"], cfg["overpass_hours_lst"]
    for year in cfg["years"]:
        todo = [st for st in states if not (RAW_CEMS_DIR / f"cems_{year}_{st.lower()}.parquet").exists()]
        if not todo:
            print(f"{year}: already prepared")
            continue
        print(f"{year}: preparing states {todo}")
        if year <= 2022:
            frames = []
            for st in todo:
                csv = RAW_CEMS_DIR / f"epacems-{year}-{st.lower()}.csv"
                if csv.exists():
                    with open(csv, "rb") as fh:
                        frames.append(_filter_csv(fh, [st], hours))
                else:
                    zpath = RAW_CEMS_DIR / f"epacems-{year}-{st.lower()}.zip"
                    _download(ZENODO.format(rec=cfg["sources"]["cems_state_record"], name=zpath.name), zpath)
                    with zipfile.ZipFile(zpath) as zf:
                        for name in zf.namelist():
                            with zf.open(name) as fh:
                                frames.append(_filter_csv(fh, [st], hours))
                    zpath.unlink()
            _save_by_state(pd.concat(frames, ignore_index=True), year, todo)
        else:
            zpath = RAW_CEMS_DIR / f"epacems-{year}.zip"
            _download(ZENODO.format(rec=cfg["sources"]["cems_national_record"], name=zpath.name), zpath)
            frames = []
            with zipfile.ZipFile(zpath) as zf:
                for name in sorted(zf.namelist()):
                    print(f"  filtering {name}")
                    with zf.open(name) as fh:
                        frames.append(_filter_csv(fh, todo, hours))
            zpath.unlink()
            _save_by_state(pd.concat(frames, ignore_index=True), year, todo)


if __name__ == "__main__":
    main()
