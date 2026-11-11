"""Shared configuration and path helpers for AirCross.

Every script loads settings through `load_config()` so that `config/config.yaml`
is the single source of truth.
"""

from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

RAW_DIR = BASE_DIR / "data" / "01_raw"
RAW_CEMS_DIR = RAW_DIR / "cems"
RAW_METEO_DIR = RAW_DIR / "meteo"
RAW_EIA_DIR = RAW_DIR / "eia860"
PROCESSED_DIR = BASE_DIR / "data" / "02_processed"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
PAPER_DIR = BASE_DIR / "paper"
TABLES_DIR = PAPER_DIR / "tables"
FIGURES_DIR = PAPER_DIR / "figures"

for _d in (RAW_CEMS_DIR, RAW_METEO_DIR, RAW_EIA_DIR, PROCESSED_DIR, EXPERIMENTS_DIR, TABLES_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
