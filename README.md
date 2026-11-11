# AirCross: Meteorology-Conditioned Attention for Hourly Stack NOx Prediction (EPA CEMS + ERA5, 2020--2025)

## Overview
AirCross is an empirical deep learning benchmark for power-plant emissions modeling across contrasting atmospheric transport and regional operational regimes.

**Authors**: Cristhian David Cáceres Mateus and Talha Ali Khan (corresponding author: talhaali.khan@ue-germany.de)  
**Affiliation**: Department of Business, University of Europe for Applied Sciences, 14469 Potsdam, Germany  

This repository implements **Phase A**: an empirical multi-year benchmark predicting hourly smokestack $\text{NO}_x$ mass emissions ($\text{kg/h}$) at afternoon satellite-overpass hours (`13:00` and `14:00` Local Standard Time) across industrial power facilities in Ohio, Pennsylvania, and Texas (2020--2025). It combines:
- **EPA CAMPD / Part 75 Continuous Emission Monitoring System (CEMS)** hourly stack measurements,
- **EIA Form 860 (Schedule 2)** official plant coordinates, and
- **ECMWF ERA5** hourly meteorological reanalysis (via Open-Meteo archive API, explicitly in $\text{m/s}$ and aligned to Local Standard Time).

> **Scope & Roadmap (Phase A vs. Phase B):**
> - **Phase A (Current Benchmark)**: Evaluates meteorology-conditioned cross-attention against strong tabular and neural baselines (`Global mean`, `Per-facility mean`, `Load regression`, `HistGradientBoostingRegressor`, `MLP`, `1D-CNN`, and an attention ablation) under temporal hold-out (2024--2025) and 5-fold grouped spatial cross-validation across unseen facilities.
> - **Phase B (Planned Extension)**: Prototype spatial Vision Transformer and cross-sectional flux modules for real Sentinel-5P TROPOMI L2 column patches are isolated in `src/models/phase_b/` and are **not** used in any reported Phase A result.

---

## Repository Structure
```text
AirCross/
├── README.md
├── requirements.txt
├── LICENSE
├── config/
│   └── config.yaml                     # Configuration settings
├── data/
│   ├── 01_raw/                         # Raw CEMS, ERA5 cache, EIA-860 zip
│   └── 02_processed/                   # facility_registry.csv, cems_era5_synchronized.parquet
├── src/
│   ├── config.py                       # Config and path loader
│   ├── data/
│   │   ├── registry.py                 # EIA-860 coordinate lookup & registry loader
│   │   ├── download_cems.py            # CEMS download helper
│   │   ├── process_cems.py             # Facility selection & unit aggregation
│   │   ├── fetch_era5.py               # ERA5 query (UTC -> LST) & merge
│   │   └── dataset.py                  # Windows, normalizer, temporal & 5-fold spatial splits
│   ├── models/
│   │   ├── aircross.py                 # Phase A AirCross architecture
│   │   ├── baselines.py                # MLP, 1D-CNN, GlobalMean, FacilityMean, LoadRegression, GradientBoosting
│   │   └── phase_b/                    # Isolated Phase B prototypes
│   ├── training/
│   │   └── train.py                    # Multi-seed training with validation early stopping
│   └── evaluation/
│       ├── metrics.py                  # MAE, RMSE, R2, logMAE, and facility bootstrap CIs
│       ├── generate_latex_tables.py    # Writes paper/tables/*.tex and paper/tables/macros.tex
│       └── make_figures.py             # Generates paper/figures/*.png
├── scripts/
│   └── check_paper_numbers.py          # Verifies every decimal number in paper/main.tex
├── tests/
│   └── test_pipeline.py                # 18 unit & integration tests (pytest)
├── experiments/
│   └── <run_id>/                       # Saved predictions, splits, and summaries
└── paper/
    ├── main.tex                        # IEEEtran conference paper
    ├── references.bib
    ├── figures/
    └── tables/
```

---

## Reproduction Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ingestion & pipeline data preparation (optional, processed cache shipped)
python -m src.data.download_cems
python -m src.data.process_cems
python -m src.data.fetch_era5

# 3. Model training & validation (optional, precomputed predictions shipped)
python -m src.training.train

# 4. Run test suite (18 unit & integration tests)
pytest tests/test_pipeline.py

# 5. Generate tables and numerical macros
python -c "import sys; sys.path.insert(0, '.'); import src.evaluation.generate_latex_tables as g; g.main()"

# 6. Generate publication figures
python -c "import sys; sys.path.insert(0, '.'); import src.evaluation.make_figures as f; f.main()"

# 7. Verify paper numbers match pipeline outputs
python scripts/check_paper_numbers.py

# 8. Compile paper with Tectonic
tectonic paper/main.tex --outdir paper/
# (or using local tool executable: ../tools/tectonic.exe paper/main.tex --outdir paper/)

# 9. Lint paper prose with Vale
vale paper/main.tex
# (or using local tool executable: ../tools/vale.exe paper/main.tex)
```
