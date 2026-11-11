# Reproducibility Log: AirCross Benchmark

**Manuscript Title**: An Empirical Benchmark of Operational Load and Meteorology Conditioning for Hourly Power-Plant $\text{NO}_x$ Prediction from EPA CEMS and ERA5 Reanalysis  
**Authors**: Cristhian David Cáceres Mateus, Talha Ali Khan (corresponding author: talhaali.khan@ue-germany.de)  
**Affiliation**: Department of Business, University of Europe for Applied Sciences, 14469 Potsdam, Germany  
**Run ID**: `20261020_141142`  
**Log Updated**: October 2026  

---

## 1. Reproduction Steps

To reproduce all tables, figures, and macros from the processed artifacts:

```bash
# 1. Environment requirements
pip install -r requirements.txt

# 2. Run the unit and property test suite (18 tests)
pytest tests/test_pipeline.py

# 3. Regenerate LaTeX tables and numerical macros
python -c "import sys; sys.path.insert(0, 'AirCross'); import src.evaluation.generate_latex_tables as g; g.main()"

# 4. Regenerate publication figures
python -c "import sys; sys.path.insert(0, 'AirCross'); import src.evaluation.make_figures as f; f.main()"

# 5. Verify all numerical claims in main.tex against the macro dictionary
python scripts/check_paper_numbers.py

# 6. Compile the manuscript PDF
tools/tectonic.exe paper/main.tex --outdir paper/

# 7. Lint manuscript prose
tools/vale.exe paper/main.tex
```

---

## 2. Data Sources and Provenance

- **Emissions & Load Data**: US EPA Clean Air Markets Program Data (CAMPD) 40 CFR Part 75 Continuous Emission Monitoring Systems (CEMS) hourly observations (2020--2025).
- **Geographic Coordinates**: US Energy Information Administration (EIA) Form 860 (Schedule 2, Plant file).
- **Atmospheric Reanalysis**: ECMWF ERA5 hourly surface reanalysis via Open-Meteo API (`models=era5`, `wind_speed_unit=ms`, `timezone=GMT`).
- **Facility Selection**: Baseload fossil-fired power stations in Ohio, Pennsylvania, and Texas with:
  1. Operating afternoon mean $\text{NO}_x \ge 200\text{ kg/h}$.
  2. $\ge 300$ operating afternoon hours in each of the six years (2020--2025).
  3. Verified coordinates in EIA Form 860.
- **Resulting Cohort**: 19 facilities (4 in OH, 2 in PA, 13 in TX), totaling 83,296 synchronized facility-hours (72,298 operating with complete $\text{NO}_x$, 10,998 non-operating, 0 unrecorded hours).
- **Imputation Rules**: Missing gross load is treated as idle (`fillna(0.0)`) and excluded from operating hours. Unrecorded $\text{NO}_x$ during operating hours is not imputed with zeros.

---

## 3. Evaluation Protocols and Statistical Results

1. **Temporal Hold-Out (2024--2025)**:
   - Training: 2020--2022 (48,432 facility-hours across all 19 facilities).
   - Validation: 2023 (16,056 facility-hours for early stopping).
   - Test: 2024--2025 (18,808 facility-hours across all 19 facilities).
   - Top models: Gradient Boosting MAE = 160.1 kg/h ($R^2 = 0.60$), AirCross MAE = 170.7 $\pm$ 3.0 kg/h ($R^2 = 0.55$), Load Regression MAE = 209.1 kg/h ($R^2 = 0.44$).
2. **Spatial Hold-Out (50 km Cluster-Independent 5-Fold Grouped CV)**:
   - Facilities within 50 km are grouped into 16 connected spatial clusters.
   - 5 disjoint folds evaluate generalization to geographically unseen facilities.
   - Statistical uncertainty derived from 2,000-replicate cluster bootstrap resampling across facilities.
   - Top models: MLP MAE = 205.6 $\pm$ 2.5 kg/h, Gradient Boosting MAE = 207.0 kg/h, Load Regression MAE = 216.4 kg/h, AirCross MAE = 217.5 $\pm$ 5.8 kg/h.
   - Paired difference tests against Load Regression span zero for all learned models (AirCross: [-17.8, 22.5] kg/h; Gradient Boosting: [-26.2, 8.9] kg/h).
3. **Subgroup Analysis**:
   - Evaluated across atmospheric wind regimes (Calm $w < 2$ m/s vs Advective $w \ge 2$ m/s) and regional fleets (Texas 13 plants vs Ohio/PA 6 plants).
   - Computed from seed-averaged predictions across all 5 seeds.
   - In Texas, Gradient Boosting achieves 139.4 kg/h vs AirCross 151.2 kg/h (95% CI: [2.3, 23.5] kg/h).
   - In Ohio/PA, AirCross achieves 202.9 kg/h vs Gradient Boosting 206.9 kg/h (difference 95% CI: [-30.5, 24.1] kg/h, statistically tied).

---

## 4. Verification Checklist

- [x] **Zero unrecorded hours imputed as zeros**: Operating units with missing $\text{NO}_x$ are explicitly excluded rather than filled.
- [x] **Strict time-zone alignment**: UTC ERA5 reanalysis shifted by standard local offsets (-5 h OH/PA, -6 h TX) before joining.
- [x] **Dynamic macro derivation**: All in-text numbers, table cells, and bootstrap bounds are exported to `tables/macros.tex`.
- [x] **Number checking**: `scripts/check_paper_numbers.py` parses `main.tex` with fixed regex `(?<!\\)%.*$` and validates all decimal values against `macros.tex`.
- [x] **No credential leakage**: Zero API tokens, keys, or credentials present.
- [x] **Test suite**: 18 passed tests in `tests/test_pipeline.py`.
- [x] **Tectonic compile**: 0 fatal errors, 1,007.8 KiB PDF.
- [x] **Vale lint**: 0 errors, 0 warnings.
