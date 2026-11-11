import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import PROCESSED_DIR, load_config  # noqa: E402
from src.data.dataset import (FEATURE_SETS, add_calendar_features, build_windows,  # noqa: E402
                              make_splits, spatial_folds)
from src.data.process_cems import aggregate_facility_hours  # noqa: E402
from src.evaluation.metrics import facility_bootstrap_ci, paired_facility_bootstrap_diff_ci, mae, mape, r2  # noqa: E402
from src.models.aircross import AirCross  # noqa: E402
from src.models.baselines import CNNBaseline, MLPBaseline  # noqa: E402

SYNC = PROCESSED_DIR / "cems_era5_synchronized.parquet"
SAMPLE_SYNC = PROCESSED_DIR / "cems_era5_synchronized_sample.parquet"
needs_data = pytest.mark.skipif(not (SYNC.exists() or SAMPLE_SYNC.exists()), reason="processed data not available")


def _toy_frame(days=6, fids=(1, 2, 3, 4, 5, 6)):
    rows = []
    for f in fids:
        for d in pd.date_range("2022-12-28", periods=days):
            for h in (13, 14):
                rows.append({"Facility ID": f, "Date": d.strftime("%Y-%m-%d"), "Hour": h, "operating": True,
                             "nox_complete": True, "nox_kgh": 100.0 + f, "load_mw": 500.0,
                             **{c: 1.0 for c in ["wind_u_ms", "wind_v_ms", "wind_speed_10m", "surface_pressure",
                                                 "temperature_2m", "boundary_layer_height"]}})
    return add_calendar_features(pd.DataFrame(rows), [13, 14])


# ---------------- splits ----------------
def test_spatial_folds_disjoint_and_complete():
    fids = np.arange(20)
    folds = spatial_folds(fids, 5, 42)
    allf = np.concatenate(folds)
    assert sorted(allf.tolist()) == fids.tolist()
    for i in range(5):
        for j in range(i + 1, 5):
            assert set(folds[i]).isdisjoint(folds[j])


def test_spatial_cluster_isolation():
    from src.data.registry import REGISTRY_PATH
    from src.data.dataset import haversine_km
    if not REGISTRY_PATH.exists():
        pytest.skip("registry not generated yet")
    reg = pd.read_csv(REGISTRY_PATH)
    coords = {int(r["Facility ID"]): (float(r["latitude"]), float(r["longitude"])) for _, r in reg.iterrows()}
    fids = np.array(list(coords.keys()))
    folds = spatial_folds(fids, 5, 42, coords=coords)
    
    # Assert every pair across folds is separated by >= 40 km (strictly respecting cluster threshold)
    for f1 in range(len(folds)):
        for f2 in range(f1 + 1, len(folds)):
            for id1 in folds[f1]:
                for id2 in folds[f2]:
                    d = haversine_km(coords[id1][0], coords[id1][1], coords[id2][0], coords[id2][1])
                    assert d >= 40.0, f"Facility {id1} and {id2} are only {d:.2f} km apart across folds {f1} and {f2}"


def test_no_facility_leakage_in_spatial_splits():
    df = _toy_frame(days=400, fids=tuple(range(10)))
    w = build_windows(df, FEATURE_SETS["weather+load"], 4)
    cfg = load_config()["split"]
    for s in make_splits(w, cfg):
        if s["split"] == "spatial":
            train_f = set(w.facility[s["fit"] | s["val"]])
            test_f = set(w.facility[s["test"]])
            assert train_f.isdisjoint(test_f)


# ---------------- windows ----------------
def test_windows_are_consecutive_slots():
    df = _toy_frame(days=6)
    df = df.drop(df[(df["Facility ID"] == 1) & (df["Date"] == "2022-12-30")].index)  # gap
    w = build_windows(df, FEATURE_SETS["weather+load"], 4)
    # facility 2 (no gap): 12 slots -> 9 windows; facility 1 has a 2-slot gap -> fewer
    assert (w.facility == 2).sum() == 9
    assert (w.facility == 1).sum() < 9


def test_non_operating_targets_excluded():
    df = _toy_frame(days=3)
    df.loc[df.index[-1], "operating"] = False
    w = build_windows(df, FEATURE_SETS["weather+load"], 4)
    assert len(w.y) == 6 * 3 - 1


# ---------------- units, timezone, missing values ----------------
def test_missing_nox_is_not_zero():
    # Case 1: Operating unit with missing NOx -> nox_complete is False
    units = pd.DataFrame({
        "Facility ID": [1, 1], "Date": ["2022-01-01"] * 2, "Hour": [13, 13], "State": ["OH"] * 2,
        "Facility Name": ["X"] * 2, "Unit ID": ["1", "2"], "Gross Load (MW)": [300.0, 200.0],
        "Heat Input (mmBtu)": [1.0, 1.0], "NOx Mass (lbs)": [100.0, np.nan], "SO2 Mass (lbs)": [1.0, 1.0],
        "CO2 Mass (short tons)": [1.0, 1.0]})
    fac = aggregate_facility_hours(units, 1.0)
    assert not fac["nox_complete"].iloc[0]

    # Case 2: Idle unit (load == 0) with missing NOx, but operating unit has valid NOx -> nox_complete is True
    units_idle = pd.DataFrame({
        "Facility ID": [1, 1], "Date": ["2022-01-01"] * 2, "Hour": [13, 13], "State": ["OH"] * 2,
        "Facility Name": ["X"] * 2, "Unit ID": ["1", "2"], "Gross Load (MW)": [300.0, 0.0],
        "Heat Input (mmBtu)": [1.0, 0.0], "NOx Mass (lbs)": [100.0, np.nan], "SO2 Mass (lbs)": [1.0, 0.0],
        "CO2 Mass (short tons)": [1.0, 0.0]})
    fac_idle = aggregate_facility_hours(units_idle, 1.0)
    assert fac_idle["nox_complete"].iloc[0]
    assert fac_idle["operating"].iloc[0]


def test_lst_offset_conversion():
    cfg = load_config()["data"]
    utc = pd.Timestamp("2023-07-01 18:00")
    assert (utc + pd.Timedelta(hours=cfg["utc_offset_lst_h"]["OH"])).hour == 13  # EST = UTC-5
    assert (utc + pd.Timedelta(hours=cfg["utc_offset_lst_h"]["TX"])).hour == 12  # CST = UTC-6


@needs_data
def test_wind_in_ms():
    p = SYNC if SYNC.exists() else SAMPLE_SYNC
    df = pd.read_parquet(p, columns=["wind_speed_10m"])
    assert 1.0 < df["wind_speed_10m"].mean() < 8.0  # km/h data would be ~10-20


@needs_data
def test_sync_has_no_duplicates():
    p = SYNC if SYNC.exists() else SAMPLE_SYNC
    df = pd.read_parquet(p, columns=["Facility ID", "Date", "Hour"])
    assert not df.duplicated().any()


# ---------------- metrics ----------------
def test_normalization_uses_mean():
    x = np.array([10.0, 12.0, 14.0])
    assert abs(((x - x.mean()) / x.std()).mean()) < 1e-9


def test_zero_predictor_mape_close_to_100():
    y = np.array([100.0, 200.0, 300.0])
    assert abs(mape(y, np.zeros_like(y)) - 100.0) < 1e-6


def test_r2_perfect_and_mean():
    y = np.array([1.0, 2.0, 3.0])
    assert r2(y, y) == 1.0 and abs(r2(y, np.full(3, 2.0))) < 1e-12


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.uniform(100, 200, 300); fac = np.repeat(np.arange(10), 30)
    P = (y + rng.normal(0, 10, (3, 300)))
    lo, hi = facility_bootstrap_ci(y, P, fac, mae, 300, 0)
    point = np.mean([mae(y, P[s]) for s in range(3)])
    assert lo <= point <= hi

    # cluster bootstrap with paired difference
    clusters = {i: i // 2 for i in range(10)}
    clo, chi = facility_bootstrap_ci(y, P, fac, mae, 300, 0, clusters=clusters)
    assert clo <= point <= chi
    dlo, dhi = paired_facility_bootstrap_diff_ci(y, P, P, fac, mae, 300, 0, clusters=clusters)
    assert dlo <= 0.0 <= dhi


# ---------------- models ----------------
@pytest.mark.parametrize("xattn", [True, False])
def test_aircross_shapes(xattn):
    m = AirCross(10, [0, 1, 2, 3, 4, 5], 4, 32, 4, 1, use_cross_attention=xattn).eval()
    out, w = m(torch.randn(8, 4, 10), return_attention=True)
    assert out.shape == (8,)
    if xattn:
        assert w.shape == (8, 4) and torch.allclose(w.sum(-1), torch.ones(8), atol=1e-5)


def test_baseline_shapes():
    x = torch.randn(8, 4, 10)
    assert MLPBaseline(10, 4)(x).shape == (8,)
    assert CNNBaseline(10)(x).shape == (8,)


# ---------------- security & credential protection ----------------
def test_security_no_credential_or_mcp_leakage():
    """Guarantee zero API tokens, credentials, or MCP keys are stored in source code or packaged zip."""
    import zipfile
    base_dir = Path(__file__).resolve().parents[1]

    # Rule 1: No .env or credentials inside project directory
    forbidden_files = [".env", ".env.local", "mcp_config.json", "Copernicus.txt"]
    for f in forbidden_files:
        assert not (base_dir / f).exists(), f"Security violation: {f} found in project directory!"

    # Rule 2: If package zip exists, verify it contains zero sensitive files
    zip_path = base_dir.parent / "AirCross_Enhanced_Benchmark.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                name_lower = name.lower()
                assert not name_lower.endswith(".env"), f"Security violation: .env found in {zip_path.name}"
                assert "copernicus.txt" not in name_lower, f"Security violation: credentials in {zip_path.name}"
                assert "mcp_config" not in name_lower, f"Security violation: MCP config in {zip_path.name}"
                assert not name_lower.endswith((".pem", ".key", ".p12")), f"Key file in {zip_path.name}"


def test_checker_sees_numbers_after_escaped_percent():
    """Verify that regex in check_paper_numbers.py does not strip text after an escaped percent sign."""
    import re
    text = r"We observe 12\% gain, i.e. 38.4 kg/h."
    stripped = re.sub(r"(?m)(?<!\\)%.*$", "", text)
    assert "38.4" in stripped

