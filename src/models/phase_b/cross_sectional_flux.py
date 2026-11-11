"""
PHASE B (NOT USED IN ANY REPORTED RESULT).

Cross-sectional flux (CSF) emission estimate from an OBSERVED satellite NO2 column field.
This replaces the earlier `GaussianPlumeInversion`, which received a hard-coded constant
column and therefore predicted ~0 kg/h. It must only be called with real TROPOMI columns.

Q [mol/s] = u [m/s] * integral over a cross-wind transect of (VCD - background) [mol/m^2] dy [m]
Q [kg/h]  = Q [mol/s] * M_NO2 [kg/mol] * 3600 [s/h]
NOx/NO2 scaling and NO2 lifetime corrections are left to the Phase B study and must be
reported explicitly when used.
"""

import numpy as np

M_NO2_KG_PER_MOL = 0.0460055
S_PER_H = 3600.0


def line_density_mol_per_m(vcd_mol_m2: np.ndarray, background_mol_m2: float, pixel_width_m: float) -> float:
    """Integrate a cross-wind transect of column enhancements (mol/m^2) into a line density (mol/m)."""
    enh = np.asarray(vcd_mol_m2, dtype=float) - background_mol_m2
    enh = np.where(np.isfinite(enh), enh, 0.0)
    return float(np.sum(enh) * pixel_width_m)


def cross_sectional_flux_kg_h(line_density: float, wind_speed_ms: float) -> float:
    """Q = line density [mol/m] * wind speed [m/s] -> kg NO2 per hour."""
    q_mol_s = line_density * wind_speed_ms
    return q_mol_s * M_NO2_KG_PER_MOL * S_PER_H
