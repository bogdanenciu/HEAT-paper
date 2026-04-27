"""
Single source of truth for HEAT / comparison cosmology and constants.

Zero-parameter HEAT:
    a0(z) = c H(z) / (2 pi),
with H(z) fixed by Planck 2018 LCDM parameters.  No free parameter.

The background E(z) is written in the factorised form
    E_heat(z) = sqrt( Om_b * F0 * (1+z)^3  +  Om_L ),
which is algebraically identical to LCDM because F0 is the
*derived* constant F0 = (1 - Om_L) / Om_b = Om_m / Om_b that
enforces E_heat(0) = 1 and Om_b * F0 = Om_m.  F0 is therefore a
normalisation identity, not a fit parameter.
"""
from __future__ import annotations

import numpy as np

G = 6.67430e-11          # CODATA 2018 gravitational constant [m^3 kg^-1 s^-2]
c = 299792458.0          # exact speed of light [m/s]
M_sun = 1.989e30         # IAU nominal solar mass [kg]
pc_to_m = 3.086e16       # parsec in metres
Mpc_to_m = 3.086e22      # megaparsec in metres
LN10 = np.log(10.0)

H0_km_s_Mpc = 67.4       # Planck 2018 (Table 2, TT+TE+EE+lowE+lensing)
H0 = (H0_km_s_Mpc * 1000.0) / Mpc_to_m

Om_b = 0.049             # Planck 2018 baryon density parameter
# Flat LCDM: Om_m, Om_L match paper_heat_letter.tex (Planck 2018 TT,TE,EE+lowE+lensing, Om_m=0.315)
Om_m = 0.315
Om_L = 0.685
Om_r = 9.24e-5           # radiation density (photons + 3 massless neutrinos)

F0 = (1.0 - Om_L) / Om_b  # derived: Om_b * F0 = Om_m; enforces E_heat(0) = 1
z_cmb = 1100


def E_heat(z):
    z = np.asarray(z, dtype=float)
    return np.sqrt(Om_b * F0 * (1.0 + z) ** 3 + Om_L)


def E_lcdm(z):
    z = np.asarray(z, dtype=float)
    return np.sqrt(Om_m * (1.0 + z) ** 3 + Om_L)


def hubble_parameter(z):
    return H0 * E_heat(z)


def a0_hie(z):
    """Zero-parameter emergent acceleration scale: a0(z) = c H(z) / (2 pi)."""
    z = np.asarray(z, dtype=float)
    return (c * hubble_parameter(z)) / (2.0 * np.pi)


def a0_z0_mond_comparison():
    """
    Reproducible anchor: a0_hie(0) vs standard MOND a0 = 1.2e-10 m/s^2.

    Returns dict with SI values and fractional difference (HEAT - MOND) / MOND.
    """
    from .heat_physics import a0_mond

    a_heat = float(a0_hie(0.0))
    a_m = float(a0_mond())
    return {
        "a0_hie_z0_m_s2": a_heat,
        "a0_mond_m_s2": a_m,
        "fractional_diff": (a_heat - a_m) / a_m,
    }


def estimate_redshift(distance_mpc):
    """Hubble law; valid for z << 1 (local SPARC)."""
    D_m = distance_mpc * Mpc_to_m
    return (H0 * D_m) / c
