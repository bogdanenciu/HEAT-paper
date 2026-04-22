"""HEAT / MOND phenomenology helpers."""
from __future__ import annotations

import numpy as np

from .heat_cosmology import G, LN10, M_sun, a0_hie, pc_to_m


def a0_mond() -> float:
    return 1.2e-10


def g_heat_from_g_bar(g_bar: np.ndarray, a0: np.ndarray | float) -> np.ndarray:
    g_bar = np.maximum(np.asarray(g_bar, dtype=float), 1e-45)
    a0 = np.maximum(np.asarray(a0, dtype=float), 1e-45)
    return g_bar / (1.0 - np.exp(-np.sqrt(g_bar / a0)))


def calc_v(M_kg, a0):
    return ((G * M_kg * a0) ** 0.25) / 1000.0


def calc_r(M_kg, a0):
    return np.sqrt((G * M_kg) / a0) / (1000.0 * pc_to_m)


def v_lcdm_simple(M_stars, z):
    M_halo = np.where(M_stars < 1e9, M_stars * 100, M_stars * 40)
    v_zero = 160.0
    return v_zero * (M_halo / 1e12) ** (1.0 / 3.0) * np.sqrt(1.0 + z)


def stellar_to_halo_mass(M_star):
    return M_star * (30.0 + 70.0 * (M_star < 1e9))


def v_lcdm_improved(M_star, z):
    M_halo = stellar_to_halo_mass(M_star)
    return 200.0 * (M_halo / 1e12) ** (1.0 / 3.0) * (1.0 + z) ** 0.3


def sigma_v_z(z: float) -> float:
    return float(np.sqrt(0.08**2 + (0.01 * z) ** 2))


def sigma_log10_v_kms(z: float, v_kms: float) -> float:
    """
    Uncertainty in log10(v) from fractional kinematic error sigma_v_z(z) on v.
    sigma_v_z is treated as δv/v (dimensionless).
    """
    frac = sigma_v_z(z)
    v_kms = max(float(v_kms), 1e-6)
    return frac / LN10


def sigma_log10_g_from_dg(sigma_g: np.ndarray, g: np.ndarray) -> np.ndarray:
    g = np.maximum(np.asarray(g, dtype=float), 1e-45)
    sigma_g = np.asarray(sigma_g, dtype=float)
    return sigma_g / (g * LN10)
