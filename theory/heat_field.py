"""
Local field correction phi(r) on top of the MOND-like HEAT baseline.

Reference (HEAT_THEORY.md Level 4), element-wise at each radius:
  phi = lambda * log(1+r/Rs) * log(1+g_bar/a0) / (1+(g_bar/a0)^n)

Defaults match HEAT_THEORY.md parameter table.
"""
from __future__ import annotations

import numpy as np

from .heat_cosmology import a0_hie, estimate_redshift
from .heat_physics import g_heat_from_g_bar

# Level-4 parameters (documented in HEAT_THEORY.md)
LAMBDA_DEFAULT = 0.06
N_DEFAULT = 4.0


def phi_reference(
    r: np.ndarray,
    g_bar: np.ndarray,
    a0: np.ndarray | float,
    lam: float = LAMBDA_DEFAULT,
    n: float = N_DEFAULT,
    r_scale: float | None = None,
) -> np.ndarray:
    """
    Reference local correction phi(r), same shape as r and g_bar.

    Rs defaults to median(r) if r_scale is None.
    """
    r = np.asarray(r, dtype=float)
    g_bar = np.asarray(g_bar, dtype=float)
    a0 = np.maximum(np.asarray(a0, dtype=float), 1e-45)
    if r_scale is None:
        r_scale = float(np.median(r))
    r_scale = max(r_scale, 1e-30)

    G = np.log1p(r / r_scale)
    ratio = g_bar / a0
    log_term = np.log1p(ratio)
    T = 1.0 / (1.0 + ratio**n)
    return lam * G * log_term * T


def g_heat_total(
    g_bar: np.ndarray | float,
    r_m: np.ndarray | float,
    z: float,
    *,
    lam: float = LAMBDA_DEFAULT,
    n: float = N_DEFAULT,
    r_scale: float | None = None,
) -> np.ndarray:
    """
    Full HEAT (Levels 2–4): g_total = g_base * (1 + phi) at cosmic redshift z.

    Uses phi_reference with the same r_m, g_bar arrays (e.g. one radius per galaxy).
    If r_scale is None, Rs = median(r_m) (single-point rows use Rs = r).
    """
    g_bar = np.asarray(g_bar, dtype=float)
    r_m = np.asarray(r_m, dtype=float)
    a0 = a0_hie(z)
    g_b = g_heat_from_g_bar(g_bar, a0)
    rs = r_scale
    if rs is None and r_m.size:
        rs = float(np.median(r_m))
    phi = phi_reference(r_m, g_bar, a0, lam=lam, n=n, r_scale=rs)
    return g_b * (1.0 + phi)


def g_heat_total_point(
    g_bar: float,
    r_m: float,
    z: float,
    *,
    lam: float = LAMBDA_DEFAULT,
    n: float = N_DEFAULT,
    r_scale: float | None = None,
) -> float:
    """Scalar wrapper for catalog rows (one g_bar, one characteristic r)."""
    out = g_heat_total(g_bar, r_m, z, lam=lam, n=n, r_scale=r_scale)
    return float(np.asarray(out).reshape(-1)[0])


def phi_legacy_adaptive(r: np.ndarray, g_bar: np.ndarray, a0_local: float) -> np.ndarray:
    """
    Previous heuristic: amplitude from mean(g_bar)/a0, no gate T.
    Kept for regression / comparison only; not the reference theory.
    """
    sigma_ratio = np.mean(g_bar) / a0_local
    eps = 0.05 * np.log1p(sigma_ratio)
    R_scale = float(np.median(r))
    return eps * np.log1p(r / max(R_scale, 1e-30))


def g_heat_adaptive(
    g_bar: np.ndarray,
    r: np.ndarray,
    dist_mpc: float,
    *,
    use_legacy_phi: bool = False,
    lam: float = LAMBDA_DEFAULT,
    n: float = N_DEFAULT,
    r_scale: float | None = None,
) -> np.ndarray:
    """
    g_total = g_base * (1 + phi) with z from distance; a0 = a0_hie(z).

    By default uses phi_reference. Set use_legacy_phi=True for phi_legacy_adaptive.
    """
    z = estimate_redshift(dist_mpc)
    a0_local = a0_hie(z)
    g_m = g_heat_from_g_bar(g_bar, a0_local)
    if use_legacy_phi:
        phi = phi_legacy_adaptive(r, g_bar, float(a0_local))
    else:
        phi = phi_reference(r, g_bar, a0_local, lam=lam, n=n, r_scale=r_scale)
    return g_m * (1.0 + phi)
