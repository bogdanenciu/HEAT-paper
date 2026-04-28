"""
One-off analysis: fit HEAT a0(z) = K * c * H(z) to the recent published
intermediate-z RAR data (Ciocan+2026 + Varasteanu+2025 + SPARC z=0).

Tests:
  1. Best-fit K vs HEAT's horizon-thermodynamic K_HEAT = 1/(2 pi).
  2. chi2 of HEAT shape (free K) vs constant-a0 vs Ciocan's linear-in-z.
  3. Residuals at each data point.
  4. Sensitivity to the V'ar'asteanu point (which sits high relative to SPARC).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

import sys
sys.path.insert(0, ".")
from theory.heat_cosmology import c, H0, Om_m, Om_L, E_lcdm

# ---------------------------------------------------------------------
# Data points (a0 in 1e-10 m/s^2)
# Each row: (label, z, a0_central, a0_sigma_low, a0_sigma_high)
# ---------------------------------------------------------------------
DATA = [
    # SPARC z=0 anchor (McGaugh et al. 2016, N=171)
    ("SPARC z=0",         0.00, 1.20, 0.26, 0.26),
    # MIGHTEE-HI low-z (Varasteanu et al. 2025, z<0.08, take 0.04 as representative)
    ("Varasteanu z<0.08", 0.04, 1.69, 0.13, 0.13),
    # MUSE-DARK Ciocan+2026 global fit (N=79, 0.33<z<1.44, take z~1 as effective)
    ("Ciocan z~1 global", 1.00, 2.38, 0.10, 0.12),
    # Ciocan+2026 lowest-z bin (text: "rising from ~1.99 in lowest z-bin")
    # bin centre estimated at z~0.55 (quartiles of 0.33-1.44)
    ("Ciocan bin1 z~0.55", 0.55, 1.99, 0.20, 0.20),
    # Ciocan+2026 highest-z bin (text: "to 2.71 in highest")
    # bin centre estimated at z~1.30
    ("Ciocan bin4 z~1.30", 1.30, 2.71, 0.25, 0.25),
]


def H_of_z(z):
    """H(z) in SI [1/s]."""
    return H0 * E_lcdm(z)


def a0_HEAT_shape(z, K):
    """a0(z) = K * c * H(z) in 1e-10 m/s^2."""
    return K * c * H_of_z(z) / 1e-10


def chi2(K, data):
    s = 0.0
    for (_, z, a0, slo, shi) in data:
        pred = a0_HEAT_shape(z, K)
        sigma = (slo + shi) / 2.0
        s += ((a0 - pred) / sigma) ** 2
    return s


def chi2_const(a0_const, data):
    s = 0.0
    for (_, z, a0, slo, shi) in data:
        sigma = (slo + shi) / 2.0
        s += ((a0 - a0_const) / sigma) ** 2
    return s


def chi2_linear(a0_0, a1, data):
    s = 0.0
    for (_, z, a0, slo, shi) in data:
        pred = a0_0 + a1 * z
        sigma = (slo + shi) / 2.0
        s += ((a0 - pred) / sigma) ** 2
    return s


def main():
    print("=" * 78)
    print("Cosmology check (paper assumptions):")
    print(f"  H0 = {H0*1e18:.4f} x 10^-18 s^-1   (= 67.4 km/s/Mpc)")
    print(f"  Om_m = {Om_m},  Om_L = {Om_L}")
    print(f"  c*H0 / (2*pi)        = {c*H0/(2*np.pi)/1e-10:.4f} x 10^-10 m/s^2")
    print(f"  c*H0  (i.e. K=1)     = {c*H0/1e-10:.4f} x 10^-10 m/s^2")
    print(f"  c*H(z=1)             = {c*H_of_z(1)/1e-10:.4f} x 10^-10 m/s^2")
    print(f"  E(z=1)               = {E_lcdm(1):.4f}")
    print(f"  E(z=0.55)            = {E_lcdm(0.55):.4f}")
    print(f"  E(z=1.30)            = {E_lcdm(1.30):.4f}")

    print()
    print("=" * 78)
    print("Data points used:")
    for (label, z, a0, slo, shi) in DATA:
        print(f"  {label:<24s}  z={z:5.2f}  a0={a0:.2f} +{shi:.2f}/-{slo:.2f}")

    print()
    print("=" * 78)
    print("HEAT prediction at K = 1/(2*pi) (zero-parameter, paper):")
    K_paper = 1.0 / (2.0 * np.pi)
    print(f"  K_paper = 1/(2*pi) = {K_paper:.5f}")
    for (label, z, a0, slo, shi) in DATA:
        pred = a0_HEAT_shape(z, K_paper)
        sigma = (slo + shi) / 2.0
        resid = (a0 - pred) / sigma
        print(f"  {label:<24s}  z={z:4.2f}  obs={a0:.2f}  HEAT={pred:.2f}  "
              f"resid={resid:+.2f}sigma  ratio={a0/pred:.3f}")
    chi2_paper = chi2(K_paper, DATA)
    print(f"  chi2 = {chi2_paper:.2f}  (dof = {len(DATA)})")

    print()
    print("=" * 78)
    print("HEAT shape with FREE K (best-fit normalization):")
    res = minimize_scalar(chi2, bounds=(0.01, 1.0), method="bounded", args=(DATA,))
    K_fit = float(res.x)
    chi2_fit = float(res.fun)
    print(f"  K_fit = {K_fit:.5f}   (compare K_HEAT = 1/(2*pi) = {K_paper:.5f})")
    print(f"  K_fit / K_HEAT = {K_fit/K_paper:.3f}")
    print(f"  chi2 = {chi2_fit:.2f}  (dof = {len(DATA)-1})")
    print()
    print("  Residuals at K_fit:")
    for (label, z, a0, slo, shi) in DATA:
        pred = a0_HEAT_shape(z, K_fit)
        sigma = (slo + shi) / 2.0
        resid = (a0 - pred) / sigma
        print(f"    {label:<24s}  z={z:4.2f}  obs={a0:.2f}  HEAT_fit={pred:.2f}  "
              f"resid={resid:+.2f}sigma  ratio={a0/pred:.3f}")

    print()
    print("=" * 78)
    print("Constant-a0 MOND null (best-fit constant):")
    res2 = minimize_scalar(chi2_const, bounds=(0.5, 5.0), method="bounded",
                           args=(DATA,))
    a0_const = float(res2.x)
    chi2_const_val = float(res2.fun)
    print(f"  a0_const_fit = {a0_const:.3f} x 10^-10 m/s^2")
    print(f"  chi2 = {chi2_const_val:.2f}  (dof = {len(DATA)-1})")
    print(f"  Delta chi2 (const - HEAT_fit) = {chi2_const_val - chi2_fit:.2f}")

    print()
    print("=" * 78)
    print("Ciocan linear fit a0(z) = a0(0) + a1*z [their reported best-fit]:")
    a0_0_lin = 1.0
    a1_lin = 1.59
    chi2_lin = chi2_linear(a0_0_lin, a1_lin, DATA)
    print(f"  a0(0) = {a0_0_lin}, a1 = {a1_lin}")
    print(f"  chi2 = {chi2_lin:.2f}  (dof = {len(DATA)-2})")
    print()
    print("  Residuals under Ciocan linear fit:")
    for (label, z, a0, slo, shi) in DATA:
        pred = a0_0_lin + a1_lin * z
        sigma = (slo + shi) / 2.0
        resid = (a0 - pred) / sigma
        print(f"    {label:<24s}  z={z:4.2f}  obs={a0:.2f}  Lin={pred:.2f}  "
              f"resid={resid:+.2f}sigma")

    print()
    print("=" * 78)
    print("Sensitivity test: drop Varasteanu point (the high outlier)")
    DATA_no_var = [d for d in DATA if d[0] != "Varasteanu z<0.08"]
    res3 = minimize_scalar(chi2, bounds=(0.01, 1.0), method="bounded",
                           args=(DATA_no_var,))
    K_fit_no_var = float(res3.x)
    chi2_no_var = float(res3.fun)
    print(f"  K_fit (no Varasteanu) = {K_fit_no_var:.5f}")
    print(f"  K_fit / K_HEAT        = {K_fit_no_var/K_paper:.3f}")
    print(f"  chi2 (no Varasteanu)  = {chi2_no_var:.2f}  (dof = {len(DATA_no_var)-1})")

    print()
    print("=" * 78)
    print("Sensitivity test: SPARC + Ciocan global only (drop binned + Varasteanu)")
    DATA_clean = [d for d in DATA if d[0] in ("SPARC z=0", "Ciocan z~1 global")]
    res4 = minimize_scalar(chi2, bounds=(0.01, 1.0), method="bounded",
                           args=(DATA_clean,))
    K_fit_clean = float(res4.x)
    chi2_clean = float(res4.fun)
    print(f"  K_fit (SPARC + Ciocan global) = {K_fit_clean:.5f}")
    print(f"  K_fit / K_HEAT                = {K_fit_clean/K_paper:.3f}")
    print(f"  chi2 (2 points, 1 dof)        = {chi2_clean:.2f}")

    print()
    print("=" * 78)
    print("Family scan: a0(z) = K * c * H^n(z),  fit (K, n) jointly")

    def chi2_powern(params, data):
        K, n = params
        s = 0.0
        for (_, z, a0, slo, shi) in data:
            pred = K * c * H0 * (E_lcdm(z) ** n) / 1e-10
            sigma = (slo + shi) / 2.0
            s += ((a0 - pred) / sigma) ** 2
        return s

    from scipy.optimize import minimize
    res5 = minimize(chi2_powern, x0=[K_paper, 1.0], args=(DATA,),
                    method="Nelder-Mead",
                    options={"xatol": 1e-6, "fatol": 1e-6})
    K_fit_n, n_fit = res5.x
    print(f"  K_fit = {K_fit_n:.5f}   n_fit = {n_fit:.3f}")
    print(f"  K_fit / K_HEAT = {K_fit_n/K_paper:.3f}")
    print(f"  chi2 = {res5.fun:.2f}  (dof = {len(DATA)-2})")
    print()
    print("  Compare to Paper's claim: n must be 1.00 +/- 0.07 from vdW exponent.")
    print(f"  This data alone gives n = {n_fit:.3f} (not as constrained as vdW).")


if __name__ == "__main__":
    main()
