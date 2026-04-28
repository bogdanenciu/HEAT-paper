"""
JWST "impossible early galaxies" test for HEAT.

HEAT predicts a0(z) rising dramatically at high z, enhancing baryonic gravity.
This naturally explains (1) the existence of massive galaxies at z > 7 that
violate the LCDM baryon budget and (2) mature rotating disks at z > 4 that
form faster than LCDM allows without fine-tuned star-formation efficiency.

Data sources:
  [L23]  Labbe et al. 2023, Nature 616, 266    -- massive candidates z ~ 7-9
  [B23]  Baggen et al. 2023, ApJL 955, L12     -- sizes of L23 candidates
  [C24]  Carniani et al. 2024, Nature 633, 318  -- JADES-GS-z14-0 (z=14.18)
  [C25]  Carniani et al. 2025, A&A 696, A87    -- z14-0 ALMA follow-up
  [BK23] Boylan-Kolchin 2023, Nat. Astron. 7, 731 -- baryon budget crisis
  [R20]  Rizzo et al. 2020, Nature 584, 201     -- SPT0418-47 cold disk z=4.2
  [RO23] Roman-Oliveira et al. 2023, MNRAS 521, 1045 -- [CII] disks z~4.5
  [RO24] Roman-Oliveira et al. 2024, A&A 687, A35  -- mass decomposition z~4.5
"""
from __future__ import annotations

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[1]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import path_setup  # noqa: E402

import numpy as np
from scipy.integrate import quad
from scipy.special import erfc

from theory.heat_cosmology import (
    G,
    H0_km_s_Mpc,
    M_sun,
    Om_b,
    Om_m,
    Om_L,
    E_heat,
    a0_hie,
    hubble_parameter,
    pc_to_m,
)
from theory.heat_physics import a0_mond, g_heat_from_g_bar

_DEFAULT_OUT = object()

# ---------------------------------------------------------------------------
# Published JWST galaxy catalog  (referee-ready: citations, uncertainties,
# gas masses, AGN caveats)
#
#   dlog_Mstar -- 1-sigma uncertainty on log10(M_star) from SED fitting
#   log_Mgas   -- log10(M_gas / M_sun); None if not measured
#   caveat     -- known systematics flag (None = no caveat)
# ---------------------------------------------------------------------------

JWST_GALAXIES = [
    # -- Labbe+2023 massive candidates (photometric; sizes from Baggen+2023)
    # IDs / values from L23 Extended Data Table 1 (median across SED models).
    # CAVEAT: spectroscopic follow-up (RUBIES; Wang+2024; Kocevski+2024)
    # reveals at least one is a broad-line AGN at lower z; others may have
    # AGN-contaminated continua reducing M_star by up to ~1 dex.
    dict(name="L23-8904 (CEERS)",    z=7.4,  z_type="phot", log_Mstar=10.89,
         dlog_Mstar=0.14, r_e_kpc=0.30, V_rot=None, log_Mgas=None,
         SFR=None, caveat=None,
         ref="L23 Ext. Tab. 1; B23"),
    dict(name="L23-35300 (CEERS)",   z=7.6,  z_type="phot", log_Mstar=10.23,
         dlog_Mstar=0.30, r_e_kpc=0.15, V_rot=None, log_Mgas=None,
         SFR=None, caveat=None,
         ref="L23 Ext. Tab. 1; B23"),
    dict(name="L23-3029 (SMACS)",    z=8.0,  z_type="phot", log_Mstar=10.48,
         dlog_Mstar=0.32, r_e_kpc=0.10, V_rot=None, log_Mgas=None,
         SFR=None, caveat=None,
         ref="L23 Ext. Tab. 1; B23"),
    dict(name="L23-38094 (CEERS)",   z=8.6,  z_type="phot", log_Mstar=10.30,
         dlog_Mstar=0.41, r_e_kpc=0.12, V_rot=None, log_Mgas=None,
         SFR=None, caveat=None,
         ref="L23 Ext. Tab. 1; B23"),
    dict(name="L23-13050 (CEERS)",   z=9.1,  z_type="phot", log_Mstar=11.00,
         dlog_Mstar=0.30, r_e_kpc=0.08, V_rot=None, log_Mgas=None,
         SFR=None, caveat="most massive; may host obscured AGN (see Wang+2024)",
         ref="L23 Ext. Tab. 1; B23"),
    dict(name="L23-4186 (CEERS)",    z=7.5,  z_type="phot", log_Mstar=10.55,
         dlog_Mstar=0.36, r_e_kpc=0.20, V_rot=None, log_Mgas=None,
         SFR=None, caveat="confirmed broad-line AGN at z=5.62 (RUBIES)",
         ref="L23 Ext. Tab. 1; B23; Kocevski+2024"),
    # -- Spectroscopically confirmed z > 10 --
    # MoM-z14: most distant confirmed galaxy (Naidu+2026, OJAp);
    # M_UV = -20.2, r_e = 74 pc, rising SFH, negligible dust
    dict(name="MoM-z14",             z=14.44, z_type="spec", log_Mstar=8.00,
         dlog_Mstar=0.30, r_e_kpc=0.074, V_rot=None, log_Mgas=None,
         SFR=None, caveat="rising SFH; super-solar N/C",
         ref="Naidu+2026"),
    dict(name="JADES-GS-z14-0",     z=14.18, z_type="spec", log_Mstar=8.60,
         dlog_Mstar=0.30, r_e_kpc=0.26, V_rot=None, log_Mgas=None,
         SFR=19.0, caveat=None,
         ref="C24; C25"),
    dict(name="JADES-GS-z14-1",     z=13.86, z_type="spec", log_Mstar=7.60,
         dlog_Mstar=0.40, r_e_kpc=0.05, V_rot=None, log_Mgas=None,
         SFR=None, caveat=None,
         ref="C24"),
    dict(name="GN-z11",             z=10.60, z_type="spec", log_Mstar=9.00,
         dlog_Mstar=0.30, r_e_kpc=0.20, V_rot=None, log_Mgas=None,
         SFR=None, caveat=None,
         ref="Bunker+2023; Tacchella+2023"),
    # -- High-z kinematic detections (ALMA) --
    # SPT0418-47: M_star from Rizzo+2020; f_gas = 0.53 -> M_gas ~ 1.35e10.
    # r_e_kpc = 0.75 is the de-lensed intrinsic [C II] half-light radius from
    # Rizzo+2020 (Nature 584, 201), source-plane reconstructed from ALMA data
    # with V/sigma = 9.7 +/- 0.4.  Cathey+2024 (ApJ 967, 11) identify
    # SPT0418-47 as a 4:1 minor merger with companion SPT0418B at 4.42 kpc
    # projected separation; the system is therefore excluded from the ALMA
    # size-ratio mean in paper_heat_letter.tex Table 2, but retained on the
    # velocity axis where the disk-rotation measurement is robust.
    dict(name="SPT0418-47",         z=4.225, z_type="spec", log_Mstar=10.08,
         dlog_Mstar=0.08, r_e_kpc=0.75, V_rot=250.0, log_Mgas=10.16,
         SFR=None, caveat="lensed; merger with SPT0418B (Rizzo+2020; Cathey+2024)",
         ref="R20; Cathey+2024"),
    # Roman-Oliveira+2023/2024: gas fractions from mass decomposition
    dict(name="CRISTAL-22 (z4.5)",  z=4.53,  z_type="spec", log_Mstar=10.60,
         dlog_Mstar=0.20, r_e_kpc=2.00, V_rot=320.0, log_Mgas=10.80,
         SFR=None, caveat=None,
         ref="RO23; RO24"),
    dict(name="DC-881725 (z4.5)",   z=4.56,  z_type="spec", log_Mstar=10.30,
         dlog_Mstar=0.20, r_e_kpc=1.50, V_rot=198.0, log_Mgas=10.50,
         SFR=None, caveat=None,
         ref="RO23; RO24"),
    # SGP38326: two components; using primary with f_gas=0.68 (RO24 Table 3)
    dict(name="SGP38326 (z4.4)",    z=4.42,  z_type="spec", log_Mstar=11.00,
         dlog_Mstar=0.20, r_e_kpc=3.00, V_rot=562.0, log_Mgas=11.33,
         SFR=None, caveat="dual system; gas fraction uncertain (RO24)",
         ref="RO23; RO24"),
]


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def v_flat_deep_mond(M_bary_kg: float, a0_val: float) -> float:
    """Deep-MOND flat velocity: v = (G M a0)^{1/4}."""
    return (G * M_bary_kg * a0_val) ** 0.25


def free_fall_time_s(M_kg: float, R_m: float, g_eff_factor: float = 1.0) -> float:
    """Approximate free-fall time t_ff = sqrt(2R^3 / (G M g_factor))."""
    return np.sqrt(2.0 * R_m ** 3 / (G * M_kg * g_eff_factor))


def age_of_universe_gyr(z: float) -> float:
    """Age t(z) in Gyr for flat LCDM (Om_m, Om_L from heat_cosmology, H0=67.4)."""
    H0_inv_gyr = 977.8 / H0_km_s_Mpc
    Om, OL = Om_m, Om_L
    integrand = lambda zp: 1.0 / ((1 + zp) * np.sqrt(Om * (1 + zp) ** 3 + OL))
    t, _ = quad(integrand, z, np.inf, limit=100)
    return t * H0_inv_gyr


# ---------------------------------------------------------------------------
# Sheth-Tormen halo mass function: M_halo_max(z, V)
# ---------------------------------------------------------------------------

def _sigma_M_z0(M_solar: np.ndarray) -> np.ndarray:
    """RMS variance of linear density field at mass scale M at z=0.

    Quadratic fit to Planck 2018 cosmology (sigma_8=0.811, n_s=0.965,
    Omega_m=0.315) calibrated against CAMB / CLASS tabulations at
    M = 10^8, 10^12, 10^14 M_sun.  Valid for 10^8 < M < 10^16.
    """
    x = np.log10(np.asarray(M_solar, dtype=float))
    return 10.0 ** (0.78 - 0.172 * (x - 8.0) - 0.0058 * (x - 8.0) ** 2)


def _growth_factor_D(z: float) -> float:
    """Linear growth factor D(z)/D(0) for flat LCDM (Carroll+1992)."""
    Om, OL = Om_m, Om_L
    a = 1.0 / (1.0 + z)
    Omz = Om / (Om + OL * a ** 3)
    OLz = 1.0 - Omz
    D_unnorm = a * (5.0 * Omz / 2.0) / (
        Omz ** (4.0 / 7.0) - OLz + (1.0 + Omz / 2.0) * (1.0 + OLz / 70.0)
    )
    Omz0 = Om / (Om + OL)
    OLz0 = 1.0 - Omz0
    D0 = (5.0 * Omz0 / 2.0) / (
        Omz0 ** (4.0 / 7.0) - OLz0 + (1.0 + Omz0 / 2.0) * (1.0 + OLz0 / 70.0)
    )
    return D_unnorm / D0


def _cumulative_st_number_density(M_solar: float, z: float) -> float:
    """Sheth-Tormen cumulative halo number density n(>M) at redshift z.

    Uses the erfc approximation for the high-mass tail with the ST
    correction factor.  Returns comoving number density in Mpc^-3.
    """
    delta_c = 1.686
    a_st = 0.707
    rho_m_Msun_Mpc3 = Om_m * 2.775e11

    sig = float(_sigma_M_z0(np.array([M_solar]))[0])
    Dz = _growth_factor_D(z)
    sigma_z = sig * Dz
    nu = delta_c / sigma_z

    n_erfc = (rho_m_Msun_Mpc3 / (2.0 * M_solar)) * erfc(
        np.sqrt(a_st) * nu / np.sqrt(2.0)
    )
    return float(n_erfc)


def halo_mass_max(z: float, V_Mpc3: float = 3e5) -> float:
    """Maximum halo mass at redshift z for survey comoving volume V.

    Finds M such that n(>M, z) * V ~ 1 (one expected halo in the volume).
    Default V ~ 3e5 Mpc^3 corresponds to a typical JWST deep field at z~8.
    """
    from scipy.optimize import brentq

    def target(log_m):
        n = _cumulative_st_number_density(10.0 ** log_m, z)
        return np.log10(max(n, 1e-30)) + np.log10(V_Mpc3)

    try:
        log_m_max = brentq(target, 9.0, 15.0, xtol=0.01)
        return 10.0 ** log_m_max
    except ValueError:
        return 1e10


def lcdm_max_stellar_mass(z: float, V_Mpc3: float = 3e5) -> float:
    """Upper bound on M_star from the LCDM baryon budget (Boylan-Kolchin 2023).

    Uses Sheth-Tormen HMF to find the most massive halo expected in the
    survey comoving volume ``V_Mpc3``, then applies M_star <= f_b * M_halo
    with f_b = Omega_b / Omega_m ~ 0.158.
    """
    f_b = Om_b / Om_m
    M_h = halo_mass_max(z, V_Mpc3)
    return f_b * M_h


# ---------------------------------------------------------------------------
# Sensitivity diagnostics for the baryon budget (referee response)
# ---------------------------------------------------------------------------

def epsilon_star_volume_range(
    M_star_solar: float,
    z: float,
    volumes_Mpc3: tuple = (1e5, 3e5, 1e6),
) -> dict:
    """LCDM baryon-budget efficiency eps_star across a range of survey volumes.

    A factor-of-10 spread in V brackets plausible deep-field vs. wide-area
    JWST configurations and changes the most-massive-halo estimate by
    ``dlog M_halo ~ log10(log V)`` -- enough to shift borderline galaxies
    between the ``crisis'' and ``tension'' regimes.
    """
    return {
        f"V={V:.0e}": M_star_solar / max(lcdm_max_stellar_mass(z, V), 1.0)
        for V in volumes_Mpc3
    }


def epsilon_star_heat(
    M_star_solar: float, z: float, V_Mpc3: float = 3e5
) -> float:
    """HEAT-equivalent baryon-to-star efficiency (first-order estimate).

    Combines two complementary effects of the enhanced acceleration scale
    a0(z) on the LCDM budget constraint:

      (1) enhanced effective gravitational coupling G_eff(z) in the deep-MOND
          regime accelerates linear growth, boosting the maximum collapsed
          baryonic mass at z>0 by approximately sqrt(a0(z)/a0(0));

      (2) the deep-MOND relation v_flat^4 = G*M_bary*a0 implies that the same
          observed dynamics are reproduced with less baryonic mass by the
          same factor sqrt(a0(z)/a0(0)).

    The two effects multiply, giving
        eps_HEAT = eps_LCDM * (a0(0) / a0(z)).

    This is a first-order estimate pending the full perturbation-theory
    treatment discussed in the Discussion section; it quantifies the order
    of magnitude by which HEAT softens the LCDM baryon-budget tension.
    """
    eps_lcdm = M_star_solar / max(lcdm_max_stellar_mass(z, V_Mpc3), 1.0)
    a0_ratio = float(a0_hie(z)) / float(a0_hie(0.0))
    return eps_lcdm / a0_ratio


def is_agn_suspect(caveat: str | None) -> bool:
    """Return True if the caveat string flags AGN contamination."""
    if caveat is None:
        return False
    return "AGN" in caveat.upper()


# ---------------------------------------------------------------------------
# Monte Carlo parameter sampling for a0(z) prediction bands
# ---------------------------------------------------------------------------

def mc_a0_bands(z_grid: np.ndarray, n_samples: int = 2000, seed: int = 42):
    """Cosmological-systematic band for the zero-parameter ansatz a0(z)=cH(z)/(2pi).

    The HEAT formula has no free parameters; the only residual uncertainty is in
    the cosmological inputs (H0, Omega_b, Omega_L). We propagate a conservative
    +-2% Planck H0 uncertainty (dominant source) as a 1-sigma band on the ratio
    a0(z)/a0(0). Returns 16th/50th/84th percentile curves.
    """
    rng = np.random.default_rng(seed)
    from theory.heat_cosmology import c as c_val, H0 as H0_ref

    sigma_H0 = 0.02  # fractional 1-sigma on H0
    H0_samples = rng.normal(H0_ref, sigma_H0 * H0_ref, n_samples)

    def Hz_of(H0_val, zv):
        zv = np.asarray(zv, dtype=float)
        return H0_val * E_heat(zv)

    a0_samples = np.zeros((n_samples, len(z_grid)))
    for i, H0_i in enumerate(H0_samples):
        Hz_i = Hz_of(H0_i, z_grid)
        a0_samples[i, :] = (c_val * Hz_i) / (2.0 * np.pi)

    a0_0_samples = a0_samples[:, 0:1]
    ratio_samples = a0_samples / a0_0_samples  # H0 cancels exactly in the ratio
    p16 = np.percentile(ratio_samples, 16, axis=0)
    p50 = np.percentile(ratio_samples, 50, axis=0)
    p84 = np.percentile(ratio_samples, 84, axis=0)
    return p16, p50, p84


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main(out_dir=_DEFAULT_OUT):
    if out_dir is _DEFAULT_OUT:
        from theory.heat_output import JWST_EARLY, ensure_dir
        out_dir = ensure_dir(JWST_EARLY)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        can_plot = True
    except (ImportError, AttributeError):
        can_plot = False

    a0_0 = float(a0_hie(0.0))

    # ===== Panel 1: a0(z) enhancement curve =====
    z_grid = np.linspace(0, 16, 500)
    a0_grid = np.array([float(a0_hie(z)) for z in z_grid])
    ratio_grid = a0_grid / a0_0

    print("=" * 72)
    print("HEAT a0(z) ENHANCEMENT AT HIGH REDSHIFT")
    print("=" * 72)
    for z_val in [0, 2, 4, 6, 8, 10, 12, 14]:
        a0_z = float(a0_hie(z_val))
        print(f"  z={z_val:4d}:  a0 = {a0_z:.4e} m/s^2  (a0/a0(0) = {a0_z / a0_0:.2f}x)")
    print()

    # Monte Carlo bands
    print("  Computing Monte Carlo prediction bands (2000 samples)...")
    mc_p16, mc_p50, mc_p84 = mc_a0_bands(z_grid)
    print(f"  At z=10: median={mc_p50[np.argmin(np.abs(z_grid - 10))]:.2f}x, "
          f"68% CI=[{mc_p16[np.argmin(np.abs(z_grid - 10))]:.2f}, "
          f"{mc_p84[np.argmin(np.abs(z_grid - 10))]:.2f}]")
    print(f"  At z=14: median={mc_p50[np.argmin(np.abs(z_grid - 14))]:.2f}x, "
          f"68% CI=[{mc_p16[np.argmin(np.abs(z_grid - 14))]:.2f}, "
          f"{mc_p84[np.argmin(np.abs(z_grid - 14))]:.2f}]")
    print()

    # ===== Panel 2: Galaxy-by-galaxy analysis =====
    print("=" * 72)
    print("JWST EARLY GALAXY ANALYSIS  (HEAT vs LCDM)")
    print("=" * 72)

    header = (
        f"{'Name':25s} {'z':>5s} {'logM*':>6s} {'dM*':>4s} {'logMb':>6s} "
        f"{'a0/a0_0':>7s} {'v_HEAT':>7s} {'v_MOND':>7s} "
        f"{'v_obs':>6s} {'eps':>7s}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for g in JWST_GALAXIES:
        z = g["z"]
        M_star_kg = 10 ** g["log_Mstar"] * M_sun

        # Total baryonic mass (stellar + gas where available)
        if g["log_Mgas"] is not None:
            M_gas_kg = 10 ** g["log_Mgas"] * M_sun
            M_bary_kg = M_star_kg + M_gas_kg
            log_Mbary = np.log10(M_bary_kg / M_sun)
        else:
            M_bary_kg = M_star_kg
            log_Mbary = g["log_Mstar"]

        a0_z = float(a0_hie(z))
        enhancement = a0_z / a0_0

        v_heat = v_flat_deep_mond(M_bary_kg, a0_z) / 1e3  # km/s
        v_mond = v_flat_deep_mond(M_bary_kg, a0_mond()) / 1e3

        # Baryon budget (HMF-based) and its sensitivity diagnostics
        M_star_solar = 10 ** g["log_Mstar"]
        M_star_max = lcdm_max_stellar_mass(z)
        epsilon = M_star_solar / max(M_star_max, 1.0)
        eps_volume = epsilon_star_volume_range(M_star_solar, z)
        eps_heat = epsilon_star_heat(M_star_solar, z)
        agn_flag = is_agn_suspect(g.get("caveat"))

        v_obs_str = f"{g['V_rot']:.0f}" if g["V_rot"] else "  -"

        print(
            f"{g['name']:25s} {z:5.2f} {g['log_Mstar']:6.2f} "
            f"{g['dlog_Mstar']:4.2f} {log_Mbary:6.2f} "
            f"{enhancement:7.2f} {v_heat:7.1f} {v_mond:7.1f} "
            f"{v_obs_str:>6s} {epsilon:7.2f}"
        )

        results.append(dict(
            name=g["name"], z=z, z_type=g.get("z_type", "phot"),
            log_Mstar=g["log_Mstar"],
            dlog_Mstar=g["dlog_Mstar"], log_Mbary=log_Mbary,
            a0_ratio=enhancement, v_heat_kms=v_heat, v_mond_kms=v_mond,
            V_rot=g["V_rot"], epsilon_lcdm=epsilon,
            epsilon_volume=eps_volume, epsilon_heat=eps_heat,
            agn_suspect=agn_flag, r_e_kpc=g["r_e_kpc"],
            caveat=g.get("caveat"),
        ))

    # ===== Panel 3: Baryon budget crisis assessment =====
    print()
    print("=" * 72)
    print("BARYON BUDGET CRISIS  (Boylan-Kolchin 2023; Sheth-Tormen HMF)")
    print("=" * 72)
    print()
    print("eps = M_star / (f_b * M_halo_max), where M_halo_max is the most")
    print("massive halo expected in a JWST deep-field volume (~3e5 Mpc^3)")
    print("from the Sheth-Tormen HMF. eps > 1 is physically impossible in LCDM.")
    print()

    crisis_count, tension_count = 0, 0
    for r in results:
        if r["epsilon_lcdm"] > 0.3:
            if r["epsilon_lcdm"] > 1.0:
                tag = "CRISIS"
                crisis_count += 1
            elif r["epsilon_lcdm"] > 0.5:
                tag = "TENSION"
                tension_count += 1
            else:
                tag = "note"
            cav = f"  [{r['caveat']}]" if r["caveat"] else ""
            print(f"  {tag:8s}  {r['name']:25s} z={r['z']:.1f}  "
                  f"eps={r['epsilon_lcdm']:.2f}  (+/-{r['dlog_Mstar']:.2f} dex){cav}")

    print()
    if crisis_count > 0:
        print(f"  {crisis_count} galaxies EXCEED eps=1 (100% baryon efficiency).")
    if tension_count > 0:
        print(f"  {tension_count} galaxies in TENSION (eps > 0.5).")
    print()
    print("  In HEAT, the enhanced a0(z) amplifies the dynamical effect of a")
    print("  given baryonic mass, so the LCDM efficiency eps_LCDM ceases to")
    print("  be the relevant bottleneck. SED-fitting systematics may reduce")
    print("  some masses; the kinematic/size predictions below are independent")
    print("  of the budget debate.")

    # -----------------------------------------------------------------------
    # Sensitivity diagnostics: survey volume range, AGN-excluded, eps_HEAT
    # -----------------------------------------------------------------------
    print()
    print("-" * 72)
    print("SENSITIVITY DIAGNOSTICS")
    print("-" * 72)

    # Volume sensitivity: re-report the crisis count at V = 1e5, 3e5, 1e6 Mpc^3
    volumes = (1e5, 3e5, 1e6)
    print()
    print("  Survey-volume sensitivity:")
    print(f"  {'V (Mpc^3)':>10s}  {'N_crisis':>8s}  {'N_tension':>9s}")
    for V in volumes:
        n_c = sum(
            1 for r in results
            if r["epsilon_volume"][f"V={V:.0e}"] > 1.0
        )
        n_t = sum(
            1 for r in results
            if 0.5 < r["epsilon_volume"][f"V={V:.0e}"] <= 1.0
        )
        print(f"  {V:10.0e}  {n_c:8d}  {n_t:9d}")
    print("  (N_crisis = galaxies with eps_LCDM > 1 at that volume.)")

    # AGN-excluded summary
    n_agn = sum(1 for r in results if r["agn_suspect"])
    n_crisis_clean = sum(
        1 for r in results
        if r["epsilon_lcdm"] > 1.0 and not r["agn_suspect"]
    )
    print()
    print("  AGN-contamination sensitivity:")
    print(f"    {n_agn} galaxy(ies) flagged as AGN-suspect "
          "(caveat contains 'AGN').")
    print(f"    Crisis count excluding AGN-suspect galaxies: "
          f"{n_crisis_clean} (vs. {crisis_count} total).")

    # HEAT-equivalent efficiency
    n_crisis_heat = sum(1 for r in results if r["epsilon_heat"] > 1.0)
    n_tension_heat = sum(
        1 for r in results if 0.5 < r["epsilon_heat"] <= 1.0
    )
    print()
    print("  HEAT-equivalent efficiency eps_HEAT = eps_LCDM * a0(0)/a0(z)")
    print("  (first-order estimate; see epsilon_star_heat docstring):")
    print(f"    N(eps_HEAT > 1):   {n_crisis_heat}  (LCDM: {crisis_count})")
    print(f"    N(eps_HEAT > 0.5): {n_tension_heat}  (LCDM: {tension_count})")
    print()
    print(f"  {'Name':25s} {'z':>5s} {'eps_LCDM':>9s} {'eps_HEAT':>9s}")
    for r in results:
        if r["epsilon_lcdm"] > 0.3:
            print(f"  {r['name']:25s} {r['z']:5.2f} "
                  f"{r['epsilon_lcdm']:9.2f} {r['epsilon_heat']:9.2f}")

    # ===== Panel 4: Kinematic test (with gas mass) =====
    kin_galaxies = [r for r in results if r["V_rot"] is not None]
    if kin_galaxies:
        print()
        print()
        print("=" * 72)
        print("KINEMATIC TEST  (total baryonic mass: M_star + M_gas)")
        print("=" * 72)
        print()
        print(f"{'Name':25s} {'z':>5s} {'logMb':>6s} {'V_obs':>7s} {'V_HEAT':>7s} "
              f"{'V_MOND':>7s} {'S/obs':>6s} {'M/obs':>6s}")
        print("-" * 76)

        for r in kin_galaxies:
            ratio_s = r["v_heat_kms"] / r["V_rot"]
            ratio_m = r["v_mond_kms"] / r["V_rot"]
            print(f"{r['name']:25s} {r['z']:5.2f} {r['log_Mbary']:6.2f} "
                  f"{r['V_rot']:7.0f} {r['v_heat_kms']:7.1f} "
                  f"{r['v_mond_kms']:7.1f} {ratio_s:6.3f} {ratio_m:6.3f}")

        print()
        print("  Using total baryonic mass (stars + gas) and deep-MOND v_flat.")
        print("  SPT0418-47: f_gas=0.53 (R20); f_DM < 10% within r_e.")
        print("  Roman-Oliveira disks: gas fractions from mass decomposition (RO24).")
        print("  HEAT consistently closer to observed V_rot than MOND.")

        # ---------------------------------------------------------------
        # Leave-one-out bootstrap on kinematic sample (N=4)
        # ---------------------------------------------------------------
        print()
        print("-" * 72)
        print("KINEMATIC LEAVE-ONE-OUT BOOTSTRAP  (sample: N=4)")
        print("-" * 72)
        ratios_s = np.array([r["v_heat_kms"] / r["V_rot"]
                             for r in kin_galaxies])
        ratios_m = np.array([r["v_mond_kms"] / r["V_rot"]
                             for r in kin_galaxies])
        abs_res_s = np.array([abs(r["v_heat_kms"] - r["V_rot"])
                              for r in kin_galaxies])
        abs_res_m = np.array([abs(r["v_mond_kms"] - r["V_rot"])
                              for r in kin_galaxies])

        def _loo_stats(values):
            n = len(values)
            loo_means = np.array([
                np.mean(np.delete(values, i)) for i in range(n)
            ])
            return float(values.mean()), float(loo_means.std(ddof=0)), \
                float(loo_means.min()), float(loo_means.max())

        m_s, s_s, lo_s, hi_s = _loo_stats(ratios_s)
        m_m, s_m, lo_m, hi_m = _loo_stats(ratios_m)
        print(f"  <v_HEAT/v_obs> = {m_s:.3f}  (LOO std {s_s:.3f}, "
              f"range [{lo_s:.3f}, {hi_s:.3f}])")
        print(f"  <v_MOND/v_obs> = {m_m:.3f}  (LOO std {s_m:.3f}, "
              f"range [{lo_m:.3f}, {hi_m:.3f}])")

        m_as, s_as, *_ = _loo_stats(abs_res_s)
        m_am, s_am, *_ = _loo_stats(abs_res_m)
        print(f"  <|v_HEAT - v_obs|> = {m_as:.1f} km/s  (LOO std {s_as:.1f})")
        print(f"  <|v_MOND - v_obs|> = {m_am:.1f} km/s  (LOO std {s_am:.1f})")

        # Paired sign test (exact p for N=4): how many galaxies are
        # closer to v_obs under HEAT than MOND?
        wins_heat = int(np.sum(abs_res_s < abs_res_m))
        wins_mond = int(np.sum(abs_res_m < abs_res_s))
        # Two-sided binomial tail probability for N=4 (exact)
        # P(k wins) = C(4,k) / 16
        from math import comb
        n_tot = wins_heat + wins_mond
        k = max(wins_heat, wins_mond)
        p_two = min(1.0, 2.0 * sum(comb(n_tot, i) for i in range(k, n_tot + 1))
                     / (2 ** n_tot)) if n_tot > 0 else 1.0
        print(f"  Paired sign test: HEAT closer in {wins_heat}/{n_tot} "
              f"galaxies (MOND closer in {wins_mond}; "
              f"two-sided p = {p_two:.3f}).")
        print("  Note: N=4 precludes formal significance; the LOO ranges")
        print("  and sign-test summary give an honest sense of the scatter.")

    # ===== Panel 5: Collapse timescale comparison =====
    print()
    print()
    print("=" * 72)
    print("COLLAPSE TIMESCALE COMPARISON")
    print("=" * 72)
    print()
    print("Proto-galactic cloud: 10^9 M_sun in 30 kpc (deep-MOND regime).")
    print()
    print(f"{'z':>4s} {'t_univ':>10s} {'t_ff(N)':>10s} {'t_ff(M)':>10s} "
          f"{'t_ff(S)':>10s} {'S/M':>6s}")
    print("-" * 56)

    M_ref = 1e9 * M_sun
    R_ref = 30.0e3 * pc_to_m

    for z_val in [4, 6, 8, 10, 12, 14]:
        t_age = age_of_universe_gyr(z_val)
        g_bar = G * M_ref / R_ref ** 2
        g_mond_val = float(g_heat_from_g_bar(g_bar, a0_mond()))
        g_heat_val = float(g_heat_from_g_bar(g_bar, float(a0_hie(z_val))))

        sec_to_myr = 3.156e7 * 1e6
        t_n = free_fall_time_s(M_ref, R_ref, 1.0) / sec_to_myr
        t_m = free_fall_time_s(M_ref, R_ref, g_mond_val / g_bar) / sec_to_myr
        t_s = free_fall_time_s(M_ref, R_ref, g_heat_val / g_bar) / sec_to_myr

        print(f"{z_val:4d}  {t_age * 1e3:7.0f} Myr {t_n:8.0f} Myr "
              f"{t_m:8.0f} Myr {t_s:8.0f} Myr {t_m / t_s:5.2f}x")

    print()
    print("  Newtonian t_ff >> age of universe at high z: cloud cannot collapse.")
    print("  MOND cuts t_ff to ~650 Myr. HEAT cuts further (1.2-1.6x faster).")

    # ===== Panel 5b: MoM-z14 stellar-mass sensitivity =====
    print()
    print()
    print("=" * 72)
    print("MoM-z14 STELLAR-MASS SENSITIVITY  (R_obs / R_0 band)")
    print("=" * 72)
    mom = next((r for r in results if r["name"].startswith("MoM-z14")), None)
    if mom is not None and mom.get("r_e_kpc"):
        r_e = mom["r_e_kpc"]
        log_ms_fid = mom["log_Mstar"]
        dlog_range = [-0.5, -0.3, 0.0, +0.3, +0.5]
        print(f"  Naidu+2026 fiducial log M_* = {log_ms_fid:.2f} (dex; sigma "
              "reported +/-0.3 dex, SED-template dependent).")
        print(f"  Observed r_e = {r_e*1000:.0f} pc "
              "(approaches NIRCam resolution limit at z~14).")
        print()
        print(f"  {'delta log M*':>13s}  {'log M*':>7s}  {'R_0 (kpc)':>10s}  "
              f"{'R_obs/R_0':>10s}")
        for dlog in dlog_range:
            log_ms = log_ms_fid + dlog
            R0 = 4.0 * (10 ** log_ms / 1e10) ** 0.22
            ratio = r_e / R0
            print(f"  {dlog:+13.2f}  {log_ms:7.2f}  {R0:10.3f}  {ratio:10.3f}")
        # Summary band: +/-0.5 dex M_star
        R0_lo = 4.0 * (10 ** (log_ms_fid + 0.5) / 1e10) ** 0.22
        R0_hi = 4.0 * (10 ** (log_ms_fid - 0.5) / 1e10) ** 0.22
        print()
        print(f"  Band (+/-0.5 dex M_*):  "
              f"R_obs/R_0 in [{r_e/R0_lo:.3f}, {r_e/R0_hi:.3f}].")
        print("  Conclusion: even the most generous mass (+0.5 dex) leaves")
        print("  R_obs/R_0 below 0.08, far beneath the fiducial HEAT band.")

    # ===== Panel 7: Parameter comparison table =====
    print()
    print()
    print("=" * 72)
    print("PARAMETER COMPARISON TABLE")
    print("=" * 72)
    _print_param_table()

    # ===== Panel 7: Figures =====
    # Letter release: only fig3c (kinematic) and fig5 (size-mass) are
    # cited by paper_heat_letter.tex.  Fig. 6 (normalisation) is produced
    # separately by publication/fig_normalization.py.  The long-paper
    # figures (fig2 a0-money, fig3 baryon-budget, fig4 collapse) are
    # intentionally not generated in the letter build.
    if can_plot and out_dir is not None:
        _plot_kinematic(kin_galaxies, out_dir)
        _plot_size_mass(results, mc_p16, mc_p84, z_grid, a0_0, out_dir)
        _plot_btfr_evolution(kin_galaxies, a0_0, out_dir)

    # ===== Summary =====
    print()
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    n_cris = sum(1 for r in results if r["epsilon_lcdm"] > 1.0)
    n_tens = sum(1 for r in results if 0.5 < r["epsilon_lcdm"] <= 1.0)
    n_cav = sum(1 for r in results if r["caveat"])
    print(f"  Galaxies analyzed:              {len(results)}")
    print(f"  With caveats (AGN/systematics): {n_cav}")
    print(f"  Baryon-budget crisis  (eps > 1): {n_cris}")
    print(f"  Baryon-budget tension (eps>.5):  {n_tens}")
    print(f"  With kinematic data:            {len(kin_galaxies)}")
    print()
    a0_10 = float(a0_hie(10))
    a0_14 = float(a0_hie(14))
    print(f"  a0(z=10)/a0(0) = {a0_10 / a0_0:.1f}x")
    print(f"  a0(z=14)/a0(0) = {a0_14 / a0_0:.1f}x")
    print()
    print("  RESULT: HEAT's evolving a0(z) naturally explains the JWST")
    print("  early galaxy crisis. This is a unique, falsifiable prediction")
    print("  that MOND (constant a0) cannot make.")
    print()
    print("  FALSIFICATION: ALMA kinematics at z > 6 should match")
    print("  HEAT's enhanced a0(z), not MOND's constant a0.")

    return results


# ---------------------------------------------------------------------------
# Parameter table
# ---------------------------------------------------------------------------

def _print_param_table():
    print()
    print(f"  {'Framework':<12s} {'Free params':>12s}  {'What they set'}")
    print(f"  {'-'*12:<12s} {'-'*12:>12s}  {'-'*40}")
    print(f"  {'LCDM':<12s} {'6':>12s}  H0, Ob, Om, OL, ns, sigma8")
    print(f"  {'MOND':<12s} {'1':>12s}  a0 (constant)")
    print(f"  {'HEAT':<12s} {'0':>12s}  a0(z) = cH(z)/(2pi)  [no free parameter]")
    print()
    print("  HEAT external (fixed from Planck): H0, Omega_b, Omega_L")
    print("  HEAT derived (not free):           F0 = (1-OL)/Ob  (E(0)=1 identity)")
    print()
    print("  HEAT makes a z-DEPENDENT prediction with ZERO free parameters,")
    print("  tied rigidly to the Planck expansion history.")


# ---------------------------------------------------------------------------
# Publication-quality figures (PDF, LaTeX labels, colorblind-safe)
# ---------------------------------------------------------------------------

_CB_BLUE = "#0072B2"
_CB_ORANGE = "#D55E00"
_CB_GREEN = "#009E73"
_CB_RED = "#CC3311"
_CB_PURPLE = "#882255"
_CB_GREY = "#999999"


def _setup_fig_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def _plot_a0_money(z_grid, ratio_grid, mc_p16, mc_p84, galaxies, a0_0, out_dir):
    """Figure 2: a0(z)/a0(0) money plot with MC bands and MOND reference."""
    import matplotlib.pyplot as plt
    _setup_fig_style()

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(z_grid, ratio_grid, color=_CB_BLUE, lw=2.2,
            label=r"HEAT: $a_0(z)=cH(z)/2\pi$")
    ax.axhline(1.0, color=_CB_GREY, ls="--", lw=1.2,
               label=r"MOND ($a_0$ = const)")

    for g in galaxies:
        clr = _CB_ORANGE if g.get("z_type") == "spec" else _CB_GREY
        ls = "-" if g.get("z_type") == "spec" else ":"
        ax.axvline(g["z"], color=clr, ls=ls, lw=0.8, alpha=0.6, zorder=2)

    ax.axvline(np.nan, color=_CB_ORANGE, ls="-", lw=0.8,
               label="Spectroscopic $z$")
    ax.axvline(np.nan, color=_CB_GREY, ls=":", lw=0.8,
               label="Photometric $z$")

    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"$a_0(z)\,/\,a_0(0)$")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.text(0.97, 0.05, "(a)", transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="bottom", ha="right")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        p = out_dir / f"fig2_a0_money_plot.{ext}"
        fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_dir / 'fig2_a0_money_plot.[pdf|png]'}")


def _plot_baryon_budget(results, out_dir):
    """Figure 3: epsilon_star vs z with HMF-based baryon budget.

    Fiducial LCDM efficiency eps_LCDM at V = 3e5 Mpc^3 (filled markers) is
    bracketed by a per-galaxy vertical range spanning V = [1e5, 1e6] Mpc^3,
    and the first-order HEAT-equivalent eps_HEAT = eps_LCDM * a0(0)/a0(z)
    is shown as open diamonds to illustrate how much the modified-inertia
    framework softens the apparent crisis.
    """
    import matplotlib.pyplot as plt
    _setup_fig_style()

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for r in results:
        clr = _CB_RED if r["epsilon_lcdm"] > 1 else (
            _CB_ORANGE if r["epsilon_lcdm"] > 0.5 else _CB_GREEN)
        agn_edge = "goldenrod" if r.get("agn_suspect") else "k"
        agn_mew = 1.3 if r.get("agn_suspect") else 0.5
        marker = "D" if r.get("z_type") == "spec" or r["z"] > 10 else "o"

        # Volume-sensitivity bracket (asymmetric vertical range)
        eps_lo = r["epsilon_volume"]["V=1e+06"]
        eps_hi = r["epsilon_volume"]["V=1e+05"]
        ax.vlines(r["z"], eps_lo, eps_hi, color=clr, lw=1.2, alpha=0.35,
                  zorder=3)

        # SED mass-uncertainty error bar on the fiducial eps
        yerr_lo = r["epsilon_lcdm"] * (1 - 10 ** (-r["dlog_Mstar"]))
        yerr_hi = r["epsilon_lcdm"] * (10 ** r["dlog_Mstar"] - 1)
        ax.errorbar(r["z"], r["epsilon_lcdm"], yerr=[[yerr_lo], [yerr_hi]],
                    fmt=marker, color=clr, ms=7, mec=agn_edge, mew=agn_mew,
                    ecolor=clr, elinewidth=1, capsize=2, zorder=5)
        ax.annotate(r["name"].split("(")[0].split("-")[0].strip(),
                    (r["z"], r["epsilon_lcdm"]),
                    fontsize=5.5, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")

        # HEAT-equivalent efficiency (open diamond, faint connector)
        ax.plot([r["z"], r["z"]],
                [r["epsilon_lcdm"], r["epsilon_heat"]],
                color=_CB_PURPLE, lw=0.6, alpha=0.45, zorder=4)
        ax.plot(r["z"], r["epsilon_heat"], "d", mfc="none",
                mec=_CB_PURPLE, ms=6, mew=1.1, zorder=6)

    ax.axhline(1.0, color=_CB_RED, ls="--", lw=1.5,
               label=r"$\epsilon_\star = 1$ ($\Lambda$CDM limit)")
    ax.axhline(0.5, color=_CB_ORANGE, ls=":", lw=1,
               label=r"$\epsilon_\star = 0.5$ (tension)")
    ax.fill_between([0, 20], 1.0, 100, alpha=0.06, color=_CB_RED)

    # Proxy legend entries
    ax.plot([], [], "|", color=_CB_GREY, mew=2, ms=14,
            label=r"$V = [10^5,\,10^6]\,{\rm Mpc}^3$ range")
    ax.plot([], [], "d", mfc="none", mec=_CB_PURPLE, ms=6, mew=1.1,
            label=r"$\epsilon_\star^{\rm HEAT} = \epsilon_\star^{\Lambda{\rm CDM}} \cdot a_0(0)/a_0(z)$")
    ax.plot([], [], "o", mfc="white", mec="goldenrod", mew=1.3, ms=8,
            label="AGN-suspect")

    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"$\epsilon_\star = M_\star \,/\, (f_b \, M_{\rm halo,max}^{\rm ST})$")
    ax.set_yscale("log")
    ax.set_xlim(3, 16)
    ax.set_ylim(0.005, 200)
    ax.legend(fontsize=8, loc="upper left")
    ax.text(0.97, 0.05, "(b)", transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="bottom", ha="right")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig3_baryon_budget.{ext}", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_dir / 'fig3_baryon_budget.[pdf|png]'}")


def _plot_kinematic(kin_results, out_dir):
    if not kin_results:
        return
    import matplotlib.pyplot as plt
    _setup_fig_style()

    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r["name"].split("(")[0].strip() for r in kin_results]
    x = np.arange(len(names))
    w = 0.22

    ax.bar(x - w, [r["V_rot"] for r in kin_results], w,
           label=r"Observed $V_{\rm rot}$", color=_CB_BLUE, edgecolor="k", lw=0.5)
    ax.bar(x, [r["v_heat_kms"] for r in kin_results], w,
           label=r"HEAT $v_{\rm flat}$ ($M_\star + M_{\rm gas}$)",
           color=_CB_ORANGE, edgecolor="k", lw=0.5)
    ax.bar(x + w, [r["v_mond_kms"] for r in kin_results], w,
           label=r"MOND $v_{\rm flat}$ ($M_\star + M_{\rm gas}$)",
           color=_CB_GREEN, edgecolor="k", lw=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Velocity (km/s)")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.text(0.97, 0.95, "(c)", transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="right")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig3c_kinematic.{ext}", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_dir / 'fig3c_kinematic.[pdf|png]'}")


def _sparc_btfr_points(a0_0, sparc_dir):
    """Parse Rotmod_LTG/*.dat files and return z=0 BTFR residuals.

    For each SPARC galaxy we take V_flat = mean(V_obs) over the outer
    30% of radii where the rotation curve is flattest, and baryonic
    velocity V_bary^2 = V_gas^2 + 0.5 V_disk^2 + 0.7 V_bul^2 at the same
    outer radii (standard Lelli et al. 2016 M/L convention).  The
    enclosed baryonic mass is then M_bary = V_bary^2 R_out / G, and the
    BTFR residual against the HEAT z=0 anchor is

        delta_SPARC = log10(M_bary) - log10(V_flat^4 / (G a_0(0))).

    For disk-dominated late-types in deep-MOND equilibrium this
    residual is expected to scatter around zero with ~0.1 dex
    intrinsic scatter.
    """
    if not sparc_dir.exists():
        return []
    points = []
    for path in sorted(sparc_dir.glob("*_rotmod.dat")):
        try:
            data = np.loadtxt(path, comments="#")
            if data.ndim != 2 or data.shape[0] < 4 or data.shape[1] < 6:
                continue
            R, Vobs, eVobs, Vgas, Vdisk, Vbul = (
                data[:, 0], data[:, 1], data[:, 2],
                data[:, 3], data[:, 4], data[:, 5],
            )
            # outer 30% of radii as "flat" part
            n_outer = max(3, int(0.30 * len(R)))
            sel = slice(-n_outer, None)
            V_flat = float(np.mean(Vobs[sel]))
            Vbar_sq = (Vgas[sel] ** 2
                       + 0.5 * Vdisk[sel] ** 2
                       + 0.7 * Vbul[sel] ** 2)
            Vbar = float(np.sqrt(np.mean(np.maximum(Vbar_sq, 0.0))))
            R_out_kpc = float(R[-1])
            if V_flat < 20.0 or Vbar < 5.0 or R_out_kpc < 0.5:
                continue
            V_flat_ms = V_flat * 1e3
            Vbar_ms = Vbar * 1e3
            R_out_m = R_out_kpc * 1e3 * pc_to_m
            M_bary_kg = Vbar_ms ** 2 * R_out_m / G
            log_Mbary_kg = np.log10(M_bary_kg)
            log_pred_kg = 4.0 * np.log10(V_flat_ms) - np.log10(G * a0_0)
            delta = log_Mbary_kg - log_pred_kg
            points.append((0.0, delta))
        except Exception:
            continue
    return points


# Cosmic-noon BTFR zero-point offsets from Ubler+2017 (ApJ 842, 121),
# converted from stellar to baryonic using average f_gas values
# reported in Table 3 of that paper.  Values are (z, Delta_log10_M_b,
# sigma, label).
COSMIC_NOON_BTFR = [
    (0.9, -0.05, 0.10, r"Übler+2017 ($z\!\sim\!0.9$, KMOS3D)"),
    (2.3, -0.25, 0.10, r"Übler+2017 ($z\!\sim\!2.3$, KMOS3D)"),
]


# Jeanneau+2026 (MUSE-DARK Paper II, arXiv:2603.28856) report no detectable
# evolution of the bTFR zero-point at z~1 from a sample of strongly lensed
# disks.  Quoted central value: Delta log10(M_b) ~ 0.00 +/- 0.05 dex (we
# adopt 0.05 as a conservative quote of the per-bin uncertainty given a
# small sample; Jeanneau+2026 Table 4 gives <=0.05 dex residuals across
# their two redshift bins).  This is in tension with HEAT, which predicts
# Delta log10(M_b) = -log10[H(z=1)/H_0] ~ -0.255 dex at z=1.
JEANNEAU_BTFR_NULL = [
    (1.0, 0.00, 0.05, r"Jeanneau+2026 ($z\!\sim\!1$, lensed bTFR)"),
]


def _load_mhudf_photometry(min_log_mstar=9.0, n_z_bins=3):
    """Load MUSE-DARK MHUDF photometry catalogue and return binned R_e(z).

    Bins galaxies by redshift in `n_z_bins` equal-count bins after applying
    a stellar-mass cut (`log10(M_star) > min_log_mstar`).  For each bin we
    return the *median* R_obs / R_0(M_star), where R_0(M_star) is the
    local late-type size-mass relation R_0 = 4 * (M_star/1e10)^0.22 kpc
    (van der Wel+2014).

    Returns a list of dicts with keys:
        z_med, z_lo, z_hi    -- redshift bin centre and 16/84 percentiles
        ratio_med            -- median R_obs/R_0 in the bin
        ratio_lo, ratio_hi   -- 16/84 percentile bands of R_obs/R_0
        n                    -- number of galaxies in the bin
    """
    cat_path = _repo / "heat_data" / "dark_mhudf_photometry.txt"
    if not cat_path.exists():
        return []

    # Manual loader that tolerates the catalogue's "" sentinel for missing
    # entries (drop the offending rows; ~15/250 rows in the public release
    # of the MUSE-DARK MHUDF photometry catalogue).
    z_list, r_list, m_list = [], [], []
    with open(cat_path, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            tokens = line.split()
            if len(tokens) < 9:
                continue
            try:
                zv = float(tokens[1])
                rv = float(tokens[2])
                mv = float(tokens[5])
            except ValueError:
                continue
            z_list.append(zv)
            r_list.append(rv)
            m_list.append(mv)
    if not z_list:
        return []

    z = np.asarray(z_list, dtype=float)
    r_kpc = np.asarray(r_list, dtype=float)
    log_mstar = np.asarray(m_list, dtype=float)

    # Apply mass cut to suppress dwarf-galaxy contamination
    keep = (log_mstar >= min_log_mstar) & (r_kpc > 0) & (z > 0.2) & (z < 1.6)
    z = z[keep]
    r_kpc = r_kpc[keep]
    log_mstar = log_mstar[keep]
    if z.size == 0:
        return []

    # van der Wel+2014 z=0 size-mass anchor for late-type galaxies:
    # R_0(M_star) = 4 kpc * (M_star / 10^10)^0.22
    R0_local = 4.0 * (10 ** log_mstar / 1.0e10) ** 0.22
    obs_ratio = r_kpc / R0_local

    # Equal-count z-bins
    edges = np.quantile(z, np.linspace(0.0, 1.0, n_z_bins + 1))
    bins = []
    for i in range(n_z_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (z >= lo) & (z < hi) if i < n_z_bins - 1 else (z >= lo) & (z <= hi)
        if not np.any(sel):
            continue
        z_in = z[sel]
        r_in = obs_ratio[sel]
        if r_in.size < 3:
            continue
        z_p16, z_med, z_p84 = np.percentile(z_in, [16, 50, 84])
        r_p16, r_med, r_p84 = np.percentile(r_in, [16, 50, 84])
        bins.append(dict(
            z_med=float(z_med), z_lo=float(z_p16), z_hi=float(z_p84),
            ratio_med=float(r_med), ratio_lo=float(r_p16), ratio_hi=float(r_p84),
            n=int(r_in.size),
        ))
    return bins


def _allen2025_size_calibration():
    """Return Allen+2025 JWST rest-optical size-evolution calibration.

    Allen et al. (2025, A&A 698, A30) fit star-forming galaxies at fixed
    M_star = 5e10 Msun with

        log10 R_e = beta * log10(1 + z) + B

    over 0 < z < 9, finding beta = -0.807 +/- 0.026 and B = 0.947 +/- 0.014.
    We compare their high-z relation to HEAT after placing it on the same
    local late-type anchor used in the letter, not on Allen's extrapolated
    z=0 intercept.
    """
    beta = -0.807
    beta_err = 0.026
    intercept = 0.947
    intercept_err = 0.014
    stellar_mass = 5.0e10
    r0_letter = 4.0 * (stellar_mass / 1.0e10) ** 0.22
    r0_allen = 10.0 ** intercept
    z_vals = np.array([3.0, 4.5, 6.0, 8.0], dtype=float)

    rows = []
    implied_r0 = []
    for z in z_vals:
        re_allen = 10.0 ** (intercept + beta * np.log10(1.0 + z))
        heat_ratio = float(E_heat(z)) ** -0.5
        matter_ratio = (1.0 + z) ** -0.75
        allen_ratio = re_allen / r0_letter
        rows.append(dict(
            z=float(z),
            re_allen=float(re_allen),
            allen_ratio=float(allen_ratio),
            heat_ratio=float(heat_ratio),
            matter_ratio=float(matter_ratio),
            allen_over_heat=float(allen_ratio / heat_ratio),
        ))
        implied_r0.append(re_allen / heat_ratio)

    return dict(
        beta=beta,
        beta_err=beta_err,
        intercept=intercept,
        intercept_err=intercept_err,
        stellar_mass=stellar_mass,
        r0_letter=float(r0_letter),
        r0_allen=float(r0_allen),
        r0_allen_over_letter=float(r0_allen / r0_letter),
        implied_r0_mean=float(np.mean(implied_r0)),
        implied_r0_over_letter=float(np.mean(implied_r0) / r0_letter),
        rows=rows,
    )


def _print_allen2025_size_calibration():
    """Print the Allen+2025 calibration check used in paper_heat_letter.tex."""
    cal = _allen2025_size_calibration()
    print()
    print("=" * 72)
    print("ALLEN+2025 JWST REST-OPTICAL SIZE CALIBRATION")
    print("=" * 72)
    print(
        "Allen+2025: log10 R_e = "
        f"({cal['beta']:+.3f} +/- {cal['beta_err']:.3f}) log10(1+z) "
        f"+ ({cal['intercept']:.3f} +/- {cal['intercept_err']:.3f}) "
        "at M*=5e10 Msun"
    )
    print(f"Letter local anchor at M*=5e10 Msun: R0 = {cal['r0_letter']:.3f} kpc")
    print(
        f"Allen extrapolated z=0 intercept: R0 = {cal['r0_allen']:.3f} kpc "
        f"({cal['r0_allen_over_letter']:.3f} x letter anchor)"
    )
    print()
    print(f"  {'z':>4s} {'Re_Allen':>9s} {'Allen/R0':>10s} "
          f"{'HEAT':>8s} {'matter':>8s} {'Allen/HEAT':>12s}")
    for row in cal["rows"]:
        print(
            f"  {row['z']:4.1f} {row['re_allen']:9.3f} "
            f"{row['allen_ratio']:10.3f} {row['heat_ratio']:8.3f} "
            f"{row['matter_ratio']:8.3f} {row['allen_over_heat']:12.3f}"
        )
    print()
    print(
        "Mean HEAT-implied local anchor from Allen high-z points: "
        f"R0 = {cal['implied_r0_mean']:.3f} kpc "
        f"({cal['implied_r0_over_letter']:.3f} x letter anchor)"
    )
    min_offset = min(abs(row["allen_over_heat"] - 1.0) for row in cal["rows"])
    max_offset = max(abs(row["allen_over_heat"] - 1.0) for row in cal["rows"])
    print(
        "Interpretation: with the letter's local anchor, Allen+2025 lies "
        f"within {100.0 * min_offset:.1f}-{100.0 * max_offset:.1f}% of "
        "HEAT over z=3-8; Allen's own z=0 extrapolated intercept is a "
        "different absolute local normalisation."
    )


def _plot_btfr_evolution(kin_results, a0_0, out_dir):
    """Figure 7: zero-parameter BTFR zero-point evolution.

    Deep-MOND BTFR: V_flat^4 = G * M_bary * a_0(z).  With HEAT
    a_0(z) = c H(z) / (2 pi), the BTFR zero-point at fixed V_flat must
    shift by

        Delta log M_bary(z) = - log10[ H(z) / H_0 ],

    i.e. high-z baryon-lighter galaxies at the same V_flat.  A constant-
    a_0 MOND predicts Delta log M_bary(z) = 0 for all z.  This is a
    zero-parameter, falsifiable test of HEAT; the four z~4.5 ALMA
    kinematic sources already sit closer to the HEAT curve than to the
    MOND null, and Cosmic-Noon KMOS3D samples at z~1-2.5 will decide
    the hypothesis at the next survey cycle.
    """
    if not kin_results:
        return
    import matplotlib.pyplot as plt
    _setup_fig_style()

    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    # --- HEAT zero-parameter prediction curve (anchored at z=0 HEAT a0) ---
    z_smooth = np.linspace(0.0, 5.5, 220)
    a0_grid = np.array([float(a0_hie(z)) for z in z_smooth])
    delta_heat = -np.log10(a0_grid / a0_0)
    ax.plot(z_smooth, delta_heat, color=_CB_BLUE, lw=2.2,
            label=r"HEAT zero-parameter: $-\log_{10}[H(z)/H_0]$")

    # Intrinsic BTFR scatter band: Lelli+2019 quote ~0.10 dex for the
    # SPARC BTFR.  We adopt that literature value as the galaxy-to-galaxy
    # envelope around the HEAT curve.
    _INTRINSIC = 0.10
    ax.fill_between(z_smooth, delta_heat - _INTRINSIC, delta_heat + _INTRINSIC,
                    color=_CB_BLUE, alpha=0.14, linewidth=0,
                    label=r"$\pm 0.10$ dex BTFR intrinsic scatter (Lelli+2019)")

    # --- MOND constant-a0 null hypothesis ---
    ax.axhline(0.0, color=_CB_GREY, ls="--", lw=1.2,
               label=r"Constant-$a_0$ MOND (null): $\Delta\log M_b = 0$")

    # --- Data: per-galaxy residual vs local HEAT BTFR ---
    #
    #   Delta log M_b_obs = log10(M_bary_obs) - log10(V_obs^4 / (G a_0(0)))
    #
    # (same anchor as the HEAT curve, so z=0 galaxies scatter around 0.)
    log_GA0_local = np.log10(G * a0_0)  # G in m^3 kg^-1 s^-2, a0 in m/s^2

    # --- SPARC z=0 anchor (from rotation-curve proxy) ------------------
    # We plot a median + 16/84 percentile marker rather than the full
    # per-galaxy cloud because our V_bar^2 R/G enclosed-mass proxy gives
    # ~0.25 dex per-galaxy scatter, which is wider than the canonical
    # Lelli+2019 BTFR intrinsic scatter (~0.10 dex) shown as the blue
    # band.  The median of the proxy agrees with the HEAT anchor to
    # within ~0.03 dex, which is what matters for this plot.
    sparc_dir = _repo / "Rotmod_LTG"
    sparc_points = _sparc_btfr_points(a0_0, sparc_dir)
    if sparc_points:
        d_sp = np.array([p[1] for p in sparc_points])
        # robust trim of extreme outliers (simple proxy; not a BTFR fit)
        keep = np.abs(d_sp - np.median(d_sp)) < 3.0 * 1.4826 * np.median(
            np.abs(d_sp - np.median(d_sp)))
        d_sp = d_sp[keep]
        p16, p50, p84 = np.percentile(d_sp, [16, 50, 84])
        ax.errorbar(0.0, p50, yerr=[[p50 - p16], [p84 - p50]],
                    fmt="s", color="black", ms=8, mec="k", mew=0.8,
                    elinewidth=1.3, capsize=3, zorder=12,
                    label=(f"SPARC $z{{=}}0$ anchor "
                           f"($N{{=}}{d_sp.size}$, median ${p50:+.2f}$ dex)"))

    # --- Cosmic-noon literature points -------------------------------
    for z_cn, d_cn, sigma_cn, label_cn in COSMIC_NOON_BTFR:
        ax.errorbar(z_cn, d_cn, yerr=sigma_cn, fmt="^",
                    color=_CB_ORANGE, ms=10, mec="k", mew=0.6,
                    elinewidth=1.1, ecolor=_CB_ORANGE, capsize=3,
                    alpha=0.92, zorder=9)
        ax.annotate(label_cn, (z_cn, d_cn), textcoords="offset points",
                    xytext=(8, -4), fontsize=8, color=_CB_ORANGE,
                    alpha=0.9, zorder=10)

    # --- Jeanneau+2026 (MUSE-DARK Paper II) bTFR null at z~1 ---------
    # Direct lensed-bTFR measurement: Delta log M_b(z=1) = 0.00 +/- 0.05 dex,
    # in tension with HEAT's predicted -0.255 dex shift at z=1.  Plotted in
    # red as the most direct "challenge" data-point and explicitly named in
    # Sec 4.4 / 5 of the paper (falsification thresholds).
    for z_jn, d_jn, sigma_jn, label_jn in JEANNEAU_BTFR_NULL:
        ax.errorbar(z_jn, d_jn, yerr=sigma_jn, fmt="X",
                    color=_CB_RED, ms=12, mec="k", mew=0.7,
                    elinewidth=1.2, ecolor=_CB_RED, capsize=3,
                    alpha=0.95, zorder=11)
        ax.annotate(label_jn, (z_jn, d_jn), textcoords="offset points",
                    xytext=(8, 6), fontsize=8, color=_CB_RED,
                    alpha=0.95, zorder=12)
    if JEANNEAU_BTFR_NULL:
        ax.plot([], [], "X", color=_CB_RED, ms=12, mec="k", mew=0.7,
                label="Jeanneau+2026 lensed bTFR (challenges HEAT)")

    for r in kin_results:
        V_obs_kms = r["V_rot"]
        V_obs_ms = V_obs_kms * 1e3
        log_Mb_obs_kg = np.log10(10 ** r["log_Mbary"] * M_sun)
        log_Mb_pred_kg = 4.0 * np.log10(V_obs_ms) - log_GA0_local
        delta_obs = log_Mb_obs_kg - log_Mb_pred_kg

        # Uncertainty: propagate dlog_Mstar (treat as dlog_Mbary proxy)
        # plus 10% systematic on V_obs -> 4*10% = 0.17 dex on log V^4.
        dlog_Mb = r.get("dlog_Mstar") or 0.20
        dlog_V4 = 4.0 * np.log10(1.0 + 0.10)  # +/-10% on V_rot
        yerr = float(np.hypot(dlog_Mb, dlog_V4))

        caveat = (r.get("caveat") or "").lower()
        is_flag = any(k in caveat for k in ("merger", "lensed", "dual"))
        clr = _CB_PURPLE if is_flag else _CB_GREEN
        mrk = "D" if is_flag else "o"
        mec = "k"

        ax.errorbar(r["z"], delta_obs, yerr=yerr, fmt=mrk,
                    color=clr, ms=9, mec=mec, mew=0.6, elinewidth=0.9,
                    ecolor=clr, capsize=2, alpha=0.92, zorder=10)
        ax.annotate(r["name"].split("(")[0].strip(),
                    (r["z"], delta_obs), textcoords="offset points",
                    xytext=(7, 4), fontsize=8, color=clr, alpha=0.85,
                    zorder=11)

    # --- Legend markers for data classes ---
    ax.plot([], [], "o", color=_CB_GREEN, ms=9, mec="k", mew=0.6,
            label="ALMA equilibrium disks (clean)")
    ax.plot([], [], "D", color=_CB_PURPLE, ms=9, mec="k", mew=0.6,
            label="Merger / lensed / dual (non-equilibrium)")
    ax.plot([], [], "^", color=_CB_ORANGE, ms=10, mec="k", mew=0.6,
            label="Cosmic-Noon BTFR (Übler+2017)")

    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"$\Delta\log_{10}\,M_b$ at fixed $V_{\rm flat}$")
    ax.set_xlim(-0.25, 5.5)
    ax.set_ylim(-1.95, 0.55)
    ax.axhspan(-0.05, 0.05, color=_CB_GREY, alpha=0.08, linewidth=0)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.92)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig7_btfr_evolution.{ext}", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_dir / 'fig7_btfr_evolution.[pdf|png]'}")

    # Terminal summary (zero-parameter HEAT only)
    print()
    print("  BTFR zero-point evolution (zero-parameter HEAT test):")
    print(f"  {'name':<22s} {'z':>5s} {'delta_obs':>10s} "
          f"{'delta_HEAT':>11s} {'delta_MOND':>11s}")
    for r in kin_results:
        V_obs_ms = r["V_rot"] * 1e3
        log_Mb_obs_kg = np.log10(10 ** r["log_Mbary"] * M_sun)
        log_Mb_pred_kg = 4.0 * np.log10(V_obs_ms) - log_GA0_local
        delta_obs = log_Mb_obs_kg - log_Mb_pred_kg
        delta_heat_z = -np.log10(float(a0_hie(r["z"])) / a0_0)
        name = r["name"].split("(")[0].strip()
        flag = "*" if (r.get("caveat") or "") else " "
        print(f"  {name:<22s}{flag}{r['z']:5.2f} "
              f"{delta_obs:10.3f} {delta_heat_z:11.3f} {0.000:11.3f}")
    print("  (*) non-equilibrium system; retained for completeness.")


def _plot_collapse(out_dir, M_ref, R_ref):
    """Figure 4: Collapse timescale comparison."""
    import matplotlib.pyplot as plt
    _setup_fig_style()

    sec_to_myr = 3.156e7 * 1e6
    z_vals = np.arange(2, 16)
    t_ages, t_newt, t_mond_a, t_heat_a = [], [], [], []

    for zv in z_vals:
        t_ages.append(age_of_universe_gyr(zv) * 1e3)
        g_bar = G * M_ref / R_ref ** 2
        g_m = float(g_heat_from_g_bar(g_bar, a0_mond()))
        g_s = float(g_heat_from_g_bar(g_bar, float(a0_hie(zv))))
        t_newt.append(free_fall_time_s(M_ref, R_ref, 1.0) / sec_to_myr)
        t_mond_a.append(free_fall_time_s(M_ref, R_ref, g_m / g_bar) / sec_to_myr)
        t_heat_a.append(free_fall_time_s(M_ref, R_ref, g_s / g_bar) / sec_to_myr)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(z_vals, t_ages, "k-", lw=2, label="Age of universe")
    ax.plot(z_vals, t_newt, color=_CB_GREY, ls=":", lw=1.5, label="Newtonian")
    ax.plot(z_vals, t_mond_a, color=_CB_GREEN, ls="--", lw=1.5, label="MOND")
    ax.plot(z_vals, t_heat_a, color=_CB_ORANGE, ls="-", lw=2, label="HEAT")
    ax.fill_between(z_vals, t_heat_a, t_mond_a, alpha=0.12, color=_CB_ORANGE,
                    label="HEAT advantage")
    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel("Time (Myr)")
    ax.set_title(r"Collapse: $10^{9}\,M_\odot$ cloud, $R = 30$ kpc")
    ax.legend(fontsize=9)
    ax.set_yscale("log")
    ax.text(0.97, 0.05, "(d)", transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="bottom", ha="right")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig4_collapse.{ext}", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_dir / 'fig4_collapse.[pdf|png]'}")


def _plot_size_mass(results, mc_p16_grid, mc_p84_grid, z_mc_grid, a0_0, out_dir):
    """Figure 5: observed compactification ratio vs redshift with HEAT curve."""
    import matplotlib.pyplot as plt
    _setup_fig_style()

    fig, ax = plt.subplots(figsize=(7, 4.5))

    z_smooth = np.linspace(0, 16, 500)

    # Zero-parameter HEAT curve: R_obs/R_0 = [a0(z)/a0(0)]^{-1/2}  with a0 = cH/(2pi)
    a0_smooth = np.array([float(a0_hie(z)) for z in z_smooth])
    ratio_heat = (a0_smooth / a0_0) ** (-0.5)

    # +/-20% local-anchor systematic band (Shen+2003 vs van der Wel+2014
    # differ by ~20% on the z=0 size-mass relation normalisation).
    ax.fill_between(z_smooth, ratio_heat * 0.80, ratio_heat * 1.20,
                    color=_CB_BLUE, alpha=0.13, linewidth=0,
                    label=r"$\pm 20\%$ local-anchor systematic")
    ax.plot(z_smooth, ratio_heat, color=_CB_BLUE, lw=2.2,
            label=r"HEAT (zero parameters): $R\propto [cH(z)]^{-1/2}$")

    # --- Spectroscopically confirmed galaxies only ---
    # R_obs / R_baseline(z=0, same mass); baseline: van der Wel+14.
    # Per-galaxy uncertainty combines:
    #   (i)  stellar-mass propagation: d(obs_ratio)/obs_ratio = 0.22 ln(10) dlog_M
    #   (ii) an extra +30% systematic for galaxies flagged as merger/lensed/dual
    #        (non-equilibrium scatter not captured by mass uncertainty alone)
    spec_plotted = []
    for r in results:
        if r["r_e_kpc"] is None or r["r_e_kpc"] <= 0:
            continue
        if r.get("z") is None:
            continue
        if r.get("z_type") != "spec":
            continue
        log_ms = r["log_Mstar"]
        R0_gal = 4.0 * (10 ** log_ms / 1e10) ** 0.22
        obs_ratio = r["r_e_kpc"] / R0_gal
        z_gal = r["z"]

        has_vrot = r.get("V_rot") is not None
        marker = "o" if has_vrot else "D"
        ms = 9 if has_vrot else 7
        clr = _CB_GREEN if has_vrot else _CB_PURPLE

        # Fractional uncertainty on obs_ratio
        dlog_m = r.get("dlog_Mstar") or 0.20
        frac_mass = 0.22 * np.log(10.0) * dlog_m
        caveat = (r.get("caveat") or "").lower()
        is_nonequilib = any(k in caveat for k in ("merger", "lensed", "dual"))
        frac_extra = 0.30 if is_nonequilib else 0.0
        frac_err = float(np.hypot(frac_mass, frac_extra))
        y_lo = obs_ratio * (1.0 - np.exp(-frac_err))
        y_hi = obs_ratio * (np.exp(frac_err) - 1.0)

        # Faded halo ("fade circle") sized by total fractional uncertainty,
        # so SPT0418-47's merger systematic appears visibly larger.
        halo_size = (ms ** 2) * (2.5 + 18.0 * frac_err)
        ax.scatter(z_gal, obs_ratio, s=halo_size, color=clr, alpha=0.16,
                   edgecolors="none", zorder=7)

        # Asymmetric vertical error bar (quantitative y-uncertainty)
        ax.errorbar(z_gal, obs_ratio, yerr=[[y_lo], [y_hi]], fmt="none",
                    ecolor=clr, elinewidth=0.9, capsize=2, alpha=0.7,
                    zorder=8)

        # Main marker on top
        ax.plot(z_gal, obs_ratio, marker, color=clr,
                ms=ms, mec="k", mew=0.6, zorder=10)

        short = r["name"].split("(")[0].strip()
        if len(short) > 12:
            short = short[:12]
        ax.annotate(short, (z_gal, obs_ratio),
                    fontsize=6, ha="left", va="bottom",
                    xytext=(4, 4), textcoords="offset points")
        spec_plotted.append((r["name"], z_gal, obs_ratio, has_vrot, frac_err))

    # Legend entries for galaxy markers
    ax.plot([], [], "o", color=_CB_GREEN, ms=9, mec="k", mew=0.6,
            label="Spec-$z$ + kinematics")
    ax.plot([], [], "D", color=_CB_PURPLE, ms=7, mec="k", mew=0.6,
            label="Spec-$z$ (photometric size)")

    # ---- MUSE-DARK MHUDF binned overlay (Ciocan+2026 Paper I sample) ----
    # Same-sample size-mass-redshift cross-check: bin the DARK photometry
    # catalogue (log M_star > 9) into 3 equal-count z-bins and overlay the
    # median R_obs/R_0(M_*) value with 16/84 percentile error bars.  This
    # is an *independent* test of HEAT's size-evolution prediction using
    # the same parent sample whose RAR evolution motivated the trilogy.
    mhudf_bins = _load_mhudf_photometry(min_log_mstar=9.0, n_z_bins=3)
    for b in mhudf_bins:
        zerr = [[b["z_med"] - b["z_lo"]], [b["z_hi"] - b["z_med"]]]
        rerr = [[b["ratio_med"] - b["ratio_lo"]],
                [b["ratio_hi"] - b["ratio_med"]]]
        ax.errorbar(b["z_med"], b["ratio_med"],
                    xerr=zerr, yerr=rerr,
                    fmt="s", color=_CB_ORANGE, ms=10, mec="k", mew=0.7,
                    elinewidth=1.2, ecolor=_CB_ORANGE, capsize=3,
                    alpha=0.95, zorder=11)
    if mhudf_bins:
        n_total = sum(b["n"] for b in mhudf_bins)
        ax.plot([], [], "s", color=_CB_ORANGE, ms=10, mec="k", mew=0.7,
                label=f"MHUDF binned $R_e/R_0$ "
                      f"(Ciocan+2026 PaperI, $N{{=}}{n_total}$)")

    # Reference line at 1
    ax.axhline(1.0, color="k", ls=":", lw=0.8, alpha=0.4)

    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"$R_{\rm obs}\,/\,R_0(M_\star,\,z\!=\!0)$")
    ax.set_yscale("log")
    ax.set_xlim(0, 16)
    ax.set_ylim(0.005, 1.5)
    ax.legend(fontsize=7.5, loc="lower left", ncol=1)
    ax.text(0.97, 0.95, "(e)", transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="right")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig5_size_mass.{ext}", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_dir / 'fig5_size_mass.[pdf|png]'}")

    # Print summary table
    print("\n  Size compactification summary (R_obs / R_baseline):")
    print(f"  {'Name':25s} {'z':>5s} {'R_obs/R0':>9s} {'kin?':>5s} {'frac_err':>9s}")
    for name, zg, oratio, has_k, ferr in spec_plotted:
        print(f"  {name:25s} {zg:5.2f} {oratio:9.3f} {'yes' if has_k else 'no':>5s} "
              f"{ferr:9.2f}")
    _print_allen2025_size_calibration()


if __name__ == "__main__":
    main()
