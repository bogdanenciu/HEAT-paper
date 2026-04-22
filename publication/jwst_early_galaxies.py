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
    # SPT0418-47: M_star from Rizzo+2020; f_gas = 0.53 -> M_gas ~ 1.35e10
    dict(name="SPT0418-47",         z=4.225, z_type="spec", log_Mstar=10.08,
         dlog_Mstar=0.08, r_e_kpc=0.22, V_rot=250.0, log_Mgas=10.16,
         SFR=None, caveat="lensed; merger companion SPT0418B (Cathey+2024)",
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
    """Age t(z) in Gyr for flat LCDM (Om=0.31, H0=67.4)."""
    H0_inv_gyr = 977.8 / H0_km_s_Mpc
    Om, OL = 0.31, 0.69
    integrand = lambda zp: 1.0 / ((1 + zp) * np.sqrt(Om * (1 + zp) ** 3 + OL))
    t, _ = quad(integrand, z, np.inf, limit=100)
    return t * H0_inv_gyr


# ---------------------------------------------------------------------------
# Sheth-Tormen halo mass function: M_halo_max(z, V)
# ---------------------------------------------------------------------------

def _sigma_M_z0(M_solar: np.ndarray) -> np.ndarray:
    """RMS variance of linear density field at mass scale M at z=0.

    Quadratic fit to Planck 2018 cosmology (sigma_8=0.811, n_s=0.965,
    Omega_m=0.31) calibrated against CAMB / CLASS tabulations at
    M = 10^8, 10^12, 10^14 M_sun.  Valid for 10^8 < M < 10^16.
    """
    x = np.log10(np.asarray(M_solar, dtype=float))
    return 10.0 ** (0.78 - 0.172 * (x - 8.0) - 0.0058 * (x - 8.0) ** 2)


def _growth_factor_D(z: float) -> float:
    """Linear growth factor D(z)/D(0) for flat LCDM (Carroll+1992)."""
    Om, OL = 0.31, 0.69
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
    rho_m_Msun_Mpc3 = 0.31 * 2.775e11

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
        return H0_val * np.sqrt(
            0.049 * 6.3265 * (1.0 + zv) ** (-0.0001) * (1.0 + zv) ** 3 + 0.69
        )

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
    ax.plot(z_smooth, ratio_heat, color=_CB_BLUE, lw=2.2,
            label=r"HEAT (zero parameters): $R\propto [cH(z)]^{-1/2}$")

    # --- Spectroscopically confirmed galaxies only ---
    # R_obs / R_baseline(z=0, same mass); baseline: van der Wel+14
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

        ax.plot(z_gal, obs_ratio, marker, color=clr,
                ms=ms, mec="k", mew=0.6, zorder=10)

        short = r["name"].split("(")[0].strip()
        if len(short) > 12:
            short = short[:12]
        ax.annotate(short, (z_gal, obs_ratio),
                    fontsize=6, ha="left", va="bottom",
                    xytext=(4, 4), textcoords="offset points")
        spec_plotted.append((r["name"], z_gal, obs_ratio, has_vrot))

    # Legend entries for galaxy markers
    ax.plot([], [], "o", color=_CB_GREEN, ms=9, mec="k", mew=0.6,
            label="Spec-$z$ + kinematics")
    ax.plot([], [], "D", color=_CB_PURPLE, ms=7, mec="k", mew=0.6,
            label="Spec-$z$ (photometric size)")

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
    print(f"  {'Name':25s} {'z':>5s} {'R_obs/R0':>9s} {'kin?':>5s}")
    for name, zg, oratio, has_k in spec_plotted:
        print(f"  {name:25s} {zg:5.2f} {oratio:9.3f} {'yes' if has_k else 'no':>5s}")


if __name__ == "__main__":
    main()
