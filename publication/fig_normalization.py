"""
Two-panel figure for the HEAT letter:

  Panel A:  Absolute size evolution R_eff(z)/R_0 vs redshift.
            Compares the zero-parameter HEAT curve
              R_HEAT(z)/R_0 = [H(z)/H_0]^{-1/2}
            against the matter-era power law (1+z)^{-3/4}.
            The two curves diverge by the closed-form factor
              R(z) = [Omega_m(z)/Omega_m(0)]^{1/4}.

  Panel B:  Normalisation ratio  R(z) = R_HEAT / R_powerlaw.
            Shows the closed-form curve, its z->inf asymptote
            Omega_m^{-1/4} = 1.335 (Planck 2018, Om=0.315),
            and the baseline (matter-only, R=1).
            ALMA kinematic-sample mean at z ~ 4.5 is overlaid as
            illustrative current data; a shaded 20% band brackets
            the local-anchor systematic.

Output:  heat_output/jwst_early_galaxies/fig6_normalization.[pdf|png]
"""
from __future__ import annotations

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[1]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import path_setup  # noqa: E402

import numpy as np  # noqa: E402

from theory.heat_cosmology import (  # noqa: E402
    a0_hie,
    hubble_parameter,  # noqa: F401  (kept for consumers of this module)
)

# Planck 2018 (TT,TE,EE+lowE+lensing; Table 2) matter density fraction.
# This is the value quoted in paper_heat_letter.tex (Om_m = 0.315),
# giving the asymptote R_inf = Om_m^{-1/4} = 1.335.  heat_cosmology.py
# carries an effective Om_m = Om_b * F0 = 0.310; for the closed-form
# ratio R(z) in this figure we adopt the exact Planck reference so the
# annotated value matches the text.
Om_m = 0.315
Om_L = 0.685


# Colour-blind palette (matches jwst_early_galaxies.py)
_CB_BLUE = "#0173b2"
_CB_ORANGE = "#de8f05"
_CB_GREEN = "#029e73"
_CB_RED = "#cc3311"
_CB_GREY = "#999999"


def Omega_m_of_z(z):
    """Flat LCDM matter density parameter at redshift z."""
    z = np.asarray(z, dtype=float)
    a3 = (1.0 + z) ** 3
    return Om_m * a3 / (Om_m * a3 + Om_L)


def R_closed_form(z):
    """Zero-parameter HEAT normalisation ratio R(z) = [Om_m(z)/Om_m(0)]^{1/4}."""
    return (Omega_m_of_z(z) / Om_m) ** 0.25


def R_heat_over_R0(z, a0_0):
    """Absolute HEAT size evolution: R/R_0 = [a0(z)/a0(0)]^{-1/2}."""
    z = np.asarray(z, dtype=float)
    a0_z = np.array([float(a0_hie(zi)) for zi in z.flat]).reshape(z.shape)
    return (a0_z / a0_0) ** (-0.5)


def R_power_law(z, exponent=-0.75):
    """Matter-era empirical power law: (1+z)^{-3/4}."""
    return (1.0 + np.asarray(z, dtype=float)) ** exponent


def _alma_sample_point():
    """
    ALMA kinematic-sample mean R_obs/R_0 at z ~ 4.5, excluding lensed SPT0418-47.

    Uses the three non-lensed Roman-Oliveira+2023/2024 galaxies:
      CRISTAL-22    (z=4.53, R_e=2.00 kpc, logM=10.60)
      DC-881725     (z=4.56, R_e=1.50 kpc, logM=10.30)
      SGP38326      (z=4.42, R_e=3.00 kpc, logM=11.00)

    R_0(M*, z=0) = 4.0 * (M/1e10)^0.22 kpc  (Shen+03 SDSS local baseline).
    """
    sample = [
        (4.53, 2.00, 10.60),
        (4.56, 1.50, 10.30),
        (4.42, 3.00, 11.00),
    ]
    z_mean = np.mean([s[0] for s in sample])
    ratios = [re / (4.0 * (10.0 ** (lm - 10.0)) ** 0.22) for (_, re, lm) in sample]
    R_mean = float(np.mean(ratios))
    R_err = float(np.std(ratios, ddof=1) / np.sqrt(len(ratios)))
    return z_mean, R_mean, R_err


def build_figure(out_dir: Path):
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    z_grid = np.linspace(0.0, 5.0, 401)
    a0_0 = float(a0_hie(0.0))

    R_heat = R_heat_over_R0(z_grid, a0_0)
    R_pl = R_power_law(z_grid)
    R_ratio = R_closed_form(z_grid)
    R_inf = float(Om_m ** (-0.25))

    z_alma, R_alma, R_alma_err = _alma_sample_point()
    R_pl_at_alma = float(R_power_law(z_alma))
    alma_ratio = R_alma / R_pl_at_alma
    alma_ratio_err = R_alma_err / R_pl_at_alma

    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(7.0, 7.4), sharex=True,
        gridspec_kw=dict(height_ratios=[1.15, 1.0], hspace=0.08),
    )

    # --- Panel A: absolute size evolution ------------------------------------
    axA.fill_between(z_grid, 0.8 * R_heat, 1.2 * R_heat,
                     color=_CB_BLUE, alpha=0.12,
                     label=r"HEAT $\pm 20\%$ (local anchor)")
    axA.plot(z_grid, R_heat, color=_CB_BLUE, lw=2.3,
             label=r"HEAT: $R/R_0 = [H(z)/H_0]^{-1/2}$")
    axA.plot(z_grid, R_pl, color=_CB_GREY, lw=1.8, ls="--",
             label=r"Matter-era power law: $(1+z)^{-3/4}$")

    axA.errorbar([z_alma], [R_alma], yerr=[R_alma_err],
                 fmt="o", color=_CB_ORANGE, ms=9, mec="k", mew=0.7,
                 capsize=3, zorder=10,
                 label=r"ALMA kinematic mean ($z\!\approx\!4.5$, $N{=}3$)")
    axA.annotate(r"Roman-Oliveira+23/24",
                 xy=(z_alma, R_alma), xytext=(z_alma - 0.05, R_alma - 0.09),
                 fontsize=8, color=_CB_ORANGE, ha="right", va="top")

    axA.axhline(1.0, color="k", ls=":", lw=0.6, alpha=0.4)
    axA.set_ylabel(r"$R_{\rm eff}(z)\,/\,R_0$")
    axA.set_xlim(0, 5.0)
    axA.set_ylim(0.15, 1.1)
    axA.legend(loc="upper right", framealpha=0.9)
    axA.text(0.02, 0.05, "(a) Absolute evolution", transform=axA.transAxes,
             fontsize=11, fontweight="bold", va="bottom", ha="left")

    # --- Panel B: normalisation ratio ----------------------------------------
    axB.axhspan(1.0 - 0.25, 1.0 + 0.25, color=_CB_GREY, alpha=0.10,
                label=r"Current data systematic ($\pm 25\%$)")
    axB.plot(z_grid, R_ratio, color=_CB_BLUE, lw=2.3,
             label=r"HEAT: $\mathcal{R}(z)=[\Omega_m(z)/\Omega_m(0)]^{1/4}$")
    axB.axhline(1.0, color=_CB_GREY, ls="--", lw=1.4,
                label=r"Matter-era baseline ($\mathcal{R}=1$)")
    axB.axhline(R_inf, color=_CB_BLUE, ls=":", lw=1.2, alpha=0.7)
    axB.text(4.85, R_inf + 0.01,
             rf"$\mathcal{{R}}_\infty=\Omega_m^{{-1/4}}={R_inf:.3f}$",
             color=_CB_BLUE, fontsize=9, ha="right", va="bottom")

    axB.errorbar([z_alma], [alma_ratio], yerr=[alma_ratio_err],
                 fmt="o", color=_CB_ORANGE, ms=9, mec="k", mew=0.7,
                 capsize=3, zorder=10,
                 label=r"ALMA ratio $R_{\rm obs}/(1+z)^{-3/4}$")

    axB.text(2.5, 1.48,
             "Euclid/Roman target: percent-level $\\mathcal{R}(z)$",
             fontsize=8.5, color=_CB_GREEN, style="italic",
             ha="center")

    axB.set_xlabel(r"Redshift $z$")
    axB.set_ylabel(r"$\mathcal{R}(z)\equiv R_{\rm eff}(z)/[R_0(1+z)^{-3/4}]$")
    axB.set_xlim(0, 5.0)
    axB.set_ylim(0.75, 1.55)
    axB.legend(loc="lower right", framealpha=0.9)
    axB.text(0.02, 0.05, "(b) Normalisation ratio", transform=axB.transAxes,
             fontsize=11, fontweight="bold", va="bottom", ha="left")

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig6_normalization.{ext}", dpi=200)
    plt.close(fig)

    print(f"Saved: {out_dir / 'fig6_normalization.[pdf|png]'}")
    print()
    print("Sanity check values:")
    print(f"  Omega_m          = {Om_m}")
    print(f"  Omega_m^(-1/4)   = {R_inf:.4f}")
    for zi in [0.25, 0.9, 1.5, 2.0, 3.0, 4.5]:
        r = float(R_closed_form(zi))
        print(f"  R(z={zi:4.2f})     = {r:.4f}")
    print(f"  ALMA mean z      = {z_alma:.3f}")
    print(f"  ALMA R_obs/R_0   = {R_alma:.3f} +/- {R_alma_err:.3f}")
    print(f"  ALMA / powerlaw  = {alma_ratio:.3f} +/- {alma_ratio_err:.3f}")
    print(f"  HEAT R(z_alma)   = {float(R_closed_form(z_alma)):.3f}")


def main():
    out_dir = _repo / "heat_output" / "jwst_early_galaxies"
    build_figure(out_dir)


if __name__ == "__main__":
    main()
