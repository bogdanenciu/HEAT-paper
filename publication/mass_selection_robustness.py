"""
Mass-selection robustness figure for the HEAT letter.

Demonstrates that an evolving baryonic-mass selection at fixed
observed stellar mass cannot mimic the closed-form HEAT
normalisation curve

    R_HEAT(z) = [Omega_m(z) / Omega_m(0)]^{1/4}.

Setup.  Suppose a static-MOND universe with constant a0 in which
the underlying baryonic mass at fixed observed M_star evolves as

    M_b(z) = M_b(0) * (1 + z)^alpha,   alpha >= 0,

reflecting evolving gas fractions / stellar-to-baryonic mass
ratios.  The deep-MOND equilibrium R \\propto (G M_b / a0)^{1/2}
then yields

    R_mimic(z) / R_0 / (1+z)^{-3/4}  =  (1+z)^{alpha/2}

at fixed observed M_star.  We plot this family for
alpha in {0, 0.1, 0.2, 0.3, 0.4, 0.5} alongside the
HEAT closed-form curve, illustrating that:

  (i)  the HEAT curve saturates at Omega_m^{-1/4} ~ 1.34 with a
       Lambda-sourced break near z ~ 0.3, while the power-law
       family is monotone, unsaturated, and Lambda-independent;
  (ii) no single alpha reproduces both the location of the
       break and the asymptotic value;
  (iii) the alpha that matches the asymptote at z=4 (alpha ~ 0.36)
       under-predicts the rise at z ~ 0.5 by ~10%.

Output:  heat_output/jwst_early_galaxies/fig8_mass_selection.[pdf|png]
"""
from __future__ import annotations

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[1]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import path_setup  # noqa: E402

import numpy as np  # noqa: E402

from theory.heat_cosmology import Om_L, Om_m  # noqa: E402

_CB_BLUE = "#0173b2"
_CB_ORANGE = "#de8f05"
_CB_GREEN = "#029e73"
_CB_RED = "#cc3311"
_CB_GREY = "#999999"
_CB_PURPLE = "#7e2f8e"


def Omega_m_of_z(z):
    """Flat LCDM matter density parameter at redshift z."""
    z = np.asarray(z, dtype=float)
    a3 = (1.0 + z) ** 3
    return Om_m * a3 / (Om_m * a3 + Om_L)


def R_heat(z):
    """HEAT closed-form normalisation R(z) = [Om_m(z)/Om_m(0)]^{1/4}."""
    return (Omega_m_of_z(z) / Om_m) ** 0.25


def R_mimic(z, alpha):
    """Power-law mass-evolution mimic: (1+z)^{alpha/2} at fixed M_star."""
    z = np.asarray(z, dtype=float)
    return (1.0 + z) ** (0.5 * alpha)


def alpha_match_asymptote(z_anchor=4.0):
    """alpha such that (1+z_anchor)^{alpha/2} = Omega_m^{-1/4}."""
    target = Om_m ** (-0.25)
    return 2.0 * np.log(target) / np.log(1.0 + z_anchor)


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
    R_inf = float(Om_m ** (-0.25))
    R_h = R_heat(z_grid)

    alpha_match = alpha_match_asymptote(z_anchor=4.0)

    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    cmap = plt.get_cmap("viridis")
    norm_a = plt.Normalize(vmin=0.0, vmax=0.55)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    for a in alphas:
        ax.plot(z_grid, R_mimic(z_grid, a), color=cmap(norm_a(a)), lw=1.4,
                ls="--", alpha=0.85,
                label=rf"$\alpha={a:.1f}$")

    ax.plot(z_grid, R_mimic(z_grid, alpha_match),
            color=_CB_RED, lw=2.0, ls=(0, (3, 1, 1, 1)),
            label=rf"asymptote-tuned $\alpha={alpha_match:.2f}$ (still wrong shape)")

    ax.plot(z_grid, R_h, color=_CB_BLUE, lw=2.6,
            label=r"HEAT: $\mathcal{R}(z)=[\Omega_m(z)/\Omega_m(0)]^{1/4}$")

    ax.axhline(R_inf, color=_CB_BLUE, ls=":", lw=1.0, alpha=0.7)
    ax.text(4.85, R_inf + 0.012,
            rf"$\mathcal{{R}}_\infty=\Omega_m^{{-1/4}}={R_inf:.3f}$",
            color=_CB_BLUE, fontsize=9, ha="right", va="bottom")
    ax.axhline(1.0, color="k", ls=":", lw=0.6, alpha=0.4)

    z_break = 0.30
    ax.axvline(z_break, color=_CB_GREY, ls=":", lw=0.7, alpha=0.5)
    ax.text(z_break + 0.05, 1.42,
            rf"$z_{{\rm eq,\Lambda}}\!\approx\!{z_break:.2f}$",
            fontsize=8.5, color=_CB_GREY, ha="left", va="top")

    ax.set_xlabel(r"Redshift $z$")
    ax.set_ylabel(r"$R_{\rm eff}(z)/[R_{0}(1+z)^{-3/4}]$")
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(0.92, 1.55)

    leg = ax.legend(loc="upper left", framealpha=0.9, ncol=2, fontsize=8.5,
                    title=r"static-MOND mimics: $M_b(z)\!=\!M_b(0)(1+z)^{\alpha}$")
    leg.get_title().set_fontsize(8.5)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig8_mass_selection.{ext}", dpi=200)
    plt.close(fig)

    print(f"Saved: {out_dir / 'fig8_mass_selection.[pdf|png]'}")
    print()
    print("Sanity check values:")
    print(f"  Omega_m                  = {Om_m}")
    print(f"  R_inf = Omega_m^(-1/4)   = {R_inf:.4f}")
    print(f"  alpha matching asymptote at z=4: {alpha_match:.3f}")
    print()
    z_probe = [0.3, 0.5, 1.0, 2.0, 3.0, 4.0]
    header = f"  {'z':>5}  {'R_HEAT(z)':>10}"
    for a in alphas:
        header += f"  a={a:.1f}".rjust(7)
    header += f"  a={alpha_match:.2f}".rjust(7)
    print(header)
    for zi in z_probe:
        rh = float(R_heat(zi))
        row = f"  {zi:5.2f}  {rh:10.4f}"
        for a in alphas:
            row += f"  {float(R_mimic(zi, a)):7.4f}"
        row += f"  {float(R_mimic(zi, alpha_match)):7.4f}"
        print(row)


def main():
    out_dir = _repo / "heat_output" / "jwst_early_galaxies"
    build_figure(out_dir)


if __name__ == "__main__":
    main()
