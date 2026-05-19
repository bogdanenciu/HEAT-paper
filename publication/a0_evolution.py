"""
Figure 9 (a0(z) and Sigma_DM(z) evolution) for HEAT Letter.

Tests the HEAT a0(z) propto H(z) shape directly against the
MUSE-DARK trilogy, with the c H(z) / (2 pi) normalisation retained
as a secondary anchor reference:
  Paper I  (Ciocan+2026, arXiv:2506.19721) -- halo central density evolution
  Paper II (Jeanneau+2026, arXiv:2603.28856) -- lensed bTFR zero-evolution result (treated in Fig 7)
  Paper III (Ciocan+2026, arXiv:2604.22613) -- intermediate-z RAR a0(z) evolution

Clean N=7 cohort:
  z=0.00 SPARC RAR-fit median        (McGaugh+2016)
  z=0.00 SPARC HMC joint inference   (Desmond+2023, MLS IF -- matches Paper III)
  z=0.04 MIGHTEE+SPARC RAR combined  (Varasteanu+2025, RAR-framework value)
  z=0.55-1.30 four MUSE-DARK bins    (Ciocan+2026 Paper III, Fig.3)

Two-panel layout:
  (a) a0(z) -- TWO HEAT bands shown as a systematic uncertainty envelope:
       upper solid (raw cohort, beta = 1.251 +/- 0.037) and lower solid
       (+0.11 dex M/L correction applied only to the four Ciocan bins,
       beta = 1.032 +/- 0.031), with a shaded blue band between them.
       The bare HEAT prediction beta = 1 (i.e. a0(z) = c H(z) / (2 pi))
       is shown as a dashed blue reference and lies inside the band.
       Constant-a0 MOND (1.20 e-10) is shown as a dashed grey reference;
       it is excluded at >9 sigma everywhere in the band.  The three
       Ciocan linear-in-z fits (DC14 uniform, best-fit per-galaxy,
       MOND framework) are shown as dotted lines.
  (b) Delta log Sigma_DM vs log(1+z) -- HEAT-implied scaling
       Delta log Sigma_DM = log10[a0(z)/a0(0)] = log10[H(z)/H_0].
       Paper I MHUDF point at z~0.85 is overlaid.

Outputs:
  heat_output/jwst_early_galaxies/fig9_a0_evolution.pdf (and .png)
  heat_output/jwst_early_galaxies/a0_evolution_stats.txt
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

try:
    import path_setup  # noqa: F401  # ensures repo root on sys.path
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from theory.heat_cosmology import a0_hie, hubble_parameter, H0, c
from theory.heat_output import HEAT_DATA_ROOT, JWST_EARLY, ensure_dir


_CB_BLUE = "#0072B2"
_CB_ORANGE = "#D55E00"
_CB_GREEN = "#009E73"
_CB_PURPLE = "#882255"
_CB_GREY = "#999999"
_CB_RED = "#CC3311"

# Uniform stellar M/L offset applied only to Ciocan high-z bins to
# probe the (beta, n) degeneracy reported in Sec 4.5; same value as
# publication/a0_ml_degeneracy.py.
ML_DEX = 0.11

# Clean N=7 cohort: rows in heat_data/ciocan2026_a0z.csv to include.
# section + label uniquely identify each row.
COHORT_KEYS = {
    (1, "SPARC z=0"),                              # McGaugh+2016
    (5, "Desmond2023 SPARC"),                      # Desmond+2023 HMC
    (5, "Varasteanu2025 MIGHTEE+SPARC combined"),  # combined RAR fit
    (1, "Ciocan bin1"),
    (1, "Ciocan bin2"),
    (1, "Ciocan bin3"),
    (1, "Ciocan bin4"),
}

# Rows from section 1 we DROP from the cohort: the Varasteanu MIGHTEE-only
# row (a0=1.69) is superseded by the combined RAR fit (a0=1.32) in section 5.
DROPPED_LEGACY = {(1, "Varasteanu z<0.08")}


def _setup_fig_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def _load_csv():
    """Read heat_data/ciocan2026_a0z.csv into structured rows."""
    path = HEAT_DATA_ROOT / "ciocan2026_a0z.csv"
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(
            (line for line in f if not line.lstrip().startswith("#"))
        )
        for r in reader:
            r["z"] = float(r["z"])
            r["a0_1e_minus10"] = float(r["a0_1e_minus10"])
            r["sigma_lo"] = float(r["sigma_lo"])
            r["sigma_hi"] = float(r["sigma_hi"])
            r["section"] = int(r["section"])
            rows.append(r)
    return rows


# Ciocan+2026 Paper III linear-fit triplets (a0(z) = a0(0) + a1 * z)
# values in 10^-10 m/s^2
CIOCAN_LINEAR_FITS = {
    "DC14 uniform": dict(a0=1.00, sa0=0.04, a1=1.59, sa1=0.11,
                         color=_CB_PURPLE, label="Ciocan DC14 uniform"),
    "DC14 per-galaxy": dict(a0=1.05, sa0=0.05, a1=1.63, sa1=0.12,
                            color=_CB_GREEN, label="Ciocan best-fit per-galaxy"),
    "MOND": dict(a0=1.03, sa0=0.05, a1=1.20, sa1=0.10,
                 color=_CB_ORANGE, label="Ciocan MOND framework"),
}


def heat_a0_at(z):
    """HEAT prediction in 10^-10 m/s^2 (bare, beta = 1)."""
    return float(a0_hie(z)) / 1e-10


def linear_a0(z, a0, a1):
    return a0 + a1 * z


def _is_ciocan_bin(label: str) -> bool:
    return label.startswith("Ciocan bin")


def _apply_ml(rows: list, ml_dex: float) -> tuple:
    """Return (a0, sigma) arrays after scaling Ciocan-bin entries by 10^(-ml_dex)."""
    scales = np.array([10.0 ** (-ml_dex) if _is_ciocan_bin(r["label"]) else 1.0
                       for r in rows])
    a0 = np.array([r["a0_1e_minus10"] for r in rows]) * scales
    sig = np.array([(r["sigma_lo"] + r["sigma_hi"]) / 2.0
                    for r in rows]) * scales
    return a0, sig


def _format_stats(rows):
    """Compute the clean-cohort (N=7) chi^2 ladder, raw and M/L-corrected."""
    cohort = [r for r in rows
              if (r["section"], r["label"]) in COHORT_KEYS]
    # Sort by redshift for stable display
    cohort.sort(key=lambda r: r["z"])
    zs = np.array([r["z"] for r in cohort])
    labels = [r["label"] for r in cohort]
    N = len(cohort)

    heat_pred = np.array([heat_a0_at(z) for z in zs])

    def _ladder(ml_dex: float) -> dict:
        a0, sig = _apply_ml(cohort, ml_dex)
        w = 1.0 / sig ** 2
        # free-K best fit
        y = heat_pred * (2.0 * np.pi)
        K_fit = float(np.sum(w * y * a0) / np.sum(w * y * y))
        beta = K_fit / (1.0 / (2.0 * np.pi))
        sigma_beta = float(1.0 / np.sqrt(np.sum(w * heat_pred ** 2)))
        chi2_freeK = float(np.sum(((a0 - beta * heat_pred) / sig) ** 2))
        chi2_heat = float(np.sum(((a0 - heat_pred) / sig) ** 2))
        chi2_mond = float(np.sum(((a0 - 1.20) / sig) ** 2))
        lin = {name: float(np.sum(
            ((a0 - np.array([linear_a0(zi, fit["a0"], fit["a1"]) for zi in zs]))
             / sig) ** 2))
            for name, fit in CIOCAN_LINEAR_FITS.items()}
        return dict(a0=a0, sig=sig, K_fit=K_fit, beta=beta,
                    sigma_beta=sigma_beta,
                    chi2_freeK=chi2_freeK, chi2_heat=chi2_heat,
                    chi2_mond=chi2_mond, lin_chi2=lin)

    raw = _ladder(0.0)
    mlc = _ladder(ML_DEX)

    K_heat = 1.0 / (2.0 * np.pi)
    lines = []
    lines.append("=" * 78)
    lines.append(f"HEAT a0(z) test against clean N={N} cohort")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Cohort (sorted by z):")
    for i, (lab, z) in enumerate(zip(labels, zs)):
        lines.append(f"  {lab:38s} z = {z:5.2f}   a0_raw = {raw['a0'][i]:.3f} +/- {raw['sig'][i]:.3f}")
    lines.append("")
    lines.append(f"K_HEAT = 1 / (2 pi) = {K_heat:.5f}")
    lines.append(f"Raw free-K fit:         beta = {raw['beta']:.3f} +/- {raw['sigma_beta']:.3f}    K = {raw['K_fit']:.5f}")
    lines.append(f"+{ML_DEX:.2f} dex M/L fit:     beta = {mlc['beta']:.3f} +/- {mlc['sigma_beta']:.3f}    K = {mlc['K_fit']:.5f}")
    lines.append("")
    lines.append(f"  chi^2 ladder (clean N={N} cohort, raw and +{ML_DEX:.2f} dex M/L applied")
    lines.append("                only to Ciocan high-z bins):")
    lines.append("")
    lines.append(f"  {'model':38s}  {'raw N=7':>10s}   {'+0.11 dex M/L':>15s}")
    lines.append("  " + "-" * 70)
    lines.append(f"  {'HEAT shape, free K':38s}  "
                 f"{raw['chi2_freeK']:10.2f}   {mlc['chi2_freeK']:15.2f}")
    lines.append(f"  {'HEAT, fixed K = 1/(2 pi)':38s}  "
                 f"{raw['chi2_heat']:10.2f}   {mlc['chi2_heat']:15.2f}")
    lines.append(f"  {'Constant-a0 MOND (1.20e-10)':38s}  "
                 f"{raw['chi2_mond']:10.2f}   {mlc['chi2_mond']:15.2f}")
    for name in CIOCAN_LINEAR_FITS:
        lines.append(f"  {('Ciocan linear: ' + name):38s}  "
                     f"{raw['lin_chi2'][name]:10.2f}   {mlc['lin_chi2'][name]:15.2f}")
    lines.append("")
    lines.append("Note: the +0.11 dex M/L correction reduces beta from 1.25 +/- 0.04")
    lines.append("to 1.03 +/- 0.03; the bare HEAT prediction (beta = 1, no free")
    lines.append("parameters) is then recovered to 1 sigma.  See companion")
    lines.append("a0_ml_degeneracy.txt for the full (beta, n) degeneracy region")
    lines.append("and the M/L scan.")
    lines.append("")
    lines.append("Cross-framework comparison at z=0 (intercept of Ciocan linear fits):")
    lines.append(f"  HEAT a0(0) = c H_0 / (2 pi)              = {heat_a0_at(0.0):.3f} e-10 m/s^2")
    for name, fit in CIOCAN_LINEAR_FITS.items():
        a00 = fit["a0"]
        ratio = heat_a0_at(0.0) / a00
        lines.append(f"  Ciocan {name:24s}  a0(0)={a00:.2f}  -> HEAT/Ciocan = {ratio:.3f}")
    lines.append("")
    lines.append("The Ciocan MOND-framework intercept a0(0) = 1.03 +/- 0.05 agrees")
    lines.append("with the bare HEAT prediction (1.04) to ~1%.  Both invert the same")
    lines.append("MOND interpolation function on the same kinematics, so this is the")
    lines.append("apples-to-apples z=0 anchor for any Ciocan-framework analysis.")
    lines.append("=" * 78)

    return "\n".join(lines), dict(
        cohort=cohort, zs=zs, labels=labels, heat_pred=heat_pred,
        raw=raw, mlc=mlc, K_heat=K_heat, N=N,
    )


def _plot(rows, stats, out_dir):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple
    _setup_fig_style()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.4),
                             gridspec_kw=dict(width_ratios=[1.35, 1.0]))
    ax_a0, ax_sig = axes

    # ----- Panel (a): a0(z) with dual-band uncertainty envelope -----
    z_smooth = np.linspace(0.0, 1.7, 220)
    a0_curve = np.array([heat_a0_at(z) for z in z_smooth])

    beta_raw = stats["raw"]["beta"]
    beta_mlc = stats["mlc"]["beta"]
    a0_upper = a0_curve * beta_raw    # raw cohort best fit
    a0_lower = a0_curve * beta_mlc    # +0.11 dex M/L corrected

    # Shaded systematic envelope between raw and ML-corrected best fits
    ax_a0.fill_between(z_smooth, a0_lower, a0_upper,
                       color=_CB_BLUE, alpha=0.20, linewidth=0)
    # Upper edge: raw cohort best fit (solid)
    ax_a0.plot(z_smooth, a0_upper, color=_CB_BLUE, lw=2.0, alpha=0.95)
    # Lower edge: ML-corrected best fit (solid)
    ax_a0.plot(z_smooth, a0_lower, color=_CB_BLUE, lw=2.0, alpha=0.95,
               ls=(0, (4, 1.5)))
    # Bare HEAT prediction (beta = 1) -- dashed reference, lies inside band
    ax_a0.plot(z_smooth, a0_curve, color=_CB_BLUE, lw=1.4, ls=":", alpha=0.9)

    # Constant-a0 MOND baseline
    ax_a0.axhline(1.20, color=_CB_GREY, ls="--", lw=1.3)

    # Ciocan linear fits
    ciocan_line_handles = []
    for name, fit in CIOCAN_LINEAR_FITS.items():
        zz = np.linspace(0.0, 1.7, 50)
        ln, = ax_a0.plot(zz, fit["a0"] + fit["a1"] * zz,
                         color=fit["color"], ls=":", lw=1.5, alpha=0.85)
        ciocan_line_handles.append(ln)

    # Plot the clean cohort data points with the markers we describe in
    # the LaTeX caption: square = McGaugh, diamond = Desmond, plus =
    # Varasteanu combined, circles = Ciocan bins.
    for r in stats["cohort"]:
        lab = r["label"]
        if "McGaugh" in lab or lab == "SPARC z=0":
            mrk, clr, ms = "s", "k", 9
        elif "Desmond" in lab:
            mrk, clr, ms = "D", "k", 8
        elif "Varasteanu" in lab:
            mrk, clr, ms = "P", _CB_RED, 10
        else:
            mrk, clr, ms = "o", _CB_PURPLE, 8
        ax_a0.errorbar(r["z"], r["a0_1e_minus10"],
                       yerr=[[r["sigma_lo"]], [r["sigma_hi"]]],
                       fmt=mrk, color=clr, ms=ms, mec="k", mew=0.6,
                       elinewidth=1.0, capsize=2, zorder=11)

    # Section-2: Ciocan global (z=1) across three frameworks
    section2 = [r for r in rows if r["section"] == 2]
    for r in section2:
        if "MOND" in r["label"]:
            mrk, clr = "*", _CB_ORANGE
        elif "per-galaxy" in r["label"]:
            mrk, clr = "v", _CB_GREEN
        else:
            mrk, clr = "^", _CB_PURPLE
        ax_a0.errorbar(r["z"], r["a0_1e_minus10"],
                       yerr=[[r["sigma_lo"]], [r["sigma_hi"]]],
                       fmt=mrk, color=clr, ms=12, mec="k", mew=0.7,
                       elinewidth=1.0, capsize=2, alpha=0.85, zorder=12)

    # ---- chi^2 inset (upper-left): raw vs ML-corrected, N=7 cohort ----
    raw = stats["raw"]
    mlc = stats["mlc"]
    chi2_text = (
        r"$\chi^{2}$  ($N\!=\!%d$ raw / $+%.2f$ dex M/L):" "\n"
        r"  HEAT free $K$ ($\beta\!=\!%.2f/%.2f$): %4.1f / %4.1f""\n"
        r"  HEAT $\beta\!=\!1$  : %4.1f / %4.1f""\n"
        r"  Const-$a_0$ MOND   : %4.1f / %4.1f""\n"
        r"  Ciocan DC14 unif.  : %4.1f / %4.1f""\n"
        r"  Ciocan best-fit p-g: %4.1f / %4.1f""\n"
        r"  Ciocan MOND fwk    : %4.1f / %4.1f"
    ) % (
        stats["N"], ML_DEX,
        beta_raw, beta_mlc,
        raw["chi2_freeK"], mlc["chi2_freeK"],
        raw["chi2_heat"], mlc["chi2_heat"],
        raw["chi2_mond"], mlc["chi2_mond"],
        raw["lin_chi2"]["DC14 uniform"], mlc["lin_chi2"]["DC14 uniform"],
        raw["lin_chi2"]["DC14 per-galaxy"], mlc["lin_chi2"]["DC14 per-galaxy"],
        raw["lin_chi2"]["MOND"], mlc["lin_chi2"]["MOND"],
    )
    ax_a0.text(
        0.015, 0.985, chi2_text,
        transform=ax_a0.transAxes,
        fontsize=7.5, va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white",
                  ec="0.4", lw=0.7, alpha=0.94),
        zorder=20,
    )

    # Annotation: Ciocan-MOND <-> HEAT 1%-match at z=0
    ax_a0.annotate(
        r"Ciocan MOND $a_0(0)=1.03\pm0.05$" "\n"
        r"HEAT $cH_0/(2\pi)=1.04$" "\n"
        r"agreement to $\sim 1\%$",
        xy=(0.0, 1.04), xytext=(0.10, 0.50),
        fontsize=9, ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="w", ec=_CB_BLUE, alpha=0.9),
        arrowprops=dict(arrowstyle="->", color=_CB_BLUE, lw=1.0),
    )

    ax_a0.set_xlabel("Redshift $z$")
    ax_a0.set_ylabel(r"$a_0$ $[10^{-10}\,\mathrm{m\,s^{-2}}]$")
    ax_a0.set_xlim(-0.05, 1.6)
    ax_a0.set_ylim(0.4, 3.4)
    ax_a0.grid(True, alpha=0.25, linewidth=0.5)
    ax_a0.set_title(r"(a) $a_0(z)$: HEAT vs Ciocan+2026 (Paper III)",
                    fontsize=11)

    # ---- Legend (below panel) ----
    handles = [
        Line2D([0], [0], color=_CB_BLUE, lw=2.0,
               label=(r"HEAT shape, $\beta\!=\!%.2f\!\pm\!%.2f$ "
                      r"(raw $N\!=\!%d$ best fit)"
                      % (beta_raw, raw["sigma_beta"], stats["N"]))),
        Line2D([0], [0], color=_CB_BLUE, lw=2.0,
               ls=(0, (4, 1.5)),
               label=(r"HEAT shape, $\beta\!=\!%.2f\!\pm\!%.2f$ "
                      r"($+%.2f$ dex M/L corrected)"
                      % (beta_mlc, mlc["sigma_beta"], ML_DEX))),
        Line2D([0], [0], color=_CB_BLUE, lw=1.4, ls=":",
               label=(r"Bare HEAT prediction: "
                      r"$a_0(z)\!=\!cH(z)/(2\pi)$ ($\beta\!=\!1$)")),
        Line2D([0], [0], color=_CB_GREY, lw=1.3, ls="--",
               label=r"Constant-$a_0$ MOND: $1.20\!\times\!10^{-10}$"),
        tuple(ciocan_line_handles),
        Line2D([0], [0], marker="s", color="w", mfc="k",
               ms=9, mec="k", mew=0.6,
               label="SPARC RAR-fit median (McGaugh+2016)"),
        Line2D([0], [0], marker="D", color="w", mfc="k",
               ms=8, mec="k", mew=0.6,
               label="SPARC HMC joint inference (Desmond+2023, MLS IF)"),
        Line2D([0], [0], marker="P", color="w", mfc=_CB_RED,
               ms=10, mec="k", mew=0.6,
               label=(r"V$\check{\rm a}$ra$\rm s$teanu+2025 "
                      r"(MIGHTEE+SPARC combined, RAR-IF)")),
        Line2D([0], [0], marker="o", color="w", mfc=_CB_PURPLE,
               ms=8, mec="k", mew=0.6,
               label="Ciocan+2026 binned $a_0$ (Paper III Fig.3)"),
        Line2D([0], [0], marker="^", color="w", mfc=_CB_PURPLE,
               ms=12, mec="k", mew=0.7,
               label=r"Ciocan global $z\sim 1$ (DC14)"),
        Line2D([0], [0], marker="v", color="w", mfc=_CB_GREEN,
               ms=12, mec="k", mew=0.7,
               label=r"Ciocan global $z\sim 1$ (best-fit per-gal.)"),
        Line2D([0], [0], marker="*", color="w", mfc=_CB_ORANGE,
               ms=14, mec="k", mew=0.7,
               label=r"Ciocan global $z\sim 1$ (MOND framework)"),
    ]
    labels = [h.get_label() if hasattr(h, "get_label") else
              "Ciocan+2026 linear fits (DC14 unif. / best-fit per-gal. / MOND)"
              for h in handles]

    ax_a0.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        fontsize=8.0,
        framealpha=0.92,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
        handlelength=2.4,
        columnspacing=1.4,
    )

    # ----- Panel (b): Delta log Sigma_DM(z) -----
    z_b = np.linspace(0.0, 2.0, 200)
    a0_z = np.array([heat_a0_at(z) for z in z_b])
    a0_0 = heat_a0_at(0.0)
    delta_heat = np.log10(a0_z / a0_0)
    log1pz = np.log10(1.0 + z_b)
    ax_sig.plot(log1pz, delta_heat, color=_CB_BLUE, lw=2.4,
                label=r"HEAT: $\Delta\log_{10}\,a_0(z) = \log_{10}\,H(z)/H_0$")

    ax_sig.axhline(0.0, color=_CB_GREY, ls="--", lw=1.2,
                   label=r"Constant-$a_0$ / no DM-density evolution")

    slope = 0.54
    slope_err = 0.31
    band_hi = slope * log1pz + slope_err * log1pz
    band_lo = slope * log1pz - slope_err * log1pz
    ax_sig.fill_between(log1pz, band_lo, band_hi,
                        color=_CB_PURPLE, alpha=0.18, linewidth=0,
                        label=r"Ciocan+2026 PaperI: $\rho_s\propto(1+z)^{0.54\pm 0.31}$")
    ax_sig.plot(log1pz, slope * log1pz, color=_CB_PURPLE, lw=1.2, ls="-.",
                alpha=0.9)

    section4 = [r for r in rows if r["section"] == 4]
    for r in section4:
        x = np.log10(1.0 + r["z"])
        y = r["a0_1e_minus10"]
        yerr = (r["sigma_lo"] + r["sigma_hi"]) / 2.0
        ax_sig.errorbar(x, y, yerr=yerr, fmt="D", color=_CB_PURPLE,
                        ms=10, mec="k", mew=0.7, elinewidth=1.1,
                        capsize=3, zorder=11,
                        label=r"Ciocan+2026 PaperI MHUDF $z\!\sim\!0.85$")

    ax_sig.set_xlabel(r"$\log_{10}(1+z)$")
    ax_sig.set_ylabel(r"$\Delta\log_{10}\,\Sigma_{\rm DM}(z)$  /  $\Delta\log_{10}\,a_0(z)$")
    ax_sig.set_xlim(-0.02, 0.50)
    ax_sig.set_ylim(-0.05, 0.55)
    ax_sig.grid(True, alpha=0.25, linewidth=0.5)
    ax_sig.legend(fontsize=8.0, loc="upper left", framealpha=0.92)
    ax_sig.set_title(r"(b) Halo-density evolution: HEAT vs Paper I",
                     fontsize=11)

    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.subplots_adjust(bottom=0.30, wspace=0.28)
    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig9_a0_evolution.{ext}", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_dir / 'fig9_a0_evolution.[pdf|png]'}")


def main():
    rows = _load_csv()
    summary, stats = _format_stats(rows)
    print(summary)

    out_dir = ensure_dir(JWST_EARLY)
    txt_path = out_dir / "a0_evolution_stats.txt"
    txt_path.write_text(summary + "\n", encoding="utf-8")
    print(f"\nWrote stats: {txt_path}")

    _plot(rows, stats, out_dir)


if __name__ == "__main__":
    main()
