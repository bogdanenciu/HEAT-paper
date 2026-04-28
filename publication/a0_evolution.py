"""
Figure 9 (a0(z) and Sigma_DM(z) evolution) for HEAT Letter.

Tests the zero-parameter HEAT prediction a0(z) = c H(z) / (2 pi) directly
against the MUSE-DARK trilogy:
  Paper I  (Ciocan+2026, arXiv:2506.19721) -- halo central density evolution
  Paper II (Jeanneau+2026, arXiv:2603.28856) -- lensed bTFR null (treated in Fig 7)
  Paper III (Ciocan+2026, arXiv:2604.22613) -- intermediate-z RAR a0(z) evolution

Two-panel layout:
  (a) a0(z) -- zero-parameter HEAT curve vs Ciocan multi-framework data
       (DC14 uniform, DC14 per-galaxy, MOND) and SPARC + Varasteanu anchors.
       Constant-a0 MOND null is shown as a dashed grey line.  We annotate
       the 1% match between HEAT prediction and Ciocan-MOND framework
       extrapolated to z=0.
  (b) Delta log Sigma_DM vs log(1+z) -- HEAT-implied scaling
       Delta log Sigma_DM = log10[a0(z)/a0(0)] = log10[H(z)/H_0] (since,
       under the deep-MOND <-> equivalent-DM mapping, the dynamical
       acceleration scale is a0(z)).  Paper I MHUDF point at z~0.85
       is overlaid.

Outputs:
  heat_output/jwst_early_galaxies/fig9_a0_evolution.pdf (and .png)
  heat_output/jwst_early_galaxies/a0_evolution_stats.txt
"""
from __future__ import annotations

import csv
from pathlib import Path

import path_setup  # noqa: F401  # ensures repo root on sys.path

import numpy as np

from theory.heat_cosmology import a0_hie, hubble_parameter, H0, c
from theory.heat_output import HEAT_DATA_ROOT, JWST_EARLY, ensure_dir


_CB_BLUE = "#0072B2"
_CB_ORANGE = "#D55E00"
_CB_GREEN = "#009E73"
_CB_PURPLE = "#882255"
_CB_GREY = "#999999"
_CB_RED = "#CC3311"


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
                            color=_CB_GREEN, label="Ciocan DC14 per-galaxy"),
    "MOND": dict(a0=1.03, sa0=0.05, a1=1.20, sa1=0.10,
                 color=_CB_ORANGE, label="Ciocan MOND framework"),
}


def heat_a0_at(z):
    """HEAT prediction in 10^-10 m/s^2."""
    return float(a0_hie(z)) / 1e-10


def linear_a0(z, a0, a1):
    return a0 + a1 * z


def chi2_for_model(zs, obs, sig, model_fn):
    """Chi^2 of model_fn(z) vs (obs +/- sig)."""
    pred = np.array([model_fn(zi) for zi in zs])
    return float(np.sum(((np.asarray(obs) - pred) / np.asarray(sig)) ** 2))


def _format_stats(rows):
    """Compute residual table (in sigma units) and chi^2 across models.

    Two chi^2 cohorts are reported in parallel:
      * 'full' (N=6): SPARC + Varasteanu + 4 Ciocan bins.
      * 'RC-only' (N=5): SPARC + 4 Ciocan bins, excluding the bTFR-derived
        Varasteanu+2025 point.  bTFR-derived a0 values are known to differ
        from rotation-curve-derived a0 values at the ~30% level on the
        same galaxies (Rodrigues+2018; McGaugh+2018), so the methodology
        split is justified independently of the residual.
    """
    section1 = [r for r in rows if r["section"] == 1]
    zs = np.array([r["z"] for r in section1])
    obs = np.array([r["a0_1e_minus10"] for r in section1])
    sig = np.array([(r["sigma_lo"] + r["sigma_hi"]) / 2.0 for r in section1])
    labels = [r["label"] for r in section1]

    # Boolean mask for the RC-derived (rotation-curve-modelled) subset.
    is_rc = np.array(["Varasteanu" not in lab for lab in labels])

    # Models
    heat_pred = np.array([heat_a0_at(z) for z in zs])
    mond_const = np.full_like(zs, 1.20, dtype=float)  # canonical SPARC value

    # Free-K best fit on the FULL N=6 sample:  a0_fit(z) = K * c * H(z)
    y = heat_pred * (2.0 * np.pi)  # K=1 prediction (in 10^-10 m/s^2 units)
    w = 1.0 / sig ** 2
    K_fit = float(np.sum(w * y * obs) / np.sum(w * y * y))
    K_heat = 1.0 / (2.0 * np.pi)
    free_K_pred = K_fit * y

    # Free-K best fit on the RC-only N=5 sample (drops Varasteanu).
    y_rc = y[is_rc]
    obs_rc = obs[is_rc]
    sig_rc = sig[is_rc]
    w_rc = 1.0 / sig_rc ** 2
    K_fit_rc = float(np.sum(w_rc * y_rc * obs_rc)
                     / np.sum(w_rc * y_rc * y_rc))
    free_K_pred_rc = K_fit_rc * y_rc

    # Linear-in-z fits: chi^2 over the FULL sample for the headline table,
    # and over the RC-only sample for the methodology-split summary.
    lin_chi2 = {}
    lin_chi2_rc = {}
    for name, fit in CIOCAN_LINEAR_FITS.items():
        pred = np.array([linear_a0(zi, fit["a0"], fit["a1"]) for zi in zs])
        lin_chi2[name] = float(np.sum(((obs - pred) / sig) ** 2))
        lin_chi2_rc[name] = float(
            np.sum(((obs[is_rc] - pred[is_rc]) / sig[is_rc]) ** 2)
        )

    chi2_heat = float(np.sum(((obs - heat_pred) / sig) ** 2))
    chi2_mond = float(np.sum(((obs - mond_const) / sig) ** 2))
    chi2_freeK = float(np.sum(((obs - free_K_pred) / sig) ** 2))

    chi2_heat_rc = float(np.sum(((obs[is_rc] - heat_pred[is_rc])
                                 / sig[is_rc]) ** 2))
    chi2_mond_rc = float(np.sum(((obs[is_rc] - mond_const[is_rc])
                                 / sig[is_rc]) ** 2))
    chi2_freeK_rc = float(np.sum(((obs[is_rc] - free_K_pred_rc)
                                  / sig[is_rc]) ** 2))

    # Same RC-only chi^2 but anchored at the Ciocan-MOND-framework intercept
    # (1.03 +/- 0.05) instead of the SPARC RC-derived value (1.20 +/- 0.26).
    # This is the apples-to-apples z=0 anchor for any analysis whose
    # high-z points are themselves derived in the same (Ciocan-MOND)
    # interpolation framework.
    obs_mond_anchor = obs[is_rc].copy()
    sig_mond_anchor = sig[is_rc].copy()
    sparc_idx = np.where([("SPARC" in lab) for lab in
                          [labels[i] for i, m in enumerate(is_rc) if m]])[0]
    if len(sparc_idx):
        obs_mond_anchor[sparc_idx[0]] = 1.03
        sig_mond_anchor[sparc_idx[0]] = 0.05
    chi2_heat_rc_mondAnchor = float(np.sum(
        ((obs_mond_anchor - heat_pred[is_rc]) / sig_mond_anchor) ** 2
    ))

    lines = []
    lines.append("=" * 78)
    lines.append("HEAT a0(z) test against Ciocan+2026 + Varasteanu+2025 + SPARC")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Per-bin residuals (zero-parameter HEAT, K = 1/(2*pi)):")
    lines.append(f"  K_HEAT = 1/(2 pi)              = {K_heat:.5f}")
    lines.append(f"  K_fit  (best free-K HEAT shape, N=6)   = {K_fit:.5f}")
    lines.append(f"  K_fit  (best free-K, N=5 RC-only)      = {K_fit_rc:.5f}")
    lines.append(f"  K_fit/K_HEAT  (full)    = {K_fit / K_heat:+.3f}")
    lines.append(f"  K_fit/K_HEAT  (RC-only) = {K_fit_rc / K_heat:+.3f}")
    lines.append("")
    lines.append(f"  {'label':30s} {'z':>5s} {'a0_obs':>8s} "
                 f"{'a0_HEAT':>8s} {'sig_HEAT':>10s} {'ratio':>7s}  RC?")
    lines.append("  " + "-" * 78)
    for i, lab in enumerate(labels):
        sig_units = (obs[i] - heat_pred[i]) / sig[i]
        ratio = obs[i] / heat_pred[i]
        rc_flag = "yes" if is_rc[i] else "no (bTFR)"
        lines.append(f"  {lab:30s} {zs[i]:5.2f} "
                     f"{obs[i]:8.3f} {heat_pred[i]:8.3f} "
                     f"{sig_units:+9.2f}s {ratio:7.3f}  {rc_flag}")
    lines.append("")
    lines.append(
        "  chi^2 ladder ('full' = SPARC+Varasteanu+4Ciocan, N=6;"
    )
    lines.append(
        "                'RC'   = SPARC+4Ciocan, N=5, drops bTFR-derived Varasteanu)"
    )
    lines.append("")
    lines.append(f"  {'model':38s}  {'full N=6':>10s}   {'RC N=5':>10s}")
    lines.append("  " + "-" * 64)
    lines.append(f"  {'HEAT, fixed K=1/(2 pi)':38s}  "
                 f"{chi2_heat:10.2f}   {chi2_heat_rc:10.2f}")
    lines.append(f"  {'HEAT shape with free K':38s}  "
                 f"{chi2_freeK:10.2f}   {chi2_freeK_rc:10.2f}")
    lines.append(f"  {'constant-a0 MOND (1.20e-10)':38s}  "
                 f"{chi2_mond:10.2f}   {chi2_mond_rc:10.2f}")
    for name in CIOCAN_LINEAR_FITS:
        lines.append(f"  {('Ciocan linear: ' + name):38s}  "
                     f"{lin_chi2[name]:10.2f}   {lin_chi2_rc[name]:10.2f}")
    lines.append("")
    lines.append(
        f"  chi^2 (HEAT, RC-only, MOND-framework z=0 anchor 1.03+/-0.05) "
        f"= {chi2_heat_rc_mondAnchor:.2f}"
    )
    lines.append(
        "  -- using the apples-to-apples z=0 anchor for the Ciocan-MOND framework."
    )
    lines.append("")

    # Cross-framework headline: HEAT vs each Ciocan-extrapolated z=0
    lines.append("Cross-framework comparison at z=0 (intercept of Ciocan linear fits):")
    lines.append(f"  HEAT a0(0) = c H_0 / (2 pi)        = {heat_a0_at(0.0):.3f} e-10 m/s^2")
    for name, fit in CIOCAN_LINEAR_FITS.items():
        a00 = fit["a0"]
        ratio = heat_a0_at(0.0) / a00
        lines.append(f"  Ciocan {name:20s}  a0(0)={a00:.2f}  -> HEAT/Ciocan = {ratio:.3f}")
    lines.append("")
    lines.append("Note: Ciocan MOND-framework extrapolation gives a0(0)=1.03(0.05)")
    lines.append("which agrees with the HEAT zero-parameter prediction (1.04) to ~1%.")
    lines.append("This is the apples-to-apples z=0 anchor: both invert the same MOND")
    lines.append("interpolation function on the same kinematics. The SPARC RAR-fit")
    lines.append("value (1.20+/-0.26) uses a different inversion (galaxy-by-galaxy")
    lines.append("RAR median); HEAT agrees with it within its 1-sigma envelope.")
    lines.append("=" * 78)

    return "\n".join(lines), dict(
        zs=zs, obs=obs, sig=sig, labels=labels, is_rc=is_rc,
        heat_pred=heat_pred, mond_const=mond_const,
        K_fit=K_fit, K_fit_rc=K_fit_rc, K_heat=K_heat,
        free_K_pred=free_K_pred, free_K_pred_rc=free_K_pred_rc,
        chi2_heat=chi2_heat, chi2_freeK=chi2_freeK, chi2_mond=chi2_mond,
        chi2_heat_rc=chi2_heat_rc, chi2_freeK_rc=chi2_freeK_rc,
        chi2_mond_rc=chi2_mond_rc,
        chi2_heat_rc_mondAnchor=chi2_heat_rc_mondAnchor,
        lin_chi2=lin_chi2, lin_chi2_rc=lin_chi2_rc,
    )


def _plot(rows, stats, out_dir):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple
    _setup_fig_style()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.4),
                             gridspec_kw=dict(width_ratios=[1.35, 1.0]))
    ax_a0, ax_sig = axes

    # ----- Panel (a): a0(z) -----
    z_smooth = np.linspace(0.0, 1.7, 220)
    a0_curve = np.array([heat_a0_at(z) for z in z_smooth])
    ax_a0.plot(z_smooth, a0_curve, color=_CB_BLUE, lw=2.4)

    # 13% systematic band (SPARC posterior vs HEAT z=0 anchor); described
    # in the figure caption rather than the legend to keep the latter clean.
    ax_a0.fill_between(z_smooth, a0_curve * 0.87, a0_curve * 1.13,
                       color=_CB_BLUE, alpha=0.13, linewidth=0)

    # Free-K HEAT shape: same a0(z) propto H(z) scaling, K floated.
    # The headline observable R(z)/R_0 is invariant under K (K-invariance
    # argument, sec:exp), so this curve illustrates that the *shape* of
    # the data is consistent with HEAT once the absolute normalisation
    # is allowed to drift.  K_fit / K_HEAT ~ 1.35 in current data.
    K_fit = float(stats["K_fit"])
    K_heat = float(stats["K_heat"])
    a0_freeK = a0_curve * (K_fit / K_heat)
    ax_a0.plot(z_smooth, a0_freeK, color=_CB_BLUE, lw=1.6, ls="--",
               alpha=0.85)

    # Constant-a0 MOND null
    ax_a0.axhline(1.20, color=_CB_GREY, ls="--", lw=1.3)

    # Ciocan linear fits (plotted with their own colors but grouped into
    # a single multi-color legend entry below via HandlerTuple).
    ciocan_line_handles = []
    for name, fit in CIOCAN_LINEAR_FITS.items():
        zz = np.linspace(0.0, 1.7, 50)
        ln, = ax_a0.plot(zz, fit["a0"] + fit["a1"] * zz,
                         color=fit["color"], ls=":", lw=1.5, alpha=0.85)
        ciocan_line_handles.append(ln)

    # Section-1 data (binned + anchors)
    section1 = [r for r in rows if r["section"] == 1]
    for r in section1:
        if "SPARC" in r["label"]:
            mrk, clr, ms = "s", "k", 9
        elif "Varasteanu" in r["label"]:
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

    # ---- chi^2 comparison inset (upper-left) ----
    # Pulls live values from stats so the figure stays in sync with
    # the printed summary and the LaTeX table tab:a0z_chi2.  We report
    # two cohorts: 'full' (N=6) and 'RC-only' (N=5, drops bTFR-derived
    # Varasteanu+2025 per Rodrigues+2018, McGaugh+2018 methodology).
    chi2_h = float(stats["chi2_heat"])
    chi2_fk = float(stats["chi2_freeK"])
    chi2_m = float(stats["chi2_mond"])
    chi2_h_rc = float(stats["chi2_heat_rc"])
    chi2_fk_rc = float(stats["chi2_freeK_rc"])
    chi2_m_rc = float(stats["chi2_mond_rc"])
    lin = stats["lin_chi2"]
    lin_rc = stats["lin_chi2_rc"]
    n_full = int(np.sum(np.ones_like(stats["zs"])))
    n_rc = int(np.sum(stats["is_rc"]))
    chi2_text = (
        r"$\chi^{2}$  (full $N\!=\!%d$ / RC $N\!=\!%d$):" "\n"
        r"  HEAT, $K{=}1/(2\pi)$ : %5.1f / %5.1f""\n"
        r"  HEAT free $K$        : %5.1f / %5.1f  (best)""\n"
        r"  Const-$a_0$ MOND     : %5.1f / %5.1f""\n"
        r"  Ciocan DC14 unif.    : %5.1f / %5.1f""\n"
        r"  Ciocan DC14 per-gal  : %5.1f / %5.1f""\n"
        r"  Ciocan MOND fwk      : %5.1f / %5.1f"
    ) % (
        n_full, n_rc,
        chi2_h, chi2_h_rc,
        chi2_fk, chi2_fk_rc,
        chi2_m, chi2_m_rc,
        lin["DC14 uniform"], lin_rc["DC14 uniform"],
        lin["DC14 per-galaxy"], lin_rc["DC14 per-galaxy"],
        lin["MOND"], lin_rc["MOND"],
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

    # ---- Manually-built legend, placed below the panel ----
    # Group the three Ciocan linear-fit lines into a single multi-color
    # legend entry via HandlerTuple to free up vertical legend real estate.
    K_fit_disp = float(stats["K_fit"])
    K_fit_rc_disp = float(stats["K_fit_rc"])
    K_heat_disp = float(stats["K_heat"])

    # Existing curves (HEAT zero-K solid blue, free-K dashed blue, MOND
    # dashed grey) + the band already carry labels via their plot calls;
    # we re-collect them and append our composite Ciocan handle plus
    # marker proxies so a single ax.legend() call orders everything.
    handles = [
        Line2D([0], [0], color=_CB_BLUE, lw=2.4,
               label=r"HEAT zero-parameter: $a_0(z)=cH(z)/(2\pi)$"),
        Line2D([0], [0], color=_CB_BLUE, lw=1.6, ls="--",
               label=(r"HEAT shape, free $K$: $K\!\approx\!%.2f\,K_{\rm HEAT}$ "
                      r"(full); $K\!\approx\!%.2f\,K_{\rm HEAT}$ (RC-only)" %
                      (K_fit_disp / K_heat_disp,
                       K_fit_rc_disp / K_heat_disp))),
        Line2D([0], [0], color=_CB_GREY, lw=1.3, ls="--",
               label=r"Constant-$a_0$ MOND: $1.20\!\times\!10^{-10}$"),
        tuple(ciocan_line_handles),  # combined Ciocan linear fits
        Line2D([0], [0], marker="s", color="w", mfc="k",
               ms=9, mec="k", mew=0.6,
               label="SPARC $z=0$ (McGaugh+2016, RC-derived)"),
        Line2D([0], [0], marker="P", color="w", mfc=_CB_RED,
               ms=10, mec="k", mew=0.6,
               label=(r"V$\check{\rm a}$ra$\rm s$teanu+2025 "
                      r"(HI bTFR-derived; not in RC $\chi^{2}$)")),
        Line2D([0], [0], marker="o", color="w", mfc=_CB_PURPLE,
               ms=8, mec="k", mew=0.6,
               label="Ciocan+2026 binned $a_0$ (Paper III Fig.3)"),
        Line2D([0], [0], marker="^", color="w", mfc=_CB_PURPLE,
               ms=12, mec="k", mew=0.7,
               label=r"Ciocan global $z\sim 1$ (DC14)"),
        Line2D([0], [0], marker="v", color="w", mfc=_CB_GREEN,
               ms=12, mec="k", mew=0.7,
               label=r"Ciocan global $z\sim 1$ (DC14 per-gal.)"),
        Line2D([0], [0], marker="*", color="w", mfc=_CB_ORANGE,
               ms=14, mec="k", mew=0.7,
               label=r"Ciocan global $z\sim 1$ (MOND framework)"),
    ]
    labels = [h.get_label() if hasattr(h, "get_label") else
              "Ciocan+2026 linear fits (DC14 unif. / DC14 per-gal. / MOND)"
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
    # HEAT-implied scaling: Delta log Sigma_DM(z) = log10[a0(z)/a0(0)]
    z_b = np.linspace(0.0, 2.0, 200)
    a0_z = np.array([heat_a0_at(z) for z in z_b])
    a0_0 = heat_a0_at(0.0)
    delta_heat = np.log10(a0_z / a0_0)
    log1pz = np.log10(1.0 + z_b)
    ax_sig.plot(log1pz, delta_heat, color=_CB_BLUE, lw=2.4,
                label=r"HEAT: $\Delta\log_{10}\,a_0(z) = \log_{10}\,H(z)/H_0$")

    # Reference null: no evolution
    ax_sig.axhline(0.0, color=_CB_GREY, ls="--", lw=1.2,
                   label=r"Constant-$a_0$ / no DM-density evolution")

    # Paper I central halo density evolution: rho_s ~ (1+z)^0.54 +/- 0.31
    # so Delta log rho_s(z) = 0.54 * log10(1+z) +/- 0.31 * log10(1+z) ...
    # we plot as a shaded power-law band using their best-fit slope
    slope = 0.54
    slope_err = 0.31
    band_hi = slope * log1pz + slope_err * log1pz
    band_lo = slope * log1pz - slope_err * log1pz
    ax_sig.fill_between(log1pz, band_lo, band_hi,
                        color=_CB_PURPLE, alpha=0.18, linewidth=0,
                        label=r"Ciocan+2026 PaperI: $\rho_s\propto(1+z)^{0.54\pm 0.31}$")
    ax_sig.plot(log1pz, slope * log1pz, color=_CB_PURPLE, lw=1.2, ls="-.",
                alpha=0.9)

    # Section-4 data point (Paper I MHUDF z=0.85)
    section4 = [r for r in rows if r["section"] == 4]
    for r in section4:
        x = np.log10(1.0 + r["z"])
        y = r["a0_1e_minus10"]   # interpreted as Delta log Sigma_DM
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

    # Reserve space below panel (a) for the external legend; rcParams
    # savefig.bbox="tight" then expands as needed for clean PDF/PNG output.
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
