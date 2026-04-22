"""
SPARC / Rotmod — publication-style pipeline (fixed assumptions, no per-galaxy tuning).

Methodology (intended for methods sections and supplementary material)
----------------------------------------------------------------------
- **Data:** SPARC `*_rotmod.dat` files (columns: r kpc, Vobs, err, Vgas, Vdisk, Vbulge).
- **Distance:** nominal value from file header only (`# Distance = X Mpc`); no distance or M/L grid search.
- **Mass-to-light:** fixed disk and bulge M/L in solar units (McGaugh et al. / SPARC convention):
  M/L_disk = 0.5, M/L_bulge = 0.7 (adjust constants below if you cite a specific reference).
- **Baryonic acceleration:** g_bar = (V_gas^2 + M/L_d V_disk^2 + M/L_b V_bulge^2) / r  (same r for all terms).
- **Observed acceleration:** g_obs = V_obs^2 / r.
- **Cosmology for HEAT:** z = H0 * D / c (local); a0(z) from theory.heat_cosmology.a0_hie.
- **Models compared:**
  - **MOND:** standard kernel with constant a0 = 1.2e-10 m/s^2.
  - **HEAT baseline (L2–3):** same kernel with a0(z) only (no phi).
  - **HEAT full (L2–4):** g_total = g_base * (1 + phi_reference); Rs = median(r) per galaxy.
- **Uncertainties on log10(g):** sigma_log10 = (delta g) / (g_obs ln 10), with delta g = 2 V_obs err / r from SPARC errors.

Outputs: printed summary statistics, optional CSV and figures under `--out-dir` (default: ``heat_output/sparc_publication/``).

**Data location:** folder with `*_rotmod.dat`. Resolution order: CLI argument, then ``SPARC_PATH``,
then ``<repo>/Rotmod_LTG`` (repository root from ``path_setup.REPO_ROOT``, not the process cwd).

This does not replace a full paper pipeline (morphology splits, alternative M/L, external distances);
it is a **reproducible baseline** comparable across MOND vs HEAT without hidden tuning.
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_repo = _Path(__file__).resolve().parents[1]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import path_setup  # noqa: E402

from theory.heat_output import SPARC_PUBLICATION

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from theory.heat_cosmology import LN10, a0_hie, estimate_redshift, pc_to_m
from theory.heat_field import g_heat_total
from theory.heat_physics import a0_mond, g_heat_from_g_bar
from theory.sparc_io import extract_distance

# Default for main() when out_dir is omitted (CLI "none" passes None explicitly to skip files).
_DEFAULT_SPARC_OUT = object()

# Fixed M/L (solar); common SPARC analysis choice — cite when publishing.
ML_DISK = 0.5
ML_BULGE = 0.7
A0_MOND = a0_mond()


def resolve_sparc_data_path(raw: str | None = None) -> str:
    """
    Folder containing SPARC *_rotmod.dat files.

    Uses ``SPARC_PATH`` when set. Otherwise the default name ``Rotmod_LTG`` is resolved under the
    **repository root** (parent of ``publication/``), so running this script with cwd ``publication/``
    still finds ``STTSSimulator/Rotmod_LTG``, not ``publication/Rotmod_LTG``.
    Any other relative path is resolved from the process current working directory.
    """
    repo_root: Path = path_setup.REPO_ROOT

    if raw is None or not str(raw).strip():
        env = os.environ.get("SPARC_PATH")
        if env and env.strip():
            return os.path.abspath(os.path.expanduser(env.strip()))
        return str((repo_root / "Rotmod_LTG").resolve())

    raw_s = str(raw).strip()
    # Same default string as run_all_heat_tests / CLI convention: anchor to repo, not cwd.
    if raw_s == "Rotmod_LTG":
        return str((repo_root / "Rotmod_LTG").resolve())

    return os.path.abspath(os.path.expanduser(raw_s))


def sigma_log10_from_dg(sigma_g: np.ndarray, g_obs: np.ndarray) -> np.ndarray:
    return sigma_g / (np.maximum(g_obs, 1e-45) * LN10)


@dataclass
class GalaxyFit:
    name: str
    n_radial: int
    chi2_red_mond: float
    chi2_red_heat_base: float
    chi2_red_heat_full: float
    mean_dex_res_mond: float
    mean_dex_res_heat_base: float
    mean_dex_res_heat_full: float
    # Regime-split chi^2: radial points with g_bar > a0 (Newtonian)
    # and g_bar < a0 (deep-MOND); NaN if no radial points fall in
    # that regime for this galaxy.
    n_radial_high: int
    n_radial_low: int
    chi2_red_mond_high: float
    chi2_red_mond_low: float
    chi2_red_heat_base_high: float
    chi2_red_heat_base_low: float


def process_one_galaxy(filepath: str) -> GalaxyFit | None:
    data = np.loadtxt(filepath)
    r = data[:, 0] * 1000.0 * pc_to_m
    Vobs = data[:, 1] * 1000.0
    Verr = data[:, 2] * 1000.0
    Vgas = data[:, 3] * 1000.0
    Vdisk = data[:, 4] * 1000.0
    Vbulge = data[:, 5] * 1000.0

    mask = (r > 0) & (Vobs > 0)
    r, Vobs, Verr = r[mask], Vobs[mask], Verr[mask]
    Vgas, Vdisk, Vbulge = Vgas[mask], Vdisk[mask], Vbulge[mask]

    dist_mpc = extract_distance(filepath)
    if dist_mpc is None or len(r) < 5:
        return None

    z = float(estimate_redshift(dist_mpc))
    a0_z = a0_hie(z)

    g_obs = (Vobs**2) / r
    g_bar = (Vgas**2 + ML_DISK * Vdisk**2 + ML_BULGE * Vbulge**2) / r

    sigma_g = (2.0 * Vobs * Verr) / r
    m = sigma_g > 0
    if not np.any(m):
        return None
    sigma_g = np.where(sigma_g > 0, sigma_g, np.median(sigma_g[m]))
    sigma_log = sigma_log10_from_dg(sigma_g, g_obs)

    g_mond = g_heat_from_g_bar(g_bar, A0_MOND)
    g_sb = g_heat_from_g_bar(g_bar, a0_z)
    g_sf = g_heat_total(g_bar, r, z)

    def chi2_red(g_model: np.ndarray) -> float:
        res = np.log10(g_obs) - np.log10(g_model)
        chi2 = np.sum((res / sigma_log) ** 2)
        return float(chi2 / len(r))

    def chi2_red_masked(g_model: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        res = np.log10(g_obs[mask]) - np.log10(g_model[mask])
        chi2 = np.sum((res / sigma_log[mask]) ** 2)
        return float(chi2 / np.sum(mask))

    def mean_dex(g_model: np.ndarray) -> float:
        return float(np.mean(np.log10(g_obs) - np.log10(g_model)))

    # Regime split: points where g_bar > a0 are Newtonian (no HEAT/MOND
    # distinction possible); points where g_bar < a0 are deep-MOND
    # (where the modified-inertia kernel actually matters).  Using the
    # local a0(z) for the cut captures both frameworks consistently.
    mask_high = g_bar > float(a0_z)
    mask_low = ~mask_high

    name = os.path.basename(filepath)
    return GalaxyFit(
        name=name,
        n_radial=len(r),
        chi2_red_mond=chi2_red(g_mond),
        chi2_red_heat_base=chi2_red(g_sb),
        chi2_red_heat_full=chi2_red(g_sf),
        mean_dex_res_mond=mean_dex(g_mond),
        mean_dex_res_heat_base=mean_dex(g_sb),
        mean_dex_res_heat_full=mean_dex(g_sf),
        n_radial_high=int(np.sum(mask_high)),
        n_radial_low=int(np.sum(mask_low)),
        chi2_red_mond_high=chi2_red_masked(g_mond, mask_high),
        chi2_red_mond_low=chi2_red_masked(g_mond, mask_low),
        chi2_red_heat_base_high=chi2_red_masked(g_sb, mask_high),
        chi2_red_heat_base_low=chi2_red_masked(g_sb, mask_low),
    )


def run_publication_pipeline(
    data_path: str | None = None,
    out_dir: str | None = None,
) -> dict:
    data_path = resolve_sparc_data_path(data_path)
    try:
        files = sorted(f for f in os.listdir(data_path) if f.endswith(".dat"))
    except FileNotFoundError:
        print(
            f"Data path not found: {data_path}\n"
            "  Expected a folder with SPARC *_rotmod.dat files.\n"
            "  Options:\n"
            "    • Set environment variable SPARC_PATH to that folder (e.g. in PyCharm Run/Debug env).\n"
            "    • Pass the folder as the first CLI argument.\n"
            "    • Put a Rotmod_LTG folder next to theory/ and publication/ (repo root), or set SPARC_PATH.\n"
            "  Data: SPARC release (Lelli+ 2016 AJ 152 157); see https://sparc.mpiau-garching.mpg.de/"
        )
        return {}

    results: list[GalaxyFit] = []
    for fname in files:
        fp = os.path.join(data_path, fname)
        try:
            g = process_one_galaxy(fp)
            if g is not None:
                results.append(g)
        except (OSError, ValueError, IndexError):
            continue

    if not results:
        print(f"No galaxies processed. Check data path (no usable *_rotmod.dat): {data_path}")
        return {}

    n = len(results)
    n_radial_total = sum(g.n_radial for g in results)

    chi_m = np.array([g.chi2_red_mond for g in results])
    chi_b = np.array([g.chi2_red_heat_base for g in results])
    chi_f = np.array([g.chi2_red_heat_full for g in results])

    # Pooled dex residuals: approximate global scatter from per-galaxy mean dex (secondary)
    print("=" * 60)
    print("SPARC publication-quality run (fixed M/L, no grid search)")
    print("=" * 60)
    print(f"Galaxies: {n}  |  Total radial points: {n_radial_total}")
    print(f"M/L_disk = {ML_DISK}, M/L_bulge = {ML_BULGE} (solar)")
    print()

    for label, arr in (
        ("MOND (a0 fixed)", chi_m),
        ("HEAT baseline L2–3 (a0(z))", chi_b),
        ("HEAT full L2–4 (+ phi)", chi_f),
    ):
        print(f"{label}")
        print(f"  median χ²_red per galaxy: {np.median(arr):.4f}")
        print(f"  mean χ²_red per galaxy:   {np.mean(arr):.4f}")
        print(f"  25–75%: {np.percentile(arr, 25):.4f} – {np.percentile(arr, 75):.4f}")
        print()

    # Paired comparisons (same galaxies)
    d_mb = chi_m - chi_b
    d_mf = chi_m - chi_f
    d_bf = chi_b - chi_f
    def _wilcoxon_p(diff: np.ndarray) -> str:
        try:
            if np.allclose(diff, 0):
                return "n/a (identical)"
            return f"{stats.wilcoxon(diff, alternative='two-sided').pvalue:.4e}"
        except ValueError:
            return "n/a"

    print("Paired differences in χ²_red (positive = second model better / lower χ²)")
    print(f"  MOND − HEAT baseline:  mean={np.mean(d_mb):.4f},  Wilcoxon p={_wilcoxon_p(d_mb)}")
    print(f"  MOND − HEAT full:      mean={np.mean(d_mf):.4f},  Wilcoxon p={_wilcoxon_p(d_mf)}")
    print(f"  HEAT base − HEAT full: mean={np.mean(d_bf):.4f},  Wilcoxon p={_wilcoxon_p(d_bf)}")

    wins_base = np.sum(chi_b < chi_m)
    wins_full = np.sum(chi_f < chi_m)
    print()
    print(f"Galaxies with lower χ²_red than MOND: HEAT baseline {wins_base}/{n}, HEAT full {wins_full}/{n}")

    # Regime-split summary: low-acceleration (g_bar < a0) vs
    # high-acceleration (g_bar > a0).  The high regime is Newtonian for
    # both HEAT and MOND by construction; differences can only arise in
    # the low-acceleration regime, where the modified-inertia kernel
    # applies.
    print()
    print("Regime-split χ²_red (radial points with g_bar above/below a0)")
    hi_m = np.array([g.chi2_red_mond_high for g in results])
    lo_m = np.array([g.chi2_red_mond_low for g in results])
    hi_b = np.array([g.chi2_red_heat_base_high for g in results])
    lo_b = np.array([g.chi2_red_heat_base_low for g in results])
    n_hi = np.array([g.n_radial_high for g in results]).sum()
    n_lo = np.array([g.n_radial_low for g in results]).sum()
    print(f"  Radial points total:   high-g (Newtonian) = {n_hi}, "
          f"low-g (deep-MOND) = {n_lo}")

    def _med_str(arr):
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return "n/a"
        return f"{np.median(arr):.3f}"

    print(f"  MOND median χ²_red   high-g: {_med_str(hi_m):>7s}   "
          f"low-g: {_med_str(lo_m):>7s}")
    print(f"  HEAT median χ²_red   high-g: {_med_str(hi_b):>7s}   "
          f"low-g: {_med_str(lo_b):>7s}")
    diff_low = lo_m - lo_b
    diff_low = diff_low[np.isfinite(diff_low)]
    if diff_low.size > 5:
        try:
            p_low = stats.wilcoxon(diff_low, alternative="two-sided").pvalue
            print(f"  Wilcoxon (MOND − HEAT, low-g only): p = {p_low:.3e}  "
                  f"(median Δ = {np.median(diff_low):+.4f})")
        except ValueError:
            pass

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "sparc_publication_per_galaxy.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "galaxy",
                    "n_radial",
                    "chi2_red_mond",
                    "chi2_red_heat_baseline",
                    "chi2_red_heat_full",
                    "mean_dex_res_mond",
                    "mean_dex_res_heat_baseline",
                    "mean_dex_res_heat_full",
                ]
            )
            for g in results:
                w.writerow(
                    [
                        g.name,
                        g.n_radial,
                        f"{g.chi2_red_mond:.6f}",
                        f"{g.chi2_red_heat_base:.6f}",
                        f"{g.chi2_red_heat_full:.6f}",
                        f"{g.mean_dex_res_mond:.6f}",
                        f"{g.mean_dex_res_heat_base:.6f}",
                        f"{g.mean_dex_res_heat_full:.6f}",
                    ]
                )
        print(f"Wrote {csv_path}")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, arr, title in zip(
            axes,
            [chi_m, chi_b, chi_f],
            ["MOND", "HEAT L2–3", "HEAT L2–4"],
        ):
            ax.hist(np.log10(np.clip(arr, 1e-6, None)), bins=40, color="steelblue", edgecolor="white")
            ax.set_title(title + r" $\log_{10}\chi^2_{\rm red}$")
            ax.set_xlabel(r"$\log_{10}$ χ²_red")
        plt.tight_layout()
        fig_path = os.path.join(out_dir, "sparc_chi2_red_histograms.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Wrote {fig_path}")

    return {
        "n_galaxies": n,
        "n_radial": n_radial_total,
        "median_chi2": {"mond": float(np.median(chi_m)), "heat_base": float(np.median(chi_b)), "heat_full": float(np.median(chi_f))},
        "results": results,
    }


def main(data_path: str | None = None, out_dir: Any = _DEFAULT_SPARC_OUT):
    """CLI default writes under ``heat_output/sparc_publication/``; pass ``out_dir=None`` to skip files."""
    if out_dir is _DEFAULT_SPARC_OUT:
        out_dir = str(SPARC_PUBLICATION)
    return run_publication_pipeline(data_path, out_dir=out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SPARC publication-quality MOND vs HEAT comparison.")
    p.add_argument(
        "data_path",
        nargs="?",
        default=None,
        help="Folder with *_rotmod.dat (default: SPARC_PATH env var, else Rotmod_LTG under cwd)",
    )
    p.add_argument(
        "--out-dir",
        default=str(SPARC_PUBLICATION),
        help=f"Write CSV and figures here (set 'none' to skip). Default: {SPARC_PUBLICATION}",
    )
    args = p.parse_args()
    od = args.out_dir
    if od and str(od).lower() in ("none", "-"):
        od = None
    main(args.data_path, out_dir=od)
