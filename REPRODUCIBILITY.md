# Reproducibility Notes

This note records the release-level checks for regenerating the HEAT letter
figures from a clean checkout.

## Command

Run from the repository root:

```bash
python run_all_heat_tests.py all --agg
```

The `--agg` flag selects a non-interactive Matplotlib backend.

## Environment Checked

The latest local regeneration was run with:

```text
Python 3.10.1
numpy 2.2.6
scipy 1.15.3
matplotlib 3.10.8
```

These versions satisfy `requirements.txt`.

## Expected Artifacts

The command should regenerate the manuscript-facing artifacts below:

```text
heat_output/jwst_early_galaxies/fig3c_kinematic.pdf
heat_output/jwst_early_galaxies/fig5_size_mass.pdf
heat_output/jwst_early_galaxies/fig6_normalization.pdf
heat_output/jwst_early_galaxies/fig7_btfr_evolution.pdf
heat_output/jwst_early_galaxies/fig8_mass_selection.pdf
heat_output/jwst_early_galaxies/fig9_a0_evolution.pdf
heat_output/jwst_early_galaxies/a0_evolution_stats.txt
heat_output/jwst_early_galaxies/a0_ml_degeneracy.txt
heat_output/jwst_early_galaxies/a0_robustness.txt
heat_output/sparc_publication/sparc_chi2_red_histograms.png
heat_output/sparc_publication/sparc_publication_per_galaxy.csv
```

The latest local check found all listed artifacts present after regeneration.

## Pipeline Map

- `sparc-pub` runs `publication/sparc_publication_quality.py` and generates
  the SPARC local-anchor comparison.
- `jwst` runs `publication/jwst_early_galaxies.py` and generates the ALMA
  kinematic, size-mass, and BTFR figures.
- `fig-norm` runs `publication/fig_normalization.py` and generates the
  `R(z)/R_0` normalisation figure.
- `fig-mass` runs `publication/mass_selection_robustness.py` and generates
  the one-parameter mass-selection mimic diagnostic.
- `a0-evol` runs `publication/a0_evolution.py` and generates the direct
  `a0(z)` / inferred dark-matter surface-density diagnostic (Fig.~9 and
  `a0_evolution_stats.txt`).
- `a0-ml` runs `publication/a0_ml_degeneracy.py` (text-only, no figure
  output) and produces `a0_ml_degeneracy.txt`, the source for the M/L
  systematic scan, the shape-discrimination ladder, the
  subsample-stability table and the (beta, n) sensitivity region quoted
  in Sec. 4.5 of the HEAT Letter.
- `a0-robust` runs `publication/a0_robustness_tests.py` (text-only) and
  produces `a0_robustness.txt`, six independent robustness tests of
  the Sec. 4.5 conclusions: (A) joint (beta, n) MCMC posterior,
  (B) cosmology grid scan over H0 in [60, 76] and Om_m in [0.27, 0.35],
  (C) likelihood-ratio test for the HEAT n=1 shape, (D) 10000-draw
  bootstrap, (E) bidirectional predictive cross-validation under
  forced HEAT n=1, and (F) z-dependent M/L profile sensitivity.
  Runtime is roughly 3-4 minutes (MCMC + bootstrap dominate).
  All three of `a0-evol`, `a0-ml`, and `a0-robust` share the same
  Planck-2018 cosmology imported from `theory.heat_cosmology`.

## Notes for Reviewers

The scripts are deterministic apart from Monte Carlo diagnostic bands printed
by `publication/jwst_early_galaxies.py`; the paper-facing figures are generated
from fixed catalog rows and fixed Planck 2018 background parameters.

