# Data Provenance

This directory contains small, paper-facing catalog extracts used as overlays
in the HEAT letter figures. The files are not survey catalog replacements; they
record the exact values consumed by the plotting scripts, with source notes in
each row.

## `ciocan2026_a0z.csv`

Used by:

- `publication/a0_evolution.py`
- Figure 9 in `publication/paper_heat_letter.tex`

Columns:

- `section`: row role used by the plotting script.
- `label`: short label used in printed summaries and figure annotations.
- `z`: redshift or representative bin redshift.
- `a0_1e_minus10`: acceleration value in units of `10^-10 m s^-2`.
- `sigma_lo`, `sigma_hi`: lower and upper one-sigma uncertainties in the same
  units.
- `source`: row-level provenance.

Row groups:

- `section = 1`: data points used in the main chi-squared ladder. This includes
  the SPARC local anchor, the Varasteanu low-redshift bTFR-derived anchor, and
  four Ciocan Paper III binned RAR points.
- `section = 2`: Ciocan Paper III global framework measurements shown as
  comparison markers, not as additional independent bins in the main
  chi-squared ladder.
- `section = 4`: Ciocan Paper I halo-density counterpart used in panel (b) of
  Figure 9.

Important caveats:

- The four Ciocan Paper III binned values are visual extractions from Paper III
  Figure 3, as noted in the CSV header.
- The Ciocan linear-fit triplets listed in the CSV comments are external fit
  coefficients from Paper III and are consumed directly in
  `publication/a0_evolution.py` as reference curves.
- The Varasteanu row is bTFR-derived rather than rotation-curve-derived, so the
  manuscript reports both the full cohort and the rotation-curve-only cohort.
- The file is intended to make the plotted values auditable; users should cite
  the original papers for scientific reuse.

