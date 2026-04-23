# HEAT Letter — Hubble-Emergent Acceleration Theory

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper: CC BY 4.0](https://img.shields.io/badge/Paper-CC%20BY%204.0-lightgrey.svg)](LICENSE-PAPER)
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

Companion code and LaTeX source for the letter:

> A zero-parameter derivation of the late-type galaxy size exponent and
> normalisation from `a_0(z) = c H(z) / (2 pi)`.

HEAT identifies the MOND acceleration scale with the surface gravity of the
FRW apparent cosmological horizon, $\kappa_A(z) = c H(z)$, promoted to all
redshifts:

$$a_0(z) = \frac{\kappa_A(z)}{2\pi} = \frac{c H(z)}{2\pi}.$$

This is a **zero-parameter** model — no $\beta$, no amplitude fit, no
interpolation. It recovers the empirical MOND value at $z = 0$ to within
roughly 13% and, combined with the deep-MOND kernel
$R \propto a_0^{-1/2}$, forces the late-type size–redshift relation

$$\frac{R_{\mathrm{HEAT}}(z)}{R_0} = (1+z)^{-3/4} \left[\frac{\Omega_m(z)}{\Omega_m(0)}\right]^{1/4}$$

with a matter-era asymptote $\Omega_m^{-1/4} \approx 1.34$ and no free
parameter at any step.

## Repository structure

| Path | Contents |
|------|----------|
| `theory/` | Core physics: cosmology, MOND-family kernel, SPARC I/O, local field correction |
| `publication/` | Analysis scripts and LaTeX source of the letter |
| `Rotmod_LTG/` | SPARC rotation-curve data (171 galaxies, Lelli et al. 2016) |
| `heat_output/` | Pre-generated letter figures and CSV summaries |
| `HEAT_THEORY.md` | Formal specification |

## Quick start

```bash
git clone <repo-url>
cd HEAT-paper
pip install -r requirements.txt
```

Python 3.10+ is assumed.

## Reproducing the letter figures

```bash
python run_all_heat_tests.py all --agg     # all three pipelines
python run_all_heat_tests.py list          # show available keys

# Individual analyses:
python run_all_heat_tests.py jwst          # Figures 3c (kinematic) and 5 (size-mass)
python run_all_heat_tests.py sparc-pub     # Figure 4 (SPARC chi^2 histograms)
python run_all_heat_tests.py fig-norm      # Figure 6 (R(z)/R_0 normalisation)
```

Output is written to `heat_output/` subfolders.

## Key modules

- **`theory/heat_cosmology.py`** — Planck 2018 parameters, HEAT Friedmann
  equation, zero-parameter $a_0(z) = c H(z) / (2\pi)$.
- **`theory/heat_physics.py`** — MOND-family interpolating function and
  deep-MOND velocity/radius helpers.
- **`theory/heat_field.py`** — Optional local field correction $\phi(r)$;
  used by the SPARC analysis, not by the letter's headline predictions.
- **`publication/jwst_early_galaxies.py`** — ALMA $z \sim 4.5$ joint
  velocity–size test (Fig. 3c) and stellar-mass-normalised
  compactification (Fig. 5).
- **`publication/sparc_publication_quality.py`** — SPARC 171-galaxy
  local-anchor $\chi^2$ comparison (Fig. 4).
- **`publication/fig_normalization.py`** — Zero-parameter
  $R(z) / R_0$ normalisation figure with matter-era asymptote (Fig. 6).

## The letter

The full LaTeX source of the letter lives at
`publication/paper_heat_letter.tex`.  Compile with `pdflatex` (or Overleaf)
after the analysis scripts have populated `heat_output/`.

## Citation

If you use this code or the letter, please cite:

```bibtex
@misc{heat2026letter,
  title        = {A zero-parameter derivation of the late-type galaxy
                  size exponent and normalisation from a_0(z) = c H(z) / (2 pi)},
  author       = {Bogdan Cosmin Enciu},
  year         = {2026},
  howpublished = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

The DOI will be minted on Zenodo and inserted above before release.

## License

- **Code** (`*.py`, `requirements*.txt`, `run_all_heat_tests.py`,
  shell helpers): MIT — see [`LICENSE`](LICENSE).
- **Paper, figures, and documentation** (`publication/*.tex`,
  `heat_output/**`, `HEAT_THEORY.md`, `README.md`): CC BY 4.0 —
  see [`LICENSE-PAPER`](LICENSE-PAPER).

## Authors

- **Bogdan Cosmin Enciu** — Independent Researcher — [bogdanenciu.author@gmail.com](mailto:bogdanenciu.author@gmail.com) — [ORCID: 0009-0008-7329-7200](https://orcid.org/0009-0008-7329-7200)
