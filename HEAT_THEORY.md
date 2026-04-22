# HEAT — Formal specification (letter release)

This document is the condensed theory specification accompanying
`publication/paper_heat_letter.tex` and the zero-parameter HEAT release.

## 1. Acceleration scale

The central ansatz is an identification of the MOND acceleration scale
with the surface gravity of the FRW apparent cosmological horizon,
divided by \(2\pi\):

\[
  a_0(z) \;=\; \frac{\kappa_A(z)}{2\pi} \;=\; \frac{c\,H(z)}{2\pi}.
\]

No free parameter. The factor \(2\pi\) is the Euclidean period of the
horizon (Hawking–Unruh / Gibbons–Hawking). The static \(z=0\) limit
reproduces Milgrom (1999)'s identification of the MOND scale with the
de Sitter horizon temperature.

**Code:** [`theory/heat_cosmology.py`](theory/heat_cosmology.py) —
`a0_hie`, `hubble_parameter`, `E_heat`, `E_lcdm`, `a0_z0_mond_comparison`.

## 2. Background cosmology

HEAT's background is held strictly degenerate with ΛCDM. In factorised
form,

\[
  E_{\rm HEAT}(z)^2
    \;=\; \Omega_b\,F_0\,(1+z)^3 \;+\; \Omega_\Lambda,
  \qquad F_0 \equiv \frac{1-\Omega_\Lambda}{\Omega_b}
                  \;=\; \frac{\Omega_m}{\Omega_b},
\]

so that \(E_{\rm HEAT}(z) \equiv E_{\Lambda\rm CDM}(z)\) and
\(E_{\rm HEAT}(0) = 1\) by construction. \(F_0\) is a derived
normalisation identity, **not** a fit parameter.

All background numbers use Planck 2018 (TT+TE+EE+lowE+lensing):

| Symbol | Value | Role |
|--------|-------|------|
| \(H_0\) | \(67.4\,\mathrm{km\,s^{-1}\,Mpc^{-1}}\) | expansion rate today |
| \(\Omega_b\) | \(0.049\) | baryon density |
| \(\Omega_m\) | \(0.31\) | matter density |
| \(\Omega_\Lambda\) | \(0.69\) | cosmological constant |
| \(\Omega_r\) | \(9.24\times10^{-5}\) | radiation |

## 3. Deep-MOND kernel and size prediction

In deep-MOND equilibrium \(R\propto(GM_b/a_0)^{1/2}\). Combined with
\(a_0(z)=cH(z)/(2\pi)\) at fixed baryonic mass,

\[
  \frac{R(z)}{R_0} \;=\; \left[\frac{H(z)}{H_0}\right]^{-1/2}
  \;=\; (1+z)^{-3/4}
        \,\left[\frac{\Omega_m(z)}{\Omega_m(0)}\right]^{1/4},
\]

with matter-era exponent \(-3/4\) (matching
van der Wel et al. 2014) and normalisation asymptote
\(\mathcal{R}_\infty=\Omega_m^{-1/4}\approx 1.34\).

## 4. Local regime

On galactic scales HEAT reduces to standard MOND at \(z\approx 0\) by
construction. The SPARC 171-galaxy comparison in
`publication/sparc_publication_quality.py` uses the optional local
field correction \(\phi(r)\) from
[`theory/heat_field.py`](theory/heat_field.py) (Level-4 interpolating
function). The HEAT and constant-\(a_0\) MOND reduced-\(\chi^2\)
distributions are statistically indistinguishable
(paired Wilcoxon \(p=0.47\)).

## 5. Validation pipeline

| Key | Module | Figure |
|-----|--------|--------|
| `jwst` | `publication/jwst_early_galaxies.py` | Fig. 3c (ALMA kinematic), Fig. 5 (size–mass) |
| `sparc-pub` | `publication/sparc_publication_quality.py` | Fig. 4 (\(\chi^2_{\rm red}\) histograms) |
| `fig-norm` | `publication/fig_normalization.py` | Fig. 6 (normalisation curve) |

Run all three with `python run_all_heat_tests.py all --agg`.

## 6. Scope of this release

This repository is the code/data companion to the **letter**. Extended
analyses (JWST baryon-budget re-interpretation, collapse timescales,
redshift-binned RAR, CMB/CAMB comparison, cluster hydrostatic-mass
audit) are deferred to a longer treatment and are not included here.
