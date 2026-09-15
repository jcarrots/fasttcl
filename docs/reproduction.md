# Reproducing the numerical figures

The `reproduce/` directory contains a small, standalone data package and a
portable renderer for the four numerical figures in the paper. It reproduces
the approved plotted values without running simulations, TEMPO, or performance
benchmarks. Only NumPy and Matplotlib are needed; no TACO installation, GPU,
network connection, external fonts, or LaTeX installation is used.

From the package root, run:

```sh
python reproduce/replot.py --output-dir reproduced-figures
```

This writes `figure3.png`, `figure4.png`, `figure5.png`, and `figure6.png`.
The command works from another directory when given an absolute path to the
script. Optional arguments are `--figure 3` (or `4`, `5`, `6`), `--dpi 300`, and
`--verify-only`. Every run verifies SHA-256 hashes of its bundled data and
parameter files, as well as each array's shape, dtype, and byte content.

| Figure | Bundled plotted values | Meaning |
| --- | --- | --- |
| 3 | Six direct/reduced term norms and their relative discrepancies | Complete Hermitian/outer-commutator contributions already include the physical factor `lambda^6`. |
| 4 | All 72 measured timings, comparison guides, and 54 kernel MAD values | Historical single-core measurements; this command does not time the local installation. |
| 5 | Real population/coherence entries and full-matrix Frobenius discrepancies, for both models | Cumulative TCL2/TCL4/TCL6 dynamics and the raw finite-resolution TEMPO trajectories. |
| 6 | Normalized spectral phase spaces, downward rates, excited populations, and the kinetic comparison | The structured-bath example at `lambda^2=0.05`. |

All plotted samples are retained. Figure 3 norms and Figure 5 observables are
extracted using the exact transformations in the approved manuscript renderers.
Their unused full generator/density matrices are omitted to keep the download
small. Figure 6 includes exactly the rate interval plotted in the paper
(`Omega*t >= 120`), along with the complete displayed population arrays. There
is no additional decimation or interpolation.

The original figures used Arial for Figures 3, 5, and 6. The portable renderer
uses Matplotlib's bundled DejaVu Sans for all figures, so font metrics and raster
pixels can differ while the numerical curves, styles, limits, and panel layout
are preserved. It writes PNG only and does not modify manuscript figures.

## Parameters and provenance

`reproduce/data/figureN.json` records the model, coupling convention, numerical
grids, plotted-array semantics, and relevant qualifications for each figure.
`reproduce/provenance.json` gives the SHA-256 hash of every selected canonical
paper input and every bundled output array. Source paths are relative to the
paper repository and serve only as provenance; the renderer does not access
them. Raw checkpoints, scheduler records, machine-specific paths, unused
alternative calculations, and operational logs are not bundled.

The interaction convention is `H_int = lambda A tensor B`, with fixed `A` and
`C_lambda = lambda^2 C_B`. Archived inputs called this strength `alpha`, so
`alpha = lambda^2`. Figure 3's stored term maps already include `lambda^6`;
the package never multiplies them again. Figure 5 contains physical cumulative
trajectories and Figure 6 contains physical cumulative rates. In the runtime
fixture, strength `0.05` is already included in the damped-mode correlation,
with the explicit fixed coupling matrix given in its parameter JSON.

The replotting assets reproduce the following distinct numerical settings:

- Figure 3: `lambda^2=0.1`, cutoff `omega_c=10`, `dt=0.02`, 1,024 grid points,
  and 20 direct checkpoints; the approved direct GPU/reduced CuPy array pair is
  the source of the shown discrepancies.
- Figure 4: one Intel Xeon Gold 6226 core, `complex128`, grids `2^8` through
  `2^25`, and full-generator `dt=0.00625`. Kernel points are medians of five
  samples with MAD bars. Full-generator points are medians of three through
  `2^20` and one sample thereafter. Their workloads are distinct and their
  times must not be added. Dashed curves are anchored guides, not fitted
  exponents.
- Figure 5: `lambda^2=0.2`, `omega_c=10`, generator `dt=5e-5`, propagated-state
  `dt=1e-4`, ending at `Omega*t=50`. The approved displayed TCL arrays contain
  20,001 samples at `dt=0.0025`. TEMPO has 2,001 samples at `dt=0.025`, memory
  time 50, and relative SVD tolerance `1e-11` in OQuPy 0.5.0. Comparisons use
  coincident samples, without interpolation or state correction.
- Figure 6: the normalized one-boson band is `[1/5,7/15]`, `lambda^2=0.05`,
  zero temperature, and transverse coupling in the energy basis. Rates use
  `dt=0.01` through 360; the population generator has `dt=0.01`, while the
  displayed densities have `dt=0.02` through 1,400. The independent rate
  coefficient uses 512 Gauss-Legendre nodes in each integration variable.

TEMPO in Figure 5 is a finite-resolution comparison, not a certified
continuum-converged reference. Its plotted residuals are discrepancies, not
certified TCL errors. The black population curve in Figure 6 uses the independent
leading transition-matrix rate but is anchored to the TCL6 population at
`Omega*t=240`; it is not an independent full trajectory. Raw lower-order
populations are retained without positivity repair. The normalized spectral
phase spaces show energy support rather than complete transition rates.
The kinetic curve's archived NaNs before the anchor are retained, so this
unmodeled interval is not drawn. The manifest records this missing-value mask.

## Refreshing the extracted assets

Maintainers with the canonical paper repository may recreate the small data
package from its accepted inputs:

```sh
python reproduce/build_assets.py --paper-root /path/to/TCL6-Paper
python reproduce/replot.py --verify-only
```

This optional extraction step reads the canonical paper's active plotting
inputs and manifests, checks the accepted source hashes, and updates only the
assets under `reproduce/`. It does not retrieve historical manuscripts or run
the numerical engines. The standalone plotting command needs neither the paper
checkout nor this extraction step. Regenerating simulations and reproducing
historical timings are separate tasks from reproducing these approved figures.
