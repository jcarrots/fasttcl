# FastTCL

FastTCL is a standalone CPU package for time-convolutionless master equations through
sixth order. 

The package provides full nonsecular TCL2, a convolution-based TCL4
implementation, and the corrected 124-group Hadamard-reduced TCL6 evaluator.
All three orders use Python, NumPy, and SciPy; sixth-order preparation also
uses SymPy. No TACO installation, C++ compiler, CUDA, or cluster account is
needed.

## Installation

Use Python 3.10 or newer. Clone the repository and install the package:

```sh
git clone https://github.com/jcarrots/fasttcl.git
cd fasttcl
python -m pip install .
# Include plotting and test tools:
python -m pip install ".[plot,test]"
```

Source archives and wheels are also available on the
[GitHub releases page](https://github.com/jcarrots/fasttcl/releases).

Version 0.2.0 renames the distribution from `tcl6-hr` to `fasttcl` and the
Python import from `tcl6_hr` to `fasttcl`. Update existing imports to
`from fasttcl import ...`. The earlier version 0.1.0 release remains available.

## First calculation

```python
import numpy as np
from fasttcl import OhmicBath, prepare_model, solve

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.diag([1., -1.])
model = prepare_model(0.5*sx, 0.5*sz)
bath = OhmicBath(omega_c=10.0)  # unit-strength C_B
rho0 = np.diag([1., 0.])

result = solve(model, bath, rho0, coupling_strength=np.sqrt(0.2),
               dt=0.02, n_steps=50, order=6)
rho2 = result.trajectories[2]
rho4 = result.trajectories[4]
rho6 = result.rho
```

`order=2` calculates TCL2 only; `order=4` calculates TCL2 and TCL4; `order=6`
calculates all three truncations. Lower-order runs never prepare the TCL6
compiler. Returned density matrices are in the same basis as `rho0` and the
input Hamiltonian. This small example checks usage; use finer grids when
assessing physical accuracy.

```sh
python examples/spin_boson.py --order 6 --output-dir outputs/example
python examples/runtime.py --max-power 9 --output outputs/runtime.csv
python reproduce/replot.py --output-dir outputs/paper
python -m pytest
```

The plot command uses the supplied numerical data. It does not rerun the
original long-time calculations or TEMPO simulations.

## Example notebooks

- [Compare TCL2, TCL4, and TCL6](examples/01_tcl2_tcl4_tcl6_comparison.ipynb):
  unbiased and biased spin-boson dynamics, plots, diagnostics, and a short
  time-step refinement check.
- [Custom baths and generator contributions](examples/02_custom_bath_and_generators.ipynb):
  callable and sampled correlations, separate generator orders, and plan reuse.

From the repository root:

```sh
python -m pip install ".[notebook]"
python -m jupyter lab examples/
```

Select the Python environment containing the package and run each notebook
from the first cell. Saved plots are included for GitHub previews. The
notebooks use short CPU calculations and require no external simulation data.
To execute both from the command line, use
`python examples/check_notebooks.py --output-dir outputs/notebooks`.

## Interface

- `prepare_model(H, A)`: prepare a finite-dimensional Hermitian Hamiltonian
  and one Hermitian coupling operator.
- `generator_series(model, bath, dt=..., n_steps=..., order=...)`: return
  separate corrections and cumulative generators on the specified grid.
- `solve(model, bath, rho0, dt=..., n_steps=..., order=...)`: propagate through
  the requested order, using generators calculated at the RK4 midpoint times.
- `compile_plan(model)`: prepare TCL6 once; reuse with `plan=...` across baths,
  grids, and coupling strengths for the same model.
- `OhmicBath` and `SampledBath`: convenient unit-strength bath inputs.
  A callable `bath(times)` is also accepted.

The default TCL6 preparation selects an exact-fused plan for sparse two-level
couplings and the source plan for dense two-level couplings. This avoids a
potentially long symbolic preparation for small dense calculations. Both
use the same fast temporal contractions and complete HR expression. Use
`compile_plan(model, fusion="exact")` to request exact fusion explicitly and
reuse that plan across runs. General-dimensional preparation uses its
existing temporal-reuse path. No small matrix entries are rounded to zero.

Use `coupling_strength` for **lambda** in `H_int = lambda A tensor B`.
Supply the unit-strength correlation `C_B`; the solver inserts lambda squared,
fourth, and sixth powers exactly once. See [conventions](docs/conventions.md)
before importing data from another code.

## Contents and scope

- `src/fasttcl/`: the public solver, standalone lower orders, retained CPU TCL6
  kernels, and bundled canonical generator data.
- `examples/`: two runnable notebooks, a small order comparison script,
  and a local timing experiment.
- `tests/`: independent lower-order sums, frozen TCL6 reference data, and
  end-to-end convention/propagation checks.
- `reproduce/`: compact accepted numerical data and portable figure scripts.
- `docs/`: conventions, reproduction details, and the local validation report.

The physical setting is a finite-dimensional system with one Hermitian
coupling to a centered, stationary Gaussian scalar bath and an initially
factorized state. The paper's numerical examples are two-level models.
General-dimensional evaluation retains the same HR expression; computational
cost can grow rapidly with system dimension. This release implements orders
2, 4, and 6. It does not claim a general TCL8 solver or implement the paper's
finite-exponential acceleration as a public backend.

The solver does not enforce positivity by modifying the result. Finite-order
TCL equations and finite time steps both introduce errors; check time-step
convergence for each application. The conventions page describes the exact
discretization used here.

## Development and citation

Run `python -m pip install -e ".[plot,test]"` for source development. The
GitHub Actions workflow installs the package, runs its numerical tests, and
executes both notebooks.
The source archive contains plotting data and examples; the smaller wheel
contains the solver and its required generator data.

Citation metadata is in [CITATION.cff](CITATION.cff). Please cite the work if you used this package in your research.
