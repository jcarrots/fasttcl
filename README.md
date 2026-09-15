# FastTCL

FastTCL is a CPU package for time-convolutionless master equations through
sixth order. It provides nonsecular TCL2, convolution-based TCL4, and
Hadamard-reduced TCL6 for a finite-dimensional system coupled to a Gaussian
bath. TCL2 and TCL4 use NumPy and SciPy; TCL6 preparation also uses SymPy.

## Installation

Use Python 3.10 or newer:

```sh
git clone https://github.com/jcarrots/fasttcl.git
cd fasttcl
python -m pip install .
```

Wheels and source archives are available on the
[releases page](https://github.com/jcarrots/fasttcl/releases).

## First calculation

```python
import numpy as np
from fasttcl import OhmicBath, prepare_model, solve

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.diag([1., -1.])
model = prepare_model(0.5*sx, 0.5*sz)
bath = OhmicBath(omega_c=10.0)
rho0 = np.diag([1., 0.])

result = solve(model, bath, rho0, coupling_strength=np.sqrt(0.2),
               dt=0.02, n_steps=50, order=6)
rho2 = result.trajectories[2]
rho4 = result.trajectories[4]
rho6 = result.rho
```

`order=2`, `4`, or `6` selects the highest order and returns all lower-order
trajectories too. TCL2 and TCL4 runs skip TCL6 preparation. Density matrices
are returned in the input basis.

Use `coupling_strength` for lambda in `H_int = lambda A tensor B`. Supply
the unscaled operator `A` and bath correlation `C_B`; the solver applies the
lambda powers. See [conventions](docs/conventions.md) for spectral
normalization, generator arrays, and time grids.

For repeated TCL6 calculations with the same Hamiltonian and coupling
operator, call `plan = compile_plan(model)` once and pass `plan=plan` to
`solve` or `generator_series`. The bath, time grid, and lambda may change.
`generator_series` returns separate corrections and cumulative generators
without propagating a state.

## Examples

- [TCL2, TCL4, and TCL6 comparison](examples/01_tcl2_tcl4_tcl6_comparison.ipynb):
  unbiased and biased spin-boson dynamics and a time-step check.
- [Custom baths and generators](examples/02_custom_bath_and_generators.ipynb):
  callable and sampled correlations, generator contributions, and plan reuse.

```sh
python -m pip install ".[notebook]"
python -m jupyter lab examples/
```

The notebooks include saved plots and use short grids. Command-line examples
are described in [examples/README.md](examples/README.md).

## Physical setting

The system couples through one Hermitian operator to a centered, stationary
Gaussian scalar bath, with an initially factorized state. The examples use
two-level systems; larger system dimensions can increase the cost substantially.

Check time-step convergence for each application. A finite TCL truncation
does not guarantee positivity or convergence with perturbative order.

## Testing

```sh
python -m pip install -e ".[test,notebook]"
python -m pytest
python examples/check_notebooks.py --output-dir outputs/notebooks
```

Tests compare TCL2 and TCL4 with direct sums, TCL6 with stored reference
values, and check coupling conventions and propagation. The
[GitHub Actions workflow](.github/workflows/tests.yml) runs the tests and
both notebooks.

## Citation

If you use **TCL4**, please cite paper 1. If you use **TCL6**, please cite
**both papers 1 and 2**.

1. Jiahao Chen, Elyana Crowder, Lian Xiang, and Dragomir Davidovic.
   [Benchmarking TCL4: Assessing the usability and reliability of fourth-order approximations](https://doi.org/10.1063/5.0255350).
   *APL Quantum* **2**, 026109 (2025).
2. Jiahao Chen, Sirui Chen, and Dragomir Davidovic.
   *Fast Evaluation of the Sixth-Order Time-Convolutionless Master-Equation Generator and Beyond*.
   Manuscript (2026).

Citation metadata is in [CITATION.cff](CITATION.cff).
