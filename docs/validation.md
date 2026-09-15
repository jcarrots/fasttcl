# FastTCL validation

Current automated results for the Linux/Windows matrix and notebooks are on
[GitHub Actions](https://github.com/jcarrots/fasttcl/actions).

## Version 0.2.0 rename validation

Validated locally on Windows on 2026-09-15 after installing the rebuilt
`fasttcl` 0.2.0 wheel in the isolated Python 3.12 environment. The former
`tcl6_hr` package and TACO were absent. All **56 tests passed**, and all
**13 notebook code cells passed**. The five regenerated plots match the
previously inspected plots exactly. Dependency checks found no broken
requirements. The solver implementation and scientific input data are
byte-identical to version 0.1.0 apart from the package version declaration;
the Python namespace, distribution name, and documentation changed.

## Historical validation: version 0.1.0

The checks below were performed locally on Windows on 2026-09-14 for the
former `tcl6-hr` distribution and `tcl6_hr` import. They describe version 0.1.0.

### Installation and numerical checks

- Built a source distribution and a platform-independent `py3-none-any` wheel.
- Installed the wheel into an isolated Python 3.12 environment. Imports resolve
  to the installed `site-packages` directory, not the source tree. TACO is absent.
- Ran the entire test suite against that wheel: **56 passed**, no skips.
- Ran all 25 public API tests with Python 3.10 as an additional compatibility check.
- Dependency checks reported no broken requirements.

The lower-order suite compares complete TCL2/TCL4 arrays with independent
direct-sum formulas, including complex Hermitian couplings, three-level models,
degenerate gaps, endpoint weights, and the frequency mirror in nonsecular TCL2.
The TCL6 suite uses frozen outputs from the original canonical HR124 runtime:
the sparse exact-fusion and dense source paths match their reference arrays.
General-dimensional source/reuse evaluations also agree within floating-point
tolerance. These are regression and discrete-formula checks, not an independent
proof of every physical sixth-order coefficient.

The API tests cover lambda powers, separate/cumulative generators, input-basis
covariance, calculated RK4 midpoint stages, zero-coupling unitary dynamics,
sampled bath equivalence, and scalar input rejection. An independent Gaussian
pure-dephasing solution is reproduced within 2.2e-7 on the tested grid, with
the fourth- and sixth-order corrections zero to numerical precision.
Subprocess tests actively block TACO and CuPy imports. Lower-order tests also
block the entire TCL6 engine to verify that TCL2/TCL4 do not depend on it.

### Examples

Both unbiased and biased short spin-boson examples run at TCL2, TCL4, and TCL6.
The standalone timing example runs through all three orders with preparation
reported separately. These small runs are usability checks; their grids are
not claimed to reproduce the paper's high-resolution trajectories.

Both Jupyter notebooks were executed from the first cell with the installed
wheel: **13 code cells passed**, producing five saved figures. The notebooks
check all three truncations, trace and Hermiticity, a TCL4 time-step comparison,
callable/sampled bath agreement for generators and trajectories, and the
lambda powers of each generator correction. Callable and sampled dynamics
agreed exactly for the custom bath example. Every saved figure was inspected.
The notebook runner uses a fresh kernel with the same Python interpreter as
the command, and fails on a cell error or failed assertion.

### Tested environment

The complete installed-wheel run used Python 3.12.14, NumPy 2.5.3,
SciPy 1.18.1, SymPy 1.14.0, pytest 9.1.1, and Matplotlib 3.11.2.
Build tooling was setuptools 84.0.0 and wheel 0.48.0.
The Python 3.10 API run used the existing development interpreter.
Notebook execution used nbclient 0.11.0, nbformat 5.11.1, and ipykernel 7.3.0
in the isolated Python 3.12 environment.

The standard runtime requires only NumPy, SciPy, and SymPy. Matplotlib and
pytest are optional plotting/testing dependencies; Jupyter is available through
the optional `notebook` extra. No native TACO build,
GPU backend, persistent compiler cache, or external generator-data directory
is needed.
