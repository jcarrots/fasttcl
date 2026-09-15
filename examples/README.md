# FastTCL examples

Install from the package repository root:

```sh
python -m pip install ".[notebook]"
python -m jupyter lab examples/
```

Choose the Python environment containing `fasttcl` and run the notebook from
the first cell. Saved outputs also allow the notebooks to be read on GitHub.

| Notebook | Contents |
| --- | --- |
| [01_tcl2_tcl4_tcl6_comparison.ipynb](01_tcl2_tcl4_tcl6_comparison.ipynb) | Unbiased and biased spin-boson dynamics, populations and coherences, trace and Hermiticity checks, and a TCL4 time-step comparison. |
| [02_custom_bath_and_generators.ipynb](02_custom_bath_and_generators.ipynb) | A custom bath callable, sampled bath inputs, separate generator corrections, lambda powers, and reuse of a TCL6 plan. |

These small examples run on a CPU without TACO. For a TCL2- or TCL4-only
calculation, call `solve(..., order=2)` or `solve(..., order=4)` without
constructing or supplying a TCL6 plan. The sixth-order engine is then unused.

To execute both notebooks in fresh kernels using the current Python environment:

```sh
python examples/check_notebooks.py --output-dir outputs/notebooks
```

This stops if any cell or numerical assertion fails. The executed copies are
written to the output directory. Maintainers can add `--in-place` to refresh
the saved outputs in the source notebooks.

Command-line alternatives are `spin_boson.py` for dynamics and `runtime.py`
for a small timing run. The notebooks use short grids to demonstrate the API;
the paper figures and their provenance are described in
[reproduction.md](../docs/reproduction.md).
