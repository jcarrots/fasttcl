# Examples

From the repository root:

```sh
python -m pip install ".[notebook]"
python -m jupyter lab examples/
```

Select the environment containing FastTCL and run the cells in order.

| Notebook | Contents |
| --- | --- |
| [TCL2, TCL4, and TCL6 comparison](01_tcl2_tcl4_tcl6_comparison.ipynb) | Spin-boson dynamics, populations and coherences, and a time-step check. |
| [Custom baths and generators](02_custom_bath_and_generators.ipynb) | Callable and sampled correlations, generator contributions, and plan reuse. |

To execute both notebooks and save the results:

```sh
python examples/check_notebooks.py --output-dir outputs/notebooks
```

Add `--in-place` to update the saved notebook outputs. Command-line examples
are also available:

```sh
python examples/spin_boson.py --order 6 --output-dir outputs/example
python examples/runtime.py --max-power 9 --output outputs/runtime.csv
```

The examples use short grids; check convergence when choosing parameters
for your own calculation.
