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
python examples/spin_boson.py --order 6 --temperature 0.5 --output-dir outputs/thermal
python examples/bath_catalog.py --temperature 0.5 --steps 8
python examples/runtime.py --max-power 9 --output outputs/runtime.csv
```

`--temperature` defaults to zero. Positive values select a thermal Ohmic
bath, in frequency/energy units with hbar=kB=1.

`bath_catalog.py` runs power-law, Brownian, UV-regularized Drude-Lorentz,
discrete-mode, exponential, and custom spectral baths. It defaults to TCL2;
pass `--order 6` to reuse one TCL6 plan across the bath catalogue. See the
[bath guide](../docs/baths.md) for the exact definitions.

The examples use short grids; check convergence when choosing parameters
for your own calculation.
