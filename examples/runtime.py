"""Small all-orders CPU benchmark. Reports preparation separately from evaluation."""
import argparse
import csv
from pathlib import Path
from time import perf_counter
import numpy as np
from tcl6_hr import OhmicBath, prepare_model, compile_plan, generator_series


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-power", type=int, default=9)
    p.add_argument("--output", type=Path, default=Path("outputs/runtime.csv"))
    args = p.parse_args()
    if not 4 <= args.max_power <= 25:
        p.error("--max-power must be between 4 and 25; large grids can take hours")
    model = prepare_model([[0, .5], [.5, 0]], [[.5, 0], [0, -.5]])
    start = perf_counter()
    plan = compile_plan(model)
    prep = perf_counter()-start
    rows = []
    for power in range(4, args.max_power+1):
        nt = 2**power
        for order in (2, 4, 6):
            start = perf_counter()
            result = generator_series(model, OhmicBath(10), dt=.01, n_steps=nt-1,
                                      order=order, coupling_strength=np.sqrt(.2),
                                      plan=plan if order == 6 else None)
            elapsed = perf_counter()-start
            rows.append(dict(nt=nt, order=order, seconds=elapsed,
                             tcl6_preparation_seconds=prep if order == 6 else 0))
            print(f"Nt={nt}, through TCL{order}: {elapsed:.4f} s")
            del result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
