"""Small standalone TCL2/TCL4/TCL6 comparison; no saved simulation is needed."""
import argparse
from pathlib import Path
import numpy as np
from tcl6_hr import OhmicBath, prepare_model, solve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--order", type=int, choices=(2, 4, 6), default=6)
    p.add_argument("--biased", action="store_true")
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/example"))
    args = p.parse_args()
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.diag([1., -1.])
    h = 0.4*sx+0.3*sz if args.biased else 0.5*sx
    model = prepare_model(h, 0.5*sz)
    result = solve(model, OhmicBath(10), np.diag([1., 0.]),
                   coupling_strength=np.sqrt(0.2), dt=args.dt,
                   n_steps=args.steps, order=args.order)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir/"trajectories.npz", times=result.times,
                        **{f"rho_tcl{k}": r for k, r in result.trajectories.items()})
    for k, r in result.trajectories.items():
        print(f"TCL{k}: final rho00={r[-1,0,0].real:.8f}; "
              f"max trace error={np.max(np.abs(np.trace(r,axis1=1,axis2=2)-1)):.3g}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install the plot extra to also create the figure.")
        return
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    for k, r in result.trajectories.items():
        ax.plot(result.times, r[:,0,0].real, label=f"TCL{k}")
    ax.set(xlabel="Time", ylabel=r"$\mathrm{Re}\,\rho_{00}$")
    ax.legend()
    fig.savefig(args.output_dir/"comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
