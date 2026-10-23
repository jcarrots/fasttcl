"""Run different stationary Gaussian bath models through one FastTCL interface."""
import argparse

import numpy as np

from fasttcl import (
    BrownianBath, DiscreteBath, DrudeLorentzBath, ExponentialBath,
    OhmicBath, PowerLawBath, SpectralDensityBath, compile_plan, prepare_model, solve,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--order", type=int, choices=(2, 4, 6), default=2)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    temperature = args.temperature
    baths = {
        "Ohmic": OhmicBath(2.0, temperature=temperature),
        "Sub-Ohmic": PowerLawBath(2.0, exponent_s=0.5, temperature=temperature),
        "Super-Ohmic": PowerLawBath(2.0, exponent_s=3.0, temperature=temperature),
        "Brownian": BrownianBath(omega_0=2.0, damping=0.3, temperature=temperature),
        "Drude + UV cutoff": DrudeLorentzBath(gamma=1.0, uv_cutoff=10.0, temperature=temperature),
        "Thermal modes": DiscreteBath([1.0, 2.5], [0.8, 0.2], temperature=temperature),
        "Exponential fit": ExponentialBath([0.6, 0.4], [0.3+1.2j, 0.3-1.2j]),
        "Custom spectrum": SpectralDensityBath(lambda w: w*np.exp(-w/3),
                                               temperature=temperature, omega_max=60),
    }
    model = prepare_model(np.diag([-.5, .5]), [[0, .5], [.5, 0]])
    plan = compile_plan(model) if args.order == 6 else None
    for name, bath in baths.items():
        result = solve(model, bath, np.diag([0., 1.]), dt=.025,
                       n_steps=args.steps, order=args.order,
                       coupling_strength=.2, plan=plan)
        trace_error = np.max(np.abs(np.trace(result.rho, axis1=1, axis2=2)-1))
        print(f"{name}: C_B(0)={complex(bath.correlation(0)).real:.6g}; "
              f"final rho11={result.rho[-1,1,1].real:.8f}; "
              f"max trace error={trace_error:.3g}")


if __name__ == "__main__":
    main()
