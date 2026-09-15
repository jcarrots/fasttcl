"""Unit-strength stationary bath correlations; lambda is supplied to the solver."""
from dataclasses import dataclass
from numbers import Real
import numpy as np


def _real_scalar(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


@dataclass(frozen=True)
class OhmicBath:
    r"""Zero-temperature Ohmic bath: C_B(t)=wc^2/[2(1+i*wc*t)^2].

    This corresponds to J_B(w)=w exp(-w/wc) and
    C_B(t)=(1/2) integral_0^infinity J_B(w) exp(-i*w*t) dw.
    Coupling strength is deliberately absent: pass lambda as
    ``coupling_strength`` to ``generator_series`` or ``solve``.
    """
    omega_c: float = 10.0

    def __post_init__(self):
        object.__setattr__(self, "omega_c", _real_scalar(self.omega_c, "omega_c", positive=True))

    def correlation(self, times):
        times = np.asarray(times, dtype=float)
        return self.omega_c**2 / (2 * (1 + 1j*self.omega_c*times)**2)

    @property
    def report(self):
        return {"bath": "zero-temperature Ohmic", "omega_c": float(self.omega_c),
                "normalization": "unit C_B; spectral integral includes 1/2"}


class SampledBath:
    """A unit-strength C_B on nonnegative uniform times, starting at zero.

    Negative samples use C_B(-t)=C_B(t)*. Evaluation is restricted to the
    supplied grid: interpolation and extrapolation are never implicit.
    For ``solve(..., dt=dt, n_steps=N)`` supply 2*N+1 samples at dt/2.
    """
    def __init__(self, correlation_positive, dt):
        dt = _real_scalar(dt, "dt", positive=True)
        values = np.array(correlation_positive, dtype=np.complex128, copy=True)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError("correlation_positive must be a nonempty finite vector")
        if abs(values[0].imag) > 1e-12*max(1, abs(values[0])):
            raise ValueError("stationary Hermitian bath requires real C_B(0)")
        values.setflags(write=False)
        self.values = values
        self.dt = float(dt)

    def correlation(self, times):
        t = np.asarray(times, dtype=float)
        if not np.isfinite(t).all():
            raise ValueError("bath sample times must be finite")
        positions = np.abs(t)/self.dt
        rounded = np.rint(positions)
        if np.any(np.abs(positions-rounded) > 1e-8):
            raise ValueError("requested times are outside the supplied bath grid")
        if np.any(rounded >= len(self.values)):
            raise ValueError("bath samples do not cover the requested time interval")
        values = self.values[rounded.astype(np.int64)]
        return np.where(t < 0, values.conj(), values)

    @property
    def report(self):
        return {"bath": "sampled unit C_B", "dt": self.dt,
                "samples": len(self.values), "negative_times": "complex conjugate"}
