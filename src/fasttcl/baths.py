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


def _trigamma(z):
    """Complex trigamma for Re(z)>=1, using recurrence and an asymptotic tail.

    Shifting the real part to 16 makes the first omitted Bernoulli term
    smaller than double-precision rounding. Work is bounded independently
    of temperature and time; no Matsubara-series cutoff is needed.
    """
    shifted = np.asarray(z, dtype=np.complex128).copy()
    correction = np.zeros_like(shifted)
    for _ in range(15):
        mask = shifted.real < 16.0
        if not np.any(mask):
            break
        inverse = np.where(mask, 1.0 / shifted, 0.0)
        correction += inverse**2
        shifted = shifted + mask
    inverse = 1.0 / shifted
    tail = inverse + 0.5 * inverse**2
    bernoulli = (1/6, -1/30, 1/42, -1/30, 5/66, -691/2730, 7/6, -3617/510)
    for k, coefficient in enumerate(bernoulli, start=1):
        tail += coefficient * inverse**(2*k + 1)
    return correction + tail


@dataclass(frozen=True)
class OhmicBath:
    r"""Ohmic bath with exponential cutoff and temperature T>=0.

    With hbar=k_B=1, J_B(w)=w exp(-w/wc) and
    C_B(t)=(1/2) integral_0^infinity J_B(w)
           * [coth(w/(2*T))*cos(w*t)-i*sin(w*t)] dw.
    The default T=0 gives wc^2/[2(1+i*wc*t)^2]. For T>0 the
    thermal correction is T^2 Re trigamma(1+T/wc+i*T*t).
    Coupling strength is deliberately absent: pass lambda as
    ``coupling_strength`` to ``generator_series`` or ``solve``.
    """
    omega_c: float = 10.0
    temperature: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "omega_c", _real_scalar(self.omega_c, "omega_c", positive=True))
        temperature = _real_scalar(self.temperature, "temperature")
        if temperature < 0:
            raise ValueError("temperature must be nonnegative and finite")
        object.__setattr__(self, "temperature", temperature)

    def correlation(self, times):
        if np.iscomplexobj(times):
            raise ValueError("bath times must be finite and real")
        times = np.asarray(times, dtype=float)
        if not np.isfinite(times).all():
            raise ValueError("bath times must be finite and real")
        vacuum = self.omega_c**2 / (2 * (1 + 1j*self.omega_c*times)**2)
        if self.temperature == 0:
            return vacuum
        z = 1 + self.temperature/self.omega_c + 1j*self.temperature*times
        # Scale in two steps: T**2 can overflow even when C(t) is finite.
        return vacuum + self.temperature * (self.temperature * _trigamma(z).real)

    @property
    def report(self):
        return {"bath": "zero-temperature Ohmic" if self.temperature == 0 else "finite-temperature Ohmic",
                "omega_c": float(self.omega_c), "temperature": float(self.temperature),
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
