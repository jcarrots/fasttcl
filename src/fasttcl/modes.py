"""Discrete oscillator baths and user-supplied exponential correlations."""

from dataclasses import dataclass

import numpy as np

from .baths import _real_scalar


def _vector(value, name, *, complex_values=False):
    """Copy a finite numeric vector without silently discarding imaginary parts."""
    try:
        raw = np.asarray(value)
        allowed = "iufc" if complex_values else "iuf"
        if raw.ndim != 1 or not raw.size or raw.dtype.kind not in allowed:
            raise ValueError
        result = np.array(raw, dtype=np.complex128 if complex_values else float,
                          copy=True)
        if not np.isfinite(result).all():
            raise ValueError
    except (TypeError, ValueError, OverflowError) as error:
        qualifier = "complex" if complex_values else "real"
        raise ValueError(f"{name} must be a nonempty finite {qualifier} vector") from error
    result.setflags(write=False)
    return result


def _times(value):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError
        result = np.asarray(raw, dtype=float)
        if not np.isfinite(result).all():
            raise ValueError
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("bath times must be finite and real") from error
    return result


@dataclass(frozen=True, eq=False)
class DiscreteBath:
    r"""A finite set of thermal harmonic oscillators, with hbar=k_B=1.

    ``frequencies`` are strictly positive and ``weights`` are nonnegative,
    relative spectral weights in J_B(w)=sum_j weights_j delta(w-w_j).
    The correlation is

    C_B(t) = (1/2) sum_j weights_j
             * [(n_j+1) exp(-i*w_j*t) + n_j exp(i*w_j*t)],

    where n_j=1/(exp(w_j/temperature)-1), or zero at zero temperature.
    Supply the overall coupling lambda separately to the solver through
    ``coupling_strength``. Input vectors are copied and made read-only.
    """

    frequencies: np.ndarray
    weights: np.ndarray
    temperature: float = 0.0

    def __post_init__(self):
        frequencies = _vector(self.frequencies, "frequencies")
        weights = _vector(self.weights, "weights")
        if frequencies.shape != weights.shape:
            raise ValueError("frequencies and weights must have the same shape")
        if np.any(frequencies <= 0):
            raise ValueError("frequencies must be strictly positive")
        if np.any(weights < 0):
            raise ValueError("weights must be nonnegative")
        temperature = _real_scalar(self.temperature, "temperature")
        if temperature < 0:
            raise ValueError("temperature must be nonnegative and finite")
        thermal_weights = np.zeros_like(weights)
        if temperature > 0:
            with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
                x = frequencies / temperature
                occupation = np.exp(-x) / (-np.expm1(-x))
                # A zero-weight mode contributes zero even if its occupation
                # is too large to represent in double precision.
                np.multiply(weights, occupation, out=thermal_weights, where=weights > 0)
        if not np.isfinite(thermal_weights).all():
            raise ValueError("thermal mode weights must be finite; bath variance is not representable")
        with np.errstate(over="ignore", invalid="ignore"):
            origin = np.sum(0.5 * weights + thermal_weights)
        if not np.isfinite(origin):
            raise ValueError("discrete bath C_B(0) must be finite")
        thermal_weights.setflags(write=False)
        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "_thermal_weights", thermal_weights)

    def correlation(self, times):
        t = _times(times)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            angles = np.abs(t)[..., None] * self.frequencies
            # Writing the thermal correction as a real term avoids cancellation
            # of large thermal imaginary parts at high temperature.
            values = np.sum(0.5 * self.weights * np.exp(-1j * angles)
                            + self._thermal_weights * np.cos(angles), axis=-1)
        if not np.isfinite(values).all():
            raise ValueError("discrete bath correlation is not finite at the requested times")
        result = np.where(t < 0, values.conj(), values)
        return result[()] if t.ndim == 0 else result

    @property
    def report(self):
        return {"bath": "discrete thermal oscillators", "modes": len(self.frequencies),
                "frequencies": self.frequencies.tolist(), "weights": self.weights.tolist(),
                "temperature": self.temperature,
                "normalization": "unit C_B; spectral sum includes 1/2; weights are relative"}


@dataclass(frozen=True, eq=False)
class ExponentialBath:
    r"""A user-supplied expansion C_B(t>=0)=sum_j c_j exp(-nu_j*t).

    ``coefficients`` and ``rates`` are finite complex vectors of equal length.
    Re(nu_j)>=0 permits both decaying and undamped terms. Negative times use
    C_B(-t)=C_B(t)*. The sum of coefficients must be real and nonnegative
    with imaginary rounding error allowed up to a relative tolerance of
    1e-12 (with absolute floor 1e-12). Accepted imaginary rounding error
    at the origin is removed from C_B(0).

    This general expansion does not guarantee a positive-semidefinite
    correlation or the thermal KMS condition. It does not assign a
    temperature automatically: users must supply physically valid expansion
    coefficients for their intended bath. Overall coupling lambda is supplied
    separately to the solver. Input vectors are copied and made read-only.
    """

    coefficients: np.ndarray
    rates: np.ndarray

    def __post_init__(self):
        coefficients = _vector(self.coefficients, "coefficients", complex_values=True)
        rates = _vector(self.rates, "rates", complex_values=True)
        if coefficients.shape != rates.shape:
            raise ValueError("coefficients and rates must have the same shape")
        if np.any(rates.real < 0):
            raise ValueError("rates must have nonnegative real parts")
        with np.errstate(over="ignore", invalid="ignore"):
            origin = np.sum(coefficients)
        if not np.isfinite(origin):
            raise ValueError("sum of coefficients must be finite")
        tolerance = 1e-12 * max(1.0, abs(origin.real))
        if abs(origin.imag) > tolerance or origin.real < 0:
            raise ValueError("sum of coefficients must give real nonnegative C_B(0)")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "rates", rates)
        object.__setattr__(self, "_origin", float(origin.real))

    def correlation(self, times):
        t = _times(times)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            values = np.sum(self.coefficients * np.exp(-np.abs(t)[..., None] * self.rates),
                            axis=-1)
        values = np.where(t == 0, self._origin, values)
        if not np.isfinite(values).all():
            raise ValueError("exponential bath correlation is not finite at the requested times")
        result = np.where(t < 0, values.conj(), values)
        return result[()] if t.ndim == 0 else result

    @property
    def report(self):
        return {"bath": "user-supplied exponential expansion", "terms": len(self.rates),
                "coefficients_real": self.coefficients.real.tolist(),
                "coefficients_imag": self.coefficients.imag.tolist(),
                "rates_real": self.rates.real.tolist(), "rates_imag": self.rates.imag.tolist(),
                "negative_times": "complex conjugate",
                "normalization": "user-supplied unit C_B; lambda is separate",
                "temperature": "not assigned automatically",
                "physical_validity": "positive semidefiniteness and thermal KMS are not guaranteed"}
