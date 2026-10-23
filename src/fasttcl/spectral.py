"""Thermal scalar Gaussian baths defined by nonnegative spectral densities."""
from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import gammaln

from .baths import OhmicBath, _real_scalar


_NORMALIZATION = "unit C_B; spectral integral includes 1/2; coupling_strength supplied to solver"
_CORRELATION = "C_B(t)=1/2 integral J_B(w)[coth(w/(2*T))*cos(w*t)-i*sin(w*t)] dw"


def _times(times):
    if np.iscomplexobj(times):
        raise ValueError("bath times must be finite and real")
    values = np.asarray(times, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("bath times must be finite and real")
    return values


def _temperature(value):
    value = _real_scalar(value, "temperature")
    if value < 0:
        raise ValueError("temperature must be nonnegative and finite")
    return value


def _parameters(bath, *names):
    for name in names:
        object.__setattr__(bath, name, _real_scalar(getattr(bath, name), name, positive=True))
    object.__setattr__(bath, "temperature", _temperature(bath.temperature))
    for name in ("epsabs", "epsrel"):
        object.__setattr__(bath, name, _real_scalar(getattr(bath, name), name, positive=True))


def _checked_density(spectral_density, omega):
    value = spectral_density(float(omega))
    if (isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value)
            or not isinstance(value, Real)):
        raise ValueError("spectral_density must return a finite nonnegative real scalar")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError("spectral_density must return a finite nonnegative real scalar")
    return value


def _quad_checked(integrand, lower, upper, points, epsabs, epsrel):
    value, error, info = quad_vec(
        integrand, lower, upper, epsabs=epsabs, epsrel=epsrel,
        norm="max", points=points, full_output=True,
    )
    if not info.success or not np.isfinite(value).all() or not np.isfinite(error):
        raise RuntimeError("spectral quadrature did not converge: " + info.message)
    return value


def _integrate(spectral_density, times, temperature, omega_max, points,
               epsabs, epsrel, *, scale=1.):
    """Integrate all distinct absolute lags together, checking the variance too."""
    times = _times(times)
    if not times.size:
        return np.empty(times.shape, dtype=np.complex128)
    lags, indices = np.unique(np.concatenate(([0.], np.abs(times).ravel())),
                              return_inverse=True)

    def integrand(x):
        omega = scale * x
        density = _checked_density(spectral_density, omega)
        if density == 0:
            return np.zeros(lags.shape, dtype=np.complex128)
        vacuum_weight = .5 * density
        if temperature == 0:
            real_weight = vacuum_weight
        else:
            # Avoid overflowing coth at small w/T before multiplying by J(w).
            ratio = (.5 * omega) / temperature
            if ratio < 1e-4:
                real_weight = temperature * (density / omega) * (1 + ratio**2 / 3)
            else:
                real_weight = vacuum_weight / np.tanh(ratio)
        with np.errstate(over="ignore", invalid="ignore"):
            phase = omega * lags
            value = scale * (real_weight * np.cos(phase)
                             - 1j * vacuum_weight * np.sin(phase))
        if not np.isfinite(value).all():
            raise RuntimeError("spectral quadrature encountered a nonfinite integrand")
        return value

    scaled_points = np.asarray(points, dtype=float) / scale
    if np.isinf(omega_max):
        # The infinite-interval transform can round tiny positive frequencies
        # to zero near its finite endpoint. Keep integrable infrared
        # singularities on a finite interval, with unrestricted refinement.
        low = _quad_checked(integrand, 0., 1., scaled_points[scaled_points <= 1.],
                            epsabs/2, epsrel/2)
        high = _quad_checked(integrand, 1., np.inf, scaled_points[scaled_points >= 1.],
                             epsabs/2, epsrel/2)
        value = low + high
    else:
        value = _quad_checked(integrand, 0., omega_max / scale,
                              scaled_points, epsabs, epsrel)
    positive = value[indices[1:]].reshape(times.shape)
    return np.asarray(np.where(times < 0, positive.conj(), positive), dtype=np.complex128)


def _powerlaw_thermal(bath, times):
    """Thermal correction, with an endpoint substitution for sub-Ohmic spectra."""
    if not times.size:
        return np.empty(times.shape)
    lags, indices = np.unique(np.concatenate(([0.], np.abs(times).ravel())),
                              return_inverse=True)
    log_scale = np.log(bath.omega_c) + np.log(bath.temperature)

    def thermal_integrand(x, log_weight):
        omega = bath.omega_c * x
        ratio = omega / bath.temperature
        # ratio/(exp(ratio)-1) has limit one at zero; evaluate before
        # multiplying, including when x underflows after the substitution.
        if ratio < 1e-4:
            bose_ratio = 1 - .5*ratio + ratio**2/12
        elif ratio > 745:
            return np.zeros(lags.shape)
        elif ratio > 50:
            bose_ratio = ratio * np.exp(-ratio)
        else:
            bose_ratio = ratio / np.expm1(ratio)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            value = np.exp(log_weight) * bose_ratio * np.cos(omega * lags)
        if not np.isfinite(value).all():
            raise RuntimeError("spectral quadrature encountered a nonfinite integrand")
        return value

    def low_integrand(u):
        if bath.exponent_s >= 1:
            return thermal_integrand(u, log_scale + (bath.exponent_s-1)*np.log(u) - u)
        x = np.exp(np.log(u) / bath.exponent_s)
        return thermal_integrand(x, log_scale - np.log(bath.exponent_s) - x)

    def high_integrand(x):
        return thermal_integrand(x, log_scale + (bath.exponent_s-1)*np.log(x) - x)

    low = _quad_checked(low_integrand, 0., 1., (), bath.epsabs/2, bath.epsrel/2)
    high = _quad_checked(high_integrand, 1., np.inf,
                         (max(1., bath.exponent_s),), bath.epsabs/2, bath.epsrel/2)
    return (low + high)[indices[1:]].reshape(times.shape)


class SpectralDensityBath:
    r"""A thermal bath from a scalar callable nonnegative ``J_B(omega)``.

    ``omega_max`` is an explicit upper integration boundary. A finite value
    defines a truncated spectrum, with no implicit extrapolation. ``np.inf``
    declares that the user-supplied spectrum has an integrable ultraviolet
    tail. The thermal integral must also converge at zero frequency. Bare
    Drude-Lorentz spectra do not have finite C_B(0) and are unsuitable.

    The convention is C_B(t)=1/2 integral_0^omega_max J_B(w)
    [coth(w/(2*T))*cos(w*t)-i*sin(w*t)] dw, with hbar=k_B=1.
    At T=0, coth is replaced by one. Quadrature nodes are strictly positive;
    no user-supplied zero-frequency limit is required. Supply known peaks
    or discontinuities as ``points`` so adaptive quadrature resolves them.
    Failed convergence raises RuntimeError, rather than returning a cutoff
    approximation. Tolerances bound a shared maximum norm across time lags.
    """

    def __init__(self, spectral_density, *, temperature=0., omega_max,
                 points=(), epsabs=1e-10, epsrel=1e-9):
        if not callable(spectral_density):
            raise ValueError("spectral_density must be callable")
        if (isinstance(omega_max, (bool, np.bool_)) or np.iscomplexobj(omega_max)
                or not isinstance(omega_max, Real)):
            raise ValueError("omega_max must be positive and finite or +inf")
        omega_max = float(omega_max)
        if np.isnan(omega_max) or omega_max <= 0:
            raise ValueError("omega_max must be positive and finite or +inf")
        self.spectral_density = spectral_density
        self.temperature = _temperature(temperature)
        self.omega_max = omega_max
        self.epsabs = _real_scalar(epsabs, "epsabs", positive=True)
        self.epsrel = _real_scalar(epsrel, "epsrel", positive=True)
        try:
            points = tuple(_real_scalar(point, "points") for point in points)
        except TypeError as exc:
            raise ValueError("points must be a finite real sequence") from exc
        if any(point < 0 or point > omega_max for point in points):
            raise ValueError("points must lie within [0, omega_max]")
        self.points = tuple(sorted(set(points)))

    def correlation(self, times):
        return _integrate(self.spectral_density, times, self.temperature,
                          self.omega_max, self.points, self.epsabs, self.epsrel)

    @property
    def report(self):
        return {"bath": "thermal spectral density", "temperature": self.temperature,
                "spectral_density": "user-supplied nonnegative callable J_B(w)",
                "omega_max": self.omega_max, "points": list(self.points),
                "integration": "scipy.integrate.quad_vec; norm=max; checked convergence",
                "epsabs": self.epsabs, "epsrel": self.epsrel,
                "correlation": _CORRELATION, "normalization": _NORMALIZATION}


@dataclass(frozen=True)
class PowerLawBath:
    r"""Exponential-cutoff power-law spectrum at T>=0.

    J_B(w)=omega_c**(1-s)*w**s*exp(-w/omega_c). Positive ``exponent_s``
    gives sub-Ohmic (s<1), Ohmic (s=1), or super-Ohmic (s>1) baths whose
    thermal variance is finite. The zero-temperature correlation is exact;
    finite-temperature non-Ohmic correlations use adaptive quadrature.
    """

    omega_c: float = 10.
    exponent_s: float = 1.
    temperature: float = 0.
    epsabs: float = 1e-10
    epsrel: float = 1e-9

    def __post_init__(self):
        _parameters(self, "omega_c", "exponent_s")

    def spectral_density(self, omega):
        if omega == 0:
            return 0.
        return float(np.exp(np.log(self.omega_c)
                            + self.exponent_s * np.log(omega / self.omega_c)
                            - omega / self.omega_c))

    def correlation(self, times):
        times = _times(times)
        if self.exponent_s == 1:
            return OhmicBath(self.omega_c, self.temperature).correlation(times)
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = self.omega_c * times
            log_base = np.log(np.hypot(1., scaled)) + 1j * np.arctan(scaled)
            value = np.exp(gammaln(self.exponent_s + 1) + 2*np.log(self.omega_c)
                           - np.log(2.) - (self.exponent_s + 1)*log_base)
        if self.temperature > 0:
            value += _powerlaw_thermal(self, times)
        if not np.isfinite(value).all():
            raise RuntimeError("power-law correlation is outside floating-point range")
        return np.asarray(value, dtype=np.complex128)

    @property
    def report(self):
        method = ("exact OhmicBath correlation" if self.exponent_s == 1 else
                  "exact Gamma-function formula" if self.temperature == 0 else
                  "exact vacuum plus scipy.integrate.quad_vec thermal integral; infrared endpoint substitution; checked convergence")
        return {"bath": "exponential-cutoff power law", "omega_c": self.omega_c,
                "exponent_s": self.exponent_s, "temperature": self.temperature,
                "spectral_density": "J_B(w)=omega_c**(1-s)*w**s*exp(-w/omega_c)",
                "integration": method, "epsabs": self.epsabs, "epsrel": self.epsrel,
                "correlation": _CORRELATION, "normalization": _NORMALIZATION}


@dataclass(frozen=True)
class BrownianBath:
    r"""Thermal Brownian-oscillator spectrum with resonance omega_0.

    J_B(w)=2*damping*omega_0**3*w /
    [(omega_0**2-w**2)**2+damping**2*w**2]. The reorganization-energy
    scale is fixed to omega_0; the solver's coupling_strength supplies the
    global coupling. Overdamping is supported. Extremely narrow resonances
    below sqrt(machine epsilon) relative width are rejected because their
    spectral peak cannot be reliably resolved by this quadrature.
    The full quantum thermal integral is used, without a Matsubara cutoff.
    """

    omega_0: float = 1.
    damping: float = .2
    temperature: float = 0.
    epsabs: float = 1e-10
    epsrel: float = 1e-9

    def __post_init__(self):
        _parameters(self, "omega_0", "damping")
        if self.damping / self.omega_0 < np.sqrt(np.finfo(float).eps):
            raise ValueError("damping/omega_0 is too small for reliable Brownian quadrature; "
                             "use DiscreteBath for an unresolved isolated mode")

    def spectral_density(self, omega):
        ratio = omega / self.omega_0
        damping = self.damping / self.omega_0
        if ratio <= 1:
            return (2*self.damping*ratio
                    / ((1-ratio**2)**2 + (damping*ratio)**2))
        inverse = 1 / ratio
        return (2*self.damping*inverse**3
                / ((inverse**2-1)**2 + (damping*inverse)**2))

    def correlation(self, times):
        points = (self.omega_0, max(0., self.omega_0-self.damping),
                  self.omega_0+self.damping, self.damping,
                  self.omega_0 * min(1., self.omega_0/self.damping))
        return _integrate(self.spectral_density, times, self.temperature, np.inf,
                          points, self.epsabs, self.epsrel, scale=self.omega_0)

    @property
    def report(self):
        return {"bath": "Brownian oscillator", "omega_0": self.omega_0,
                "damping": self.damping, "temperature": self.temperature,
                "reorganization_energy_scale": self.omega_0,
                "spectral_density": "J_B(w)=2*damping*omega_0**3*w/((omega_0**2-w**2)**2+damping**2*w**2)",
                "integration": "scipy.integrate.quad_vec; dimensionless w/omega_0; resonance-aware points; checked convergence",
                "epsabs": self.epsabs, "epsrel": self.epsrel,
                "correlation": _CORRELATION, "normalization": _NORMALIZATION}


@dataclass(frozen=True, init=False)
class DrudeLorentzBath:
    r"""Explicitly ultraviolet-regularized thermal Drude-Lorentz bath.

    J_B(w)=2*gamma**2*w/(gamma**2+w**2)*exp(-w/uv_cutoff).
    The positive finite ``uv_cutoff`` is mandatory: the bare Drude-Lorentz
    spectrum has divergent C_B(0), which this solver cannot sample. The
    reorganization-energy scale before regularization is fixed to gamma.
    """

    gamma: float
    uv_cutoff: float
    temperature: float
    epsabs: float
    epsrel: float

    def __init__(self, gamma=1., *, uv_cutoff, temperature=0.,
                 epsabs=1e-10, epsrel=1e-9):
        for name, value in (("gamma", gamma), ("uv_cutoff", uv_cutoff),
                            ("temperature", temperature), ("epsabs", epsabs),
                            ("epsrel", epsrel)):
            object.__setattr__(self, name, value)
        _parameters(self, "gamma", "uv_cutoff")

    def spectral_density(self, omega):
        ratio = omega / self.gamma
        factor = ratio / (1 + ratio**2) if ratio <= 1 else (1/ratio) / (1+(1/ratio)**2)
        return 2*self.gamma*factor*np.exp(-omega/self.uv_cutoff)

    def correlation(self, times):
        return _integrate(self.spectral_density, times, self.temperature, np.inf,
                          (self.gamma, self.uv_cutoff), self.epsabs, self.epsrel,
                          scale=min(self.gamma, self.uv_cutoff))

    @property
    def report(self):
        return {"bath": "UV-regularized Drude-Lorentz", "gamma": self.gamma,
                "uv_cutoff": self.uv_cutoff, "temperature": self.temperature,
                "spectral_density": "J_B(w)=2*gamma**2*w/(gamma**2+w**2)*exp(-w/uv_cutoff)",
                "integration": "scipy.integrate.quad_vec; scaled frequency; explicit exponential UV cutoff; checked convergence",
                "epsabs": self.epsabs, "epsrel": self.epsrel,
                "correlation": _CORRELATION, "normalization": _NORMALIZATION}
