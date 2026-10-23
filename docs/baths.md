# Bath definitions

Pass any bath below to `solve` or `generator_series`. The built-in thermal
baths use hbar=kB=1 and the convention

```
C_B(t) = (1/2) integral_0^infinity J_B(w)
         * [coth(w/(2*T))*cos(w*t) - i*sin(w*t)] dw.
```

At `T=0`, replace `coth` by one. `coupling_strength` remains the separate
lambda in `H_int=lambda A tensor B`. The formulas below define the reference
spectral amplitudes; the physical spectrum is `lambda**2 * J_B`.
Temperatures are nonnegative, frequencies and damping parameters are positive,
and negative times obey `C_B(-t)=conj(C_B(t))`.

## Ohmic and power-law continua

```python
from fasttcl import OhmicBath, PowerLawBath

ohmic = OhmicBath(omega_c=10.0, temperature=0.5)
sub_ohmic = PowerLawBath(omega_c=10.0, exponent_s=0.5, temperature=0.5)
super_ohmic = PowerLawBath(omega_c=10.0, exponent_s=3.0, temperature=0.5)
```

`PowerLawBath` defines
`J_B(w)=omega_c**(1-s) * w**s * exp(-w/omega_c)`, with `s>0`.
At zero temperature its correlation is
`Gamma(s+1)*omega_c**2 / [2*(1+i*omega_c*t)**(s+1)]`.
The Ohmic case `s=1` uses `OhmicBath`, including its analytic finite-temperature
correlation. Other exponents at finite temperature use adaptive spectral
quadrature, including the integrable low-frequency contribution for `s<1`.

## Brownian oscillator continuum

```python
from fasttcl import BrownianBath

bath = BrownianBath(omega_0=2.0, damping=0.3, temperature=0.5)
```

The reference spectrum is

```
J_B(w) = 2*damping*omega_0**3*w
         / [(omega_0**2-w**2)**2 + damping**2*w**2].
```

This sets the conventional reorganization-energy parameter to `omega_0`;
the solver's `coupling_strength**2` scales its amplitude. The parameter
`damping` is the coefficient of `w` in the oscillator denominator.
The spectrum has a broadened resonance and an integrable high-frequency
tail. Its correlation uses the full thermal spectral integral, with
quadrature breakpoints near the resonance. It supports the underdamped regime
`damping < 2*omega_0` and overdamping. Relative widths
`damping/omega_0 < sqrt(machine epsilon)` are rejected because the peak
cannot be reliably resolved in double precision; `DiscreteBath` represents
an isolated undamped line. Long-time or strongly overdamped integrals may
need explicitly relaxed quadrature tolerances to converge.

## Drude-Lorentz with an explicit UV cutoff

```python
from fasttcl import DrudeLorentzBath

bath = DrudeLorentzBath(gamma=1.0, uv_cutoff=20.0, temperature=0.5)
```

This class defines the UV-regularized spectrum

```
J_B(w) = 2*gamma**2*w/(gamma**2+w**2) * exp(-w/uv_cutoff).
```

`gamma` is the Drude relaxation frequency; the reference reorganization-energy
parameter before UV regularization is `gamma`. `uv_cutoff` has no default.
The bare Drude-Lorentz tail is proportional to `1/w` and gives a divergent
`C_B(0)`. FastTCL's current sampled-correlation solver requires finite `C_B(0)`.
The extra exponential cutoff is part of this model and is recorded in
`bath.report`. Changing it changes the physical spectrum.

## Discrete thermal modes

```python
from fasttcl import DiscreteBath

bath = DiscreteBath(frequencies=[1.0, 2.5], weights=[0.8, 0.2], temperature=0.5)
```

Here `J_B(w)=sum_j weights[j]*delta(w-frequencies[j])`. Weights are
nonnegative integrated spectral weights, so

```
C_B(t) = (1/2) sum_j weights[j]
         * [(n_j+1)*exp(-i*w_j*t) + n_j*exp(+i*w_j*t)],
n_j = 1/expm1(w_j/T).
```

At `T=0`, all `n_j` are zero. An isolated thermal mode is evaluated exactly;
its finite set of modes can produce persistent oscillations and recurrences.

## Supplied exponential correlations

```python
from fasttcl import ExponentialBath

bath = ExponentialBath(coefficients=[0.6, 0.4], rates=[0.3+1.2j, 0.3-1.2j])
```

For `t>=0`, this represents `C_B(t)=sum_j coefficients[j]*exp(-rates[j]*t)`.
The real parts of the rates must be nonnegative, and `sum(coefficients)` must
be real and nonnegative. Coefficients already have correlation units and
include any spectral factor `1/2`. This supports supplied exponential fits,
finite HEOM correlation expansions, and classical exponential noise.
It does not infer a temperature or establish spectral positivity or thermal
detailed balance; those properties belong to the supplied expansion.

## Custom spectral densities

```python
import numpy as np
from fasttcl import SpectralDensityBath

bath = SpectralDensityBath(
    lambda w: w*np.exp(-w/3.0),
    temperature=0.5,
    omega_max=60.0,
    points=[3.0],
    epsabs=1e-10,
    epsrel=1e-9,
)
```

The callable accepts a scalar positive frequency and returns a finite,
nonnegative real spectral density. `omega_max` is required: a finite value
explicitly truncates the integral; `np.inf` requests an integrable infinite
tail. `points` supplies frequency breakpoints for narrow peaks or other
spectral features. Quadrature does not require evaluating a singular thermal
integrand at zero itself.

The numerical continua use adaptive vector quadrature with absolute and
relative tolerances `epsabs` and `epsrel`. A failure to converge raises an
error. Check sensitivity to tolerances and, for finite `omega_max`, the
integration endpoint. Very narrow custom features need suitable breakpoints.
The tolerances use the maximum norm across requested lags, including zero;
`epsrel` does not guarantee relative accuracy for a very small long-time tail.
`bath.report` records the integration convention and settings. For repeated
calculations on the same grid, precompute these numerical correlations into
`SampledBath`; `solve` needs `2*N+1` samples at spacing `dt/2` for `N` steps.

All these choices supply a single scalar correlation for the model's one
Hermitian coupling operator. Combining spectra for this same operator can
be done in a custom spectral callable. Multiple distinct system coupling
operators require a separate extension of the generator.

For background definitions of Brownian and Drude-Lorentz spectra, see the
[QuTiP bosonic environment documentation](https://qutip.readthedocs.io/en/stable/guide/environments/bosonic.html).
FastTCL's factor `1/2` and the reference spectral amplitudes are defined above.
