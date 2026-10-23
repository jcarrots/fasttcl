"""Finite-temperature baths through the public generator and solver interfaces."""

import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.special import loggamma

from fasttcl import (
    OhmicBath,
    SampledBath,
    compile_plan,
    generator_series,
    prepare_model,
    solve,
)


def _spectral_correlation(times, *, omega_c, temperature):
    """Integrate J(w)[coth(w/2T) cos(wt)-i sin(wt)]/2 directly.

    The integration variable is w/omega_c. This oracle does not use the
    production bath's thermal series or special-function implementation.
    """
    times = np.asarray(times, dtype=float)

    def integrand(x):
        w = omega_c * x
        thermal = 2 * temperature / omega_c if x == 0 else x / np.tanh(w / (2 * temperature))
        return 0.5 * omega_c**2 * np.exp(-x) * (
            thermal * np.cos(w * times) - 1j * x * np.sin(w * times)
        )

    values, error = quad_vec(integrand, 0, np.inf, epsabs=2e-13, epsrel=2e-13)
    assert error < 2e-11
    return values


@pytest.fixture(scope="module")
def thermal_model_and_plan():
    model = prepare_model(np.diag([-.5, .5]), [[0, .4], [.4, 0]])
    return model, compile_plan(model)


@pytest.mark.parametrize("order", [2, 4, 6])
def test_thermal_generators_and_solutions_match_independent_baths(thermal_model_and_plan, order):
    model, plan = thermal_model_and_plan
    omega_c, temperature = 2., .8
    dt, n_steps = .075, 4
    settings = dict(dt=dt, n_steps=n_steps, order=order, coupling_strength=.5, plan=plan)
    bath = OhmicBath(omega_c, temperature=temperature)

    def independent(times):
        return _spectral_correlation(times, omega_c=omega_c, temperature=temperature)

    # The solver evaluates the bath on the midpoint grid; this same sampled
    # bath also covers every generator_series request on the coarser grid.
    samples = independent(np.arange(2*n_steps + 1) * dt/2)
    sampled = SampledBath(samples, dt/2)
    series = generator_series(model, bath, **settings)
    rho0 = np.array([[.6, .2j], [-.2j, .4]])
    result = solve(model, bath, rho0, **settings)

    for reference in (independent, sampled):
        expected_series = generator_series(model, reference, **settings)
        expected_result = solve(model, reference, rho0, **settings)
        for k in range(2, order + 1, 2):
            np.testing.assert_allclose(series.corrections[k], expected_series.corrections[k], rtol=3e-10, atol=3e-14)
            np.testing.assert_allclose(result.trajectories[k], expected_result.trajectories[k], rtol=0, atol=3e-13)

    for trajectory in result.trajectories.values():
        np.testing.assert_allclose(np.trace(trajectory, axis1=1, axis2=2), 1., rtol=0, atol=2e-13)
        np.testing.assert_allclose(trajectory, trajectory.conj().transpose(0, 2, 1), rtol=0, atol=2e-13)


def test_thermal_strength_powers_and_plan_reuse(thermal_model_and_plan):
    model, plan = thermal_model_and_plan
    settings = dict(dt=.075, n_steps=4, order=6, plan=plan)
    bath = OhmicBath(2., temperature=.8)
    weak = generator_series(model, bath, coupling_strength=.3, **settings)
    strong = generator_series(model, bath, coupling_strength=.6, **settings)
    cold = generator_series(model, OhmicBath(2.), coupling_strength=.3, **settings)
    trace = np.array([1, 0, 0, 1])
    transpose = [0, 2, 1, 3]
    for k in (2, 4, 6):
        assert np.max(np.abs(weak.corrections[k])) > 1e-15
        np.testing.assert_allclose(strong.corrections[k], 2**k * weak.corrections[k], rtol=2e-12, atol=1e-17)
        np.testing.assert_allclose(np.einsum("a,tab->tb", trace, weak.corrections[k]), 0, rtol=0, atol=2e-13)
        np.testing.assert_allclose(weak.corrections[k], weak.corrections[k][:, transpose][:, :, transpose].conj(), rtol=0, atol=2e-13)
    assert np.max(np.abs(weak.corrections[2] - cold.corrections[2])) > 1e-4


def test_thermal_pure_dephasing_matches_exact_gamma_function_solution():
    model = prepare_model(np.diag([-.5, .5]), np.diag([.5, -.5]))
    omega_c, temperature, strength = 2., .8, .4
    result = solve(model, OhmicBath(omega_c, temperature=temperature), .5*np.ones((2, 2)),
                   dt=.005, n_steps=160, order=4, coupling_strength=strength)
    # The Gaussian dephasing exponent is the independent exact spectral
    # integral (1/2) int J(w)coth(w/2T)(1-cos(wt))/w**2 dw. The thermal
    # infinite product is expressed here using log Gamma for stability.
    z = 1 + temperature / omega_c
    dephasing = .25*np.log1p((omega_c*result.times)**2) + loggamma(z) - np.real(loggamma(z + 1j*temperature*result.times))
    expected = .5*np.exp(1j*result.times - strength**2*dephasing)
    np.testing.assert_allclose(result.rho[:, 0, 1], expected, rtol=0, atol=1e-7)
    np.testing.assert_allclose(result.rho[:, [0, 1], [0, 1]], .5, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.generators.corrections[4], 0, rtol=0, atol=1e-15)
