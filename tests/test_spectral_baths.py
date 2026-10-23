"""Independent spectral-integral checks for the non-Ohmic bath models."""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gamma, zeta

from fasttcl import OhmicBath
from fasttcl.spectral import (
    BrownianBath,
    DrudeLorentzBath,
    PowerLawBath,
    SpectralDensityBath,
)


def _scalar_spectral_integral(spectrum, temperature, time, boundaries):
    """Scalar QUADPACK reference, independent of production vector quadrature."""
    def real_integrand(omega):
        density = spectrum(omega)
        thermal = 1. if temperature == 0 else 1. / np.tanh(omega / (2 * temperature))
        return .5 * density * thermal * np.cos(omega * time)

    def imag_integrand(omega):
        return -.5 * spectrum(omega) * np.sin(omega * time)

    result = 0j
    for left, right in zip(boundaries, boundaries[1:]):
        if np.isinf(right) and time != 0:
            def real_amplitude(omega):
                thermal = 1. if temperature == 0 else 1. / np.tanh(omega / (2 * temperature))
                return .5 * spectrum(omega) * thermal

            real = quad(real_amplitude, left, right, weight="cos", wvar=abs(time),
                        epsabs=2e-10, limit=500, limlst=200)[0]
            imag = -math.copysign(1., time) * quad(
                lambda omega: .5*spectrum(omega), left, right, weight="sin",
                wvar=abs(time), epsabs=2e-10, limit=500, limlst=200)[0]
        else:
            real = quad(real_integrand, left, right, epsabs=2e-10,
                        epsrel=2e-10, limit=500)[0]
            imag = quad(imag_integrand, left, right, epsabs=2e-10,
                        epsrel=2e-10, limit=500)[0]
        result += real + 1j * imag
    return result


def test_finite_band_spectrum_matches_exact_integral():
    cutoff = 2.7
    bath = SpectralDensityBath(lambda omega: omega, omega_max=cutoff)
    times = np.array([[-1.3, -.2, 0.], [.2, .7, 1.3]])
    expected = np.empty(times.shape, dtype=np.complex128)
    nonzero = times != 0
    expected[~nonzero] = cutoff**2 / 4
    x = cutoff * times[nonzero]
    expected[nonzero] = (np.exp(-1j*x) * (1 + 1j*x) - 1) / (2 * times[nonzero]**2)
    actual = bath.correlation(times)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("temperature", [0., .8])
def test_custom_spectrum_with_internal_edge_matches_scalar_quadrature(temperature):
    cutoff, edge = 4., 1.1

    def spectrum(omega):
        return omega**2 * (1. if omega < edge else .15)

    bath = SpectralDensityBath(spectrum, temperature=temperature,
                               omega_max=cutoff, points=(edge,))
    times = np.array([-.7, 0., .3, .7])
    expected = [_scalar_spectral_integral(spectrum, temperature, t,
                                         [0., edge, cutoff]) for t in times]
    np.testing.assert_allclose(bath.correlation(times), expected, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("exponent", [.35, 1., 2., 3.5])
def test_power_law_zero_temperature_matches_gamma_transform(exponent):
    omega_c = 2.3
    times = np.array([[-3., -.1, 0.], [.1, .4, 3.]])
    bath = PowerLawBath(omega_c=omega_c, exponent_s=exponent)
    expected = .5 * omega_c**2 * gamma(exponent + 1) / (1 + 1j*omega_c*times)**(exponent + 1)
    np.testing.assert_allclose(bath.correlation(times), expected, rtol=5e-13, atol=1e-14)


@pytest.mark.parametrize("temperature", [0., .02, .7, 20.])
def test_power_law_s_one_matches_existing_ohmic_bath(temperature):
    times = np.array([-10., -.2, 0., .2, 10.])
    actual = PowerLawBath(omega_c=2.5, exponent_s=1., temperature=temperature).correlation(times)
    expected = OhmicBath(omega_c=2.5, temperature=temperature).correlation(times)
    np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=2e-12)


@pytest.mark.parametrize("exponent", [.35, 2.5])
def test_power_law_finite_temperature_resolves_integrable_infrared_endpoint(exponent):
    omega_c, temperature = 1.7, .6
    bath = PowerLawBath(omega_c=omega_c, exponent_s=exponent, temperature=temperature)

    def spectrum(omega):
        return omega_c**(1-exponent) * omega**exponent * np.exp(-omega/omega_c)

    times = np.array([0., .2, .8])
    expected = [_scalar_spectral_integral(spectrum, temperature, t,
                                         [0., temperature, omega_c, 10*omega_c, 80*omega_c])
                for t in times]
    np.testing.assert_allclose(bath.correlation(times), expected, rtol=2e-8, atol=2e-8)


@pytest.mark.parametrize("exponent", [.1, .35, 2.5])
def test_thermal_power_law_variance_matches_exact_real_hurwitz_zeta(exponent):
    omega_c, temperature = 1.7, .6
    expected = gamma(exponent+1) * (
        omega_c**2/2 + omega_c**(1-exponent) * temperature**(exponent+1)
        * zeta(exponent+1, 1+temperature/omega_c))
    actual = PowerLawBath(omega_c=omega_c, exponent_s=exponent,
                          temperature=temperature).correlation(0.)
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-8)


@pytest.mark.parametrize("exponent", [.1, .35])
@pytest.mark.parametrize("points", [(), (.2, 1., 1.7)])
def test_generic_infinite_spectrum_resolves_subohmic_infrared_variance(exponent, points):
    omega_c, temperature = 1.7, .6

    def spectrum(omega):
        return omega_c**(1-exponent) * omega**exponent * np.exp(-omega/omega_c)

    expected = gamma(exponent+1) * (
        omega_c**2/2 + omega_c**(1-exponent) * temperature**(exponent+1)
        * zeta(exponent+1, 1+temperature/omega_c))
    bath = SpectralDensityBath(spectrum, temperature=temperature,
                               omega_max=np.inf, points=points)
    np.testing.assert_allclose(bath.correlation(0.), expected, rtol=2e-9, atol=2e-10)


@pytest.mark.parametrize("temperature", [0., .7])
def test_brownian_spectrum_matches_scalar_integral(temperature):
    omega_0, damping = 2.3, .4
    bath = BrownianBath(omega_0=omega_0, damping=damping, temperature=temperature)

    def spectrum(omega):
        return 2*damping*omega_0**3*omega / ((omega_0**2-omega**2)**2+damping**2*omega**2)

    times = np.array([0., .25])
    boundaries = [0., omega_0-damping, omega_0, omega_0+damping, 10*omega_0, np.inf]
    expected = [_scalar_spectral_integral(spectrum, temperature, t, boundaries) for t in times]
    np.testing.assert_allclose(bath.correlation(times), expected, rtol=2e-8, atol=2e-8)


def test_narrow_brownian_peak_is_resolved_on_infinite_frequency_domain():
    omega_0, damping, temperature = 2.3, 1e-4, .7
    bath = BrownianBath(omega_0=omega_0, damping=damping, temperature=temperature)

    def spectrum(omega):
        return 2*damping*omega_0**3*omega / ((omega_0**2-omega**2)**2+damping**2*omega**2)

    boundaries = [0., omega_0-100*damping, omega_0-damping, omega_0,
                  omega_0+damping, omega_0+100*damping, 10*omega_0, np.inf]
    times = np.array([0., .25])
    expected = [_scalar_spectral_integral(spectrum, temperature, t, boundaries) for t in times]
    np.testing.assert_allclose(bath.correlation(times), expected, rtol=2e-7, atol=2e-8)
    assert bath.correlation(0.).real > 1.


def test_overdamped_brownian_spectrum_resolves_separated_frequency_scales():
    omega_0, damping, temperature = 2.3, 20., .7
    bath = BrownianBath(omega_0=omega_0, damping=damping, temperature=temperature,
                        epsabs=1e-7, epsrel=1e-7)

    def spectrum(omega):
        return 2*damping*omega_0**3*omega / ((omega_0**2-omega**2)**2+damping**2*omega**2)

    boundaries = [0., omega_0**2/damping, omega_0, damping, 10*damping, np.inf]
    times = np.array([0., .25])
    expected = [_scalar_spectral_integral(spectrum, temperature, t, boundaries) for t in times]
    np.testing.assert_allclose(bath.correlation(times), expected, rtol=2e-8, atol=2e-8)


@pytest.mark.parametrize("relative_width", [1e-16, 1e-20])
def test_unresolved_brownian_resonance_is_rejected(relative_width):
    with pytest.raises(ValueError, match="too small.*quadrature"):
        BrownianBath(omega_0=1., damping=relative_width)


@pytest.mark.parametrize("temperature", [0., .7])
def test_regularized_drude_lorentz_matches_scalar_integral(temperature):
    gamma_rate, cutoff = .8, 5.
    bath = DrudeLorentzBath(gamma=gamma_rate, uv_cutoff=cutoff, temperature=temperature)

    def spectrum(omega):
        return 2*gamma_rate**2*omega / (gamma_rate**2+omega**2) * np.exp(-omega/cutoff)

    times = np.array([-.7, 0., .3, .7])
    expected = [_scalar_spectral_integral(spectrum, temperature, t,
                                         [0., gamma_rate, cutoff, 10*cutoff, 80*cutoff])
                for t in times]
    np.testing.assert_allclose(bath.correlation(times), expected, rtol=2e-9, atol=2e-9)
    assert bath.report["uv_cutoff"] == cutoff


def test_drude_lorentz_requires_explicit_uv_cutoff_and_records_it():
    with pytest.raises(TypeError, match="uv_cutoff"):
        DrudeLorentzBath()
    low_cutoff = DrudeLorentzBath(gamma=.8, uv_cutoff=3.)
    high_cutoff = DrudeLorentzBath(gamma=.8, uv_cutoff=8.)
    assert high_cutoff.correlation(0.).real > low_cutoff.correlation(0.).real
    assert low_cutoff.report["uv_cutoff"] == 3.
    assert high_cutoff.report["uv_cutoff"] == 8.


@pytest.mark.parametrize("bath", [
    SpectralDensityBath(lambda omega: omega*np.exp(-omega), omega_max=np.inf, temperature=.7),
    PowerLawBath(omega_c=1.5, exponent_s=.6, temperature=.7),
    BrownianBath(omega_0=1.5, damping=.3, temperature=.7),
    DrudeLorentzBath(gamma=.8, uv_cutoff=4., temperature=.7),
])
def test_all_baths_preserve_shape_dtype_and_hermitian_symmetry(bath):
    times = np.array([[0., .02], [.03, .07]])
    positive = np.asarray(bath.correlation(times))
    assert positive.shape == times.shape
    assert positive.dtype == np.complex128
    assert np.isfinite(positive).all()
    assert positive[0, 0].real > 0.
    assert positive[0, 0].imag == 0.
    np.testing.assert_allclose(bath.correlation(-times), positive.conj(), rtol=1e-11, atol=1e-11)
    assert np.asarray(bath.correlation(.02)).shape == ()
    assert np.asarray(bath.correlation(np.empty((2, 0)))).shape == (2, 0)
    assert bath.report["temperature"] == .7


@pytest.mark.parametrize("factory", [
    lambda **kwargs: SpectralDensityBath(lambda omega: omega, omega_max=2., **kwargs),
    PowerLawBath,
    BrownianBath,
    lambda **kwargs: DrudeLorentzBath(uv_cutoff=4., **kwargs),
])
@pytest.mark.parametrize("temperature", [-.1, np.nan, np.inf, True, 0j, "1.0"])
def test_invalid_temperatures_rejected(factory, temperature):
    with pytest.raises((TypeError, ValueError), match="temperature"):
        factory(temperature=temperature)


@pytest.mark.parametrize("factory,parameter", [
    (PowerLawBath, "omega_c"),
    (PowerLawBath, "exponent_s"),
    (BrownianBath, "omega_0"),
    (BrownianBath, "damping"),
    (lambda **kwargs: DrudeLorentzBath(uv_cutoff=3., **kwargs), "gamma"),
    (DrudeLorentzBath, "uv_cutoff"),
])
@pytest.mark.parametrize("value", [0., -.1, np.nan, np.inf, True, 1j, "2"])
def test_builtin_scale_and_exponent_validation(factory, parameter, value):
    with pytest.raises((TypeError, ValueError), match=parameter):
        factory(**{parameter: value})


@pytest.mark.parametrize("omega_max", [0., -.1, np.nan, -np.inf, True, 1j, "2"])
def test_generic_frequency_limit_validation(omega_max):
    with pytest.raises((TypeError, ValueError), match="omega_max"):
        SpectralDensityBath(lambda omega: omega, omega_max=omega_max)


def test_generic_frequency_limit_is_required():
    with pytest.raises(TypeError, match="omega_max"):
        SpectralDensityBath(lambda omega: omega)


@pytest.mark.parametrize("parameter", ["epsabs", "epsrel"])
@pytest.mark.parametrize("value", [0., -.1, np.nan, np.inf, True, 1j, "1e-6"])
def test_quadrature_tolerances_are_strict_positive_real_scalars(parameter, value):
    factories = [lambda **kwargs: SpectralDensityBath(lambda omega: omega, omega_max=2., **kwargs),
                 PowerLawBath, BrownianBath,
                 lambda **kwargs: DrudeLorentzBath(uv_cutoff=4., **kwargs)]
    for factory in factories:
        with pytest.raises((TypeError, ValueError), match=parameter):
            factory(**{parameter: value})


@pytest.mark.parametrize("points", [None, [.1, np.nan], [np.inf], [-.1], [2.1], [True], [1j]])
def test_generic_quadrature_points_are_finite_and_within_the_frequency_domain(points):
    with pytest.raises((TypeError, ValueError), match="points"):
        SpectralDensityBath(lambda omega: omega, omega_max=2., points=points)


@pytest.mark.parametrize("density", [-1., np.nan, np.inf, 1j, True, [1.]])
def test_invalid_custom_spectral_values_rejected(density):
    with pytest.raises((TypeError, ValueError)):
        SpectralDensityBath(lambda omega: density, omega_max=2.).correlation(.1)


@pytest.mark.parametrize("spectrum", [None, 1., [1.]])
def test_generic_spectrum_must_be_callable(spectrum):
    with pytest.raises((TypeError, ValueError)):
        SpectralDensityBath(spectrum, omega_max=2.)


@pytest.mark.parametrize("times", [np.nan, np.inf, [0., np.nan], 0j, [1+0j]])
def test_nonfinite_and_complex_times_are_rejected(times):
    baths = [SpectralDensityBath(lambda omega: omega, omega_max=2.),
             PowerLawBath(exponent_s=.6), BrownianBath(),
             DrudeLorentzBath(uv_cutoff=4.)]
    for bath in baths:
        with pytest.raises((TypeError, ValueError)):
            bath.correlation(times)
