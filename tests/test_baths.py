"""Independent continuum checks for the unit-strength thermal Ohmic bath."""

import numpy as np
import pytest
from scipy.integrate import quad

from fasttcl import OhmicBath


def _spectral_correlation(omega_c, temperature, time):
    """Integrate the defining spectrum, resolving its thermal and cutoff scales."""
    def real_integrand(omega):
        thermal_weight = (
            2 * temperature if omega == 0
            else omega / np.tanh(omega / (2 * temperature))
        )
        return .5 * thermal_weight * np.exp(-omega / omega_c) * np.cos(omega * time)

    def imag_integrand(omega):
        return -.5 * omega * np.exp(-omega / omega_c) * np.sin(omega * time)

    boundaries = sorted({0., temperature, omega_c, 10 * omega_c, 80 * omega_c})
    real = sum(quad(real_integrand, left, right, epsabs=2e-12,
                    epsrel=2e-12, limit=250)[0]
               for left, right in zip(boundaries, boundaries[1:]))
    imag = quad(imag_integrand, 0., 80 * omega_c, epsabs=2e-12,
                epsrel=2e-12, limit=250)[0]
    return real + 1j * imag


@pytest.mark.parametrize("temperature", [.005, .7, 20.])
@pytest.mark.parametrize("time", [-1.3, -.2, 0., .2, 1.3])
def test_finite_temperature_matches_independent_spectral_quadrature(temperature, time):
    bath = OhmicBath(omega_c=2.5, temperature=temperature)
    expected = _spectral_correlation(2.5, temperature, time)
    np.testing.assert_allclose(bath.correlation(time), expected, rtol=3e-11, atol=3e-11)


def test_default_and_explicit_zero_temperature_preserve_original_formula_exactly():
    times = np.array([[-4., -.1, -0.], [0., .2, 3.]])
    expected = 2.5**2 / (2 * (1 + 1j * 2.5 * times)**2)
    np.testing.assert_array_equal(OhmicBath(2.5).correlation(times), expected)
    np.testing.assert_array_equal(OhmicBath(2.5, temperature=0).correlation(times), expected)
    assert OhmicBath().temperature == 0.


@pytest.mark.parametrize("temperature", [0., .7, 20.])
@pytest.mark.parametrize("times", [0.1, [-.3, 0., .3], [[-.2, .1], [0., 2.]],
                                    np.empty((2, 0))])
def test_correlation_preserves_input_shape_and_complex_dtype(temperature, times):
    actual = np.asarray(OhmicBath(temperature=temperature).correlation(times))
    assert actual.shape == np.asarray(times).shape
    assert actual.dtype == np.complex128
    assert np.isfinite(actual).all()


def test_stationary_hermitian_symmetry_and_temperature_independent_imaginary_part():
    times = np.array([0., 1e-12, .02, .3, 2., 100.])
    vacuum = OhmicBath(2.5).correlation(times)
    for temperature in (.005, .7, 20.):
        bath = OhmicBath(2.5, temperature=temperature)
        positive = bath.correlation(times)
        np.testing.assert_array_equal(bath.correlation(-times), positive.conj())
        np.testing.assert_array_equal(positive.imag, vacuum.imag)
        assert positive[0].imag == 0.
        assert positive[0].real > vacuum[0].real


@pytest.mark.parametrize("temperature", [1e-6, np.nextafter(0., 1.)])
def test_low_temperature_limit_approaches_vacuum(temperature):
    times = np.array([-10., -.1, 0., .1, 10.])
    actual = OhmicBath(2.5, temperature=temperature).correlation(times)
    np.testing.assert_allclose(actual, OhmicBath(2.5).correlation(times),
                               rtol=0., atol=1.7e-12)


def test_high_temperature_classical_limit_and_unit_normalization():
    omega_c = 2.5
    temperature = 1e5
    times = np.array([0., .1, .4, 2.])
    actual = OhmicBath(omega_c, temperature=temperature).correlation(times)
    classical = omega_c / (1 + (omega_c * times)**2)
    np.testing.assert_allclose(actual.real / temperature, classical, rtol=2e-10, atol=0.)


def test_high_temperature_does_not_overflow_when_correlation_is_representable():
    temperature = 1e155
    times = np.array([0., .1, .4, 2.])
    actual = OhmicBath(2.5, temperature=temperature).correlation(times)
    expected = 2.5 / (1 + (2.5 * times)**2)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual.real / temperature, expected, rtol=2e-14)


# Stored constants were evaluated with mpmath at 80 decimal digits from
# C_0 + T**2 Re polygamma(1, 1+T/wc+i*T*t).  They exercise long lags and a
# large T/wc without introducing a high-precision runtime dependency.
@pytest.mark.parametrize("omega_c,temperature,time,expected", [
    (2.5, .7, 1e4, 2.7999999669485713937569189695994837e-9
     - 3.999999987200000030719999934464e-13j),
    (2.5, 20., 1e4, 7.9999999871000000210124999654493126e-8
     - 3.999999987200000030719999934464e-13j),
    (2.5, 1e5, 0., 250000.00002604166666341145833478655 + 0j),
    (2.5, 1e5, .4, 124999.99999348958333374023437509083 - 1.5625j),
    (10., .1, 1e7, 9.9999999999949989999999983350001e-17
     - 9.9999999999999980000000000000003e-23j),
    (1., 1., 1e8, 9.99999999999999850000000000000025e-17
     - 9.9999999999999980000000000000003e-25j),
])
def test_extreme_parameters_against_stored_high_precision_values(omega_c, temperature,
                                                                time, expected):
    actual = OhmicBath(omega_c, temperature=temperature).correlation(time)
    np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=0.)


@pytest.mark.parametrize("temperature", [0., .7, np.float64(20.)])
def test_report_records_temperature_and_retains_unit_strength(temperature):
    bath = OhmicBath(2.5, temperature=temperature)
    assert bath.report["temperature"] == float(temperature)
    assert bath.report["omega_c"] == 2.5
    assert "unit C_B" in bath.report["normalization"]


@pytest.mark.parametrize("temperature", [-.1, np.nan, np.inf, -np.inf,
                                          True, np.bool_(False), 1j, 0j,
                                          np.complex64(.1), "1.0", None, [1.]])
def test_invalid_temperatures_are_rejected(temperature):
    with pytest.raises((TypeError, ValueError), match="temperature"):
        OhmicBath(temperature=temperature)


@pytest.mark.parametrize("times", [np.nan, np.inf, -np.inf,
                                    [0., np.nan], [0., np.inf],
                                    1j, 0j, [1 + 0j], np.array([.1j]),
                                    ["invalid"], [[0.], [0., 1.]]])
def test_nonfinite_or_nonreal_times_are_rejected(times):
    with pytest.raises((TypeError, ValueError)):
        OhmicBath(temperature=.7).correlation(times)
