"""Independent finite-mode and supplied-correlation bath checks."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from fasttcl.modes import DiscreteBath, ExponentialBath


def test_discrete_zero_temperature_matches_finite_spectral_sum():
    frequencies = [0.5, 2.0, 7.0]
    weights = [3.0, 0.0, 0.4]
    times = np.array([[-2.0, -0.1, 0.0], [0.3, 1.0, 4.0]])
    expected = sum(weight / 2 * (np.cos(frequency * times)
                                - 1j * np.sin(frequency * times))
                   for frequency, weight in zip(frequencies, weights))
    bath = DiscreteBath(frequencies, weights)
    assert bath.temperature == 0.0
    np.testing.assert_allclose(bath.correlation(times), expected, rtol=2e-15, atol=1e-15)


@pytest.mark.parametrize("temperature", [0.03, 0.7, 20.0])
def test_discrete_thermal_modes_match_independent_hyperbolic_formula(temperature):
    frequencies = [0.1, 1.3, 3.0]
    weights = [0.4, 2.0, 0.8]
    times = np.array([-2.0, -0.2, 0.0, 0.3, 1.8])
    expected = sum(weight / 2 * (np.cos(frequency * times)
                                / np.tanh(frequency / (2 * temperature))
                                - 1j * np.sin(frequency * times))
                   for frequency, weight in zip(frequencies, weights))
    actual = DiscreteBath(frequencies, weights, temperature).correlation(times)
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=2e-15)


@pytest.mark.parametrize("frequency,temperature", [(0.3, 0.7), (2.0, 1.0), (4.0, 20.0)])
def test_discrete_single_mode_has_thermal_detailed_balance(frequency, temperature):
    bath = DiscreteBath([frequency], [2.7], temperature)
    origin = bath.correlation(0.0).real
    quadrature_imaginary = bath.correlation(np.pi / (2 * frequency)).imag
    emission = (origin - quadrature_imaginary) / 2
    absorption = (origin + quadrature_imaginary) / 2
    np.testing.assert_allclose(absorption / emission, np.exp(-frequency / temperature),
                               rtol=2e-15, atol=1e-16)
    np.testing.assert_allclose(emission - absorption, 2.7 / 2)


def test_discrete_thermal_imaginary_part_and_low_temperature_limit():
    times = np.array([0.0, 0.1, 1.0, 100.0])
    vacuum = DiscreteBath([0.2, 4.0], [3.0, 0.8]).correlation(times)
    cold = DiscreteBath([0.2, 4.0], [3.0, 0.8], np.nextafter(0.0, 1.0))
    np.testing.assert_array_equal(cold.correlation(times), vacuum)
    hot = DiscreteBath([0.2, 4.0], [3.0, 0.8], 1e10)
    np.testing.assert_array_equal(hot.correlation(times).imag, vacuum.imag)
    np.testing.assert_allclose(hot.correlation(0).real / 1e10, 3.0 / 0.2 + 0.8 / 4.0,
                               rtol=2e-15)


def test_discrete_report_records_relative_weights_and_temperature():
    report = DiscreteBath([1, 2], [0.5, 3], 0.7).report
    assert report["frequencies"] == [1.0, 2.0]
    assert report["weights"] == [0.5, 3.0]
    assert report["temperature"] == 0.7
    assert report["modes"] == 2
    assert "1/2" in report["normalization"]
    assert "relative" in report["normalization"]


def test_discrete_rejects_unrepresentable_thermal_weights_and_variance():
    with pytest.raises(ValueError, match="thermal mode weights"):
        DiscreteBath([np.nextafter(0.0, 1.0)], [1.0], np.finfo(float).max)
    with pytest.raises(ValueError, match=r"C_B\(0\)"):
        DiscreteBath([1, 2, 3, 4], [1e308] * 4)
    # Modes with exactly zero spectral weight still contribute zero.
    assert DiscreteBath([np.nextafter(0.0, 1.0)], [0.0],
                        np.finfo(float).max).correlation(0.0) == 0j


@pytest.mark.parametrize("frequencies,weights", [
    ([], []), ([1], []), ([1, 2], [1]), ([0], [1]), ([-1], [1]),
    ([1], [-0.1]), ([np.inf], [1]), ([np.nan], [1]), ([1], [np.inf]),
    ([1], [np.nan]), ([1 + 0j], [1]), ([1], [1 + 0j]),
    ([[1]], [1]), ([1], [[1]]), (1, [1]), ([1], 1),
    ([True], [1]), ([1], [True]), (["1"], [1]), ([1], ["1"]),
])
def test_discrete_invalid_mode_vectors_are_rejected(frequencies, weights):
    with pytest.raises(ValueError):
        DiscreteBath(frequencies, weights)


@pytest.mark.parametrize("temperature", [-0.1, np.nan, np.inf, -np.inf,
                                          True, np.bool_(False), 1j, 0j,
                                          "1.0", None, [1.0]])
def test_discrete_invalid_temperature_is_rejected(temperature):
    with pytest.raises(ValueError, match="temperature"):
        DiscreteBath([1.0], [1.0], temperature)


def test_exponential_matches_classical_real_correlation():
    times = np.array([[-5.0, -0.1, 0.0], [0.1, 1.0, 5.0]])
    bath = ExponentialBath([0.8, 1.3], [0.2, 2.0])
    expected = 0.8 * np.exp(-0.2 * np.abs(times)) + 1.3 * np.exp(-2 * np.abs(times))
    np.testing.assert_allclose(bath.correlation(times), expected, rtol=1e-15)
    np.testing.assert_array_equal(bath.correlation(times).imag, np.zeros_like(times))


def test_exponential_complex_rates_and_signed_time_continuation():
    bath = ExponentialBath([1 + 0.3j, 0.4 - 0.3j], [0.2 + 2j, 0.7 - 0.5j])
    positive_times = np.array([0.0, 0.1, 0.7, 2.0])
    expected = ((1 + 0.3j) * np.exp(-0.2 * positive_times)
                * (np.cos(2 * positive_times) - 1j * np.sin(2 * positive_times))
                + (0.4 - 0.3j) * np.exp(-0.7 * positive_times)
                * (np.cos(0.5 * positive_times) + 1j * np.sin(0.5 * positive_times)))
    np.testing.assert_allclose(bath.correlation(positive_times), expected, rtol=2e-15)
    np.testing.assert_array_equal(bath.correlation(-positive_times),
                                  bath.correlation(positive_times).conj())


def test_exponential_undamped_modes_are_supported():
    bath = ExponentialBath([1.0, 0.3], [2j, -1j])
    times = np.array([-10.0, 0.0, 0.2, 10.0])
    expected = np.exp(-2j * times) + 0.3 * np.exp(1j * times)
    np.testing.assert_allclose(bath.correlation(times), expected, rtol=1e-15)


def test_exponential_origin_is_real_after_tolerated_rounding_error():
    bath = ExponentialBath([1 + 1j, 0.2 - 1j + 2e-13j], [1, 2])
    assert bath.correlation(0.0) == 1.2 + 0j
    assert bath.correlation(-0.0) == 1.2 + 0j
    assert np.isreal(bath.correlation(0.0))


@pytest.mark.parametrize("coefficients,rates", [
    ([], []), ([1], []), ([1, 2], [1]), ([[1]], [1]), ([1], [[1]]),
    (1, [1]), ([1], 1), ([np.nan], [1]), ([np.inf], [1]),
    ([1], [np.nan]), ([1], [np.inf]), ([1], [1 + np.inf * 1j]),
    ([1], [-0.1]), ([1], [-1e-20 + 1j]), ([-0.1], [1]), ([-2e-13], [0]),
    ([1 + 0.1j], [1]), ([1e308 + 1e308j], [1]),
    (["1"], [1]), ([1], ["1"]),
    ([True], [1]), ([1], [False]),
])
def test_exponential_invalid_expansions_are_rejected(coefficients, rates):
    with pytest.raises(ValueError):
        ExponentialBath(coefficients, rates)


def test_exponential_report_is_explicit_about_user_supplied_physics():
    report = ExponentialBath([1], [1]).report
    assert "user-supplied" in report["bath"]
    assert "not assigned" in report["temperature"]
    assert "not guaranteed" in report["physical_validity"]
    assert "positive semidefiniteness" in report["physical_validity"]
    assert "KMS" in report["physical_validity"]


def test_mode_correlations_reject_nonfinite_phase_and_evaluated_sum():
    with pytest.raises(ValueError, match="correlation is not finite"):
        DiscreteBath([1e308], [1.0]).correlation(1e308)
    with pytest.raises(ValueError, match="correlation is not finite"):
        ExponentialBath([1.0], [1e308j]).correlation(1e308)
    # Finite coefficients and C(0) can still produce an overflowing sum.
    bath = ExponentialBath([1e308, -1e308], [0, 1j])
    with pytest.raises(ValueError, match="correlation is not finite"):
        bath.correlation(np.pi)


@pytest.mark.parametrize("factory", [
    lambda first, second: DiscreteBath(first, second, temperature=0.7),
    lambda first, second: ExponentialBath(first, second),
])
def test_mode_inputs_are_copied_readonly_and_attributes_are_frozen(factory):
    first, second = np.array([1.0, 2.0]), np.array([0.2, 0.7])
    bath = factory(first, second)
    expected = bath.correlation([0.0, 0.5])
    first[:] = 4
    second[:] = 8
    np.testing.assert_array_equal(bath.correlation([0.0, 0.5]), expected)
    names = ("frequencies", "weights") if isinstance(bath, DiscreteBath) else ("coefficients", "rates")
    for name in names:
        values = getattr(bath, name)
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values[0] = 4
        with pytest.raises(FrozenInstanceError):
            setattr(bath, name, np.array([1.0]))


@pytest.mark.parametrize("bath", [DiscreteBath([1, 2], [0.4, 0.7], 0.8),
                                  ExponentialBath([1 + 0.2j, 0.5 - 0.2j], [0.5 + 1j, 2])])
@pytest.mark.parametrize("times", [0.1, [0.0, 0.2], [[-0.2, 0.1], [0.0, 2.0]],
                                    np.empty((2, 0)), []])
def test_mode_correlations_preserve_shape_dtype_and_stationary_symmetry(bath, times):
    result = bath.correlation(times)
    actual = np.asarray(result)
    if np.asarray(times).ndim == 0:
        assert np.isscalar(result)
    assert actual.shape == np.asarray(times).shape
    assert actual.dtype == np.complex128
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(bath.correlation(-np.asarray(times)), actual.conj())


@pytest.mark.parametrize("bath", [DiscreteBath([1], [1]), ExponentialBath([1], [1])])
@pytest.mark.parametrize("times", [np.nan, np.inf, -np.inf, [0, np.nan], [0, np.inf],
                                    1j, 0j, [1 + 0j], ["1"], [True],
                                    [[0.0], [0.0, 1.0]], None])
def test_mode_correlations_reject_nonfinite_and_nonreal_times(bath, times):
    with pytest.raises(ValueError, match="times"):
        bath.correlation(times)
