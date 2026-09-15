"""Independent direct-sum checks of the pure-Python lower-order runtime.

Oracle functions below are adapted from the MIT-licensed TACO project's
independent MATLAB-equation regression oracle. They use no production kernel,
contraction, or realignment helper and require no TACO/native installation.
"""

import numpy as np
import pytest

from fasttcl.lower_orders import lower_order_generators, _fcr_curve

def _gamma_trapezoid(
    correlation: np.ndarray, dt: float, omegas: np.ndarray
) -> np.ndarray:
    times = dt * np.arange(correlation.size)
    integrand = correlation[:, None] * np.exp(
        1j * times[:, None] * omegas[None, :]
    )
    gamma = np.zeros_like(integrand)
    gamma[1:] = np.cumsum(
        0.5 * dt * (integrand[:-1] + integrand[1:]), axis=0
    )
    return gamma

def _matlab_nonsecular_l2_correction(
    gamma: np.ndarray,
    pair_map: np.ndarray,
    coupling: np.ndarray,
) -> np.ndarray:
    """Literal one-channel getAsymptoticALL Redfield construction."""

    dimension = coupling.shape[0]
    identity = np.eye(dimension, dtype=np.complex128)
    identity_vector = identity.reshape(-1, order="F")
    coupling_vector = coupling.reshape(-1, order="F")
    diagonal_rows = np.arange(0, dimension * dimension, dimension + 1)
    correction = np.empty(
        (gamma.shape[0], dimension * dimension, dimension * dimension),
        dtype=np.complex128,
    )

    def realign(matrix: np.ndarray) -> np.ndarray:
        return (
            matrix.reshape((dimension,) * 4, order="F")
            .transpose(0, 2, 1, 3)
            .reshape((dimension * dimension,) * 2, order="F")
        )

    for time_index in range(gamma.shape[0]):
        gamma_matrix = gamma[time_index, pair_map]
        filtered = coupling * gamma_matrix.T
        lamb_shift = coupling @ filtered / (2.0j)
        lamb_shift = lamb_shift + lamb_shift.conj().T

        choi = np.outer(
            filtered.reshape(-1, order="F"), coupling_vector.conj()
        )
        choi = choi + choi.conj().T
        jump = np.kron(filtered.conj(), coupling) + np.kron(
            coupling.conj(), filtered
        )
        loss = 0.5 * np.outer(
            identity_vector, jump[diagonal_rows].sum(axis=0)
        )
        choi = choi - loss - loss.conj().T

        unitary = -1.0j * np.outer(
            lamb_shift.reshape(-1, order="F"), identity_vector
        )
        unitary = unitary + unitary.conj().T
        correction[time_index] = realign(choi + unitary)

    return correction

def _direct_fcr(
    gamma: np.ndarray,
    omegas: np.ndarray,
    mirror: np.ndarray,
    dt: float,
    *,
    quadrature: str = "legacy-rectangle",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Literal discrete forms of the three MATLAB TCL4 kernels."""

    nt, nf = gamma.shape
    shape = (nf, nf, nf, nt)
    kernel_f = np.zeros(shape, dtype=np.complex128)
    kernel_c = np.zeros(shape, dtype=np.complex128)
    kernel_r = np.zeros(shape, dtype=np.complex128)

    for a in range(nf):
        for b in range(nf):
            for c in range(nf):
                omega_sum = omegas[a] + omegas[b] + omegas[c]
                for n in range(nt):
                    ell = np.arange(n + 1)
                    weights = np.ones(n + 1)
                    if quadrature == "composite-trapezoid":
                        if n == 0:
                            weights[0] = 0.0
                        else:
                            weights[[0, -1]] = 0.5
                    elif quadrature != "legacy-rectangle":
                        raise ValueError(quadrature)
                    reverse = n - ell
                    phase_reverse = np.exp(-1j * omega_sum * reverse * dt)
                    phase_forward = np.exp(-1j * omega_sum * ell * dt)
                    gamma_difference_reverse = gamma[n, a] - gamma[reverse, a]

                    # The transpose of the filtered Gamma matrix maps the
                    # second transition to its exact -omega bucket.
                    kernel_f[a, b, c, n] = dt * np.sum(
                        weights
                        * gamma_difference_reverse
                        * gamma[ell, mirror[b]]
                        * phase_reverse
                    )
                    kernel_c[a, b, c, n] = dt * np.sum(
                        weights
                        * gamma_difference_reverse
                        * np.conj(gamma[ell, b])
                        * phase_reverse
                    )
                    kernel_r[a, b, c, n] = dt * np.sum(
                        weights
                        * (gamma[n, a] - gamma[ell, a])
                        * gamma[ell, b]
                        * phase_forward
                    )
    return kernel_f, kernel_c, kernel_r

def _matlab_mikx(
    kernel_f: np.ndarray,
    kernel_c: np.ndarray,
    kernel_r: np.ndarray,
    pair_map: np.ndarray,
    time_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Literal index form of matlab/MIKX.m equations (12)--(15)."""

    dimension = pair_map.shape[0]
    shape4 = (dimension,) * 4
    tensor_m = np.empty(shape4, dtype=np.complex128)
    tensor_i = np.empty(shape4, dtype=np.complex128)
    tensor_k = np.empty(shape4, dtype=np.complex128)
    tensor_x = np.empty((dimension,) * 6, dtype=np.complex128)

    for j, k, p, q in np.ndindex(shape4):
        tensor_m[j, k, p, q] = (
            kernel_f[
                pair_map[j, q], pair_map[j, k], pair_map[p, j], time_index
            ]
            - kernel_r[
                pair_map[j, q], pair_map[p, q], pair_map[q, k], time_index
            ]
        )
        tensor_i[j, k, p, q] = kernel_f[
            pair_map[j, k], pair_map[q, p], pair_map[k, q], time_index
        ]
        tensor_k[j, k, p, q] = kernel_r[
            pair_map[j, k], pair_map[p, q], pair_map[q, j], time_index
        ]
        for r, s in np.ndindex((dimension, dimension)):
            frequencies = (pair_map[j, k], pair_map[p, q], pair_map[r, s])
            tensor_x[j, k, p, q, r, s] = (
                kernel_c[frequencies + (time_index,)]
                + kernel_r[frequencies + (time_index,)]
            )
    return tensor_m, tensor_i, tensor_k, tensor_x

def _matlab_nakzwan(
    tensor_m: np.ndarray,
    tensor_i: np.ndarray,
    tensor_k: np.ndarray,
    tensor_x: np.ndarray,
    coupling: np.ndarray,
) -> np.ndarray:
    """Literal one-channel transcription of matlab/NAKZWAN_v9.m."""

    dimension = coupling.shape[0]
    operator_a = coupling
    operator_b = coupling
    tensor_t = np.zeros((dimension,) * 4, dtype=np.complex128)

    for n, i, m, j in np.ndindex((dimension,) * 4):
        result = 0.0j
        for a, b in np.ndindex((dimension, dimension)):
            term1 = (
                operator_b[n, a]
                * operator_a[a, b]
                * operator_b[b, i]
                * operator_a[j, m]
                * tensor_m[b, i, n, a]
            )
            term2 = (
                operator_a[n, a]
                * operator_b[a, b]
                * operator_b[b, i]
                * operator_a[j, m]
                * tensor_i[a, n, i, b]
            )
            term3 = (
                operator_b[n, a]
                * operator_b[a, b]
                * operator_a[b, i]
                * operator_a[j, m]
                * tensor_k[i, b, n, a]
            )
            term4 = (
                operator_a[n, a]
                * operator_b[a, i]
                * operator_b[j, b]
                * operator_a[b, m]
                * tensor_x[a, n, j, b, n, i]
            )
            term5 = (
                operator_b[n, a]
                * operator_a[a, i]
                * operator_b[j, b]
                * operator_a[b, m]
                * tensor_x[i, a, j, b, n, i]
            )
            term6 = (
                operator_a[n, a]
                * operator_a[a, b]
                * operator_b[b, i]
                * operator_b[j, m]
                * tensor_x[b, a, j, m, a, i]
            )
            term7 = (
                operator_a[n, a]
                * operator_b[a, b]
                * operator_a[b, i]
                * operator_b[j, m]
                * tensor_x[i, b, j, m, a, i]
            )
            result -= term1 - term2 + term3 + term4 - term5 - term6 + term7

            if j == m:
                for c in range(dimension):
                    result += (
                        operator_a[n, a]
                        * operator_b[a, b]
                        * operator_a[b, c]
                        * operator_b[c, i]
                        * tensor_m[c, i, a, b]
                        + operator_a[n, a]
                        * operator_b[a, b]
                        * operator_b[b, c]
                        * operator_a[c, i]
                        * tensor_k[i, c, a, b]
                        - operator_a[n, a]
                        * operator_a[a, b]
                        * operator_b[b, c]
                        * operator_b[c, i]
                        * tensor_i[b, a, i, c]
                    )
        tensor_t[n, i, m, j] = result

    raw_gw = tensor_t.reshape(
        (dimension * dimension, dimension * dimension), order="F"
    )
    return raw_gw + raw_gw.conj().T

def _gw_to_liouvillian(raw_gw: np.ndarray, dimension: int) -> np.ndarray:
    # GW has (n,i;m,j), while the superoperator has (n,m;i,j).
    tensor = raw_gw.reshape((dimension,) * 4, order="F")
    return tensor.transpose(0, 2, 1, 3).reshape(
        (dimension * dimension, dimension * dimension), order="F"
    )


def _oracle(energies, a, correlation, dt):
    frequencies = np.unique(energies[:, None] - energies[None, :])
    pair = np.empty((len(energies), len(energies)), dtype=int)
    for j, k in np.ndindex(pair.shape):
        pair[j, k] = np.flatnonzero(frequencies == energies[j] - energies[k])[0]
    mirror = np.array([np.flatnonzero(frequencies == -w)[0] for w in frequencies])
    gamma = _gamma_trapezoid(correlation, dt, frequencies)
    second = _matlab_nonsecular_l2_correction(gamma, pair, a)
    f, c, r = _direct_fcr(gamma, frequencies, mirror, dt, quadrature="composite-trapezoid")
    fourth = np.empty_like(second)
    for index in range(len(correlation)):
        tensors = _matlab_mikx(f, c, r, pair, index)
        fourth[index] = _gw_to_liouvillian(_matlab_nakzwan(*tensors, a), len(a))
    return second, fourth


@pytest.mark.parametrize("energies", [
    [-0.35, 0.8],
    [-0.8, 0.17, 0.9],
    [0.0, 0.0, 1.0],
    [0.0],
])
def test_complete_generators_match_independent_matlab_direct_oracle(energies):
    energy = np.asarray(energies)
    rng = np.random.default_rng(371)
    raw = rng.normal(size=(len(energy), len(energy))) + 1j*rng.normal(size=(len(energy), len(energy)))
    a = 0.15 * (raw + raw.conj().T)
    dt = 0.037
    time = dt * np.arange(9)
    correlation = 0.73*np.exp(-(0.41+0.83j)*time) + 0.11*np.exp(-(1.17-0.29j)*time)
    actual = lower_order_generators(energy, a, correlation, dt)
    expected = _oracle(energy, a, correlation, dt)
    for calculated, direct in zip(actual, expected):
        np.testing.assert_allclose(calculated, direct, rtol=3e-11, atol=3e-14)
        np.testing.assert_array_equal(calculated[0], 0)
        assert calculated.shape == (9, len(energy)**2, len(energy)**2)
        assert calculated.dtype == np.complex128
    rho = rng.normal(size=a.shape) + 1j*rng.normal(size=a.shape)
    rho = rho + rho.conj().T
    for correction in actual:
        derivative = (correction @ rho.reshape(-1, order="F")).reshape(
            9, len(energy), len(energy), order="F"
        )
        np.testing.assert_allclose(np.trace(derivative, axis1=1, axis2=2), 0, atol=3e-13)
        np.testing.assert_allclose(derivative, derivative.conj().transpose(0,2,1), atol=3e-13)


@pytest.mark.parametrize("nt", [1, 2, 17])
def test_fft_prefix_kernels_match_literal_trapezoid_sums(nt):
    rng = np.random.default_rng(56)
    frequencies = np.array([-0.83, 0.0, 0.83])
    mirror = np.array([2, 1, 0])
    gamma = rng.normal(size=(nt, 3)) + 1j*rng.normal(size=(nt, 3))
    gamma[0] = 0
    dt = 0.021
    expected = _direct_fcr(gamma, frequencies, mirror, dt, quadrature="composite-trapezoid")
    for kind, direct in zip(("F", "C", "R"), expected):
        for a, b, c in np.ndindex((3, 3, 3)):
            phase = np.exp(-1j*(frequencies[a]+frequencies[b]+frequencies[c])*dt*np.arange(nt))
            other = gamma[:, mirror[b]] if kind == "F" else gamma[:, b]
            if kind == "C":
                other = other.conj()
            actual = _fcr_curve(kind, gamma[:, a], other, phase, dt)
            np.testing.assert_allclose(actual, direct[a,b,c], rtol=3e-12, atol=2e-14)


def test_unit_bath_and_unscaled_operator_give_correct_lambda_powers():
    e = np.array([-0.3, 0.7])
    a = np.array([[0.1, 0.3+0.2j], [0.3-0.2j, -0.2]])
    correlation = np.exp(-(0.5+0.9j)*0.03*np.arange(13))
    coupling_strength = 0.37
    reference = lower_order_generators(e, a, correlation, 0.03)
    from_bath = lower_order_generators(e, a, coupling_strength**2*correlation, 0.03)
    from_operator = lower_order_generators(e, coupling_strength*a, correlation, 0.03)
    for order, base, scaled_bath, scaled_operator in zip((2,4), reference, from_bath, from_operator):
        np.testing.assert_allclose(scaled_bath, coupling_strength**order*base, rtol=2e-11, atol=2e-15)
        np.testing.assert_allclose(scaled_operator, coupling_strength**order*base, rtol=2e-11, atol=2e-15)


def test_tcl2_uses_emission_mirror_and_skips_fourth_order(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("TCL4 compilation must be skipped")
    monkeypatch.setattr("fasttcl.lower_orders._fourth_order_weights", forbidden)
    e = np.array([-0.5, 0.5])
    a = np.array([[0, 0.5], [0.5, 0]])
    times = 0.005*np.arange(2001)
    second, fourth = lower_order_generators(e, a, np.exp(-(0.2+1j)*times), 0.005, max_order=2)
    assert fourth is None
    # Resonant Gamma(+1) drives |1> -> |0>, not the reversed transition.
    emission = second[-1, 0, 3].real
    absorption = second[-1, 3, 0].real
    assert emission > 1.0
    assert abs(absorption) < 0.1*emission


@pytest.mark.parametrize("nt", [1, 2, 7])
def test_zero_bath_and_zero_coupling(nt):
    for coupling, bath in ((np.eye(2), np.zeros(nt)), (np.zeros((2,2)), np.ones(nt))):
        for result in lower_order_generators([-0.5,0.5], coupling, bath, 0.1):
            np.testing.assert_array_equal(result, np.zeros((nt,4,4), dtype=complex))


def test_commuting_gaussian_coupling_has_no_fourth_order_correction():
    times = 0.025*np.arange(41)
    second, fourth = lower_order_generators(
        [-0.6, 0.9], np.diag([0.3,-0.7]),
        np.exp(-(0.2+0.8j)*times), 0.025,
    )
    # Pure dephasing has an exactly second-order Gaussian cumulant; all
    # populations remain constant, even though the coherences decay.
    np.testing.assert_allclose(fourth, 0, atol=2e-14)
    np.testing.assert_allclose(second[:, [0,3], :], 0, atol=2e-14)
    assert np.max(np.abs(second)) > 0.1


@pytest.mark.parametrize("overrides", [
    {"energies": []}, {"energies": [0, 1j]}, {"energies": [0, np.inf]},
    {"coupling_energy": np.ones((3,3))},
    {"coupling_energy": [[0,1j],[1j,0]]},
    {"correlation_positive": []}, {"correlation_positive": [1,np.nan]},
    {"dt": 0}, {"dt": -1}, {"dt": np.inf}, {"dt": True},
    {"max_order": 6}, {"max_order": True},
])
def test_invalid_inputs(overrides):
    arguments = dict(energies=[0,1], coupling_energy=np.eye(2), correlation_positive=[1,0.8], dt=0.1)
    arguments.update(overrides)
    with pytest.raises(ValueError):
        lower_order_generators(**arguments)
