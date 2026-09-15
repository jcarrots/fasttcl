"""Full nonsecular TCL2 and TCL4 for a single Hermitian coupling operator.

NumPy/SciPy implementation of the F/C/R kernels and tensor contractions.
The formulas follow TACO; see THIRD_PARTY_NOTICES.md for attribution.

All integrals use composite trapezoids. The returned arrays are the order-two
and order-four *coefficients*, in the energy basis and Schrodinger picture,
with column-major vectorization. Pass the unit bath correlation C_B and the
unscaled coupling A; the caller supplies lambda**2 and lambda**4, respectively.
Neither the free Hamiltonian nor an additional coupling prefactor is included.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import product

import numpy as np
from scipy.fft import fft, ifft, next_fast_len


# Limit temporary tensors during generator assembly; keep every output time.
_ASSEMBLY_CHUNK_SIZE = 8192


def _inputs(energies, coupling, correlation, dt, max_order):
    energy = np.asarray(energies)
    if energy.ndim != 1 or energy.size == 0:
        raise ValueError("energies must be a nonempty one-dimensional array")
    if np.iscomplexobj(energy) and np.any(energy.imag != 0):
        raise ValueError("energies must be real")
    energy = np.asarray(energy.real, dtype=np.float64)
    a = np.asarray(coupling, dtype=np.complex128)
    c = np.asarray(correlation, dtype=np.complex128)
    if a.shape != (energy.size, energy.size):
        raise ValueError("coupling_energy must have shape (d, d)")
    if c.ndim != 1 or c.size == 0:
        raise ValueError("correlation_positive must be a nonempty 1D array")
    if not all(np.isfinite(x).all() for x in (energy, a, c)):
        raise ValueError("energies, coupling, and correlation must be finite")
    if not np.allclose(a, a.conj().T, rtol=1e-12, atol=1e-14):
        raise ValueError("coupling_energy must be Hermitian")
    if isinstance(dt, (bool, np.bool_)) or not np.isscalar(dt) or np.iscomplexobj(dt):
        raise ValueError("dt must be a positive finite real scalar")
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a positive finite real scalar")
    if isinstance(max_order, (bool, np.bool_)) or max_order not in (2, 4):
        raise ValueError("max_order must be 2 or 4")
    return energy, a, c, dt


def _gamma(energies, correlation, dt):
    """Exact-gap buckets, positive-label Gamma, and the nonconjugating mirror."""
    gaps = energies[:, None] - energies[None, :]
    frequencies, inverse = np.unique(gaps, return_inverse=True)
    pairs = inverse.reshape(gaps.shape)
    # Each gap has its exact arithmetic negative in the transposed pair.
    mirror = np.searchsorted(frequencies, -frequencies)
    if not np.array_equal(frequencies[mirror], -frequencies):
        raise ValueError("Bohr-frequency mirror construction failed")
    times = dt * np.arange(correlation.size, dtype=np.float64)
    integrand = correlation[:, None] * np.exp(1j * times[:, None] * frequencies)
    gamma = np.zeros_like(integrand)
    gamma[1:] = np.cumsum(0.5 * dt * (integrand[:-1] + integrand[1:]), axis=0)
    return frequencies, pairs, mirror, gamma, times


def _second_order(gamma, pairs, a):
    """Redfield: B rho A + A rho B^dagger - A B rho - rho B^dagger A."""
    nt, d = gamma.shape[0], a.shape[0]
    result = np.empty((nt, d * d, d * d), dtype=np.complex128)
    eye = np.eye(d, dtype=np.complex128)
    for start in range(0, nt, _ASSEMBLY_CHUNK_SIZE):
        stop = min(start + _ASSEMBLY_CHUNK_SIZE, nt)
        # B_rc = A_rc Gamma(E_c-E_r): transpose the frequency map, not Gamma*.
        b = a * gamma[start:stop, pairs.T]
        ab = np.einsum("na,tai->tni", a, b)
        bdag_a = np.einsum("taj,am->tjm", b.conj(), a)
        tensor = (
            np.einsum("tni,jm->tnmij", b, a)
            + np.einsum("ni,tmj->tnmij", a, b.conj())
            - np.einsum("tni,jm->tnmij", ab, eye)
            - np.einsum("ni,tjm->tnmij", eye, bdag_a)
        )
        result[start:stop] = tensor.transpose(0, 2, 1, 4, 3).reshape(-1, d*d, d*d)
    return result


def _fourth_order_weights(a, pair):
    """Contract MIKX once, storing sparse terminal weights for each F/C/R curve.

    This is the one-channel NAKZWAN expression, with raw tensor axes (n,i,m,j).
    Its seven signs are -,+,-,-,+,+,-; the delta(j,m) terms are +M,+K,-I.
    No time-dependent six-index tensor or full F/C/R bank is materialized.
    """
    d = a.shape[0]
    weights = defaultdict(lambda: defaultdict(complex))

    def emit(kind, f1, f2, f3, coefficient, target):
        if coefficient != 0:
            weights[(kind, int(f1), int(f2), int(f3))][target] += coefficient

    def m(j, k, p, q, coefficient, target):
        emit("F", pair[j,q], pair[j,k], pair[p,j], coefficient, target)
        emit("R", pair[j,q], pair[p,q], pair[q,k], -coefficient, target)

    def ii(j, k, p, q, coefficient, target):
        emit("F", pair[j,k], pair[q,p], pair[k,q], coefficient, target)

    def kk(j, k, p, q, coefficient, target):
        emit("R", pair[j,k], pair[p,q], pair[q,j], coefficient, target)

    def x(j, k, p, q, r, s, coefficient, target):
        emit("C", pair[j,k], pair[p,q], pair[r,s], coefficient, target)
        emit("R", pair[j,k], pair[p,q], pair[r,s], coefficient, target)

    for n, i, out_m, j in product(range(d), repeat=4):
        target = (n + d*out_m, i + d*j)
        for aa, b in product(range(d), repeat=2):
            chain = a[n,aa] * a[aa,b] * a[b,i] * a[j,out_m]
            m(b, i, n, aa, -chain, target)
            ii(aa, n, i, b, chain, target)
            kk(i, b, n, aa, -chain, target)
            x(b, aa, j, out_m, aa, i, chain, target)
            x(i, b, j, out_m, aa, i, -chain, target)
            crossed = a[n,aa] * a[aa,i] * a[j,b] * a[b,out_m]
            x(aa, n, j, b, n, i, -crossed, target)
            x(i, aa, j, b, n, i, crossed, target)
            if j == out_m:
                for c in range(d):
                    loop = a[n,aa] * a[aa,b] * a[b,c] * a[c,i]
                    m(c, i, aa, b, loop, target)
                    kk(i, c, aa, b, loop, target)
                    ii(b, aa, i, c, -loop, target)
    return {
        key: tuple((row, col, value) for (row, col), value in bank.items() if value != 0)
        for key, bank in weights.items()
        if any(value != 0 for value in bank.values())
    }


def _fcr_curve(kind, g1, g2, phase, dt):
    """Prefix/FFT form of a composite-trapezoidal F, C, or R integral.

    g2 is already mirrored for F or conjugated for C. The phase still uses
    the three *original* frequency labels. Gamma(0) is exactly zero.
    """
    if kind == "R":
        weighted = g2 * phase
        result = dt * (g1 * np.cumsum(weighted) - np.cumsum(g1 * weighted))
    else:
        phased_first = g1 * phase
        length = next_fast_len(2 * len(g1) - 1)
        convolution = ifft(fft(phased_first, length) * fft(g2, length))[:len(g1)]
        result = dt * (
            phased_first * np.cumsum(g2 * phase.conj()) - convolution
            - 0.5 * (g1 - g1[0]) * g2
        )
    result[0] = 0.0  # The zero-length integral is exact, including after FFTs.
    return result


def lower_order_generators(
    energies,
    coupling_energy,
    correlation_positive,
    dt,
    *,
    max_order=4,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (K2, K4) coefficient curves without the free Hamiltonian.

    Parameters are real energies (d,), a Hermitian coupling (d,d), and unit
    correlation samples C_B(n*dt), including n=0. Arrays have shape
    (Nt,d*d,d*d), complex128, energy Schrodinger picture, vec_F indexing.
    The cumulative physical generator is L0 + lambda**2*K2 + lambda**4*K4.
    With max_order=2, K4 is None and no fourth-order work is performed.

    The bath is one stationary scalar Gaussian bath and the system dimension
    is finite; repeated/degenerate Bohr gaps are supported without toleranced
    frequency merging. At fixed dimension the temporal work is O(Nt log Nt).
    Contraction compilation grows with dimension, and output storage is O(Nt*d**4).
    """
    energy, a, correlation, dt = _inputs(
        energies, coupling_energy, correlation_positive, dt, max_order
    )
    nt, d = correlation.size, energy.size
    shape = (nt, d*d, d*d)
    if nt == 1 or not np.any(a) or not np.any(correlation):
        return np.zeros(shape, complex), None if max_order == 2 else np.zeros(shape, complex)
    frequencies, pair, mirror, gamma, times = _gamma(energy, correlation, dt)
    second = _second_order(gamma, pair, a)
    if max_order == 2:
        return second, None
    fourth = np.zeros(shape, dtype=np.complex128)
    for (kind, f1, f2, f3), weights in _fourth_order_weights(a, pair).items():
        omega = frequencies[f1] + frequencies[f2] + frequencies[f3]
        phase = np.exp(-1j * omega * times)
        other = gamma[:, mirror[f2]] if kind == "F" else gamma[:, f2]
        if kind == "C":
            other = other.conj()
        curve = _fcr_curve(kind, gamma[:, f1], other, phase, dt)
        for row, col, coefficient in weights:
            fourth[:, row, col] += coefficient * curve
    permutation = np.arange(d*d).reshape(d, d).T.ravel()
    for start in range(0, nt, _ASSEMBLY_CHUNK_SIZE):
        block = fourth[start:start+_ASSEMBLY_CHUNK_SIZE]
        # Realigned H.c., not Hermitization of the superoperator matrix.
        block += block[:, permutation][:, :, permutation].conj()
    return second, fourth


__all__ = ["lower_order_generators"]
