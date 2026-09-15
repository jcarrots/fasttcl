"""CPU runtime extracted from the canonical TCL6 Numerical source; see data/source-provenance.json."""

from __future__ import annotations

from typing import Any

class ConvolutionRuntimeError(RuntimeError):
    """Raised when proof artifacts cannot be verified or benchmarked."""


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ConvolutionRuntimeError(
            "NumPy is required for numerical verification and benchmarking; "
            "install the project with the 'numerical' extra"
        ) from error
    return np


def _as_complex_vector(value: Any, *, name: str) -> Any:
    np = _numpy()
    vector = np.asarray(value, dtype=np.complex128)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(vector.real).all() or not np.isfinite(vector.imag).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def next_fft_length(linear_length: int) -> int:
    """Return the smallest power of two at least ``linear_length``."""

    if isinstance(linear_length, bool) or not isinstance(linear_length, int):
        raise TypeError("linear_length must be an integer")
    if linear_length <= 0:
        raise ValueError("linear_length must be positive")
    return 1 << (linear_length - 1).bit_length()


def causal_convolution_fft(
    left: Any,
    right: Any,
    *,
    output_length: int | None = None,
) -> Any:
    """Return a zero-padded linear convolution, truncated causally.

    No circular wraparound is permitted.  For two length-``N`` time series,
    the default result contains the first ``N`` samples needed by the TCL
    generator.
    """

    np = _numpy()
    left_vector = _as_complex_vector(left, name="left")
    right_vector = _as_complex_vector(right, name="right")
    if output_length is None:
        output_length = max(left_vector.size, right_vector.size)
    if isinstance(output_length, bool) or not isinstance(output_length, int):
        raise TypeError("output_length must be an integer")
    linear_length = left_vector.size + right_vector.size - 1
    if output_length < 0 or output_length > linear_length:
        raise ValueError(
            "output_length must lie between zero and the linear-convolution "
            "length"
        )
    fft_length = next_fft_length(linear_length)
    transformed = np.fft.fft(left_vector, fft_length)
    transformed *= np.fft.fft(right_vector, fft_length)
    result = np.fft.ifft(transformed)
    return np.asarray(result[:output_length], dtype=np.complex128)

