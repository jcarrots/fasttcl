"""Apply the outer commutator and Hermitian conjugate to the TCL6 terms."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any, Sequence

from .convolution_runtime import _numpy

class PostprocessingError(ValueError):
    """Raised when a pre-wrapper map cannot be converted to TCL6."""


@dataclass(frozen=True)
class WrappedGeneratorResult:
    """Final interaction-picture TCL6 generator and structural diagnostics."""

    values: Any
    report: dict[str, Any]


def _namespace(value: Any, xp: Any | None) -> Any:
    if xp is not None:
        return xp
    return _numpy()


def transpose_permutation(dimension: int) -> tuple[int, ...]:
    """Return the column-major permutation for ``vec(X.T)``."""

    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
    ):
        raise PostprocessingError("dimension must be a positive integer")
    return tuple(
        (flat // dimension) + dimension * (flat % dimension)
        for flat in range(dimension * dimension)
    )


def realigned_adjoint(superoperator: Any, *, xp: Any | None = None) -> Any:
    r"""Return ``T^sharp = J conj(T) J``, not the matrix adjoint ``T^dagger``."""

    array_module = _namespace(superoperator, xp)
    values = array_module.asarray(superoperator, dtype=array_module.complex128)
    if values.ndim < 2 or values.shape[-1] != values.shape[-2]:
        raise PostprocessingError(
            "superoperator must end in equal square matrix dimensions"
        )
    dimension_squared = int(values.shape[-1])
    dimension = int(round(dimension_squared**0.5))
    if dimension * dimension != dimension_squared:
        raise PostprocessingError(
            "superoperator dimension must be a perfect square"
        )
    permutation = array_module.asarray(
        transpose_permutation(dimension), dtype=array_module.int64
    )
    return array_module.conj(
        array_module.take(
            array_module.take(values, permutation, axis=-2),
            permutation,
            axis=-1,
        )
    )


def interaction_picture_operator(
    coupling_operator: Sequence[Sequence[complex]],
    level_ratios: Sequence[complex],
    *,
    start: int,
    stop: int,
    xp: Any | None = None,
) -> Any:
    """Build ``A_rc(k)=A_rc*(z_r/z_c)^k`` for ``start <= k < stop``."""

    array_module = _numpy() if xp is None else xp
    coupling = array_module.asarray(
        coupling_operator, dtype=array_module.complex128
    )
    ratios = array_module.asarray(level_ratios, dtype=array_module.complex128)
    if coupling.ndim != 2 or coupling.shape[0] != coupling.shape[1]:
        raise PostprocessingError("coupling_operator must be square")
    dimension = int(coupling.shape[0])
    if ratios.shape != (dimension,):
        raise PostprocessingError(
            "level_ratios must contain one value per energy level"
        )
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(stop, bool)
        or not isinstance(stop, int)
    ):
        raise PostprocessingError("start and stop must be integers")
    if start < 0 or stop < start:
        raise PostprocessingError("invalid interaction-picture index interval")
    if bool(array_module.any(ratios == 0).item()):
        raise PostprocessingError("level ratios must be nonzero")
    indices = array_module.arange(start, stop, dtype=array_module.float64)
    entry_ratios = ratios[:, None] / ratios[None, :]
    return coupling[None, :, :] * array_module.power(
        entry_ratios[None, :, :], indices[:, None, None]
    )


def apply_hr_outer_wrapper(
    prewrapper: Any,
    *,
    coupling_operator: Sequence[Sequence[complex]],
    level_ratios: Sequence[complex],
    chunk_size: int = 65_536,
    xp: Any | None = None,
) -> WrappedGeneratorResult:
    r"""Apply ``-[A(t),S(t)rho] + H.c.`` and the declared realignment.

    Both input and output use column-major matrix-entry flattening:
    ``row = n + d*m`` and ``column = i + d*j``.  The H.c. completion is
    ``T + J*conj(T)*J``; it is not the ordinary adjoint of the superoperator.
    """

    array_module = _namespace(prewrapper, xp)
    values = array_module.asarray(prewrapper, dtype=array_module.complex128)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise PostprocessingError(
            "prewrapper must have shape (Nt, d^2, d^2)"
        )
    dimension_squared = int(values.shape[1])
    dimension = int(round(dimension_squared**0.5))
    if dimension * dimension != dimension_squared:
        raise PostprocessingError(
            "prewrapper superoperator dimension must be a perfect square"
        )
    coupling_shape = array_module.asarray(coupling_operator).shape
    ratio_shape = array_module.asarray(level_ratios).shape
    if coupling_shape != (dimension, dimension):
        raise PostprocessingError(
            "coupling_operator shape does not match the prewrapper"
        )
    if ratio_shape != (dimension,):
        raise PostprocessingError(
            "level_ratios shape does not match the prewrapper"
        )
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise PostprocessingError("chunk_size must be a positive integer")

    nt = int(values.shape[0])
    output = array_module.empty_like(values)
    for start in range(0, nt, chunk_size):
        stop = min(start + chunk_size, nt)
        coupling_t = interaction_picture_operator(
            coupling_operator,
            level_ratios,
            start=start,
            stop=stop,
            xp=array_module,
        )
        source = values[start:stop]
        raw = array_module.zeros_like(source)
        # Direct index form of -(I kron A - A.T kron I) @ S avoids
        # constructing a time-indexed Kronecker tensor.
        for output_row in range(dimension):
            for output_column in range(dimension):
                output_flat = output_row + dimension * output_column
                for contracted in range(dimension):
                    raw[:, output_flat, :] -= (
                        coupling_t[:, output_row, contracted, None]
                        * source[
                            :, contracted + dimension * output_column, :
                        ]
                    )
                    raw[:, output_flat, :] += (
                        source[:, output_row + dimension * contracted, :]
                        * coupling_t[:, contracted, output_column, None]
                    )
        output[start:stop] = raw + realigned_adjoint(raw, xp=array_module)

    report = {
        "picture": "interaction",
        "hilbert_dimension": dimension,
        "output_shape": [nt, dimension_squared, dimension_squared],
        "vectorization": "column-major: n+d*m and i+d*j",
        "outer_wrapper": "-[A(t),S(t) rho] + H.c.",
        "outer_minus_applied": True,
        "hermitian_conjugate": "T + J conj(T) J",
        "transpose_permutation": list(transpose_permutation(dimension)),
        "extra_prefactor": None,
        "extra_time_step_factor": None,
        "chunk_size": chunk_size,
    }
    return WrappedGeneratorResult(values=output, report=report)


def wrapper_diagnostics(values: Any, *, xp: Any | None = None) -> dict[str, float]:
    """Measure trace preservation and realigned-H.c. symmetry."""

    array_module = _namespace(values, xp)
    generator = array_module.asarray(values, dtype=array_module.complex128)
    if generator.ndim != 3 or generator.shape[1] != generator.shape[2]:
        raise PostprocessingError(
            "generator must have shape (Nt, d^2, d^2)"
        )
    dimension_squared = int(generator.shape[1])
    dimension = int(round(dimension_squared**0.5))
    if dimension * dimension != dimension_squared:
        raise PostprocessingError(
            "generator superoperator dimension must be a perfect square"
        )
    trace_vector = array_module.eye(
        dimension, dtype=array_module.complex128
    ).reshape(-1, order="F")
    trace_rows = array_module.einsum("a,tab->tb", trace_vector, generator)
    trace_error = array_module.max(array_module.abs(trace_rows)) if generator.size else 0
    sharp_error = (
        array_module.max(
            array_module.abs(generator - realigned_adjoint(generator, xp=array_module))
        )
        if generator.size
        else 0
    )

    def scalar(value: Any) -> float:
        return float(value.item()) if hasattr(value, "item") else float(value)

    return {
        "trace_preservation_max_abs_error": scalar(trace_error),
        "realigned_hc_max_abs_error": scalar(sharp_error),
    }

