"""Evaluate interlocked time integrals by recursive FFT convolutions."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from typing import Any, TypeAlias

Scalar: TypeAlias = Any


class OverlapContractionError(ValueError):
    """Raised when an overlap contraction is not well defined."""


def _validated_sequences(
    f: Sequence[Scalar],
    g: Sequence[Scalar],
    h: Sequence[Scalar],
) -> tuple[tuple[Scalar, ...], tuple[Scalar, ...], tuple[Scalar, ...]]:
    values = (tuple(f), tuple(g), tuple(h))
    lengths = {len(sequence) for sequence in values}
    if len(lengths) != 1:
        raise OverlapContractionError("F, G, and H must have equal lengths")
    if not values[0]:
        raise OverlapContractionError("F, G, and H must be non-empty")
    return values


def _validated_output_length(output_length: int | None, size: int) -> int:
    full_length = 2 * size - 1
    if output_length is None:
        return full_length
    if isinstance(output_length, bool) or not isinstance(output_length, int):
        raise TypeError("output_length must be an integer")
    if output_length < 0 or output_length > full_length:
        raise OverlapContractionError(
            f"output_length must satisfy 0 <= output_length <= {full_length}"
        )
    return output_length


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _batched_linear_convolution_fft(
    left: Any,
    right: Any,
    *,
    fft: Callable[..., Any] | None = None,
    ifft: Callable[..., Any] | None = None,
) -> Any:
    """Convolve corresponding rows of two complex arrays in one FFT call."""

    import numpy as np

    left_values = np.asarray(left, dtype=np.complex128)
    right_values = np.asarray(right, dtype=np.complex128)
    if left_values.ndim != 2 or right_values.ndim != 2:
        raise OverlapContractionError(
            "batched convolution inputs must be two-dimensional"
        )
    if left_values.shape[0] != right_values.shape[0]:
        raise OverlapContractionError(
            "batched convolution inputs must have equal batch counts"
        )
    if left_values.shape[1] == 0 or right_values.shape[1] == 0:
        raise OverlapContractionError(
            "batched convolution rows must be non-empty"
        )
    linear_length = left_values.shape[1] + right_values.shape[1] - 1
    fft_length = _next_power_of_two(linear_length)
    forward = np.fft.fft if fft is None else fft
    inverse = np.fft.ifft if ifft is None else ifft
    transformed = forward(left_values, fft_length, axis=1)
    right_transformed = forward(right_values, fft_length, axis=1)
    transformed *= right_transformed
    del right_transformed
    return np.asarray(
        inverse(transformed, axis=1)[:, :linear_length],
        dtype=np.complex128,
    )


def overlap_batched_fft_operation_counts(
    size: int,
    *,
    leaf_size: int = 1,
) -> dict[str, int]:
    """Return exact structural operation counts for the batched FFT backend."""

    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("size must be an integer")
    if size <= 0:
        raise OverlapContractionError("size must be positive")
    padded_size = _next_power_of_two(size)
    if (
        isinstance(leaf_size, bool)
        or not isinstance(leaf_size, int)
        or not _is_power_of_two(leaf_size)
        or leaf_size > padded_size
    ):
        raise OverlapContractionError(
            "leaf_size must be a power of two no larger than padded size"
        )
    levels = (padded_size // leaf_size).bit_length() - 1
    # Every internal dyadic block uses six polynomial convolutions.  The
    # implementation batches all blocks of one level into six array calls.
    logical_convolutions = 6 * (padded_size // leaf_size - 1)
    return {
        "input_size": size,
        "padded_size": padded_size,
        "dyadic_levels": levels,
        "leaf_size": leaf_size,
        "batched_convolution_calls": 6 * levels,
        "batched_fft_api_calls": 18 * levels,
        "logical_scalar_convolutions": logical_convolutions,
        "logical_fft_transforms": 3 * logical_convolutions,
    }


def overlap_staged_fft_operation_counts(size: int) -> dict[str, int]:
    """Return structural counts for the three-dependency-stage backend."""

    counts = overlap_batched_fft_operation_counts(size)
    levels = counts["dyadic_levels"]
    return {
        **counts,
        "batched_convolution_calls": 3 * levels,
        "batched_fft_api_calls": 9 * levels,
    }


def _direct_overlap_leaf_blocks(
    f_vectors: Any,
    g_vectors: Any,
    h_vectors: Any,
    leaf_size: int,
) -> Any:
    """Evaluate every fixed-size leaf with vectorized exact finite sums."""

    import numpy as np

    batch_count, padded_size = f_vectors.shape
    block_count = padded_size // leaf_size
    f_blocks = f_vectors.reshape(batch_count, block_count, leaf_size)
    g_blocks = g_vectors.reshape(batch_count, block_count, leaf_size)
    h_blocks = h_vectors.reshape(batch_count, block_count, leaf_size)
    result = np.zeros(
        (batch_count, block_count, 2 * leaf_size - 1),
        dtype=np.complex128,
    )
    for shared_index in range(leaf_size):
        shared_values = h_blocks[:, :, shared_index]
        for first_index in range(shared_index, leaf_size):
            first_values = f_blocks[:, :, first_index] * shared_values
            for second_index in range(shared_index, leaf_size):
                result[:, :, first_index + second_index - shared_index] += (
                    first_values * g_blocks[:, :, second_index]
                )
    return result


def overlap_divide_conquer_fft_batched(
    f: Sequence[Scalar],
    g: Sequence[Scalar],
    h: Sequence[Scalar],
    *,
    output_length: int | None = None,
    fft: Callable[..., Any] | None = None,
    ifft: Callable[..., Any] | None = None,
) -> Any:
    """Evaluate the dyadic proof with six batched convolutions per level.

    This is algebraically the same five-case recursion as
    :func:`overlap_divide_conquer_fft`.  It builds the recursion tree
    bottom-up and batches every block at a given level along NumPy's first
    axis.  Consequently only ``18*ceil(log2(N))`` FFT API calls cross the
    Python boundary, while the total transformed data and arithmetic retain
    the certified ``O(N log(N)**2)`` bound.  Only adjacent levels and the
    three padded inputs are live, so storage is ``O(N)``.
    """

    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - environment dependent
        raise OverlapContractionError(
            "NumPy is required for the FFT overlap backend"
        ) from error

    f_values, g_values, h_values = _validated_sequences(f, g, h)
    size = len(f_values)
    requested = _validated_output_length(output_length, size)
    padded_size = _next_power_of_two(size)
    padded: list[Any] = []
    for values in (f_values, g_values, h_values):
        vector = np.zeros(padded_size, dtype=np.complex128)
        vector[:size] = values
        padded.append(vector)
    f_vector, g_vector, h_vector = padded

    # A length-one block has one admissible triple a=b=v=0.
    results = (f_vector * g_vector * h_vector).reshape(padded_size, 1)
    block_size = 2
    while block_size <= padded_size:
        half = block_size // 2
        block_count = padded_size // block_size
        f_blocks = f_vector.reshape(block_count, block_size)
        g_blocks = g_vector.reshape(block_count, block_size)
        h_blocks = h_vector.reshape(block_count, block_size)
        f_low, f_high = f_blocks[:, :half], f_blocks[:, half:]
        g_low, g_high = g_blocks[:, :half], g_blocks[:, half:]
        h_low = h_blocks[:, :half]

        output = np.zeros(
            (block_count, 2 * block_size - 1), dtype=np.complex128
        )
        child_length = 2 * half - 1
        output[:, :child_length] += results[0::2]
        output[:, half : half + child_length] += results[1::2]
        del results

        f_h_raw = _batched_linear_convolution_fft(
            f_low, h_low[:, ::-1], fft=fft, ifft=ifft
        )
        f_h_suffix = f_h_raw[:, half - 1 :]
        mixed = _batched_linear_convolution_fft(
            f_h_suffix, g_high, fft=fft, ifft=ifft
        )
        output[:, half : half + mixed.shape[1]] += mixed
        del f_h_raw, f_h_suffix, mixed

        g_h_raw = _batched_linear_convolution_fft(
            g_low, h_low[:, ::-1], fft=fft, ifft=ifft
        )
        g_h_suffix = g_h_raw[:, half - 1 :]
        mixed = _batched_linear_convolution_fft(
            f_high, g_h_suffix, fft=fft, ifft=ifft
        )
        output[:, half : half + mixed.shape[1]] += mixed
        del g_h_raw, g_h_suffix, mixed

        high_pair = _batched_linear_convolution_fft(
            f_high, g_high, fft=fft, ifft=ifft
        )
        high_low = _batched_linear_convolution_fft(
            high_pair, h_low[:, ::-1], fft=fft, ifft=ifft
        )
        output[:, half + 1 : half + 1 + high_low.shape[1]] += high_low
        del high_pair, high_low

        results = output
        block_size *= 2

    return np.asarray(results[0, :requested], dtype=np.complex128)


def overlap_divide_conquer_fft_batched_many(
    f: Any,
    g: Any,
    h: Any,
    *,
    output_length: int | None = None,
    leaf_size: int = 1,
    fft: Callable[..., Any] | None = None,
    ifft: Callable[..., Any] | None = None,
) -> Any:
    """Evaluate several independent overlap contractions in shared FFT calls.

    Inputs have shape ``(batch, N)``.  Unit and dyadic-block axes are folded
    together before every row-wise FFT, so a batch crosses the NumPy FFT API
    only ``18*ceil(log2(N))`` times while preserving each unit's exact
    five-case recursion and ``O(N)`` live storage per unit.
    """

    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - environment dependent
        raise OverlapContractionError(
            "NumPy is required for the FFT overlap backend"
        ) from error

    arrays = tuple(np.asarray(values, dtype=np.complex128) for values in (f, g, h))
    if any(values.ndim != 2 for values in arrays):
        raise OverlapContractionError(
            "batched-many overlap inputs must be two-dimensional"
        )
    shapes = {values.shape for values in arrays}
    if len(shapes) != 1:
        raise OverlapContractionError(
            "batched-many overlap inputs must have equal shapes"
        )
    batch_count, size = arrays[0].shape
    if batch_count <= 0 or size <= 0:
        raise OverlapContractionError(
            "batched-many overlap inputs must be non-empty"
        )
    requested = _validated_output_length(output_length, size)
    padded_size = _next_power_of_two(size)
    if (
        isinstance(leaf_size, bool)
        or not isinstance(leaf_size, int)
        or not _is_power_of_two(leaf_size)
        or leaf_size > padded_size
    ):
        raise OverlapContractionError(
            "leaf_size must be a power of two no larger than padded size"
        )
    padded = np.zeros(
        (3, batch_count, padded_size), dtype=np.complex128
    )
    for index, values in enumerate(arrays):
        padded[index, :, :size] = values
    f_vectors, g_vectors, h_vectors = padded

    results = _direct_overlap_leaf_blocks(
        f_vectors, g_vectors, h_vectors, leaf_size
    )
    block_size = 2 * leaf_size
    while block_size <= padded_size:
        half = block_size // 2
        block_count = padded_size // block_size
        f_blocks = f_vectors.reshape(batch_count, block_count, block_size)
        g_blocks = g_vectors.reshape(batch_count, block_count, block_size)
        h_blocks = h_vectors.reshape(batch_count, block_count, block_size)
        f_low, f_high = f_blocks[:, :, :half], f_blocks[:, :, half:]
        g_low, g_high = g_blocks[:, :, :half], g_blocks[:, :, half:]
        h_low = h_blocks[:, :, :half]

        output = np.zeros(
            (batch_count, block_count, 2 * block_size - 1),
            dtype=np.complex128,
        )
        child_length = 2 * half - 1
        output[:, :, :child_length] += results[:, 0::2]
        output[:, :, half : half + child_length] += results[:, 1::2]
        del results

        row_count = batch_count * block_count
        f_h_raw = _batched_linear_convolution_fft(
            f_low.reshape(row_count, half),
            h_low[:, :, ::-1].reshape(row_count, half),
            fft=fft,
            ifft=ifft,
        ).reshape(batch_count, block_count, -1)
        f_h_suffix = f_h_raw[:, :, half - 1 :]
        mixed = _batched_linear_convolution_fft(
            f_h_suffix.reshape(row_count, half),
            g_high.reshape(row_count, half),
            fft=fft,
            ifft=ifft,
        ).reshape(batch_count, block_count, -1)
        output[:, :, half : half + mixed.shape[2]] += mixed
        del f_h_raw, f_h_suffix, mixed

        g_h_raw = _batched_linear_convolution_fft(
            g_low.reshape(row_count, half),
            h_low[:, :, ::-1].reshape(row_count, half),
            fft=fft,
            ifft=ifft,
        ).reshape(batch_count, block_count, -1)
        g_h_suffix = g_h_raw[:, :, half - 1 :]
        mixed = _batched_linear_convolution_fft(
            f_high.reshape(row_count, half),
            g_h_suffix.reshape(row_count, half),
            fft=fft,
            ifft=ifft,
        ).reshape(batch_count, block_count, -1)
        output[:, :, half : half + mixed.shape[2]] += mixed
        del g_h_raw, g_h_suffix, mixed

        high_pair = _batched_linear_convolution_fft(
            f_high.reshape(row_count, half),
            g_high.reshape(row_count, half),
            fft=fft,
            ifft=ifft,
        ).reshape(batch_count, block_count, -1)
        high_low = _batched_linear_convolution_fft(
            high_pair.reshape(row_count, block_size - 1),
            h_low[:, :, ::-1].reshape(row_count, half),
            fft=fft,
            ifft=ifft,
        ).reshape(batch_count, block_count, -1)
        output[:, :, half + 1 : half + 1 + high_low.shape[2]] += high_low
        del high_pair, high_low

        results = output
        block_size *= 2

    return np.asarray(results[:, 0, :requested], dtype=np.complex128)


def overlap_divide_conquer_fft_staged_many(
    f: Any,
    g: Any,
    h: Any,
    *,
    output_length: int | None = None,
    fft: Callable[..., Any] | None = None,
    ifft: Callable[..., Any] | None = None,
) -> Any:
    """Evaluate overlap batches in three dependency stages per level.

    The first stage jointly evaluates ``F_low*H_low``, ``G_low*H_low``, and
    ``F_high*G_high``.  The second jointly evaluates the two mixed cases, and
    the third completes the high/high/low case.  This preserves the proved
    five-way partition while halving NumPy FFT API crossings from 18 to 9 per
    dyadic level.
    """

    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - environment dependent
        raise OverlapContractionError(
            "NumPy is required for the FFT overlap backend"
        ) from error

    arrays = tuple(np.asarray(values, dtype=np.complex128) for values in (f, g, h))
    if any(values.ndim != 2 for values in arrays):
        raise OverlapContractionError(
            "staged overlap inputs must be two-dimensional"
        )
    shapes = {values.shape for values in arrays}
    if len(shapes) != 1:
        raise OverlapContractionError(
            "staged overlap inputs must have equal shapes"
        )
    batch_count, size = arrays[0].shape
    if batch_count <= 0 or size <= 0:
        raise OverlapContractionError("staged overlap inputs must be non-empty")
    requested = _validated_output_length(output_length, size)
    padded_size = _next_power_of_two(size)
    padded = np.zeros((3, batch_count, padded_size), dtype=np.complex128)
    for index, values in enumerate(arrays):
        padded[index, :, :size] = values
    f_vectors, g_vectors, h_vectors = padded

    results = (f_vectors * g_vectors * h_vectors).reshape(
        batch_count, padded_size, 1
    )
    block_size = 2
    while block_size <= padded_size:
        half = block_size // 2
        block_count = padded_size // block_size
        row_count = batch_count * block_count
        f_blocks = f_vectors.reshape(batch_count, block_count, block_size)
        g_blocks = g_vectors.reshape(batch_count, block_count, block_size)
        h_blocks = h_vectors.reshape(batch_count, block_count, block_size)
        f_low = f_blocks[:, :, :half].reshape(row_count, half)
        f_high = f_blocks[:, :, half:].reshape(row_count, half)
        g_low = g_blocks[:, :, :half].reshape(row_count, half)
        g_high = g_blocks[:, :, half:].reshape(row_count, half)
        h_low_reversed = h_blocks[:, :, :half][:, :, ::-1].reshape(
            row_count, half
        )

        output = np.zeros(
            (batch_count, block_count, 2 * block_size - 1),
            dtype=np.complex128,
        )
        child_length = 2 * half - 1
        output[:, :, :child_length] += results[:, 0::2]
        output[:, :, half : half + child_length] += results[:, 1::2]
        del results

        stage_one = _batched_linear_convolution_fft(
            np.concatenate((f_low, g_low, f_high), axis=0),
            np.concatenate((h_low_reversed, h_low_reversed, g_high), axis=0),
            fft=fft,
            ifft=ifft,
        )
        f_h_raw, g_h_raw, high_pair = np.split(stage_one, 3, axis=0)
        del stage_one
        f_h_suffix = f_h_raw[:, half - 1 :]
        g_h_suffix = g_h_raw[:, half - 1 :]

        stage_two = _batched_linear_convolution_fft(
            np.concatenate((f_h_suffix, f_high), axis=0),
            np.concatenate((g_high, g_h_suffix), axis=0),
            fft=fft,
            ifft=ifft,
        )
        low_high, high_low = np.split(stage_two, 2, axis=0)
        del stage_two
        mixed_length = low_high.shape[1]
        output[:, :, half : half + mixed_length] += low_high.reshape(
            batch_count, block_count, mixed_length
        )
        output[:, :, half : half + mixed_length] += high_low.reshape(
            batch_count, block_count, mixed_length
        )

        high_low_values = _batched_linear_convolution_fft(
            high_pair, h_low_reversed, fft=fft, ifft=ifft
        )
        high_low_length = high_low_values.shape[1]
        output[
            :, :, half + 1 : half + 1 + high_low_length
        ] += high_low_values.reshape(
            batch_count, block_count, high_low_length
        )
        results = output
        block_size *= 2

    return np.asarray(results[:, 0, :requested], dtype=np.complex128)

