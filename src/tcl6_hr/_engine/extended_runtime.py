"""CPU runtime extracted from the canonical TCL6 Numerical source; see data/source-provenance.json."""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict

from dataclasses import dataclass

from typing import Any, Mapping, Sequence

from .convolution_runtime import _numpy, causal_convolution_fft

from .extended_evaluator import ScalarMonomialInventory, assert_publication_appendix_f_inventory

from .overlap_contraction import overlap_batched_fft_operation_counts, overlap_divide_conquer_fft_batched

EXPECTED_TEMPORAL_CORES = 16_052


EXPECTED_STRICT_TEMPORAL_CORES = 15_696


EXPECTED_OVERLAP_TEMPORAL_CORES = 356


HISTORY_DEPTHS = ("H0", "H1", "H2")


KernelIdentity = tuple[str, int, int]


PhaseVector = tuple[int, ...]


TemporalKey = tuple[
    tuple[KernelIdentity, ...], PhaseVector, PhaseVector, PhaseVector
]


class ExtendedRuntimeError(ValueError):
    """Raised when an inventory or numerical input cannot be executed."""


@dataclass(frozen=True)
class StaticContribution:
    """One sparse superoperator weight attached to a temporal curve."""

    output_flat: int
    input_flat: int
    coupling_entries: tuple[tuple[int, int], ...]
    coefficient: int
    history_depth: str | None = None


@dataclass(frozen=True)
class TemporalCore:
    """One unique arbitrary-sequence temporal expression."""

    kernel_identities: tuple[KernelIdentity, ...]
    phase_u: PhaseVector
    phase_v: PhaseVector
    phase_w: PhaseVector
    template_id: str
    contributions: tuple[StaticContribution, ...]

    @property
    def key(self) -> TemporalKey:
        return (
            self.kernel_identities,
            self.phase_u,
            self.phase_v,
            self.phase_w,
        )


@dataclass(frozen=True)
class StreamedExecutionPlan:
    """Deterministic temporal CSE plan for a scalar-monomial inventory."""

    input_monomial_count: int
    input_coefficient_l1: int
    cores: tuple[TemporalCore, ...]
    dimension: int = 2
    history_depth_manifest_sha256: str | None = None

    @property
    def report(self) -> dict[str, Any]:
        templates = Counter(core.template_id for core in self.cores)
        contribution_count = sum(
            len(core.contributions) for core in self.cores
        )
        gamma_identities = {
            identity for core in self.cores for identity in core.kernel_identities
        }
        overlap = templates.get("overlap-uv-vw-v", 0)
        report = {
            "input_scalar_monomials": self.input_monomial_count,
            "input_coefficient_l1": self.input_coefficient_l1,
            "temporal_core_count": len(self.cores),
            "strict_temporal_core_count": len(self.cores) - overlap,
            "overlap_temporal_core_count": overlap,
            "eliminated_temporal_evaluations": (
                self.input_monomial_count - len(self.cores)
            ),
            "sparse_static_contribution_count": contribution_count,
            "gamma_sequence_identity_count": len(gamma_identities),
            "template_counts": dict(sorted(templates.items())),
            "output_shape": [
                "Nt",
                self.dimension * self.dimension,
                self.dimension * self.dimension,
            ],
            "column_major_mapping": {
                "output": f"row + {self.dimension}*column",
                "density_input": (
                    f"rho_row + {self.dimension}*rho_column"
                ),
            },
            "complexity": "O(Nt log(Nt)^2)",
            "storage": "O(Nt)",
        }
        if self.dimension != 2:
            report["hilbert_dimension"] = self.dimension
        if self.history_depth_manifest_sha256 is not None:
            depth_counts = Counter(
                contribution.history_depth
                for core in self.cores
                for contribution in core.contributions
            )
            report["history_depth_instrumentation"] = {
                "enabled": True,
                "depth_manifest_sha256": self.history_depth_manifest_sha256,
                "terminal_contribution_counts": {
                    depth: depth_counts[depth] for depth in HISTORY_DEPTHS
                },
                "shared_temporal_cores_are_depth_neutral": True,
            }
        return report


@dataclass(frozen=True)
class StreamedEvaluationResult:
    """Numerical superoperator curve and execution diagnostics."""

    values: Any
    report: dict[str, Any]
    depth_values: Mapping[str, Any] | None = None
    depth_terminal_absolute_values: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class WeightedTemporalCore:
    """A temporal core whose coupling-only weights are already contracted."""

    core: TemporalCore
    weights: tuple[tuple[int, int, complex], ...]
    depth_weights: tuple[tuple[str, int, int, complex], ...] = ()


@dataclass(frozen=True)
class SpecializedExecutionPlan:
    """Coupling-specialized subset of a supplied streamed execution plan.

    Static zero detection is performed on the host before any Gamma sequence
    is normalized or transferred to an accelerator.  This is particularly
    important for the historical unbiased spin--boson publication profile:
    only 838 of its generic 16,052 temporal cores survive. These are not the
    HR production counts.
    """

    source_plan: StreamedExecutionPlan
    coupling_operator: Any
    cores: tuple[WeightedTemporalCore, ...]

    @property
    def temporal_plan(self) -> StreamedExecutionPlan:
        """Return the active temporal subset in the legacy plan container."""

        return StreamedExecutionPlan(
            input_monomial_count=self.source_plan.input_monomial_count,
            input_coefficient_l1=self.source_plan.input_coefficient_l1,
            cores=tuple(weighted.core for weighted in self.cores),
            dimension=self.source_plan.dimension,
            history_depth_manifest_sha256=(
                self.source_plan.history_depth_manifest_sha256
            ),
        )

    @property
    def report(self) -> dict[str, Any]:
        templates = Counter(
            weighted.core.template_id for weighted in self.cores
        )
        identities = {
            identity
            for weighted in self.cores
            for identity in weighted.core.kernel_identities
        }
        overlap = templates.get("overlap-uv-vw-v", 0)
        report = {
            "source_temporal_core_count": len(self.source_plan.cores),
            "active_temporal_core_count": len(self.cores),
            "active_strict_temporal_core_count": len(self.cores) - overlap,
            "active_overlap_temporal_core_count": overlap,
            "zero_weight_temporal_core_count": (
                len(self.source_plan.cores) - len(self.cores)
            ),
            "active_sparse_weight_count": sum(
                len(weighted.weights) for weighted in self.cores
            ),
            "active_gamma_sequence_identity_count": len(identities),
            "active_template_counts": dict(sorted(templates.items())),
            "specialization": "exact static coupling contraction",
        }
        if self.source_plan.history_depth_manifest_sha256 is not None:
            depth_counts = Counter(
                depth
                for weighted in self.cores
                for depth, _output, _input, _weight in weighted.depth_weights
            )
            report["history_depth_instrumentation"] = {
                "enabled": True,
                "depth_manifest_sha256": (
                    self.source_plan.history_depth_manifest_sha256
                ),
                "active_sparse_weight_counts": {
                    depth: depth_counts[depth] for depth in HISTORY_DEPTHS
                },
                "depth_only_temporal_core_count": sum(
                    not weighted.weights and bool(weighted.depth_weights)
                    for weighted in self.cores
                ),
            }
        return report


def _base_support(support: str) -> str:
    base = support[1:] if support.startswith("-") else support
    if base not in {"K", "U", "V", "W", "UV", "VW"}:
        raise ExtendedRuntimeError(f"unknown kernel support {support!r}")
    return base


def _classify_template(kernel_identities: Sequence[KernelIdentity]) -> str:
    bases = [_base_support(identity[0]) for identity in kernel_identities]
    dependencies = set(bases)
    if {"UV", "VW", "V"}.issubset(dependencies):
        if len(bases) != 3 or Counter(bases) != Counter(
            {"UV": 1, "VW": 1, "V": 1}
        ):
            raise ExtendedRuntimeError(
                "unsupported overlap sector outside the certified UV/VW/V primitive"
            )
        return "overlap-uv-vw-v"
    if "UV" in dependencies and "VW" in dependencies:
        if "U" in dependencies:
            return "prefix-overlap-u"
        if "W" in dependencies:
            return "prefix-overlap-w"
        return "prefix-overlap-unit-u"
    if "UV" in dependencies:
        return "nested-uv"
    if "VW" in dependencies:
        return "nested-vw"
    return "conv-uvw"


def _parse_monomial(
    key: tuple[Any, ...], coefficient: int, *, dimension: int = 2
) -> tuple[TemporalKey, StaticContribution]:
    if len(key) != 6:
        raise ExtendedRuntimeError("malformed scalar monomial key")
    output_row, output_column, factors, phase_u, phase_v, phase_w = key
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
    ):
        raise ExtendedRuntimeError("dimension must be a positive integer")
    valid_indices = range(dimension)
    if output_row not in valid_indices or output_column not in valid_indices:
        raise ExtendedRuntimeError(
            "output matrix index lies outside the plan dimension"
        )
    if isinstance(coefficient, bool) or not isinstance(coefficient, int):
        raise ExtendedRuntimeError("scalar monomial coefficient must be an integer")

    kernels: list[KernelIdentity] = []
    couplings: list[tuple[int, int]] = []
    densities: list[tuple[int, int]] = []
    for factor in factors:
        kind = factor[0]
        if kind == "kernel":
            if factor[1] != "Gamma":
                raise ExtendedRuntimeError(
                    f"unknown kernel family {factor[1]!r}"
                )
            support, row, column = factor[2], factor[3], factor[4]
            _base_support(support)
            if row not in valid_indices or column not in valid_indices:
                raise ExtendedRuntimeError(
                    "Gamma index lies outside the plan dimension"
                )
            kernels.append((support, row, column))
        elif kind == "coupling_operator":
            row, column = factor[1], factor[2]
            if row not in valid_indices or column not in valid_indices:
                raise ExtendedRuntimeError(
                    "coupling index lies outside the plan dimension"
                )
            couplings.append((row, column))
        elif kind == "density_operator":
            if factor[1] != "rho":
                raise ExtendedRuntimeError("unknown density-operator symbol")
            densities.append((factor[2], factor[3]))
        else:
            raise ExtendedRuntimeError(f"unknown scalar factor {factor!r}")
    if len(kernels) != 3:
        raise ExtendedRuntimeError(
            "every TCL6 scalar monomial must contain three kernels"
        )
    if len(densities) != 1 or any(
        index not in valid_indices for pair in densities for index in pair
    ):
        raise ExtendedRuntimeError(
            "every scalar monomial must contain one in-range rho entry"
        )
    if not all(
        isinstance(vector, tuple)
        and len(vector) == dimension
        and all(isinstance(value, int) and not isinstance(value, bool) for value in vector)
        for vector in (phase_u, phase_v, phase_w)
    ):
        raise ExtendedRuntimeError(
            "phase vectors must have one integer entry per energy level"
        )

    density_row, density_column = densities[0]
    temporal_key: TemporalKey = (
        tuple(sorted(kernels)), phase_u, phase_v, phase_w
    )
    contribution = StaticContribution(
        output_flat=output_row + dimension * output_column,
        input_flat=density_row + dimension * density_column,
        coupling_entries=tuple(sorted(couplings)),
        coefficient=coefficient,
    )
    return temporal_key, contribution


def compile_streamed_plan(
    monomials: ScalarMonomialInventory,
    *,
    require_complete: bool = True,
    dimension: int = 2,
    history_depth_monomials: Mapping[
        str, ScalarMonomialInventory
    ] | None = None,
    history_depth_manifest_sha256: str | None = None,
) -> StreamedExecutionPlan:
    """Combine identical temporal expressions before numerical evaluation.

    History-depth instrumentation is optional.  When enabled, the three
    supplied sector polynomials must reconstruct ``monomials`` exactly.  The
    temporal grouping key is unchanged; only terminal static contributions
    carry a depth label.
    """

    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
    ):
        raise ExtendedRuntimeError("dimension must be a positive integer")
    if require_complete and dimension != 2:
        raise ExtendedRuntimeError(
            "the frozen publication inventory constants are defined only for d=2"
        )
    if require_complete:
        assert_publication_appendix_f_inventory(monomials)
    instrumentation_enabled = history_depth_monomials is not None
    if instrumentation_enabled != (history_depth_manifest_sha256 is not None):
        raise ExtendedRuntimeError(
            "history-depth monomials and manifest digest must be supplied together"
        )
    if history_depth_manifest_sha256 is not None and (
        len(history_depth_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in history_depth_manifest_sha256
        )
    ):
        raise ExtendedRuntimeError(
            "history_depth_manifest_sha256 must be a lowercase SHA-256 digest"
        )
    if history_depth_monomials is not None:
        if set(history_depth_monomials) != set(HISTORY_DEPTHS):
            raise ExtendedRuntimeError(
                "history-depth instrumentation requires exactly H0, H1, and H2"
            )
        expected: Counter[tuple[Any, ...]] = Counter()
        for key, coefficient in monomials:
            expected[key] += coefficient
        observed: Counter[tuple[Any, ...]] = Counter()
        for depth in HISTORY_DEPTHS:
            for key, coefficient in history_depth_monomials[depth]:
                observed[key] += coefficient
        expected = Counter(
            {key: coefficient for key, coefficient in expected.items() if coefficient}
        )
        observed = Counter(
            {key: coefficient for key, coefficient in observed.items() if coefficient}
        )
        if observed != expected:
            raise ExtendedRuntimeError(
                "H0+H1+H2 must exactly reconstruct the supplied scalar inventory"
            )
    grouped: dict[
        TemporalKey,
        Counter[tuple[int, int, tuple[tuple[int, int], ...], str | None]],
    ] = defaultdict(Counter)
    coefficient_l1 = sum(abs(coefficient) for _key, coefficient in monomials)
    inputs: tuple[tuple[str | None, ScalarMonomialInventory], ...]
    if history_depth_monomials is None:
        inputs = ((None, monomials),)
    else:
        inputs = tuple(
            (depth, history_depth_monomials[depth]) for depth in HISTORY_DEPTHS
        )
    for depth, sector_monomials in inputs:
        for key, coefficient in sector_monomials:
            temporal_key, contribution = _parse_monomial(
                key, coefficient, dimension=dimension
            )
            static_key = (
                contribution.output_flat,
                contribution.input_flat,
                contribution.coupling_entries,
                depth,
            )
            grouped[temporal_key][static_key] += contribution.coefficient

    cores: list[TemporalCore] = []
    for temporal_key in sorted(grouped):
        kernel_identities, phase_u, phase_v, phase_w = temporal_key
        contributions = tuple(
            StaticContribution(
                output_flat=output_flat,
                input_flat=input_flat,
                coupling_entries=couplings,
                coefficient=coefficient,
                history_depth=depth,
            )
            for (output_flat, input_flat, couplings, depth), coefficient in sorted(
                grouped[temporal_key].items(),
                key=lambda item: (
                    item[0][0],
                    item[0][1],
                    item[0][2],
                    "" if item[0][3] is None else item[0][3],
                ),
            )
            if coefficient
        )
        if not contributions:
            continue
        cores.append(
            TemporalCore(
                kernel_identities=kernel_identities,
                phase_u=phase_u,
                phase_v=phase_v,
                phase_w=phase_w,
                template_id=_classify_template(kernel_identities),
                contributions=contributions,
            )
        )
    plan = StreamedExecutionPlan(
        input_monomial_count=len(monomials),
        input_coefficient_l1=coefficient_l1,
        cores=tuple(cores),
        dimension=dimension,
        history_depth_manifest_sha256=history_depth_manifest_sha256,
    )
    if require_complete:
        report = plan.report
        expected = {
            "temporal_core_count": EXPECTED_TEMPORAL_CORES,
            "strict_temporal_core_count": EXPECTED_STRICT_TEMPORAL_CORES,
            "overlap_temporal_core_count": EXPECTED_OVERLAP_TEMPORAL_CORES,
            "gamma_sequence_identity_count": 44,
        }
        actual = {name: report[name] for name in expected}
        if actual != expected:
            raise ExtendedRuntimeError(
                f"unexpected complete-plan inventory: {actual!r}, expected {expected!r}"
            )
    return plan


def _phase_ratio(
    level_ratios: Sequence[complex], vector: Sequence[int]
) -> complex:
    if len(level_ratios) != len(vector):
        raise ExtendedRuntimeError("phase-vector and level-ratio dimensions differ")
    value: complex = 1
    for ratio, exponent in zip(level_ratios, vector, strict=True):
        if ratio == 0:
            raise ExtendedRuntimeError("geometric level ratios must be nonzero")
        value *= ratio**exponent
    return value


class _PhaseCache:
    """Small bounded cache of geometric vectors; capacity is independent of N."""

    def __init__(
        self, level_ratios: Sequence[complex], length: int, capacity: int
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
            raise ExtendedRuntimeError("phase_cache_size must be a nonnegative integer")
        self.level_ratios = tuple(level_ratios)
        self.length = length
        self.capacity = capacity
        self.values: OrderedDict[PhaseVector, Any] = OrderedDict()
        self.index = _numpy().arange(length, dtype=_numpy().int64)

    def get(self, vector: PhaseVector) -> Any:
        np = _numpy()
        cached = self.values.get(vector)
        if cached is not None:
            self.values.move_to_end(vector)
            return cached
        ratio = _phase_ratio(self.level_ratios, vector)
        result = np.asarray(np.power(ratio, self.index), dtype=np.complex128)
        if self.capacity:
            self.values[vector] = result
            self.values.move_to_end(vector)
            while len(self.values) > self.capacity:
                self.values.popitem(last=False)
        return result


def _normalize_gamma_sequences(
    plan: StreamedExecutionPlan,
    gamma_sequences: Mapping[KernelIdentity, Sequence[complex]],
    *,
    length: int,
) -> dict[KernelIdentity, Any]:
    np = _numpy()
    required = {
        identity for core in plan.cores for identity in core.kernel_identities
    }
    normalized: dict[KernelIdentity, Any] = {}
    validated: dict[int, tuple[Any, Any]] = {}
    for identity in sorted(required):
        try:
            source = gamma_sequences[identity]
        except KeyError as error:
            raise ExtendedRuntimeError(
                f"missing independent Gamma sequence {identity!r}"
            ) from error
        cache_key = id(source)
        cached = validated.get(cache_key)
        vector = cached[1] if cached is not None and cached[0] is source else None
        if vector is None:
            values = np.asarray(source, dtype=np.complex128)
            if values.ndim != 1 or values.size < length:
                raise ExtendedRuntimeError(
                    f"Gamma sequence {identity!r} must have at least {length} samples"
                )
            vector = values[:length]
            if not np.isfinite(vector.real).all() or not np.isfinite(vector.imag).all():
                raise ExtendedRuntimeError(
                    f"Gamma sequence {identity!r} contains a nonfinite value"
                )
            validated[cache_key] = (source, vector)
        normalized[identity] = vector
    return normalized


def _raw_kernel_groups(
    core: TemporalCore,
    gamma_sequences: Mapping[KernelIdentity, Any],
) -> dict[str, Any]:
    np = _numpy()
    grouped: dict[str, Any] = {}
    for identity in core.kernel_identities:
        support = identity[0]
        base = _base_support(support)
        values = gamma_sequences[identity]
        if base in grouped:
            grouped[base] = np.multiply(grouped[base], values)
        else:
            grouped[base] = values
    return grouped


def _convolution(left: Any, right: Any, counters: Counter[str]) -> Any:
    counters["causal_convolution_calls"] += 1
    counters["fft_api_calls"] += 3
    return causal_convolution_fft(left, right, output_length=int(len(left)))


def _strict_bulk(
    template: str,
    raw: Mapping[str, Any],
    mod_u: Any,
    mod_v: Any,
    mod_w: Any,
    ones: Any,
    counters: Counter[str],
) -> Any:
    np = _numpy()

    def base(name: str) -> Any:
        return raw.get(name, ones)

    grouped = {name: base(name) for name in ("K", "U", "V", "W", "UV", "VW")}
    if template in {"prefix-overlap-u", "prefix-overlap-unit-u"}:
        # x=rv/rw, y=rw, z=ru*rw/rv.
        grouped["UV"] = grouped["UV"] * (mod_v / mod_w)
        grouped["VW"] = grouped["VW"] * mod_w
        grouped["U"] = grouped["U"] * (mod_u * mod_w / mod_v)
    elif template == "prefix-overlap-w":
        # x=ru, y=rv/ru, z=rw*ru/rv.
        grouped["UV"] = grouped["UV"] * mod_u
        grouped["VW"] = grouped["VW"] * (mod_v / mod_u)
        grouped["W"] = grouped["W"] * (mod_w * mod_u / mod_v)
    else:
        grouped["U"] = grouped["U"] * mod_u
        grouped["V"] = grouped["V"] * mod_v
        grouped["W"] = grouped["W"] * mod_w

    if template == "conv-uvw":
        core = _convolution(
            _convolution(grouped["U"], grouped["V"], counters),
            grouped["W"],
            counters,
        )
    elif template == "nested-uv":
        inner = _convolution(grouped["U"], grouped["V"], counters)
        core = _convolution(grouped["UV"] * inner, grouped["W"], counters)
    elif template == "nested-vw":
        inner = _convolution(grouped["V"], grouped["W"], counters)
        core = _convolution(grouped["U"], grouped["VW"] * inner, counters)
    elif template in {"prefix-overlap-u", "prefix-overlap-unit-u"}:
        prefix = np.cumsum(grouped["UV"])
        delayed = np.empty_like(prefix)
        delayed[0] = 0
        delayed[1:] = prefix[:-1]
        first = prefix * _convolution(
            grouped["VW"], grouped["U"], counters
        )
        second = _convolution(
            grouped["VW"], grouped["U"] * delayed, counters
        )
        core = first - second
    elif template == "prefix-overlap-w":
        prefix = np.cumsum(grouped["VW"])
        delayed = np.empty_like(prefix)
        delayed[0] = 0
        delayed[1:] = prefix[:-1]
        first = prefix * _convolution(
            grouped["UV"], grouped["W"], counters
        )
        second = _convolution(
            grouped["UV"], grouped["W"] * delayed, counters
        )
        core = first - second
    else:
        raise ExtendedRuntimeError(f"unknown strict template {template!r}")
    return grouped["K"] * core


def _quadrature_faces(
    raw: Mapping[str, Any],
    mod_u: Any,
    mod_v: Any,
    mod_w: Any,
    ones: Any,
    counters: Counter[str],
) -> tuple[Any, Any, Any, Any, Any]:
    """Return u=0, v=0, w=0, uv=0, and uw=0 face curves."""

    def base(name: str) -> Any:
        return raw.get(name, ones)

    k_values = base("K")
    u_values = base("U") * mod_u
    v_values = base("V") * mod_v
    w_values = base("W") * mod_w
    uv_values = base("UV")
    vw_values = base("VW")

    face_u = (
        k_values
        * vw_values
        * u_values[0]
        * _convolution(v_values * uv_values, w_values, counters)
    )
    face_v = (
        k_values
        * v_values[0]
        * _convolution(u_values * uv_values, w_values * vw_values, counters)
    )
    face_w = (
        k_values
        * w_values[0]
        * uv_values
        * _convolution(u_values, v_values * vw_values, counters)
    )
    intersection_uv = (
        k_values
        * u_values[0]
        * v_values[0]
        * uv_values[0]
        * w_values
        * vw_values
    )
    intersection_uw = (
        k_values
        * u_values[0]
        * w_values[0]
        * v_values
        * uv_values
        * vw_values
    )
    return face_u, face_v, face_w, intersection_uv, intersection_uw


def _evaluate_core(
    core: TemporalCore,
    *,
    gamma_sequences: Mapping[KernelIdentity, Any],
    phase_cache: _PhaseCache,
    ones: Any,
    step: complex,
    counters: Counter[str],
) -> Any:
    raw = _raw_kernel_groups(core, gamma_sequences)
    mod_u = phase_cache.get(core.phase_u)
    mod_v = phase_cache.get(core.phase_v)
    mod_w = phase_cache.get(core.phase_w)

    if core.template_id == "overlap-uv-vw-v":
        f_uv = raw["UV"] * mod_u
        g_vw = raw["VW"] * mod_w
        h_v = raw["V"] * (mod_v / (mod_u * mod_w))
        bulk = overlap_divide_conquer_fft_batched(
            f_uv, g_vw, h_v, output_length=len(ones)
        )
        operation_counts = overlap_batched_fft_operation_counts(len(ones))
        counters["overlap_primitive_calls"] += 1
        counters["overlap_batched_convolution_calls"] += operation_counts[
            "batched_convolution_calls"
        ]
        counters["fft_api_calls"] += operation_counts["batched_fft_api_calls"]
    else:
        bulk = _strict_bulk(
            core.template_id,
            raw,
            mod_u,
            mod_v,
            mod_w,
            ones,
            counters,
        )
    faces = _quadrature_faces(
        raw, mod_u, mod_v, mod_w, ones, counters
    )
    face_u, face_v, face_w, intersection_uv, intersection_uw = faces
    return (step * step) * (
        bulk
        - 0.5 * (face_u + face_v + face_w)
        + 0.25 * (intersection_uv + intersection_uw)
    )


def _validate_coupling_operator(
    value: Sequence[Sequence[complex]], *, dimension: int | None = None
) -> Any:
    np = _numpy()
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ExtendedRuntimeError("coupling_operator must be a square matrix")
    if dimension is not None and matrix.shape != (dimension, dimension):
        raise ExtendedRuntimeError(
            "coupling_operator shape does not match the plan dimension"
        )
    if not np.isfinite(matrix.real).all() or not np.isfinite(matrix.imag).all():
        raise ExtendedRuntimeError("coupling_operator contains a nonfinite value")
    return matrix


def _static_weights(
    core: TemporalCore,
    coupling_operator: Any,
    product_cache: dict[tuple[tuple[int, int], ...], complex],
) -> dict[tuple[int, int], complex]:
    weights: dict[tuple[int, int], complex] = defaultdict(complex)
    for contribution in core.contributions:
        if contribution.coupling_entries in product_cache:
            product = product_cache[contribution.coupling_entries]
        else:
            product = 1
            for row, column in contribution.coupling_entries:
                product *= coupling_operator[row, column]
            product_cache[contribution.coupling_entries] = product
        weights[(contribution.output_flat, contribution.input_flat)] += (
            contribution.coefficient * product
        )
    return {index: value for index, value in weights.items() if value != 0}


def _static_depth_weights(
    core: TemporalCore,
    coupling_operator: Any,
    product_cache: dict[tuple[tuple[int, int], ...], complex],
) -> dict[tuple[str, int, int], complex]:
    """Contract static factors without collecting across depth sectors."""

    weights: dict[tuple[str, int, int], complex] = defaultdict(complex)
    for contribution in core.contributions:
        depth = contribution.history_depth
        if depth is None:
            continue
        if depth not in HISTORY_DEPTHS:
            raise ExtendedRuntimeError(f"unknown history depth {depth!r}")
        if contribution.coupling_entries in product_cache:
            product = product_cache[contribution.coupling_entries]
        else:
            product = 1
            for row, column in contribution.coupling_entries:
                product *= coupling_operator[row, column]
            product_cache[contribution.coupling_entries] = product
        weights[(depth, contribution.output_flat, contribution.input_flat)] += (
            contribution.coefficient * product
        )
    return {index: value for index, value in weights.items() if value != 0}


def specialize_streamed_plan(
    plan: StreamedExecutionPlan,
    *,
    coupling_operator: Sequence[Sequence[complex]],
) -> SpecializedExecutionPlan:
    """Contract all time-independent coupling factors and remove exact zeros.

    The test is algebraic for the supplied complex128 coupling matrix: a core
    is retained exactly when at least one accumulated superoperator weight is
    nonzero.  No bath values, time-grid values, or floating tolerances enter
    the decision.
    """

    coupling = _validate_coupling_operator(
        coupling_operator, dimension=plan.dimension
    ).copy()
    coupling.setflags(write=False)
    product_cache: dict[tuple[tuple[int, int], ...], complex] = {}
    active: list[WeightedTemporalCore] = []
    for core in plan.cores:
        weights = _static_weights(core, coupling, product_cache)
        depth_weights = _static_depth_weights(core, coupling, product_cache)
        if not weights and not depth_weights:
            continue
        active.append(
            WeightedTemporalCore(
                core=core,
                weights=tuple(
                    (output_flat, input_flat, complex(weight))
                    for (output_flat, input_flat), weight in sorted(
                        weights.items()
                    )
                ),
                depth_weights=tuple(
                    (depth, output_flat, input_flat, complex(weight))
                    for (depth, output_flat, input_flat), weight in sorted(
                        depth_weights.items()
                    )
                ),
            )
        )
    return SpecializedExecutionPlan(
        source_plan=plan,
        coupling_operator=coupling,
        cores=tuple(active),
    )


def evaluate_specialized_plan_numpy(
    plan: SpecializedExecutionPlan,
    *,
    gamma_sequences: Mapping[KernelIdentity, Sequence[complex]],
    level_ratios: Sequence[complex],
    length: int,
    step: complex = 1,
    phase_cache_size: int = 8,
    accumulation_chunk_size: int = 65_536,
    track_history_depth_terminal_activity: bool = False,
) -> StreamedEvaluationResult:
    """Evaluate a coupling-specialized plan with the certified NumPy kernels."""

    np = _numpy()
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ExtendedRuntimeError("length must be a positive integer")
    if (
        isinstance(accumulation_chunk_size, bool)
        or not isinstance(accumulation_chunk_size, int)
        or accumulation_chunk_size <= 0
    ):
        raise ExtendedRuntimeError(
            "accumulation_chunk_size must be a positive integer"
        )
    dimension = plan.source_plan.dimension
    if len(level_ratios) != dimension:
        raise ExtendedRuntimeError(
            "level-ratio count does not match the plan dimension"
        )

    temporal_plan = plan.temporal_plan
    gamma = _normalize_gamma_sequences(
        temporal_plan, gamma_sequences, length=length
    )
    phase_cache = _PhaseCache(level_ratios, length, phase_cache_size)
    ones = np.ones(length, dtype=np.complex128)
    dimension_squared = dimension * dimension
    output = np.zeros(
        (length, dimension_squared, dimension_squared),
        dtype=np.complex128,
    )
    depth_output = (
        {depth: np.zeros_like(output) for depth in HISTORY_DEPTHS}
        if plan.source_plan.history_depth_manifest_sha256 is not None
        else None
    )
    if (
        track_history_depth_terminal_activity
        and plan.source_plan.history_depth_manifest_sha256 is None
    ):
        raise ExtendedRuntimeError(
            "terminal activity requires a history-depth instrumented plan"
        )
    depth_terminal_absolute = (
        {
            depth: np.zeros(
                (length, dimension_squared, dimension_squared),
                dtype=np.float64,
            )
            for depth in HISTORY_DEPTHS
        }
        if track_history_depth_terminal_activity
        else None
    )
    terminal_product_cache: dict[tuple[tuple[int, int], ...], complex] = {}
    counters: Counter[str] = Counter()

    for weighted in plan.cores:
        values = _evaluate_core(
            weighted.core,
            gamma_sequences=gamma,
            phase_cache=phase_cache,
            ones=ones,
            step=step,
            counters=counters,
        )
        counters["temporal_cores_evaluated"] += 1
        counters[
            "overlap_cores_evaluated"
            if weighted.core.template_id == "overlap-uv-vw-v"
            else "strict_cores_evaluated"
        ] += 1
        for output_flat, input_flat, weight in weighted.weights:
            for start in range(0, length, accumulation_chunk_size):
                stop = min(start + accumulation_chunk_size, length)
                output[start:stop, output_flat, input_flat] += (
                    weight * values[start:stop]
                )
            counters["sparse_curve_accumulations"] += 1
        if depth_output is not None:
            for depth, output_flat, input_flat, weight in weighted.depth_weights:
                for start in range(0, length, accumulation_chunk_size):
                    stop = min(start + accumulation_chunk_size, length)
                    depth_output[depth][start:stop, output_flat, input_flat] += (
                        weight * values[start:stop]
                    )
                counters["history_depth_sparse_curve_accumulations"] += 1
        if depth_terminal_absolute is not None:
            absolute_curve = np.abs(values)
            for contribution in weighted.core.contributions:
                depth = contribution.history_depth
                if depth is None:
                    continue
                entries = contribution.coupling_entries
                if entries in terminal_product_cache:
                    product = terminal_product_cache[entries]
                else:
                    product = 1
                    for row, column in entries:
                        product *= plan.coupling_operator[row, column]
                    terminal_product_cache[entries] = product
                absolute_weight = abs(contribution.coefficient * product)
                if absolute_weight == 0:
                    continue
                depth_terminal_absolute[depth][
                    :, contribution.output_flat, contribution.input_flat
                ] += absolute_weight * absolute_curve
                counters["history_depth_terminal_activity_accumulations"] += 1

    output[0, :, :] = 0
    if depth_output is not None:
        for sector in depth_output.values():
            sector[0, :, :] = 0
    if depth_terminal_absolute is not None:
        for sector in depth_terminal_absolute.values():
            sector[0, :, :] = 0
    report = {
        **plan.report,
        "backend": "numpy",
        "nt": length,
        "dtype": "complex128",
        "phase_cache_capacity": phase_cache_size,
        "phase_cache_entries": len(phase_cache.values),
        "gamma_semantic_identity_count": len(gamma),
        "gamma_host_buffer_count": len({id(values) for values in gamma.values()}),
        "gamma_finite_scan_count": len({id(values) for values in gamma.values()}),
        "accumulation_chunk_size": accumulation_chunk_size,
        "no_quadratic_arrays": True,
        "execution_counts": dict(sorted(counters.items())),
        "output_semantics": "pre-outer-wrapper column-major superoperator",
    }
    if depth_output is not None:
        report["history_depth_accumulation"] = {
            "enabled": True,
            "depth_manifest_sha256": (
                plan.source_plan.history_depth_manifest_sha256
            ),
            "temporal_curves_evaluated_once": True,
            "resident_sector_array_count": len(depth_output),
            "per_word_dense_arrays": 0,
            "terminal_absolute_activity_enabled": (
                depth_terminal_absolute is not None
            ),
            "terminal_absolute_activity_semantics": (
                "sum of absolute model-bound terminal contributions before "
                "temporal-core collection and before the outer wrapper"
            ),
        }
    return StreamedEvaluationResult(
        values=output,
        report=report,
        depth_values=depth_output,
        depth_terminal_absolute_values=depth_terminal_absolute,
    )

