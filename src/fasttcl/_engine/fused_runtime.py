"""CPU runtime extracted from the canonical TCL6 Numerical source; see data/source-provenance.json."""

from __future__ import annotations

from array import array

from collections import Counter, OrderedDict, defaultdict

from dataclasses import dataclass

from fractions import Fraction

import hashlib

import heapq

from itertools import combinations

import math

from time import perf_counter

from typing import Any, Callable, Literal, Mapping, Sequence

import sympy as sp

from .convolution_runtime import _numpy, causal_convolution_fft, next_fft_length

from .extended_runtime import ExtendedRuntimeError, KernelIdentity, PhaseVector, SpecializedExecutionPlan, StreamedEvaluationResult

from .io_utils import canonical_json

from .overlap_contraction import overlap_batched_fft_operation_counts, overlap_divide_conquer_fft_batched, overlap_divide_conquer_fft_batched_many, overlap_divide_conquer_fft_staged_many, overlap_staged_fft_operation_counts

SLOT_ORDER = ("K", "U", "V", "W", "UV", "VW")


_SLOT_INDEX = {name: index for index, name in enumerate(SLOT_ORDER)}


PlainSymbolicCoordinate = tuple[int, int, tuple[tuple[int, int], ...]]


DepthSymbolicCoordinate = tuple[
    str, int, int, tuple[tuple[int, int], ...]
]


SymbolicCoordinate = PlainSymbolicCoordinate | DepthSymbolicCoordinate


@dataclass(frozen=True, order=True)
class RationalCoefficient:
    numerator: int
    denominator: int = 1

    def __post_init__(self) -> None:
        if self.denominator <= 0:
            raise ExtendedRuntimeError("a rational denominator must be positive")
        reduced = Fraction(self.numerator, self.denominator)
        if (reduced.numerator, reduced.denominator) != (
            self.numerator,
            self.denominator,
        ):
            raise ExtendedRuntimeError("rational coefficients must be reduced")

    @classmethod
    def from_fraction(cls, value: Fraction) -> "RationalCoefficient":
        return cls(value.numerator, value.denominator)

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


@dataclass(frozen=True, order=True)
class ComplexCoefficient:
    """Canonical complex128 coefficient for validated model-bound fusion.

    Exact algebraic fusion continues to use :class:`RationalCoefficient`.
    This separate type prevents a guarded numerical factorization from being
    mistaken for an exact certificate while allowing both plans to share the
    optimized temporal evaluator.
    """

    real: float
    imag: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.real) or not math.isfinite(self.imag):
            raise ExtendedRuntimeError("complex coefficients must be finite")
        if self.real == 0.0:
            object.__setattr__(self, "real", 0.0)
        if self.imag == 0.0:
            object.__setattr__(self, "imag", 0.0)

    @classmethod
    def from_complex(cls, value: complex) -> "ComplexCoefficient":
        scalar = complex(value)
        return cls(float(scalar.real), float(scalar.imag))

    @property
    def value(self) -> complex:
        return complex(self.real, self.imag)


SequenceCoefficient = RationalCoefficient | ComplexCoefficient


def _coefficient_value(value: SequenceCoefficient) -> Fraction | complex:
    if isinstance(value, RationalCoefficient):
        return value.fraction
    if isinstance(value, ComplexCoefficient):
        return value.value
    raise ExtendedRuntimeError("unknown effective-sequence coefficient type")


def _coefficient_is_zero(value: SequenceCoefficient) -> bool:
    return _coefficient_value(value) == 0


@dataclass(frozen=True, order=True)
class EffectiveMonomial:
    kernel_identities: tuple[KernelIdentity, ...]
    phase: PhaseVector


@dataclass(frozen=True, order=True)
class EffectiveSequenceTerm:
    coefficient: SequenceCoefficient
    monomial: EffectiveMonomial


@dataclass(frozen=True, order=True)
class EffectiveSequenceExpression:
    terms: tuple[EffectiveSequenceTerm, ...]

    def __post_init__(self) -> None:
        if not self.terms:
            raise ExtendedRuntimeError("an effective sequence expression cannot be empty")
        if tuple(sorted(self.terms, key=lambda term: term.monomial)) != self.terms:
            raise ExtendedRuntimeError("effective sequence terms must be canonical")
        if any(_coefficient_is_zero(term.coefficient) for term in self.terms):
            raise ExtendedRuntimeError("zero effective-sequence terms are forbidden")
        if len({term.monomial for term in self.terms}) != len(self.terms):
            raise ExtendedRuntimeError("duplicate effective monomials must be collected")


@dataclass(frozen=True)
class _SemanticSequenceNode:
    """Exact structural identity for a derived one-dimensional sequence."""

    kind: str
    operands: tuple[Any, ...]


SemanticSequenceKey = EffectiveSequenceExpression | _SemanticSequenceNode


ProductForwardKey = _SemanticSequenceNode


ConvolutionResultKey = _SemanticSequenceNode


OverlapResultKey = _SemanticSequenceNode


SemanticForwardKey = SemanticSequenceKey


_CONVOLUTION_CACHE_SCHEDULE_CAPACITY = 16


@dataclass(frozen=True)
class _CompactConvolutionSchedule:
    """Flat integer storage for a plan-ordered convolution cache schedule.

    A production four-level TCL6 plan can contain more than one million
    temporal units.  A ``dict[unit_id, tuple[slot, ...]]`` then costs several
    gigabytes even though every tuple contains only three or five small slot
    indices.  The evaluator already traverses units in plan order, so offsets
    plus one signed integer array preserve the exact schedule without the
    per-unit Python objects.
    """

    unit_offsets: array
    slots: array

    def slots_for_unit(self, unit_index: int) -> tuple[int | None, ...]:
        start = int(self.unit_offsets[unit_index])
        stop = int(self.unit_offsets[unit_index + 1])
        return tuple(
            None if slot < 0 else int(slot)
            for slot in self.slots[start:stop]
        )

    @property
    def storage_bytes(self) -> int:
        return (
            len(self.unit_offsets) * self.unit_offsets.itemsize
            + len(self.slots) * self.slots.itemsize
        )


ConvolutionCacheSchedule = (
    Mapping[str, tuple[int | None, ...]] | _CompactConvolutionSchedule
)


_CONVOLUTION_CACHE_SCHEDULES: dict[
    tuple[str, int, str, str],
    tuple[ConvolutionCacheSchedule, dict[str, Any]],
] = {}


_PHYSICAL_EXPRESSION_CACHE_CAPACITY = 4


_PHYSICAL_EXPRESSION_KEYS: OrderedDict[
    tuple[str, str],
    tuple[
        dict[EffectiveSequenceExpression, EffectiveSequenceExpression],
        dict[str, Any],
    ],
] = OrderedDict()


_PRODUCT_FORWARD_CACHE_CAPACITY = 16


_PRODUCT_FORWARD_KEYS: OrderedDict[
    tuple[str, int, str],
    tuple[frozenset[ProductForwardKey], dict[str, int]],
] = OrderedDict()


_OVERLAP_CACHE_SCHEDULE_CAPACITY = 16


_OVERLAP_CACHE_SCHEDULES: dict[
    tuple[str, int, str],
    tuple[dict[str, int | None], dict[str, int]],
] = {}


_DERIVED_FORWARD_SCHEDULE_CAPACITY = 16


_DERIVED_FORWARD_SCHEDULES: dict[
    tuple[str, int, int, int, str],
    tuple[
        dict[str, tuple[tuple[int | None, int | None], ...]],
        dict[str, int],
    ],
] = {}


_NO_CONVOLUTION_CACHE_SLOTS = {
    "strict": (None, None, None, None, None),
    "overlap": (None, None, None),
}


_NO_DERIVED_FORWARD_SLOTS = {
    "strict": ((None, None),) * 5,
    "overlap": ((None, None),) * 3,
}


@dataclass(frozen=True)
class FusedTemporalUnit:
    unit_id: str
    group_id: str
    template_id: str
    slots: tuple[EffectiveSequenceExpression, ...]
    weights: tuple[tuple[int, int, complex], ...]
    source_core_ids: tuple[str, ...]
    basis_source_core_id: str
    varying_slot: str | None
    depth_weights: tuple[tuple[str, int, int, complex], ...] = ()

    def __post_init__(self) -> None:
        if len(self.slots) != len(SLOT_ORDER):
            raise ExtendedRuntimeError("a fused temporal unit requires six slots")
        if self.varying_slot is not None and self.varying_slot not in _SLOT_INDEX:
            raise ExtendedRuntimeError("unknown varying slot")

    def slot(self, name: str) -> EffectiveSequenceExpression:
        try:
            return self.slots[_SLOT_INDEX[name]]
        except KeyError as error:
            raise ExtendedRuntimeError(f"unknown effective slot {name!r}") from error


@dataclass(frozen=True)
class FusionGroupCertificate:
    group_id: str
    template_id: str
    varying_slot: str
    source_core_ids: tuple[str, ...]
    basis_source_core_ids: tuple[str, ...]
    coefficient_matrix: tuple[tuple[RationalCoefficient, ...], ...]
    source_symbolic_rows: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class FusedExecutionPlan:
    source_plan: SpecializedExecutionPlan
    source_core_ids: tuple[str, ...]
    symbolic_coordinates: tuple[SymbolicCoordinate, ...]
    units: tuple[FusedTemporalUnit, ...]
    groups: tuple[FusionGroupCertificate, ...]
    guard_rejection_count: int
    max_abs_coefficient_guard: RationalCoefficient
    digest: str
    history_depth_manifest_sha256: str | None = None

    @property
    def required_gamma_identities(self) -> frozenset[KernelIdentity]:
        return frozenset(
            identity
            for unit in self.units
            for expression in unit.slots
            for term in expression.terms
            for identity in term.monomial.kernel_identities
        )

    @property
    def guarded_basis_rescue_count(self) -> int:
        """Count groups whose canonical basis needed a guarded replacement.

        This is derived from the version-1 certificate rather than added to its
        digest payload, so existing exact-plan fingerprints remain stable.
        Validation proves that every noncanonical basis is precisely the
        deterministic alternate chosen after the canonical basis violates the
        coefficient guard.
        """

        return _count_guarded_basis_rescues(self.groups)

    @property
    def report(self) -> dict[str, Any]:
        templates = Counter(unit.template_id for unit in self.units)
        overlap = templates.get("overlap-uv-vw-v", 0)
        coefficients = [
            coefficient.fraction
            for group in self.groups
            for row in group.coefficient_matrix
            for coefficient in row
            if coefficient.numerator
        ]
        report = {
            "profile": "template-effective-one-varying-slot-rank-v1",
            "source_active_temporal_core_count": len(self.source_core_ids),
            "fused_temporal_unit_count": len(self.units),
            "fused_strict_temporal_unit_count": len(self.units) - overlap,
            "fused_overlap_temporal_unit_count": overlap,
            "eliminated_temporal_evaluations": len(self.source_core_ids) - len(self.units),
            "fusion_group_count": len(self.groups),
            "singleton_unit_count": sum(unit.varying_slot is None for unit in self.units),
            "fused_sparse_weight_count": sum(len(unit.weights) for unit in self.units),
            "required_gamma_sequence_identity_count": len(self.required_gamma_identities),
            "template_counts": dict(sorted(templates.items())),
            "max_abs_coefficient": float(max(map(abs, coefficients), default=0)),
            "max_abs_coefficient_guard": float(self.max_abs_coefficient_guard.fraction),
            "coefficient_guard_rejections": self.guard_rejection_count,
            "guarded_basis_rescue_count": self.guarded_basis_rescue_count,
            "digest": self.digest,
        }
        if self.history_depth_manifest_sha256 is not None:
            depth_counts = Counter(
                depth
                for unit in self.units
                for depth, _output, _input, _weight in unit.depth_weights
            )
            report["history_depth_instrumentation"] = {
                "enabled": True,
                "depth_manifest_sha256": self.history_depth_manifest_sha256,
                "fused_sparse_weight_counts": {
                    depth: depth_counts[depth] for depth in ("H0", "H1", "H2")
                },
                "fusion_row_space": "depth-augmented terminal coordinates",
                "temporal_units_are_depth_neutral": True,
            }
        return report


@dataclass(frozen=True)
class _SourceRecord:
    source_id: str
    weighted: Any
    monomials: tuple[EffectiveMonomial, ...]
    symbolic: Mapping[SymbolicCoordinate, int]
    row: tuple[int, ...]


@dataclass(frozen=True)
class _SelectedGroup:
    template_id: str
    varying_index: int
    source_indices: tuple[int, ...]
    basis_indices: tuple[int, ...]
    coefficients: tuple[tuple[Fraction, ...], ...]
    guarded_basis_rescue: bool


def _vector_add(left: PhaseVector, right: PhaseVector) -> PhaseVector:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _vector_subtract(left: PhaseVector, right: PhaseVector) -> PhaseVector:
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _effective_monomials(weighted: Any) -> tuple[EffectiveMonomial, ...]:
    core = weighted.core
    grouped: dict[str, list[KernelIdentity]] = defaultdict(list)
    for identity in core.kernel_identities:
        support = identity[0]
        base = support[1:] if support.startswith("-") else support
        if base not in _SLOT_INDEX:
            raise ExtendedRuntimeError(f"unknown kernel support {support!r}")
        grouped[base].append(identity)

    zero = tuple(0 for _ in core.phase_u)
    phases: dict[str, PhaseVector] = {name: zero for name in SLOT_ORDER}
    if core.template_id in {"prefix-overlap-u", "prefix-overlap-unit-u"}:
        phases["UV"] = _vector_subtract(core.phase_v, core.phase_w)
        phases["VW"] = core.phase_w
        phases["U"] = _vector_add(
            core.phase_u, _vector_subtract(core.phase_w, core.phase_v)
        )
    elif core.template_id == "prefix-overlap-w":
        phases["UV"] = core.phase_u
        phases["VW"] = _vector_subtract(core.phase_v, core.phase_u)
        phases["W"] = _vector_add(
            core.phase_w, _vector_subtract(core.phase_u, core.phase_v)
        )
    elif core.template_id == "overlap-uv-vw-v":
        phases["UV"] = core.phase_u
        phases["VW"] = core.phase_w
        phases["V"] = _vector_subtract(
            core.phase_v, _vector_add(core.phase_u, core.phase_w)
        )
    else:
        phases["U"] = core.phase_u
        phases["V"] = core.phase_v
        phases["W"] = core.phase_w
    return tuple(
        EffectiveMonomial(tuple(sorted(grouped[name])), phases[name])
        for name in SLOT_ORDER
    )


def _symbolic_weights(
    weighted: Any, coupling_operator: Any
) -> dict[SymbolicCoordinate, int]:
    values: dict[SymbolicCoordinate, int] = defaultdict(int)
    depths = {
        contribution.history_depth
        for contribution in weighted.core.contributions
    }
    instrumented = depths != {None}
    if instrumented and (None in depths or not depths.issubset({"H0", "H1", "H2"})):
        raise ExtendedRuntimeError(
            "history-depth contributions must be uniformly tagged H0, H1, or H2"
        )
    for contribution in weighted.core.contributions:
        if any(
            coupling_operator[row, column] == 0
            for row, column in contribution.coupling_entries
        ):
            continue
        if instrumented:
            coordinate: SymbolicCoordinate = (
                contribution.history_depth,
                contribution.output_flat,
                contribution.input_flat,
                tuple(sorted(contribution.coupling_entries)),
            )
        else:
            coordinate = (
                contribution.output_flat,
                contribution.input_flat,
                tuple(sorted(contribution.coupling_entries)),
            )
        values[coordinate] += contribution.coefficient
    return {coordinate: value for coordinate, value in values.items() if value}


def _matrix_rank(rows: Sequence[Sequence[int]]) -> int:
    if not rows:
        return 0
    matrix = [[Fraction(value) for value in row] for row in rows]
    row_count = len(matrix)
    column_count = len(matrix[0])
    pivot_row = 0
    for column in range(column_count):
        pivot = next(
            (
                row
                for row in range(pivot_row, row_count)
                if matrix[row][column]
            ),
            None,
        )
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        divisor = matrix[pivot_row][column]
        matrix[pivot_row] = [value / divisor for value in matrix[pivot_row]]
        for row in range(pivot_row + 1, row_count):
            multiplier = matrix[row][column]
            if multiplier:
                matrix[row] = [
                    value - multiplier * base
                    for value, base in zip(
                        matrix[row], matrix[pivot_row], strict=True
                    )
                ]
        pivot_row += 1
        if pivot_row == row_count:
            break
    return pivot_row


def _factor_for_basis(
    rows: Sequence[tuple[int, ...]],
    source_indices: tuple[int, ...],
    basis_indices: tuple[int, ...],
) -> tuple[tuple[Fraction, ...], ...] | None:
    basis = sp.Matrix([rows[index] for index in basis_indices])
    pivot_columns = basis.rref()[1]
    if len(pivot_columns) != len(basis_indices):
        return None
    square = basis[:, list(pivot_columns)]
    if square.det() == 0:
        return None
    inverse = square.inv()
    coefficients: list[tuple[Fraction, ...]] = []
    for source_index in source_indices:
        restricted = sp.Matrix(
            [[rows[source_index][column] for column in pivot_columns]]
        )
        solved = restricted * inverse
        if solved * basis != sp.Matrix([rows[source_index]]):
            raise ExtendedRuntimeError("exact symbolic rank reconstruction failed")
        coefficients.append(
            tuple(Fraction(int(value.p), int(value.q)) for value in solved)
        )
    return tuple(coefficients)


def _canonical_independent_basis(
    rows: Sequence[tuple[int, ...]], source_indices: tuple[int, ...], rank: int
) -> tuple[int, ...]:
    basis: list[int] = []
    for source_index in source_indices:
        trial = (*basis, source_index)
        if _matrix_rank([rows[index] for index in trial]) > len(basis):
            basis.append(source_index)
        if len(basis) == rank:
            break
    if len(basis) != rank:
        raise ExtendedRuntimeError("could not select an independent symbolic basis")
    return tuple(basis)


def _count_guarded_basis_rescues(
    groups: Sequence[FusionGroupCertificate],
) -> int:
    """Return the number of certified bases that are not first-independent."""

    rescued = 0
    for group in groups:
        source_positions = {
            source_id: index
            for index, source_id in enumerate(group.source_core_ids)
        }
        selected = tuple(
            source_positions[source_id]
            for source_id in group.basis_source_core_ids
        )
        canonical = _canonical_independent_basis(
            group.source_symbolic_rows,
            tuple(range(len(group.source_core_ids))),
            len(group.basis_source_core_ids),
        )
        rescued += selected != canonical
    return rescued


def _guarded_basis_factorization(
    rows: Sequence[tuple[int, ...]],
    source_indices: tuple[int, ...],
    rank: int,
    guard: Fraction,
    weight_arities: Sequence[int],
) -> tuple[
    tuple[int, ...],
    tuple[tuple[Fraction, ...], ...],
    Fraction,
    bool,
]:
    """Factor rows with canonical-first, guard-rescue-only precedence.

    The first-independent basis in canonical source order always wins when its
    synthesis coefficients satisfy ``guard``.  Only if that basis violates the
    guard do we consider alternatives.  Feasible alternatives are ordered by
    (1) smallest maximum absolute coefficient, (2) largest retained sparse
    weight arity, and (3) lexicographically smallest source-index tuple.  The
    returned Boolean is true exactly when such an alternate rescues the group.
    """

    canonical_basis = _canonical_independent_basis(
        rows, source_indices, rank
    )
    canonical_coefficients = _factor_for_basis(
        rows, source_indices, canonical_basis
    )
    assert canonical_coefficients is not None
    canonical_maximum = max(
        (
            abs(value)
            for coefficient_row in canonical_coefficients
            for value in coefficient_row
        ),
        default=Fraction(0),
    )
    if canonical_maximum <= guard:
        return canonical_basis, canonical_coefficients, canonical_maximum, False

    combination_count = math.comb(len(source_indices), rank)
    if combination_count <= 100_000:
        candidate_bases = combinations(source_indices, rank)
    else:
        candidate_bases = iter(
            (_canonical_independent_basis(rows, source_indices, rank),)
        )
    best: tuple[
        Fraction,
        int,
        tuple[int, ...],
        tuple[tuple[Fraction, ...], ...],
    ] | None = None
    for basis_indices in candidate_bases:
        basis_tuple = tuple(basis_indices)
        coefficients = _factor_for_basis(rows, source_indices, basis_tuple)
        if coefficients is None:
            continue
        maximum = max(
            (
                abs(value)
                for coefficient_row in coefficients
                for value in coefficient_row
            ),
            default=Fraction(0),
        )
        retained_weights = sum(weight_arities[index] for index in basis_tuple)
        candidate = (maximum, -retained_weights, basis_tuple, coefficients)
        if maximum <= guard and (
            best is None or candidate[:3] < best[:3]
        ):
            best = candidate
    if best is None:
        return canonical_basis, canonical_coefficients, canonical_maximum, False
    return best[2], best[3], best[0], True


def _monomial_json(monomial: EffectiveMonomial) -> list[Any]:
    return [
        [list(identity) for identity in monomial.kernel_identities],
        list(monomial.phase),
    ]


def _candidate_key(
    template_id: str,
    fixed_monomials: tuple[EffectiveMonomial, ...],
    varying_index: int,
    source_indices: tuple[int, ...],
) -> str:
    return canonical_json(
        [
            template_id,
            [_monomial_json(monomial) for monomial in fixed_monomials],
            SLOT_ORDER[varying_index],
            list(source_indices),
        ]
    )


def _single_expression(monomial: EffectiveMonomial) -> EffectiveSequenceExpression:
    return EffectiveSequenceExpression(
        (
            EffectiveSequenceTerm(
                RationalCoefficient(1),
                monomial,
            ),
        )
    )


def _combined_expression(
    monomials: Sequence[EffectiveMonomial], coefficients: Sequence[Fraction]
) -> EffectiveSequenceExpression:
    collected: dict[EffectiveMonomial, Fraction] = defaultdict(Fraction)
    for monomial, coefficient in zip(monomials, coefficients, strict=True):
        collected[monomial] += coefficient
    terms = tuple(
        EffectiveSequenceTerm(
            RationalCoefficient.from_fraction(coefficient), monomial
        )
        for monomial, coefficient in sorted(collected.items())
        if coefficient
    )
    if not terms:
        raise ExtendedRuntimeError("fusion produced an empty varying expression")
    return EffectiveSequenceExpression(terms)


def _coefficient_json(value: RationalCoefficient) -> list[int]:
    return [value.numerator, value.denominator]


def _expression_json(expression: EffectiveSequenceExpression) -> list[Any]:
    return [
        [_coefficient_json(term.coefficient), _monomial_json(term.monomial)]
        for term in expression.terms
    ]


def _coordinate_json(coordinate: SymbolicCoordinate) -> list[Any]:
    """Serialize plain coordinates compatibly and tagged coordinates explicitly."""

    if len(coordinate) == 3:
        output, input_, coupling = coordinate
        return [output, input_, [list(entry) for entry in coupling]]
    depth, output, input_, coupling = coordinate
    return [depth, output, input_, [list(entry) for entry in coupling]]


def _digest_payload(
    *,
    source_core_ids: tuple[str, ...],
    coordinates: tuple[SymbolicCoordinate, ...],
    units: tuple[FusedTemporalUnit, ...],
    groups: tuple[FusionGroupCertificate, ...],
    guard_rejection_count: int,
    guard: RationalCoefficient,
    history_depth_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    payload = {
        "profile": "template-effective-one-varying-slot-rank-v1",
        "source_core_ids": list(source_core_ids),
        "symbolic_coordinates": [_coordinate_json(value) for value in coordinates],
        "units": [
            {
                "unit_id": unit.unit_id,
                "group_id": unit.group_id,
                "template_id": unit.template_id,
                "slots": [_expression_json(expression) for expression in unit.slots],
                "weights": [
                    [output, input_, [float(weight.real).hex(), float(weight.imag).hex()]]
                    for output, input_, weight in unit.weights
                ],
                "source_core_ids": list(unit.source_core_ids),
                "basis_source_core_id": unit.basis_source_core_id,
                "varying_slot": unit.varying_slot,
            }
            for unit in units
        ],
        "groups": [
            {
                "group_id": group.group_id,
                "template_id": group.template_id,
                "varying_slot": group.varying_slot,
                "source_core_ids": list(group.source_core_ids),
                "basis_source_core_ids": list(group.basis_source_core_ids),
                "coefficient_matrix": [
                    [_coefficient_json(value) for value in row]
                    for row in group.coefficient_matrix
                ],
                "source_symbolic_rows": [list(row) for row in group.source_symbolic_rows],
            }
            for group in groups
        ],
        "guard_rejection_count": guard_rejection_count,
        "max_abs_coefficient_guard": _coefficient_json(guard),
    }
    if history_depth_manifest_sha256 is not None:
        payload["history_depth_manifest_sha256"] = history_depth_manifest_sha256
        for unit_record, unit in zip(payload["units"], units, strict=True):
            unit_record["depth_weights"] = [
                [
                    depth,
                    output,
                    input_,
                    [float(weight.real).hex(), float(weight.imag).hex()],
                ]
                for depth, output, input_, weight in unit.depth_weights
            ]
    return payload


def fuse_specialized_plan(
    plan: SpecializedExecutionPlan,
    *,
    max_abs_coefficient: int | Fraction = 4,
) -> FusedExecutionPlan:
    """Compile a deterministic exact fused plan from active temporal cores."""

    if isinstance(max_abs_coefficient, bool):
        raise ExtendedRuntimeError("max_abs_coefficient must be positive")
    try:
        guard_fraction = Fraction(max_abs_coefficient)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ExtendedRuntimeError(
            "max_abs_coefficient must be a positive rational value"
        ) from error
    if guard_fraction <= 0:
        raise ExtendedRuntimeError("max_abs_coefficient must be positive")
    guard = RationalCoefficient.from_fraction(guard_fraction)

    if any(not weighted.core.contributions for weighted in plan.cores):
        raise ExtendedRuntimeError(
            "exact fusion requires uncontracted integer coupling-entry "
            "provenance; this model-bound plan contains only contracted "
            "complex weights"
        )

    source_ordinals = {
        core.key: ordinal
        for ordinal, core in enumerate(plan.source_plan.cores)
    }
    ordered_weighted = sorted(
        plan.cores, key=lambda weighted: source_ordinals[weighted.core.key]
    )
    source_ids = tuple(
        f"source-{source_ordinals[weighted.core.key]:05d}"
        for weighted in ordered_weighted
    )
    symbolic_maps = tuple(
        _symbolic_weights(weighted, plan.coupling_operator)
        for weighted in ordered_weighted
    )
    coordinates = tuple(
        sorted(
            {
                coordinate
                for symbolic in symbolic_maps
                for coordinate in symbolic
            }
        )
    )
    rows = tuple(
        tuple(symbolic.get(coordinate, 0) for coordinate in coordinates)
        for symbolic in symbolic_maps
    )
    records = tuple(
        _SourceRecord(
            source_id=source_id,
            weighted=weighted,
            monomials=_effective_monomials(weighted),
            symbolic=symbolic,
            row=row,
        )
        for source_id, weighted, symbolic, row in zip(
            source_ids,
            ordered_weighted,
            symbolic_maps,
            rows,
            strict=True,
        )
    )

    buckets: list[
        tuple[str, int, tuple[EffectiveMonomial, ...], tuple[int, ...]]
    ] = []
    for varying_index in range(len(SLOT_ORDER)):
        grouped: dict[
            tuple[str, tuple[EffectiveMonomial, ...]], list[int]
        ] = defaultdict(list)
        for source_index, record in enumerate(records):
            fixed = (
                record.monomials[:varying_index]
                + record.monomials[varying_index + 1 :]
            )
            grouped[(record.weighted.core.template_id, fixed)].append(
                source_index
            )
        for (template_id, fixed), source_indices in grouped.items():
            if len(source_indices) > 1:
                buckets.append(
                    (template_id, varying_index, fixed, tuple(source_indices))
                )

    unused = set(range(len(records)))
    rejected_states: set[tuple[int, tuple[int, ...]]] = set()
    selected: list[_SelectedGroup] = []
    while True:
        candidates: list[
            tuple[
                Fraction,
                int,
                str,
                int,
                str,
                int,
                tuple[int, ...],
                int,
            ]
        ] = []
        for bucket_index, (
            template_id,
            varying_index,
            fixed,
            all_source_indices,
        ) in enumerate(buckets):
            source_indices = tuple(
                index for index in all_source_indices if index in unused
            )
            state = (bucket_index, source_indices)
            if len(source_indices) < 2 or state in rejected_states:
                continue
            rank = _matrix_rank([rows[index] for index in source_indices])
            savings = len(source_indices) - rank
            if savings <= 0:
                continue
            candidates.append(
                (
                    Fraction(savings, len(source_indices)),
                    savings,
                    _candidate_key(
                        template_id, fixed, varying_index, source_indices
                    ),
                    bucket_index,
                    template_id,
                    varying_index,
                    source_indices,
                    rank,
                )
            )
        if not candidates:
            break
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        chosen: _SelectedGroup | None = None
        for (
            _ratio,
            _savings,
            _key,
            bucket_index,
            template_id,
            varying_index,
            source_indices,
            rank,
        ) in candidates:
            (
                basis_indices,
                coefficients,
                maximum,
                guarded_basis_rescue,
            ) = _guarded_basis_factorization(
                rows,
                source_indices,
                rank,
                guard_fraction,
                tuple(len(record.weighted.weights) for record in records),
            )
            if maximum > guard_fraction:
                rejected_states.add((bucket_index, source_indices))
                continue
            chosen = _SelectedGroup(
                template_id=template_id,
                varying_index=varying_index,
                source_indices=source_indices,
                basis_indices=basis_indices,
                coefficients=coefficients,
                guarded_basis_rescue=guarded_basis_rescue,
            )
            break
        if chosen is None:
            break
        selected.append(chosen)
        unused.difference_update(chosen.source_indices)

    units: list[FusedTemporalUnit] = []
    certificates: list[FusionGroupCertificate] = []
    for group_ordinal, selected_group in enumerate(selected):
        group_id = f"group-{group_ordinal:04d}"
        source_group_ids = tuple(
            records[index].source_id for index in selected_group.source_indices
        )
        basis_group_ids = tuple(
            records[index].source_id for index in selected_group.basis_indices
        )
        coefficient_records = tuple(
            tuple(RationalCoefficient.from_fraction(value) for value in row)
            for row in selected_group.coefficients
        )
        certificates.append(
            FusionGroupCertificate(
                group_id=group_id,
                template_id=selected_group.template_id,
                varying_slot=SLOT_ORDER[selected_group.varying_index],
                source_core_ids=source_group_ids,
                basis_source_core_ids=basis_group_ids,
                coefficient_matrix=coefficient_records,
                source_symbolic_rows=tuple(
                    rows[index] for index in selected_group.source_indices
                ),
            )
        )
        varying_monomials = tuple(
            records[index].monomials[selected_group.varying_index]
            for index in selected_group.source_indices
        )
        for basis_column, basis_index in enumerate(
            selected_group.basis_indices
        ):
            expressions = [
                _single_expression(monomial)
                for monomial in records[basis_index].monomials
            ]
            expressions[selected_group.varying_index] = _combined_expression(
                varying_monomials,
                tuple(
                    row[basis_column]
                    for row in selected_group.coefficients
                ),
            )
            unit_id = f"unit-{len(units):04d}"
            units.append(
                FusedTemporalUnit(
                    unit_id=unit_id,
                    group_id=group_id,
                    template_id=selected_group.template_id,
                    slots=tuple(expressions),
                    weights=records[basis_index].weighted.weights,
                    source_core_ids=source_group_ids,
                    basis_source_core_id=records[basis_index].source_id,
                    varying_slot=SLOT_ORDER[selected_group.varying_index],
                    depth_weights=(
                        records[basis_index].weighted.depth_weights
                    ),
                )
            )

    for source_index in sorted(unused):
        record = records[source_index]
        unit_id = f"unit-{len(units):04d}"
        units.append(
            FusedTemporalUnit(
                unit_id=unit_id,
                group_id=unit_id,
                template_id=record.weighted.core.template_id,
                slots=tuple(
                    _single_expression(monomial)
                    for monomial in record.monomials
                ),
                weights=record.weighted.weights,
                source_core_ids=(record.source_id,),
                basis_source_core_id=record.source_id,
                varying_slot=None,
                depth_weights=record.weighted.depth_weights,
            )
        )

    unit_tuple = tuple(units)
    certificate_tuple = tuple(certificates)
    payload = _digest_payload(
        source_core_ids=source_ids,
        coordinates=coordinates,
        units=unit_tuple,
        groups=certificate_tuple,
        guard_rejection_count=len(rejected_states),
        guard=guard,
        history_depth_manifest_sha256=(
            plan.source_plan.history_depth_manifest_sha256
        ),
    )
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    fused = FusedExecutionPlan(
        source_plan=plan,
        source_core_ids=source_ids,
        symbolic_coordinates=coordinates,
        units=unit_tuple,
        groups=certificate_tuple,
        guard_rejection_count=len(rejected_states),
        max_abs_coefficient_guard=guard,
        digest=digest,
        history_depth_manifest_sha256=(
            plan.source_plan.history_depth_manifest_sha256
        ),
    )
    if fused.guarded_basis_rescue_count != sum(
        group.guarded_basis_rescue for group in selected
    ):
        raise ExtendedRuntimeError(
            "guarded alternate-basis accounting is inconsistent"
        )
    validate_fused_plan(fused)
    return fused


def validate_fused_plan(plan: FusedExecutionPlan) -> None:
    """Validate certificates as a complete, disjoint structural proof.

    In addition to checking ``W = C D`` exactly, this proves that every varying
    unit belongs to exactly one certificate, every certificate emits exactly
    one unit per basis row, and the remaining units are genuine singletons.
    It also replays the canonical-first guarded basis rule.
    """

    source_manifest = (
        plan.source_plan.source_plan.history_depth_manifest_sha256
    )
    if plan.history_depth_manifest_sha256 != source_manifest:
        raise ExtendedRuntimeError(
            "fused plan history-depth manifest binding was mutated"
        )
    source_ordinals = {
        core.key: ordinal
        for ordinal, core in enumerate(plan.source_plan.source_plan.cores)
    }
    ordered_weighted = sorted(
        plan.source_plan.cores,
        key=lambda weighted: source_ordinals[weighted.core.key],
    )
    expected_ids = tuple(
        f"source-{source_ordinals[weighted.core.key]:05d}"
        for weighted in ordered_weighted
    )
    if plan.source_core_ids != expected_ids:
        raise ExtendedRuntimeError("fused source-core IDs are not canonical")
    symbolic_maps = tuple(
        _symbolic_weights(weighted, plan.source_plan.coupling_operator)
        for weighted in ordered_weighted
    )
    expected_coordinates = tuple(
        sorted(
            {
                coordinate
                for symbolic in symbolic_maps
                for coordinate in symbolic
            }
        )
    )
    if plan.symbolic_coordinates != expected_coordinates:
        raise ExtendedRuntimeError("fused symbolic coordinates were mutated")
    rows = tuple(
        tuple(symbolic.get(coordinate, 0) for coordinate in expected_coordinates)
        for symbolic in symbolic_maps
    )
    records = {
        source_id: (weighted, _effective_monomials(weighted), row)
        for source_id, weighted, row in zip(
            expected_ids, ordered_weighted, rows, strict=True
        )
    }
    source_order = {
        source_id: index for index, source_id in enumerate(expected_ids)
    }

    if tuple(unit.unit_id for unit in plan.units) != tuple(
        f"unit-{ordinal:04d}" for ordinal in range(len(plan.units))
    ):
        raise ExtendedRuntimeError("fused unit IDs are not canonical")

    group_by_id: dict[str, FusionGroupCertificate] = {}
    for group in plan.groups:
        if group.group_id in group_by_id:
            raise ExtendedRuntimeError("duplicate fusion group ID")
        group_by_id[group.group_id] = group
    if tuple(group_by_id) != tuple(
        f"group-{ordinal:04d}" for ordinal in range(len(plan.groups))
    ):
        raise ExtendedRuntimeError("fusion group IDs are not canonical")

    certified_units: dict[str, list[FusedTemporalUnit]] = {
        group_id: [] for group_id in group_by_id
    }
    singleton_units: list[FusedTemporalUnit] = []
    for unit in plan.units:
        if unit.varying_slot is None:
            singleton_units.append(unit)
            continue
        if unit.group_id not in group_by_id:
            raise ExtendedRuntimeError(
                "varying fused unit references an unknown fusion certificate"
            )
        certified_units[unit.group_id].append(unit)

    certified_unit_ids = {
        unit.unit_id for units in certified_units.values() for unit in units
    }
    singleton_unit_ids = {unit.unit_id for unit in singleton_units}
    all_unit_ids = {unit.unit_id for unit in plan.units}
    if (
        certified_unit_ids.intersection(singleton_unit_ids)
        or certified_unit_ids.union(singleton_unit_ids) != all_unit_ids
        or len(certified_unit_ids) + len(singleton_unit_ids) != len(plan.units)
    ):
        raise ExtendedRuntimeError(
            "certified units and singletons do not exactly partition the plan"
        )

    seen_sources: set[str] = set()
    observed_guarded_rescues = 0
    for group in plan.groups:
        if group.varying_slot not in _SLOT_INDEX:
            raise ExtendedRuntimeError("fusion certificate has an unknown slot")
        if len(group.source_core_ids) != len(set(group.source_core_ids)):
            raise ExtendedRuntimeError(
                "fusion certificate contains duplicate source IDs"
            )
        if len(group.basis_source_core_ids) != len(
            set(group.basis_source_core_ids)
        ):
            raise ExtendedRuntimeError(
                "fusion certificate contains duplicate basis IDs"
            )
        rank = len(group.basis_source_core_ids)
        if rank == 0 or len(group.source_core_ids) <= rank:
            raise ExtendedRuntimeError(
                "fusion certificate must describe a nontrivial rank reduction"
            )
        if not set(group.basis_source_core_ids).issubset(
            group.source_core_ids
        ):
            raise ExtendedRuntimeError("fusion basis is not a source-row subset")
        try:
            source_records = tuple(
                records[source_id] for source_id in group.source_core_ids
            )
            basis_rows = tuple(
                records[source_id][2]
                for source_id in group.basis_source_core_ids
            )
        except KeyError as error:
            raise ExtendedRuntimeError(
                "fusion certificate references an unknown source"
            ) from error
        if tuple(
            sorted(group.source_core_ids, key=source_order.__getitem__)
        ) != group.source_core_ids:
            raise ExtendedRuntimeError(
                "fusion certificate source IDs are not in canonical order"
            )
        if seen_sources.intersection(group.source_core_ids):
            raise ExtendedRuntimeError(
                "a source core occurs in two fusion groups"
            )
        seen_sources.update(group.source_core_ids)

        if any(
            weighted.core.template_id != group.template_id
            for weighted, _monomials, _row in source_records
        ):
            raise ExtendedRuntimeError(
                "fusion source template does not match its certificate"
            )
        varying_index = _SLOT_INDEX[group.varying_slot]
        source_monomials = tuple(
            monomials for _weighted, monomials, _row in source_records
        )
        fixed_reference = tuple(
            monomial
            for slot_index, monomial in enumerate(source_monomials[0])
            if slot_index != varying_index
        )
        if any(
            tuple(
                monomial
                for slot_index, monomial in enumerate(monomials)
                if slot_index != varying_index
            )
            != fixed_reference
            for monomials in source_monomials[1:]
        ):
            raise ExtendedRuntimeError(
                "fusion sources do not share the five fixed effective ports"
            )

        expected_rows = tuple(row for _weighted, _monomials, row in source_records)
        if group.source_symbolic_rows != expected_rows:
            raise ExtendedRuntimeError(
                "fusion certificate symbolic rows were mutated"
            )
        if len(group.coefficient_matrix) != len(group.source_core_ids):
            raise ExtendedRuntimeError(
                "fusion coefficient row count is inconsistent"
            )
        if _matrix_rank(basis_rows) != rank:
            raise ExtendedRuntimeError("fusion basis rows are not independent")
        for source_row, coefficient_row in zip(
            expected_rows, group.coefficient_matrix, strict=True
        ):
            if len(coefficient_row) != rank:
                raise ExtendedRuntimeError(
                    "fusion coefficient column count is inconsistent"
                )
            if any(
                abs(value.fraction) > plan.max_abs_coefficient_guard.fraction
                for value in coefficient_row
            ):
                raise ExtendedRuntimeError(
                    "fusion coefficient exceeds its exact guard"
                )
            reconstructed = tuple(
                sum(
                    coefficient.fraction * basis_row[column]
                    for coefficient, basis_row in zip(
                        coefficient_row, basis_rows, strict=True
                    )
                )
                for column in range(len(expected_coordinates))
            )
            if reconstructed != source_row:
                raise ExtendedRuntimeError(
                    "fusion certificate does not reconstruct W=C D"
                )

        (
            expected_basis_indices,
            expected_coefficients,
            expected_maximum,
            guarded_basis_rescue,
        ) = _guarded_basis_factorization(
            expected_rows,
            tuple(range(len(group.source_core_ids))),
            rank,
            plan.max_abs_coefficient_guard.fraction,
            tuple(len(weighted.weights) for weighted, _monomials, _row in source_records),
        )
        if expected_maximum > plan.max_abs_coefficient_guard.fraction:
            raise ExtendedRuntimeError(
                "fusion certificate has no basis satisfying its exact guard"
            )
        expected_basis_ids = tuple(
            group.source_core_ids[index] for index in expected_basis_indices
        )
        if group.basis_source_core_ids != expected_basis_ids:
            raise ExtendedRuntimeError(
                "fusion basis violates canonical guarded-selection precedence"
            )
        actual_coefficients = tuple(
            tuple(value.fraction for value in row)
            for row in group.coefficient_matrix
        )
        if actual_coefficients != expected_coefficients:
            raise ExtendedRuntimeError(
                "fusion coefficients differ from the deterministic exact factorization"
            )
        observed_guarded_rescues += guarded_basis_rescue

        group_units = tuple(certified_units[group.group_id])
        if len(group_units) != rank:
            raise ExtendedRuntimeError(
                "fusion group does not emit one unit per basis row"
            )
        for basis_column, (unit, basis_source_id) in enumerate(
            zip(group_units, group.basis_source_core_ids, strict=True)
        ):
            if (
                unit.template_id != group.template_id
                or unit.varying_slot != group.varying_slot
                or unit.source_core_ids != group.source_core_ids
                or unit.basis_source_core_id != basis_source_id
            ):
                raise ExtendedRuntimeError("fusion unit provenance was mutated")
            weighted, basis_monomials, _row = records[basis_source_id]
            if unit.weights != weighted.weights:
                raise ExtendedRuntimeError(
                    "fusion unit numeric weights were mutated"
                )
            if unit.depth_weights != weighted.depth_weights:
                raise ExtendedRuntimeError(
                    "fusion unit history-depth weights were mutated"
                )
            for slot_index, expression in enumerate(unit.slots):
                if slot_index == varying_index:
                    expected = _combined_expression(
                        tuple(
                            monomials[varying_index]
                            for monomials in source_monomials
                        ),
                        tuple(
                            coefficient_row[basis_column].fraction
                            for coefficient_row in group.coefficient_matrix
                        ),
                    )
                else:
                    expected = _single_expression(basis_monomials[slot_index])
                if expression != expected:
                    raise ExtendedRuntimeError(
                        "fusion unit effective expression was mutated"
                    )

    if observed_guarded_rescues != plan.guarded_basis_rescue_count:
        raise ExtendedRuntimeError(
            "guarded alternate-basis rescue accounting is inconsistent"
        )

    for unit in singleton_units:
        if (
            unit.group_id != unit.unit_id
            or len(unit.source_core_ids) != 1
            or unit.basis_source_core_id != unit.source_core_ids[0]
        ):
            raise ExtendedRuntimeError(
                "singleton fused-unit provenance is malformed"
            )
        source_id = unit.source_core_ids[0]
        if source_id in seen_sources:
            raise ExtendedRuntimeError(
                "a fused source is also emitted as a singleton"
            )
        try:
            weighted, monomials, _row = records[source_id]
        except KeyError as error:
            raise ExtendedRuntimeError(
                "singleton references an unknown source"
            ) from error
        seen_sources.add(source_id)
        if (
            unit.template_id != weighted.core.template_id
            or unit.weights != weighted.weights
            or unit.depth_weights != weighted.depth_weights
        ):
            raise ExtendedRuntimeError("singleton fused unit was mutated")
        if unit.slots != tuple(
            _single_expression(value) for value in monomials
        ):
            raise ExtendedRuntimeError(
                "singleton effective slots were mutated"
            )
    if seen_sources != set(expected_ids):
        raise ExtendedRuntimeError(
            "fused units do not partition all source cores"
        )

    payload = _digest_payload(
        source_core_ids=plan.source_core_ids,
        coordinates=plan.symbolic_coordinates,
        units=plan.units,
        groups=plan.groups,
        guard_rejection_count=plan.guard_rejection_count,
        guard=plan.max_abs_coefficient_guard,
        history_depth_manifest_sha256=plan.history_depth_manifest_sha256,
    )
    expected_digest = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    if plan.digest != expected_digest:
        raise ExtendedRuntimeError("fused plan digest was mutated")


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


class _CPUPhaseCache:
    def __init__(
        self,
        level_ratios: Sequence[complex],
        length: int,
        capacity: int,
        ones: Any,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
            raise ExtendedRuntimeError("phase_cache_size must be nonnegative")
        np = _numpy()
        self.level_ratios = tuple(level_ratios)
        self.capacity = capacity
        self.index = np.arange(length, dtype=np.int64)
        self.ones = ones
        self.values: OrderedDict[PhaseVector, Any] = OrderedDict()

    def get(self, vector: PhaseVector) -> Any:
        np = _numpy()
        if all(exponent == 0 for exponent in vector):
            return self.ones
        cached = self.values.get(vector)
        if cached is not None:
            self.values.move_to_end(vector)
            return cached
        ratio = _phase_ratio(self.level_ratios, vector)
        values = np.asarray(np.power(ratio, self.index), dtype=np.complex128)
        if self.capacity:
            self.values[vector] = values
            self.values.move_to_end(vector)
            while len(self.values) > self.capacity:
                self.values.popitem(last=False)
        return values


def _normalize_gamma_sequences(
    plan: FusedExecutionPlan,
    gamma_sequences: Mapping[KernelIdentity, Sequence[complex]],
    *,
    length: int,
) -> dict[KernelIdentity, Any]:
    np = _numpy()
    normalized: dict[KernelIdentity, Any] = {}
    validated: dict[int, tuple[Any, Any]] = {}
    for identity in sorted(plan.required_gamma_identities):
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


def _physical_expression_keys(
    plan: FusedExecutionPlan,
    gamma_sequences: Mapping[KernelIdentity, Sequence[complex]],
    *,
    enabled: bool,
) -> tuple[
    dict[EffectiveSequenceExpression, EffectiveSequenceExpression],
    dict[str, Any],
]:
    """Canonicalize only bound curve atoms, never HR/fusion port semantics."""

    if not enabled:
        return {}, {
            "requested": False,
            "selected": False,
            "reason": "explicitly disabled",
            "semantic_expression_count": None,
            "physical_expression_count": None,
            "collapsed_expression_count": 0,
            "term_merge_count": 0,
            "analysis_skipped": True,
            "cache_hit": False,
        }
    semantic_to_physical = getattr(
        gamma_sequences, "semantic_to_physical", None
    )
    if semantic_to_physical is None:
        return {}, {
            "requested": True,
            "selected": False,
            "reason": "Gamma mapping has no certified physical binding",
            "semantic_expression_count": None,
            "physical_expression_count": None,
            "collapsed_expression_count": 0,
            "term_merge_count": 0,
            "analysis_skipped": True,
            "cache_hit": False,
        }

    binding_digest = hashlib.sha256(
        repr(tuple(sorted(semantic_to_physical.items()))).encode("utf-8")
    ).hexdigest()
    cache_key = (plan.digest, binding_digest)
    cached = _PHYSICAL_EXPRESSION_KEYS.get(cache_key)
    if cached is not None:
        _PHYSICAL_EXPRESSION_KEYS.move_to_end(cache_key)
        mapping, profile = cached
        return mapping, {**profile, "cache_hit": True}

    expressions = {
        expression for unit in plan.units for expression in unit.slots
    }

    result: dict[EffectiveSequenceExpression, EffectiveSequenceExpression] = {}
    term_merge_count = 0
    for expression in expressions:
        coefficients: dict[EffectiveMonomial, Fraction | complex] = {}
        for term in expression.terms:
            try:
                physical_identities = tuple(
                    sorted(
                        semantic_to_physical[identity]
                        for identity in term.monomial.kernel_identities
                    )
                )
            except KeyError as error:
                return {}, {
                    "requested": True,
                    "selected": False,
                    "reason": (
                        "physical binding does not cover every expression atom"
                    ),
                    "semantic_expression_count": len(expressions),
                    "physical_expression_count": len(expressions),
                    "collapsed_expression_count": 0,
                    "term_merge_count": 0,
                    "missing_identity": repr(error.args[0]),
                }
            monomial = EffectiveMonomial(
                kernel_identities=physical_identities,
                phase=term.monomial.phase,
            )
            value = _coefficient_value(term.coefficient)
            zero: Fraction | complex = (
                Fraction(0) if isinstance(value, Fraction) else 0.0j
            )
            coefficients[monomial] = coefficients.get(monomial, zero) + value
        physical_terms = tuple(
            EffectiveSequenceTerm(
                coefficient=(
                    RationalCoefficient.from_fraction(coefficient)
                    if isinstance(coefficient, Fraction)
                    else ComplexCoefficient.from_complex(coefficient)
                ),
                monomial=monomial,
            )
            for monomial, coefficient in sorted(coefficients.items())
            if coefficient
        )
        if not physical_terms:
            return {}, {
                "requested": True,
                "selected": False,
                "reason": "physical quotient produced a zero expression",
                "semantic_expression_count": len(expressions),
                "physical_expression_count": len(expressions),
                "collapsed_expression_count": 0,
                "term_merge_count": 0,
            }
        physical = EffectiveSequenceExpression(physical_terms)
        result[expression] = physical
        term_merge_count += len(expression.terms) - len(physical.terms)
    physical_count = len(set(result.values()))
    alias_classes = Counter(result.values())
    expression_key_digest = hashlib.sha256(
        (
            "physical-expression-quotient-v1\0"
            + plan.digest
            + "\0"
            + binding_digest
        ).encode("utf-8")
    ).hexdigest()
    profile = {
        "requested": True,
        "selected": True,
        "reason": "certified stationary scalar-bath physical binding",
        "semantic_expression_count": len(expressions),
        "physical_expression_count": physical_count,
        "collapsed_expression_count": len(expressions) - physical_count,
        "alias_class_count": sum(
            count > 1 for count in alias_classes.values()
        ),
        "maximum_alias_class_size": max(alias_classes.values(), default=0),
        "term_merge_count": term_merge_count,
        "binding_digest": binding_digest,
        "expression_key_digest": expression_key_digest,
        "cache_hit": False,
    }
    if len(_PHYSICAL_EXPRESSION_KEYS) >= _PHYSICAL_EXPRESSION_CACHE_CAPACITY:
        _PHYSICAL_EXPRESSION_KEYS.popitem(last=False)
    _PHYSICAL_EXPRESSION_KEYS[cache_key] = (result, profile)
    return result, profile


class _CPUExpressionCache:
    """Cache effective expressions only within one emitted fusion group."""

    def __init__(
        self,
        gamma: Mapping[KernelIdentity, Any],
        phase_cache: _CPUPhaseCache,
        ones: Any,
        counters: Counter[str],
        persistent_monomials: bool = False,
        persistent_expressions: bool = False,
    ) -> None:
        self.gamma = gamma
        self.phase_cache = phase_cache
        self.ones = ones
        self.counters = counters
        self.persistent_monomials = persistent_monomials
        self.persistent_expressions = persistent_expressions
        self.values: dict[EffectiveSequenceExpression, Any] = {}
        self.value_sources: dict[
            EffectiveSequenceExpression, EffectiveSequenceExpression
        ] = {}
        self.monomial_values: dict[EffectiveMonomial, Any] = {}

    def clear(self) -> None:
        if not self.persistent_expressions:
            self.values.clear()
            self.value_sources.clear()

    def _monomial(self, monomial: EffectiveMonomial) -> Any:
        if self.persistent_monomials:
            cached = self.monomial_values.get(monomial)
            if cached is not None:
                self.counters["effective_monomial_cache_hits"] += 1
                return cached
        values = None
        for identity in monomial.kernel_identities:
            gamma_values = self.gamma[identity]
            values = gamma_values if values is None else values * gamma_values
        if all(exponent == 0 for exponent in monomial.phase):
            result = self.ones if values is None else values
        else:
            phase = self.phase_cache.get(monomial.phase)
            result = phase if values is None else values * phase
        if self.persistent_monomials:
            self.monomial_values[monomial] = result
            self.counters["effective_monomials_materialized"] += 1
        return result

    def get(
        self,
        expression: EffectiveSequenceExpression,
        *,
        cache_key: EffectiveSequenceExpression | None = None,
    ) -> Any:
        np = _numpy()
        key = expression if cache_key is None else cache_key
        cached = self.values.get(key)
        if cached is not None:
            self.counters["effective_expression_cache_hits"] += 1
            if self.value_sources.get(key) != expression:
                self.counters[
                    "physical_expression_alias_cache_hits"
                ] += 1
            return cached
        result = None
        for term in expression.terms:
            coefficient = _coefficient_value(term.coefficient)
            monomial = self._monomial(term.monomial)
            contribution = monomial if coefficient == 1 else coefficient * monomial
            result = contribution if result is None else result + contribution
        assert result is not None
        result = np.asarray(result, dtype=np.complex128)
        self.values[key] = result
        self.value_sources[key] = expression
        self.counters["effective_expressions_materialized"] += 1
        self.counters["effective_expression_terms"] += len(expression.terms)
        if len(expression.terms) > 1:
            self.counters["linear_combination_expressions_materialized"] += 1
            self.counters["linear_combination_terms"] += len(expression.terms)
        return result


class _CPUConvolutionEngine:
    """Causal FFT engine with an optional per-unit forward-transform cache."""

    def __init__(
        self,
        *,
        length: int,
        counters: Counter[str],
        shared_forward_transforms: bool,
        retained_product_keys: frozenset[ProductForwardKey],
        convolution_result_cache_size: int,
        overlap_result_cache_size: int,
        derived_forward_cache_size: int,
        semantic_forward_cache_size: int | None = None,
        fft_backend: str = "numpy",
        fft_workers: int = 1,
        fft_overwrite_inverse: bool = False,
        fft_overwrite_product_forward: bool = False,
    ) -> None:
        self.length = length
        self.fft_length = next_fft_length(2 * length - 1)
        self.counters = counters
        self.shared_forward_transforms = shared_forward_transforms
        self.semantic_forward_cache_size = semantic_forward_cache_size
        self.local_forward: dict[int, tuple[Any, Any]] = {}
        self.semantic_forward: OrderedDict[EffectiveSequenceExpression, Any] = (
            OrderedDict()
        )
        self.retained_product_keys = retained_product_keys
        self.product_forward: dict[ProductForwardKey, Any] = {}
        self.convolution_values: list[Any | None] = [
            None
        ] * convolution_result_cache_size
        self.convolution_keys: list[ConvolutionResultKey | None] = [
            None
        ] * convolution_result_cache_size
        self.convolution_result_cache_entries = 0
        self.overlap_values: list[Any | None] = [
            None
        ] * overlap_result_cache_size
        self.overlap_result_cache_entries = 0
        self.derived_forward_values: list[Any | None] = [
            None
        ] * derived_forward_cache_size
        self.derived_forward_cache_entries = 0
        self.fft_backend = fft_backend
        self.fft_workers = fft_workers
        self.fft_overwrite_inverse = fft_overwrite_inverse
        self.fft_overwrite_product_forward = fft_overwrite_product_forward
        if fft_backend == "scipy":
            try:
                import scipy.fft as scipy_fft
            except ImportError as error:  # pragma: no cover - optional extra
                raise ExtendedRuntimeError(
                    "fft_backend='scipy' requires the numerical SciPy extra"
                ) from error
            self._fft_module = scipy_fft
        else:
            self._fft_module = _numpy().fft

    def fft(
        self,
        values: Any,
        n: int | None = None,
        axis: int = -1,
        *,
        overwrite_input: bool = False,
    ) -> Any:
        """Execute one forward transform with the selected CPU provider."""

        if self.fft_backend == "scipy":
            return self._fft_module.fft(
                values,
                n=n,
                axis=axis,
                workers=self.fft_workers,
                overwrite_x=overwrite_input,
            )
        return self._fft_module.fft(values, n=n, axis=axis)

    def ifft(self, values: Any, n: int | None = None, axis: int = -1) -> Any:
        """Execute one inverse transform; only temporaries may be overwritten."""

        if self.fft_backend == "scipy":
            return self._fft_module.ifft(
                values,
                n=n,
                axis=axis,
                workers=self.fft_workers,
                overwrite_x=self.fft_overwrite_inverse,
            )
        return self._fft_module.ifft(values, n=n, axis=axis)

    def clear(self) -> None:
        self.local_forward.clear()

    def _forward(
        self,
        values: Any,
        semantic_key: SemanticForwardKey | None,
        derived_cache_slot: int | None,
    ) -> Any:
        np = _numpy()
        if derived_cache_slot is not None:
            derived = self.derived_forward_values[derived_cache_slot]
            if derived is not None:
                self.counters["derived_forward_fft_cache_hits"] += 1
                return derived
        elif isinstance(semantic_key, EffectiveSequenceExpression):
            semantic = self.semantic_forward.get(semantic_key)
            if semantic is not None:
                self.semantic_forward.move_to_end(semantic_key)
                self.counters["semantic_forward_fft_cache_hits"] += 1
                return semantic
        elif semantic_key in self.retained_product_keys:
            product = self.product_forward.get(semantic_key)
            if product is not None:
                self.counters["product_forward_fft_cache_hits"] += 1
                return product
        key = id(values)
        cached = self.local_forward.get(key)
        if cached is not None and cached[0] is values:
            self.counters["local_forward_fft_cache_hits"] += 1
            return cached[1]
        vector = np.asarray(values, dtype=np.complex128)
        if vector.ndim != 1 or vector.size != self.length:
            raise ExtendedRuntimeError(
                "shared causal FFT inputs must match the requested length"
            )
        overwrite_input = bool(
            self.fft_overwrite_product_forward
            and isinstance(semantic_key, _SemanticSequenceNode)
            and semantic_key.kind == "product"
        )
        transformed = self.fft(
            vector,
            self.fft_length,
            overwrite_input=overwrite_input,
        )
        if overwrite_input:
            self.counters["overwritten_product_forward_inputs"] += 1
        self.counters["forward_fft_calls"] += 1
        self.counters["fft_api_calls"] += 1
        if derived_cache_slot is not None:
            self.derived_forward_values[derived_cache_slot] = transformed
            self.derived_forward_cache_entries += 1
            self.counters["derived_forward_fft_cache_entries"] = (
                self.derived_forward_cache_entries
            )
        elif isinstance(semantic_key, EffectiveSequenceExpression):
            if self.semantic_forward_cache_size != 0:
                self.semantic_forward[semantic_key] = transformed
                self.semantic_forward.move_to_end(semantic_key)
                if (
                    self.semantic_forward_cache_size is not None
                    and len(self.semantic_forward)
                    > self.semantic_forward_cache_size
                ):
                    self.semantic_forward.popitem(last=False)
                    self.counters[
                        "semantic_forward_fft_cache_evictions"
                    ] += 1
        elif semantic_key in self.retained_product_keys:
            assert isinstance(semantic_key, _SemanticSequenceNode)
            self.product_forward[semantic_key] = transformed
        else:
            self.local_forward[key] = (values, transformed)
        self.counters["local_forward_fft_cache_peak_entries"] = max(
            self.counters["local_forward_fft_cache_peak_entries"],
            len(self.local_forward),
        )
        self.counters["semantic_forward_fft_cache_entries"] = len(
            self.semantic_forward
        )
        self.counters["product_forward_fft_cache_entries"] = len(
            self.product_forward
        )
        return transformed

    def convolve(
        self,
        left: Any,
        right: Any,
        *,
        left_key: SemanticForwardKey | None = None,
        right_key: SemanticForwardKey | None = None,
        result_cache_slot: int | None = None,
        result_cache_key: ConvolutionResultKey | None = None,
        left_derived_cache_slot: int | None = None,
        right_derived_cache_slot: int | None = None,
    ) -> Any:
        np = _numpy()
        self.counters["causal_convolution_calls"] += 1
        self.counters["semantic_fft_api_calls"] += 3
        if result_cache_slot is not None:
            cached_result = self.convolution_values[result_cache_slot]
            if (
                cached_result is not None
                and self.convolution_keys[result_cache_slot]
                == result_cache_key
            ):
                self.counters["convolution_result_cache_hits"] += 1
                return cached_result
        self.counters["executed_causal_convolution_calls"] += 1
        if not self.shared_forward_transforms and self.fft_backend == "numpy":
            self.counters["fft_api_calls"] += 3
            result = causal_convolution_fft(
                left, right, output_length=self.length
            )
        elif not self.shared_forward_transforms:
            left_values = np.asarray(left, dtype=np.complex128)
            right_values = np.asarray(right, dtype=np.complex128)
            transformed = self.fft(left_values, self.fft_length)
            transformed *= self.fft(right_values, self.fft_length)
            result = self.ifft(transformed)
            self.counters["forward_fft_calls"] += 2
            self.counters["inverse_fft_calls"] += 1
            self.counters["fft_api_calls"] += 3
            result = np.asarray(result[: self.length], dtype=np.complex128)
        else:
            transformed = self._forward(
                left, left_key, left_derived_cache_slot
            ) * self._forward(
                right, right_key, right_derived_cache_slot
            )
            result = self.ifft(transformed)
            self.counters["inverse_fft_calls"] += 1
            self.counters["fft_api_calls"] += 1
            result = np.asarray(result[: self.length], dtype=np.complex128)
        if result_cache_slot is not None:
            previous_key = self.convolution_keys[result_cache_slot]
            if previous_key is not None and previous_key != result_cache_key:
                self.counters["convolution_result_cache_evictions"] += 1
            self.convolution_values[result_cache_slot] = result
            self.convolution_keys[result_cache_slot] = result_cache_key
            # Track keyed occupancy by its delta, including unkeyed writes.
            # Scanning every slot on every write makes large model-derived
            # cache capacities quadratic in the core count.
            self.convolution_result_cache_entries += (
                int(result_cache_key is not None) - int(previous_key is not None)
            )
            self.counters["convolution_result_cache_writes"] += 1
            self.counters["convolution_result_cache_entries"] = (
                self.convolution_result_cache_entries
            )
        return result

    def convolve_many(
        self,
        calls: Sequence[
            tuple[
                Any,
                Any,
                SemanticForwardKey | None,
                SemanticForwardKey | None,
                int | None,
                ConvolutionResultKey | None,
                int | None,
                int | None,
            ]
        ],
        *,
        batch_size: int,
    ) -> tuple[Any, ...]:
        """Evaluate independent convolutions with bounded inverse FFT batches."""

        np = _numpy()
        results: list[Any | None] = [None] * len(calls)
        for start in range(0, len(calls), batch_size):
            stop = min(start + batch_size, len(calls))
            pending_indices: list[int] = []
            pending_spectra: list[Any] = []
            pending_slots: list[
                tuple[int | None, ConvolutionResultKey | None]
            ] = []
            for index in range(start, stop):
                (
                    left,
                    right,
                    left_key,
                    right_key,
                    result_cache_slot,
                    result_cache_key,
                    left_derived_cache_slot,
                    right_derived_cache_slot,
                ) = calls[index]
                self.counters["causal_convolution_calls"] += 1
                self.counters["semantic_fft_api_calls"] += 3
                if result_cache_slot is not None:
                    cached_result = self.convolution_values[
                        result_cache_slot
                    ]
                    if (
                        cached_result is not None
                        and self.convolution_keys[result_cache_slot]
                        == result_cache_key
                    ):
                        self.counters["convolution_result_cache_hits"] += 1
                        results[index] = cached_result
                        continue
                self.counters["executed_causal_convolution_calls"] += 1
                if not self.shared_forward_transforms:
                    results[index] = self.convolve(
                        left,
                        right,
                        left_key=left_key,
                        right_key=right_key,
                        result_cache_slot=result_cache_slot,
                        result_cache_key=result_cache_key,
                        left_derived_cache_slot=left_derived_cache_slot,
                        right_derived_cache_slot=right_derived_cache_slot,
                    )
                    # ``convolve`` owns the counters in this compatibility path.
                    self.counters["causal_convolution_calls"] -= 1
                    self.counters["semantic_fft_api_calls"] -= 3
                    self.counters["executed_causal_convolution_calls"] -= 1
                    continue
                pending_indices.append(index)
                pending_slots.append(
                    (result_cache_slot, result_cache_key)
                )
                pending_spectra.append(
                    self._forward(
                        left, left_key, left_derived_cache_slot
                    )
                    * self._forward(
                        right, right_key, right_derived_cache_slot
                    )
                )
            if not pending_spectra:
                continue
            if len(pending_spectra) == 1:
                transformed_results = self.ifft(pending_spectra[0])[None, :]
            else:
                transformed_results = self.ifft(
                    np.stack(pending_spectra, axis=0), axis=1
                )
                self.counters["batched_inverse_fft_api_calls"] += 1
                self.counters["batched_inverse_fft_rows"] += len(
                    pending_spectra
                )
            self.counters["inverse_fft_calls"] += len(pending_spectra)
            self.counters["fft_api_calls"] += 1
            for row, index, cache_access in zip(
                transformed_results,
                pending_indices,
                pending_slots,
                strict=True,
            ):
                result = np.asarray(row[: self.length], dtype=np.complex128)
                results[index] = result
                result_cache_slot, result_cache_key = cache_access
                if result_cache_slot is not None:
                    previous_key = self.convolution_keys[
                        result_cache_slot
                    ]
                    if (
                        previous_key is not None
                        and previous_key != result_cache_key
                    ):
                        self.counters[
                            "convolution_result_cache_evictions"
                        ] += 1
                    self.convolution_values[result_cache_slot] = result
                    self.convolution_keys[
                        result_cache_slot
                    ] = result_cache_key
                    self.convolution_result_cache_entries += (
                        int(result_cache_key is not None) - int(previous_key is not None)
                    )
                    self.counters["convolution_result_cache_writes"] += 1
                    self.counters["convolution_result_cache_entries"] = (
                        self.convolution_result_cache_entries
                    )
        if any(result is None for result in results):
            raise AssertionError("independent convolution batch was incomplete")
        return tuple(results)

    def overlap(
        self,
        first: Any,
        second: Any,
        shared: Any,
        *,
        result_cache_slot: int | None,
    ) -> Any:
        """Evaluate or reuse one exact UV/VW/V overlap contraction."""

        if result_cache_slot is not None:
            cached_result = self.overlap_values[result_cache_slot]
            if cached_result is not None:
                self.counters["overlap_result_cache_hits"] += 1
                return cached_result
        result = overlap_divide_conquer_fft_batched(
            first,
            second,
            shared,
            output_length=self.length,
            fft=self.fft,
            ifft=self.ifft,
        )
        operation_counts = overlap_batched_fft_operation_counts(self.length)
        self.counters["overlap_primitive_calls"] += 1
        self.counters["overlap_batched_convolution_calls"] += operation_counts[
            "batched_convolution_calls"
        ]
        self.counters["fft_api_calls"] += operation_counts[
            "batched_fft_api_calls"
        ]
        self.counters["semantic_fft_api_calls"] += operation_counts[
            "batched_fft_api_calls"
        ]
        if result_cache_slot is not None:
            self.overlap_values[result_cache_slot] = result
            self.overlap_result_cache_entries += 1
            self.counters["overlap_result_cache_entries"] = (
                self.overlap_result_cache_entries
            )
        return result


def _semantic_sequence_sort_key(value: SemanticSequenceKey) -> tuple[Any, ...]:
    if isinstance(value, EffectiveSequenceExpression):
        return (
            "expression",
            tuple(
                (
                    (
                        "q",
                        term.coefficient.numerator,
                        term.coefficient.denominator,
                    )
                    if isinstance(term.coefficient, RationalCoefficient)
                    else (
                        "c",
                        term.coefficient.real.hex(),
                        term.coefficient.imag.hex(),
                    )
                ,
                    term.monomial.kernel_identities,
                    term.monomial.phase,
                )
                for term in value.terms
            ),
        )
    return (
        "node",
        value.kind,
        tuple(
            _semantic_sequence_sort_key(operand)
            for operand in value.operands
        ),
    )


def _commutative_sequence_node(
    kind: str,
    left: SemanticSequenceKey,
    right: SemanticSequenceKey,
) -> _SemanticSequenceNode:
    first, second = sorted(
        (left, right), key=_semantic_sequence_sort_key
    )
    return _SemanticSequenceNode(kind, (first, second))


def _product_forward_key(
    left: SemanticSequenceKey,
    right: SemanticSequenceKey,
) -> ProductForwardKey:
    """Return a canonical semantic key for a commutative pointwise product."""

    return _commutative_sequence_node("product", left, right)


def _convolution_result_key(
    left: SemanticSequenceKey,
    right: SemanticSequenceKey,
) -> ConvolutionResultKey:
    """Return the exact commutative identity of a causal convolution."""

    return _commutative_sequence_node("convolution", left, right)


def _overlap_result_key(
    first: SemanticSequenceKey,
    second: SemanticSequenceKey,
    shared: SemanticSequenceKey,
) -> OverlapResultKey:
    """Return the exact identity of a UV/VW/V overlap contraction."""

    ordered = sorted((first, second), key=_semantic_sequence_sort_key)
    return _SemanticSequenceNode("overlap", (ordered[0], ordered[1], shared))


def _unary_sequence_node(
    kind: str, operand: SemanticSequenceKey
) -> _SemanticSequenceNode:
    return _SemanticSequenceNode(kind, (operand,))


def _select_product_forward_keys(
    plan: FusedExecutionPlan,
    capacity: int,
    expression_keys: Mapping[
        EffectiveSequenceExpression, EffectiveSequenceExpression
    ] | None = None,
) -> tuple[frozenset[ProductForwardKey], dict[str, int]]:
    """Select the most reused face-product spectra under a strict entry cap."""

    frequencies: Counter[ProductForwardKey] = Counter()
    for unit in plan.units:
        ports = {
            name: (
                expression_keys.get(expression, expression)
                if expression_keys is not None
                else expression
            )
            for name, expression in zip(SLOT_ORDER, unit.slots, strict=True)
        }
        frequencies.update(
            (
                _product_forward_key(ports["V"], ports["UV"]),
                _product_forward_key(ports["U"], ports["UV"]),
                _product_forward_key(ports["W"], ports["VW"]),
                _product_forward_key(ports["V"], ports["VW"]),
            )
        )
    recurring = sorted(
        (
            (key, count)
            for key, count in frequencies.items()
            if count > 1
        ),
        key=lambda item: (-item[1], _semantic_sequence_sort_key(item[0])),
    )
    selected = tuple(recurring[:capacity])
    return (
        frozenset(key for key, _count in selected),
        {
            "product_forward_candidate_count": len(frequencies),
            "product_forward_recurring_candidate_count": len(recurring),
            "product_forward_selected_count": len(selected),
            "product_forward_selected_occurrence_count": sum(
                count for _key, count in selected
            ),
            "product_forward_predicted_cache_hits": sum(
                count - 1 for _key, count in selected
            ),
        },
    )


def _cached_product_forward_keys(
    plan: FusedExecutionPlan,
    capacity: int,
    *,
    expression_keys: Mapping[
        EffectiveSequenceExpression, EffectiveSequenceExpression
    ] | None = None,
    expression_key_digest: str = "semantic",
) -> tuple[frozenset[ProductForwardKey], dict[str, int | bool]]:
    """Cache the plan-level product-frequency analysis, not FFT arrays."""

    cache_key = (plan.digest, capacity, expression_key_digest)
    cached = _PRODUCT_FORWARD_KEYS.get(cache_key)
    cache_hit = cached is not None
    if cached is None:
        cached = _select_product_forward_keys(
            plan, capacity, expression_keys=expression_keys
        )
        if len(_PRODUCT_FORWARD_KEYS) >= _PRODUCT_FORWARD_CACHE_CAPACITY:
            _PRODUCT_FORWARD_KEYS.popitem(last=False)
        _PRODUCT_FORWARD_KEYS[cache_key] = cached
    else:
        _PRODUCT_FORWARD_KEYS.move_to_end(cache_key)
    selected, profile = cached
    return selected, {
        **profile,
        "product_forward_schedule_cache_hit": cache_hit,
    }


def _unit_convolution_calls(
    unit: FusedTemporalUnit,
    expression_keys: Mapping[
        EffectiveSequenceExpression, EffectiveSequenceExpression
    ] | None = None,
) -> tuple[tuple[SemanticSequenceKey, SemanticSequenceKey], ...]:
    ports: dict[str, SemanticSequenceKey] = {
        name: (
            expression_keys.get(expression, expression)
            if expression_keys is not None
            else expression
        )
        for name, expression in zip(SLOT_ORDER, unit.slots, strict=True)
    }
    calls: list[tuple[SemanticSequenceKey, SemanticSequenceKey]] = []
    if unit.template_id == "conv-uvw":
        inner = _convolution_result_key(ports["U"], ports["V"])
        calls.extend(((ports["U"], ports["V"]), (inner, ports["W"])))
    elif unit.template_id == "nested-uv":
        inner = _convolution_result_key(ports["U"], ports["V"])
        calls.extend(
            (
                (ports["U"], ports["V"]),
                (
                    _product_forward_key(ports["UV"], inner),
                    ports["W"],
                ),
            )
        )
    elif unit.template_id == "nested-vw":
        inner = _convolution_result_key(ports["V"], ports["W"])
        calls.extend(
            (
                (ports["V"], ports["W"]),
                (
                    ports["U"],
                    _product_forward_key(ports["VW"], inner),
                ),
            )
        )
    elif unit.template_id in {"prefix-overlap-u", "prefix-overlap-unit-u"}:
        delayed = _unary_sequence_node(
            "delay",
            _unary_sequence_node("prefix", ports["UV"]),
        )
        calls.extend(
            (
                (ports["VW"], ports["U"]),
                (
                    ports["VW"],
                    _product_forward_key(ports["U"], delayed),
                ),
            )
        )
    elif unit.template_id == "prefix-overlap-w":
        delayed = _unary_sequence_node(
            "delay",
            _unary_sequence_node("prefix", ports["VW"]),
        )
        calls.extend(
            (
                (ports["UV"], ports["W"]),
                (
                    ports["UV"],
                    _product_forward_key(ports["W"], delayed),
                ),
            )
        )
    elif unit.template_id != "overlap-uv-vw-v":
        raise ExtendedRuntimeError(
            f"unknown fused template {unit.template_id!r}"
        )

    calls.extend(
        (
            (
                _product_forward_key(ports["V"], ports["UV"]),
                ports["W"],
            ),
            (
                _product_forward_key(ports["U"], ports["UV"]),
                _product_forward_key(ports["W"], ports["VW"]),
            ),
            (
                ports["U"],
                _product_forward_key(ports["V"], ports["VW"]),
            ),
        )
    )
    return tuple(calls)


def _unit_convolution_result_keys(
    unit: FusedTemporalUnit,
    expression_keys: Mapping[
        EffectiveSequenceExpression, EffectiveSequenceExpression
    ] | None = None,
) -> tuple[ConvolutionResultKey, ...]:
    return tuple(
        _convolution_result_key(left, right)
        for left, right in _unit_convolution_calls(
            unit, expression_keys=expression_keys
        )
    )


def _select_convolution_result_keys(
    plan: FusedExecutionPlan,
    capacity: int,
    expression_keys: Mapping[
        EffectiveSequenceExpression, EffectiveSequenceExpression
    ] | None = None,
) -> tuple[dict[str, tuple[int | None, ...]], dict[str, int]]:
    frequencies: Counter[ConvolutionResultKey] = Counter()
    for unit in plan.units:
        frequencies.update(
            _unit_convolution_result_keys(
                unit, expression_keys=expression_keys
            )
        )
    recurring = sorted(
        (
            (key, count)
            for key, count in frequencies.items()
            if count > 1
        ),
        key=lambda item: (-item[1], _semantic_sequence_sort_key(item[0])),
    )
    selected = tuple(recurring[:capacity])
    selected_slots = {
        key: slot for slot, (key, _count) in enumerate(selected)
    }
    schedule = {
        unit.unit_id: tuple(
            selected_slots.get(key)
            for key in _unit_convolution_result_keys(
                unit, expression_keys=expression_keys
            )
        )
        for unit in plan.units
    }
    return (
        schedule,
        {
            "convolution_result_candidate_count": len(frequencies),
            "convolution_result_recurring_candidate_count": len(recurring),
            "convolution_result_selected_count": len(selected),
            "convolution_result_selected_occurrence_count": sum(
                count for _key, count in selected
            ),
            "convolution_result_predicted_cache_hits": sum(
                count - 1 for _key, count in selected
            ),
        },
    )


def _select_liveness_convolution_result_slots(
    plan: FusedExecutionPlan,
    capacity: int,
    expression_keys: Mapping[
        EffectiveSequenceExpression, EffectiveSequenceExpression
    ] | None = None,
) -> tuple[_CompactConvolutionSchedule, dict[str, Any]]:
    """Assign buffers with exact liveness using compact plan-ordered arrays."""

    key_to_id: dict[ConvolutionResultKey, int] = {}
    id_to_key: list[ConvolutionResultKey] = []
    frequencies: list[int] = []
    ordered_ids = array("I")
    unit_offsets = array("Q", (0,))
    for unit in plan.units:
        for key in _unit_convolution_result_keys(
            unit, expression_keys=expression_keys
        ):
            key_id = key_to_id.get(key)
            if key_id is None:
                key_id = len(id_to_key)
                if key_id >= 2**32:
                    raise ExtendedRuntimeError(
                        "convolution dependency DAG exceeds uint32 identity capacity"
                    )
                key_to_id[key] = key_id
                id_to_key.append(key)
                frequencies.append(0)
            ordered_ids.append(key_id)
            frequencies[key_id] += 1
        unit_offsets.append(len(ordered_ids))

    occurrence_count = len(ordered_ids)
    position_code = "I" if occurrence_count < 2**32 - 1 else "Q"
    no_future = occurrence_count
    next_positions = array(position_code, (no_future,)) * occurrence_count
    last_positions = array(position_code, (no_future,)) * len(id_to_key)
    for position in range(occurrence_count - 1, -1, -1):
        key_id = ordered_ids[position]
        next_positions[position] = last_positions[key_id]
        last_positions[key_id] = position

    # Preserve the previous deterministic semantic tie-break without storing a
    # semantic key beside every occurrence.
    ranked_ids = sorted(
        range(len(id_to_key)),
        key=lambda key_id: _semantic_sequence_sort_key(id_to_key[key_id]),
    )
    tie_rank = array("I", (0,)) * len(id_to_key)
    for rank, key_id in enumerate(ranked_ids):
        tie_rank[key_id] = rank

    slot_code = "h" if capacity <= 2**15 - 1 else "i"
    scheduled_slots = array(slot_code, (-1,)) * occurrence_count
    active_slots: dict[int, int] = {}
    active_next: dict[int, int] = {}
    free_slots = list(range(capacity))
    heapq.heapify(free_slots)
    # Entries are (-next-use, -semantic-rank, key-id).  Stale entries are
    # discarded lazily and the heap is periodically rebuilt, bounding it by a
    # small multiple of the requested live-buffer capacity.
    eviction_heap: list[tuple[int, int, int]] = []
    hit_count = 0
    miss_count = 0
    eviction_count = 0
    peak_live = 0
    heap_rebuild_limit = max(1024, 4 * max(1, capacity))

    for position, key_id in enumerate(ordered_ids):
        next_position = int(next_positions[position])
        slot = active_slots.get(key_id)
        if slot is not None:
            hit_count += 1
        else:
            miss_count += 1
            if next_position == no_future or capacity == 0:
                continue
            if free_slots:
                slot = heapq.heappop(free_slots)
            else:
                while eviction_heap:
                    negative_next, _negative_rank, victim = heapq.heappop(
                        eviction_heap
                    )
                    if active_next.get(victim) == -negative_next:
                        break
                else:
                    raise AssertionError("liveness eviction heap is empty")
                slot = active_slots.pop(victim)
                active_next.pop(victim)
                eviction_count += 1
            active_slots[key_id] = slot
            peak_live = max(peak_live, len(active_slots))

        scheduled_slots[position] = slot
        if next_position == no_future:
            active_slots.pop(key_id, None)
            active_next.pop(key_id, None)
            heapq.heappush(free_slots, slot)
        else:
            active_next[key_id] = next_position
            heapq.heappush(
                eviction_heap,
                (-next_position, -int(tie_rank[key_id]), key_id),
            )
        if len(eviction_heap) > heap_rebuild_limit:
            eviction_heap = [
                (-future, -int(tie_rank[active_id]), active_id)
                for active_id, future in active_next.items()
            ]
            heapq.heapify(eviction_heap)

    schedule = _CompactConvolutionSchedule(
        unit_offsets=unit_offsets,
        slots=scheduled_slots,
    )
    return schedule, {
        "convolution_result_candidate_count": len(frequencies),
        "convolution_result_recurring_candidate_count": sum(
            count > 1 for count in frequencies
        ),
        "convolution_result_selected_count": peak_live,
        "convolution_result_selected_occurrence_count": occurrence_count,
        "convolution_result_predicted_cache_hits": hit_count,
        "convolution_result_predicted_cache_misses": miss_count,
        "convolution_result_predicted_evictions": eviction_count,
        "convolution_result_peak_live_slots": peak_live,
        "convolution_result_schedule_policy": "farthest-next-use-liveness",
        "convolution_result_schedule_encoding": "plan-ordered-flat-integers",
        "convolution_result_schedule_storage_bytes": schedule.storage_bytes,
    }


def _cached_convolution_result_schedule(
    plan: FusedExecutionPlan,
    capacity: int,
    *,
    expression_keys: Mapping[
        EffectiveSequenceExpression, EffectiveSequenceExpression
    ] | None = None,
    expression_key_digest: str = "semantic",
    policy: str = "frequency",
) -> tuple[ConvolutionCacheSchedule, dict[str, Any]]:
    """Reuse content-addressed schedule lowering across grid evaluations."""

    if policy not in {"frequency", "liveness"}:
        raise ExtendedRuntimeError(
            "convolution cache policy must be 'frequency' or 'liveness'"
        )
    cache_key = (plan.digest, capacity, expression_key_digest, policy)
    cached = _CONVOLUTION_CACHE_SCHEDULES.get(cache_key)
    cache_hit = cached is not None
    if cached is None:
        cached = (
            _select_liveness_convolution_result_slots(
                plan, capacity, expression_keys=expression_keys
            )
            if policy == "liveness"
            else _select_convolution_result_keys(
                plan, capacity, expression_keys=expression_keys
            )
        )
        if len(_CONVOLUTION_CACHE_SCHEDULES) >= (
            _CONVOLUTION_CACHE_SCHEDULE_CAPACITY
        ):
            oldest_key = next(iter(_CONVOLUTION_CACHE_SCHEDULES))
            _CONVOLUTION_CACHE_SCHEDULES.pop(oldest_key)
        _CONVOLUTION_CACHE_SCHEDULES[cache_key] = cached
    schedule, profile = cached
    return schedule, {
        **profile,
        "convolution_result_schedule_policy": (
            "farthest-next-use-liveness"
            if policy == "liveness"
            else "frequency-ranked-fixed-slots"
        ),
        "convolution_result_schedule_cache_hit": cache_hit,
    }


def _select_overlap_result_slots(
    plan: FusedExecutionPlan,
    capacity: int,
    expression_keys: Mapping[
        EffectiveSequenceExpression, EffectiveSequenceExpression
    ] | None = None,
) -> tuple[dict[str, int | None], dict[str, int]]:
    frequencies: Counter[OverlapResultKey] = Counter()
    unit_keys: dict[str, OverlapResultKey] = {}
    for unit in plan.units:
        if unit.template_id != "overlap-uv-vw-v":
            continue
        ports = {
            name: (
                expression_keys.get(expression, expression)
                if expression_keys is not None
                else expression
            )
            for name, expression in zip(SLOT_ORDER, unit.slots, strict=True)
        }
        key = _overlap_result_key(ports["UV"], ports["VW"], ports["V"])
        unit_keys[unit.unit_id] = key
        frequencies[key] += 1
    recurring = sorted(
        ((key, count) for key, count in frequencies.items() if count > 1),
        key=lambda item: (-item[1], _semantic_sequence_sort_key(item[0])),
    )
    selected = tuple(recurring[:capacity])
    selected_slots = {
        key: slot for slot, (key, _count) in enumerate(selected)
    }
    schedule = {
        unit.unit_id: selected_slots.get(unit_keys[unit.unit_id])
        if unit.unit_id in unit_keys
        else None
        for unit in plan.units
    }
    return schedule, {
        "overlap_result_candidate_count": len(frequencies),
        "overlap_result_recurring_candidate_count": len(recurring),
        "overlap_result_selected_count": len(selected),
        "overlap_result_selected_occurrence_count": sum(
            count for _key, count in selected
        ),
        "overlap_result_predicted_cache_hits": sum(
            count - 1 for _key, count in selected
        ),
    }


def _cached_overlap_result_schedule(
    plan: FusedExecutionPlan,
    capacity: int,
    *,
    expression_keys: Mapping[
        EffectiveSequenceExpression, EffectiveSequenceExpression
    ] | None = None,
    expression_key_digest: str = "semantic",
) -> tuple[dict[str, int | None], dict[str, int | bool]]:
    """Reuse exact overlap-result slot lowering across grid evaluations."""

    cache_key = (plan.digest, capacity, expression_key_digest)
    cached = _OVERLAP_CACHE_SCHEDULES.get(cache_key)
    cache_hit = cached is not None
    if cached is None:
        cached = _select_overlap_result_slots(
            plan, capacity, expression_keys=expression_keys
        )
        if len(_OVERLAP_CACHE_SCHEDULES) >= _OVERLAP_CACHE_SCHEDULE_CAPACITY:
            oldest_key = next(iter(_OVERLAP_CACHE_SCHEDULES))
            _OVERLAP_CACHE_SCHEDULES.pop(oldest_key)
        _OVERLAP_CACHE_SCHEDULES[cache_key] = cached
    schedule, profile = cached
    return schedule, {
        **profile,
        "overlap_result_schedule_cache_hit": cache_hit,
    }


def _select_derived_forward_slots(
    plan: FusedExecutionPlan,
    convolution_schedule: Mapping[str, Sequence[int | None]],
    retained_product_keys: frozenset[ProductForwardKey],
    capacity: int,
    expression_keys: Mapping[
        EffectiveSequenceExpression, EffectiveSequenceExpression
    ] | None = None,
) -> tuple[
    dict[str, tuple[tuple[int | None, int | None], ...]],
    dict[str, int],
]:
    """Rank derived operand spectra after result-cache pruning."""

    frequencies: Counter[_SemanticSequenceNode] = Counter()
    seen_results: set[int] = set()
    executed_calls = 0
    for unit in plan.units:
        calls = _unit_convolution_calls(
            unit, expression_keys=expression_keys
        )
        result_slots = convolution_schedule[unit.unit_id]
        for (left, right), result_slot in zip(
            calls, result_slots, strict=True
        ):
            executes = result_slot is None or result_slot not in seen_results
            if result_slot is not None:
                seen_results.add(result_slot)
            if not executes:
                continue
            executed_calls += 1
            for operand in (left, right):
                if (
                    isinstance(operand, _SemanticSequenceNode)
                    and operand not in retained_product_keys
                ):
                    frequencies[operand] += 1
    recurring = sorted(
        (
            (key, count)
            for key, count in frequencies.items()
            if count > 1
        ),
        key=lambda item: (-item[1], _semantic_sequence_sort_key(item[0])),
    )
    selected = tuple(recurring[:capacity])
    selected_slots = {
        key: slot for slot, (key, _count) in enumerate(selected)
    }
    schedule = {
        unit.unit_id: tuple(
            (selected_slots.get(left), selected_slots.get(right))
            for left, right in _unit_convolution_calls(
                unit, expression_keys=expression_keys
            )
        )
        for unit in plan.units
    }
    return schedule, {
        "derived_forward_executed_convolution_count": executed_calls,
        "derived_forward_candidate_count": len(frequencies),
        "derived_forward_recurring_candidate_count": len(recurring),
        "derived_forward_selected_count": len(selected),
        "derived_forward_selected_occurrence_count": sum(
            count for _key, count in selected
        ),
        "derived_forward_predicted_cache_hits": sum(
            count - 1 for _key, count in selected
        ),
    }


def _cached_derived_forward_schedule(
    plan: FusedExecutionPlan,
    *,
    convolution_schedule: Mapping[str, Sequence[int | None]],
    convolution_result_cache_size: int,
    retained_product_keys: frozenset[ProductForwardKey],
    product_forward_fft_cache_size: int,
    derived_forward_fft_cache_size: int,
    expression_keys: Mapping[
        EffectiveSequenceExpression, EffectiveSequenceExpression
    ] | None = None,
    expression_key_digest: str = "semantic",
) -> tuple[
    dict[str, tuple[tuple[int | None, int | None], ...]],
    dict[str, int | bool],
]:
    cache_key = (
        plan.digest,
        convolution_result_cache_size,
        product_forward_fft_cache_size,
        derived_forward_fft_cache_size,
        expression_key_digest,
    )
    cached = _DERIVED_FORWARD_SCHEDULES.get(cache_key)
    cache_hit = cached is not None
    if cached is None:
        cached = _select_derived_forward_slots(
            plan,
            convolution_schedule,
            retained_product_keys,
            derived_forward_fft_cache_size,
            expression_keys=expression_keys,
        )
        if len(_DERIVED_FORWARD_SCHEDULES) >= (
            _DERIVED_FORWARD_SCHEDULE_CAPACITY
        ):
            oldest_key = next(iter(_DERIVED_FORWARD_SCHEDULES))
            _DERIVED_FORWARD_SCHEDULES.pop(oldest_key)
        _DERIVED_FORWARD_SCHEDULES[cache_key] = cached
    schedule, profile = cached
    return schedule, {
        **profile,
        "derived_forward_schedule_cache_hit": cache_hit,
    }


def _evaluate_effective_ports(
    template: str,
    ports: Mapping[str, Any],
    port_keys: Mapping[str, EffectiveSequenceExpression],
    *,
    step: complex,
    counters: Counter[str],
    convolution_engine: _CPUConvolutionEngine,
    convolution_result_slots: Sequence[int | None],
    convolution_result_keys: Sequence[ConvolutionResultKey | None] | None = None,
    derived_forward_slots: Sequence[tuple[int | None, int | None]],
    overlap_result_slot: int | None,
    precomputed_overlap: Any | None,
    face_convolution_batch_size: int = 1,
) -> Any:
    np = _numpy()
    if convolution_result_keys is None:
        convolution_result_keys = (None,) * len(convolution_result_slots)
    convolution_index = 0

    def convolve(
        left: Any,
        right: Any,
        left_key: SemanticForwardKey | None = None,
        right_key: SemanticForwardKey | None = None,
    ) -> Any:
        nonlocal convolution_index
        result_cache_slot = convolution_result_slots[convolution_index]
        result_cache_key = convolution_result_keys[convolution_index]
        left_derived_slot, right_derived_slot = derived_forward_slots[
            convolution_index
        ]
        convolution_index += 1
        return convolution_engine.convolve(
            left,
            right,
            left_key=left_key,
            right_key=right_key,
            result_cache_slot=result_cache_slot,
            result_cache_key=result_cache_key,
            left_derived_cache_slot=left_derived_slot,
            right_derived_cache_slot=right_derived_slot,
        )

    def convolve_many(
        calls: Sequence[
            tuple[
                Any,
                Any,
                SemanticForwardKey | None,
                SemanticForwardKey | None,
            ]
        ],
    ) -> tuple[Any, ...]:
        nonlocal convolution_index
        scheduled = []
        for left, right, left_key, right_key in calls:
            result_cache_slot = convolution_result_slots[convolution_index]
            result_cache_key = convolution_result_keys[convolution_index]
            left_derived_slot, right_derived_slot = derived_forward_slots[
                convolution_index
            ]
            convolution_index += 1
            scheduled.append(
                (
                    left,
                    right,
                    left_key,
                    right_key,
                    result_cache_slot,
                    result_cache_key,
                    left_derived_slot,
                    right_derived_slot,
                )
            )
        return convolution_engine.convolve_many(
            scheduled, batch_size=face_convolution_batch_size
        )

    if template == "conv-uvw":
        inner = convolve(
            ports["U"], ports["V"], port_keys["U"], port_keys["V"]
        )
        outer = convolve(inner, ports["W"], right_key=port_keys["W"])
        bulk = ports["K"] * outer
    elif template == "nested-uv":
        inner = convolve(
            ports["U"], ports["V"], port_keys["U"], port_keys["V"]
        )
        outer = convolve(
            ports["UV"] * inner,
            ports["W"],
            right_key=port_keys["W"],
        )
        bulk = ports["K"] * outer
    elif template == "nested-vw":
        inner = convolve(
            ports["V"], ports["W"], port_keys["V"], port_keys["W"]
        )
        outer = convolve(
            ports["U"],
            ports["VW"] * inner,
            left_key=port_keys["U"],
        )
        bulk = ports["K"] * outer
    elif template in {"prefix-overlap-u", "prefix-overlap-unit-u"}:
        prefix = np.cumsum(ports["UV"], dtype=np.complex128)
        delayed = np.empty_like(prefix)
        delayed[0] = 0
        delayed[1:] = prefix[:-1]
        direct = convolve(
            ports["VW"],
            ports["U"],
            port_keys["VW"],
            port_keys["U"],
        )
        delayed_convolution = convolve(
            ports["VW"],
            ports["U"] * delayed,
            left_key=port_keys["VW"],
        )
        bulk = ports["K"] * (
            prefix * direct - delayed_convolution
        )
    elif template == "prefix-overlap-w":
        prefix = np.cumsum(ports["VW"], dtype=np.complex128)
        delayed = np.empty_like(prefix)
        delayed[0] = 0
        delayed[1:] = prefix[:-1]
        direct = convolve(
            ports["UV"],
            ports["W"],
            port_keys["UV"],
            port_keys["W"],
        )
        delayed_convolution = convolve(
            ports["UV"],
            ports["W"] * delayed,
            left_key=port_keys["UV"],
        )
        bulk = ports["K"] * (
            prefix * direct - delayed_convolution
        )
    elif template == "overlap-uv-vw-v":
        overlap_values = (
            precomputed_overlap
            if precomputed_overlap is not None
            else convolution_engine.overlap(
                ports["UV"],
                ports["VW"],
                ports["V"],
                result_cache_slot=overlap_result_slot,
            )
        )
        bulk = ports["K"] * overlap_values
    else:
        raise ExtendedRuntimeError(f"unknown fused template {template!r}")

    face_calls = (
        (
            ports["V"] * ports["UV"],
            ports["W"],
            _product_forward_key(port_keys["V"], port_keys["UV"]),
            port_keys["W"],
        ),
        (
            ports["U"] * ports["UV"],
            ports["W"] * ports["VW"],
            _product_forward_key(port_keys["U"], port_keys["UV"]),
            _product_forward_key(port_keys["W"], port_keys["VW"]),
        ),
        (
            ports["U"],
            ports["V"] * ports["VW"],
            port_keys["U"],
            _product_forward_key(port_keys["V"], port_keys["VW"]),
        ),
    )
    if face_convolution_batch_size == 1:
        face_u_convolution, face_v_convolution, face_w_convolution = (
            convolve(*call) for call in face_calls
        )
    else:
        (
            face_u_convolution,
            face_v_convolution,
            face_w_convolution,
        ) = convolve_many(face_calls)
    face_u = (
        ports["K"]
        * ports["VW"]
        * ports["U"][0]
        * face_u_convolution
    )
    face_v = (
        ports["K"]
        * ports["V"][0]
        * face_v_convolution
    )
    face_w = (
        ports["K"]
        * ports["W"][0]
        * ports["UV"]
        * face_w_convolution
    )
    intersection_uv = (
        ports["K"]
        * ports["U"][0]
        * ports["V"][0]
        * ports["UV"][0]
        * ports["W"]
        * ports["VW"]
    )
    intersection_uw = (
        ports["K"]
        * ports["U"][0]
        * ports["W"][0]
        * ports["V"]
        * ports["UV"]
        * ports["VW"]
    )
    if convolution_index != len(convolution_result_slots):
        raise ExtendedRuntimeError(
            "convolution-result cache schedule does not match the template"
        )
    if convolution_index != len(convolution_result_keys):
        raise ExtendedRuntimeError(
            "convolution-result key schedule does not match the template"
        )
    return (step * step) * (
        bulk
        - 0.5 * (face_u + face_v + face_w)
        + 0.25 * (intersection_uv + intersection_uw)
    )


def evaluate_fused_plan_numpy(
    plan: FusedExecutionPlan,
    *,
    gamma_sequences: Mapping[KernelIdentity, Sequence[complex]],
    level_ratios: Sequence[complex],
    length: int,
    step: complex = 1,
    phase_cache_size: int = 8,
    accumulation_chunk_size: int = 65_536,
    unit_accumulation_batch_size: int = 1,
    persistent_monomial_cache: bool = False,
    shared_forward_fft_cache: bool = False,
    semantic_forward_fft_cache_size: int | None = None,
    persistent_expression_cache: bool | None = None,
    product_forward_fft_cache_size: int = 0,
    convolution_result_cache_size: int = 0,
    convolution_result_cache_policy: Literal["frequency", "liveness"] = (
        "frequency"
    ),
    convolution_result_cache_budget_mib: float | None = None,
    overlap_result_cache_size: int = 0,
    overlap_unit_batch_size: int = 1,
    overlap_staged_fft: bool = False,
    overlap_leaf_size: int = 1,
    derived_forward_fft_cache_size: int = 0,
    fft_backend: str = "numpy",
    fft_workers: int = 1,
    fft_overwrite_inverse: bool = False,
    face_convolution_batch_size: int = 1,
    fft_overwrite_product_forward: bool = False,
    physical_gamma_cse: bool = False,
    physical_dependency_dag: bool = False,
    track_history_depth_terminal_activity: bool = False,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> StreamedEvaluationResult:
    """Evaluate an exact fused plan with NumPy FFT kernels."""

    np = _numpy()
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ExtendedRuntimeError("length must be a positive integer")
    if not isinstance(track_history_depth_terminal_activity, bool):
        raise ExtendedRuntimeError(
            "track_history_depth_terminal_activity must be boolean"
        )
    if (
        track_history_depth_terminal_activity
        and getattr(plan, "history_depth_manifest_sha256", None) is None
    ):
        raise ExtendedRuntimeError(
            "terminal activity requires a history-depth instrumented plan"
        )
    dimension = getattr(plan, "dimension", None)
    if dimension is None:
        dimension = plan.source_plan.source_plan.dimension
    if len(level_ratios) != dimension:
        raise ExtendedRuntimeError(
            "level-ratio count does not match the plan dimension"
        )
    if (
        isinstance(accumulation_chunk_size, bool)
        or not isinstance(accumulation_chunk_size, int)
        or accumulation_chunk_size <= 0
    ):
        raise ExtendedRuntimeError(
            "accumulation_chunk_size must be a positive integer"
        )
    if (
        isinstance(unit_accumulation_batch_size, bool)
        or not isinstance(unit_accumulation_batch_size, int)
        or unit_accumulation_batch_size <= 0
    ):
        raise ExtendedRuntimeError(
            "unit_accumulation_batch_size must be a positive integer"
        )
    if not isinstance(persistent_monomial_cache, bool):
        raise ExtendedRuntimeError("persistent_monomial_cache must be boolean")
    if not isinstance(shared_forward_fft_cache, bool):
        raise ExtendedRuntimeError("shared_forward_fft_cache must be boolean")
    if semantic_forward_fft_cache_size is not None and (
        isinstance(semantic_forward_fft_cache_size, bool)
        or not isinstance(semantic_forward_fft_cache_size, int)
        or semantic_forward_fft_cache_size < 0
    ):
        raise ExtendedRuntimeError(
            "semantic_forward_fft_cache_size must be None or a nonnegative integer"
        )
    if semantic_forward_fft_cache_size is not None and not shared_forward_fft_cache:
        raise ExtendedRuntimeError(
            "semantic_forward_fft_cache_size requires shared_forward_fft_cache"
        )
    if persistent_expression_cache is not None and not isinstance(
        persistent_expression_cache, bool
    ):
        raise ExtendedRuntimeError(
            "persistent_expression_cache must be None or boolean"
        )
    effective_persistent_expression_cache = (
        shared_forward_fft_cache
        if persistent_expression_cache is None
        else persistent_expression_cache
    )
    if (
        isinstance(product_forward_fft_cache_size, bool)
        or not isinstance(product_forward_fft_cache_size, int)
        or product_forward_fft_cache_size < 0
    ):
        raise ExtendedRuntimeError(
            "product_forward_fft_cache_size must be a nonnegative integer"
        )
    if product_forward_fft_cache_size and not shared_forward_fft_cache:
        raise ExtendedRuntimeError(
            "product_forward_fft_cache_size requires shared_forward_fft_cache"
        )
    if (
        isinstance(convolution_result_cache_size, bool)
        or not isinstance(convolution_result_cache_size, int)
        or convolution_result_cache_size < 0
    ):
        raise ExtendedRuntimeError(
            "convolution_result_cache_size must be a nonnegative integer"
        )
    if convolution_result_cache_policy not in {"frequency", "liveness"}:
        raise ExtendedRuntimeError(
            "convolution_result_cache_policy must be 'frequency' or 'liveness'"
        )
    if convolution_result_cache_budget_mib is not None and (
        isinstance(convolution_result_cache_budget_mib, bool)
        or not isinstance(convolution_result_cache_budget_mib, (int, float))
        or not math.isfinite(convolution_result_cache_budget_mib)
        or convolution_result_cache_budget_mib <= 0
    ):
        raise ExtendedRuntimeError(
            "convolution_result_cache_budget_mib must be a positive finite "
            "number or None"
        )
    result_curve_bytes = 16 * length
    budget_slots = (
        convolution_result_cache_size
        if convolution_result_cache_budget_mib is None
        else int(
            float(convolution_result_cache_budget_mib)
            * 1024**2
            // result_curve_bytes
        )
    )
    effective_convolution_result_cache_size = min(
        convolution_result_cache_size, budget_slots
    )
    if (
        isinstance(derived_forward_fft_cache_size, bool)
        or not isinstance(derived_forward_fft_cache_size, int)
        or derived_forward_fft_cache_size < 0
    ):
        raise ExtendedRuntimeError(
            "derived_forward_fft_cache_size must be a nonnegative integer"
        )
    if derived_forward_fft_cache_size and not shared_forward_fft_cache:
        raise ExtendedRuntimeError(
            "derived_forward_fft_cache_size requires shared_forward_fft_cache"
        )
    if derived_forward_fft_cache_size and not convolution_result_cache_size:
        raise ExtendedRuntimeError(
            "derived_forward_fft_cache_size requires "
            "convolution_result_cache_size"
        )
    if (
        derived_forward_fft_cache_size
        and not effective_convolution_result_cache_size
    ):
        raise ExtendedRuntimeError(
            "convolution result byte budget leaves no slot for derived "
            "forward caching"
        )
    if (
        derived_forward_fft_cache_size
        and convolution_result_cache_policy == "liveness"
    ):
        raise ExtendedRuntimeError(
            "derived forward caching does not yet support reusable liveness slots"
        )
    if (
        isinstance(overlap_result_cache_size, bool)
        or not isinstance(overlap_result_cache_size, int)
        or overlap_result_cache_size < 0
    ):
        raise ExtendedRuntimeError(
            "overlap_result_cache_size must be a nonnegative integer"
        )
    if (
        isinstance(overlap_unit_batch_size, bool)
        or not isinstance(overlap_unit_batch_size, int)
        or overlap_unit_batch_size <= 0
    ):
        raise ExtendedRuntimeError(
            "overlap_unit_batch_size must be a positive integer"
        )
    if overlap_unit_batch_size > 1 and not shared_forward_fft_cache:
        raise ExtendedRuntimeError(
            "overlap_unit_batch_size greater than one requires "
            "shared_forward_fft_cache"
        )
    if overlap_unit_batch_size > 1 and overlap_result_cache_size:
        raise ExtendedRuntimeError(
            "overlap unit batching and overlap-result caching are "
            "mutually exclusive"
        )
    if not isinstance(overlap_staged_fft, bool):
        raise ExtendedRuntimeError("overlap_staged_fft must be boolean")
    if overlap_staged_fft and overlap_unit_batch_size == 1:
        raise ExtendedRuntimeError(
            "overlap_staged_fft requires overlap_unit_batch_size greater "
            "than one"
        )
    if (
        isinstance(overlap_leaf_size, bool)
        or not isinstance(overlap_leaf_size, int)
        or overlap_leaf_size <= 0
        or overlap_leaf_size & (overlap_leaf_size - 1)
    ):
        raise ExtendedRuntimeError(
            "overlap_leaf_size must be a positive power of two"
        )
    if overlap_leaf_size > 1 and overlap_unit_batch_size == 1:
        raise ExtendedRuntimeError(
            "overlap_leaf_size greater than one requires overlap unit batching"
        )
    if overlap_leaf_size > 1 and overlap_staged_fft:
        raise ExtendedRuntimeError(
            "direct overlap leaves are not supported by the staged FFT schedule"
        )
    if fft_backend not in {"numpy", "scipy"}:
        raise ExtendedRuntimeError("fft_backend must be 'numpy' or 'scipy'")
    if (
        isinstance(fft_workers, bool)
        or not isinstance(fft_workers, int)
        or fft_workers <= 0
    ):
        raise ExtendedRuntimeError("fft_workers must be a positive integer")
    if not isinstance(fft_overwrite_inverse, bool):
        raise ExtendedRuntimeError("fft_overwrite_inverse must be boolean")
    if not isinstance(fft_overwrite_product_forward, bool):
        raise ExtendedRuntimeError(
            "fft_overwrite_product_forward must be boolean"
        )
    if not isinstance(physical_gamma_cse, bool):
        raise ExtendedRuntimeError("physical_gamma_cse must be boolean")
    if not isinstance(physical_dependency_dag, bool):
        raise ExtendedRuntimeError("physical_dependency_dag must be boolean")
    if progress is not None and not callable(progress):
        raise ExtendedRuntimeError("progress must be callable or None")
    if physical_dependency_dag and not physical_gamma_cse:
        raise ExtendedRuntimeError(
            "physical_dependency_dag requires physical_gamma_cse"
        )
    if fft_backend == "numpy" and fft_workers != 1:
        raise ExtendedRuntimeError(
            "fft_workers greater than one requires fft_backend='scipy'"
        )
    if fft_backend == "numpy" and fft_overwrite_inverse:
        raise ExtendedRuntimeError(
            "fft_overwrite_inverse requires fft_backend='scipy'"
        )
    if fft_backend == "numpy" and fft_overwrite_product_forward:
        raise ExtendedRuntimeError(
            "fft_overwrite_product_forward requires fft_backend='scipy'"
        )
    if (
        isinstance(face_convolution_batch_size, bool)
        or not isinstance(face_convolution_batch_size, int)
        or face_convolution_batch_size <= 0
        or face_convolution_batch_size > 3
    ):
        raise ExtendedRuntimeError(
            "face_convolution_batch_size must be 1, 2, or 3"
        )
    if face_convolution_batch_size > 1 and not shared_forward_fft_cache:
        raise ExtendedRuntimeError(
            "face convolution batching requires shared forward FFT caching"
        )

    gamma = _normalize_gamma_sequences(
        plan, gamma_sequences, length=length
    )
    expression_keys, physical_cse_profile = _physical_expression_keys(
        plan,
        gamma_sequences,
        enabled=physical_gamma_cse,
    )
    dependency_expression_keys = (
        expression_keys
        if physical_dependency_dag and physical_cse_profile["selected"]
        else None
    )
    dependency_key_digest = (
        str(physical_cse_profile["expression_key_digest"])
        if dependency_expression_keys is not None
        else "semantic"
    )
    physical_dependency_profile = {
        "requested": physical_dependency_dag,
        "selected": dependency_expression_keys is not None,
        "reason": (
            "certified physical expressions propagated through derived nodes"
            if dependency_expression_keys is not None
            else (
                "explicitly disabled"
                if not physical_dependency_dag
                else "certified physical Gamma binding unavailable"
            )
        ),
        "expression_key_digest": dependency_key_digest,
    }
    ones = np.ones(length, dtype=np.complex128)
    phase_cache = _CPUPhaseCache(
        level_ratios, length, phase_cache_size, ones
    )
    counters: Counter[str] = Counter()
    expression_cache = _CPUExpressionCache(
        gamma,
        phase_cache,
        ones,
        counters,
        persistent_monomials=persistent_monomial_cache,
        persistent_expressions=effective_persistent_expression_cache,
    )
    retained_product_keys, product_cache_profile = _cached_product_forward_keys(
        plan,
        product_forward_fft_cache_size,
        expression_keys=expression_keys,
        expression_key_digest=(
            str(physical_cse_profile.get("expression_key_digest", "semantic"))
            if expression_keys
            else "semantic"
        ),
    )
    if effective_convolution_result_cache_size:
        convolution_cache_schedule, convolution_cache_profile = (
            _cached_convolution_result_schedule(
                plan,
                effective_convolution_result_cache_size,
                expression_keys=dependency_expression_keys,
                expression_key_digest=dependency_key_digest,
                policy=convolution_result_cache_policy,
            )
        )
    else:
        convolution_cache_schedule = None
        convolution_cache_profile = {
            "convolution_result_candidate_count": 0,
            "convolution_result_recurring_candidate_count": 0,
            "convolution_result_selected_count": 0,
            "convolution_result_selected_occurrence_count": 0,
            "convolution_result_predicted_cache_hits": 0,
            "convolution_result_schedule_cache_hit": False,
        }
    if overlap_result_cache_size:
        overlap_cache_schedule, overlap_cache_profile = (
            _cached_overlap_result_schedule(
                plan,
                overlap_result_cache_size,
                expression_keys=dependency_expression_keys,
                expression_key_digest=dependency_key_digest,
            )
        )
    else:
        overlap_cache_schedule = None
        overlap_cache_profile = {
            "overlap_result_candidate_count": 0,
            "overlap_result_recurring_candidate_count": 0,
            "overlap_result_selected_count": 0,
            "overlap_result_selected_occurrence_count": 0,
            "overlap_result_predicted_cache_hits": 0,
            "overlap_result_schedule_cache_hit": False,
        }
    if derived_forward_fft_cache_size:
        assert convolution_cache_schedule is not None
        derived_forward_schedule, derived_forward_profile = (
            _cached_derived_forward_schedule(
                plan,
                convolution_schedule=convolution_cache_schedule,
                convolution_result_cache_size=(
                    effective_convolution_result_cache_size
                ),
                retained_product_keys=retained_product_keys,
                product_forward_fft_cache_size=(
                    product_forward_fft_cache_size
                ),
                derived_forward_fft_cache_size=derived_forward_fft_cache_size,
                expression_keys=dependency_expression_keys,
                expression_key_digest=dependency_key_digest,
            )
        )
    else:
        derived_forward_schedule = None
        derived_forward_profile = {
            "derived_forward_executed_convolution_count": 0,
            "derived_forward_candidate_count": 0,
            "derived_forward_recurring_candidate_count": 0,
            "derived_forward_selected_count": 0,
            "derived_forward_selected_occurrence_count": 0,
            "derived_forward_predicted_cache_hits": 0,
            "derived_forward_schedule_cache_hit": False,
        }
    convolution_engine = _CPUConvolutionEngine(
        length=length,
        counters=counters,
        shared_forward_transforms=shared_forward_fft_cache,
        semantic_forward_cache_size=semantic_forward_fft_cache_size,
        retained_product_keys=retained_product_keys,
        convolution_result_cache_size=(
            effective_convolution_result_cache_size
        ),
        overlap_result_cache_size=overlap_result_cache_size,
        derived_forward_cache_size=derived_forward_fft_cache_size,
        fft_backend=fft_backend,
        fft_workers=fft_workers,
        fft_overwrite_inverse=fft_overwrite_inverse,
        fft_overwrite_product_forward=fft_overwrite_product_forward,
    )
    dimension_squared = dimension * dimension
    superoperator_size = dimension_squared * dimension_squared
    output = np.zeros(
        (length, dimension_squared, dimension_squared),
        dtype=np.complex128,
    )
    depth_output = (
        {
            depth: np.zeros_like(output)
            for depth in ("H0", "H1", "H2")
        }
        if getattr(plan, "history_depth_manifest_sha256", None) is not None
        else None
    )
    depth_terminal_absolute = (
        {
            depth: np.zeros(
                (length, dimension_squared, dimension_squared),
                dtype=np.float64,
            )
            for depth in ("H0", "H1", "H2")
        }
        if track_history_depth_terminal_activity
        else None
    )
    output_flat_view = output.reshape(length, superoperator_size)
    pending_values: list[Any] = []
    pending_weight_rows: list[Any] = []
    precomputed_overlap_values: dict[str, Any] = {}

    if overlap_unit_batch_size > 1:
        overlap_units = tuple(
            unit
            for unit in plan.units
            if unit.template_id == "overlap-uv-vw-v"
        )
        semantic_operation_counts = overlap_batched_fft_operation_counts(length)
        operation_counts = (
            overlap_staged_fft_operation_counts(length)
            if overlap_staged_fft
            else overlap_batched_fft_operation_counts(
                length, leaf_size=overlap_leaf_size
            )
        )
        for batch_start in range(0, len(overlap_units), overlap_unit_batch_size):
            batch = overlap_units[
                batch_start : batch_start + overlap_unit_batch_size
            ]
            batch_ports = []
            for unit in batch:
                port_keys = {
                    name: expression_keys.get(expression, expression)
                    for name, expression in zip(
                        SLOT_ORDER, unit.slots, strict=True
                    )
                }
                batch_ports.append(
                    {
                        name: expression_cache.get(
                            original,
                            cache_key=port_keys[name],
                        )
                        for name, original in zip(
                            SLOT_ORDER, unit.slots, strict=True
                        )
                        if name in {"UV", "VW", "V"}
                    }
                )
            overlap_function = (
                overlap_divide_conquer_fft_staged_many
                if overlap_staged_fft
                else overlap_divide_conquer_fft_batched_many
            )
            overlap_arguments = {
                "output_length": length,
                "fft": convolution_engine.fft,
                "ifft": convolution_engine.ifft,
                **(
                    {"leaf_size": overlap_leaf_size}
                    if not overlap_staged_fft
                    else {}
                ),
            }
            batched_values = overlap_function(
                np.stack([ports["UV"] for ports in batch_ports], axis=0),
                np.stack([ports["VW"] for ports in batch_ports], axis=0),
                np.stack([ports["V"] for ports in batch_ports], axis=0),
                **overlap_arguments,
            )
            for unit, values in zip(batch, batched_values, strict=True):
                precomputed_overlap_values[unit.unit_id] = values
            counters["overlap_primitive_calls"] += len(batch)
            counters["overlap_unit_batches"] += 1
            counters["overlap_batched_convolution_calls"] += operation_counts[
                "batched_convolution_calls"
            ]
            counters["fft_api_calls"] += operation_counts[
                "batched_fft_api_calls"
            ]
            counters["semantic_fft_api_calls"] += (
                len(batch)
                * semantic_operation_counts["batched_fft_api_calls"]
            )

    def flush_accumulation_batch() -> None:
        if not pending_values:
            return
        curve_matrix = np.stack(pending_values, axis=1)
        weight_matrix = np.stack(pending_weight_rows, axis=0)
        for start in range(0, length, accumulation_chunk_size):
            stop = min(start + accumulation_chunk_size, length)
            output_flat_view[start:stop] += (
                curve_matrix[start:stop] @ weight_matrix
            )
        counters["unit_accumulation_batches"] += 1
        counters["unit_curves_batched_for_accumulation"] += len(
            pending_values
        )
        counters["dense_accumulation_weight_slots"] += int(
            weight_matrix.size
        )
        pending_values.clear()
        pending_weight_rows.clear()

    current_group: str | None = None
    unit_evaluation_started = perf_counter()
    total_unit_count = len(plan.units)

    for unit_index, unit in enumerate(plan.units):
        convolution_engine.clear()
        if unit.group_id != current_group:
            expression_cache.clear()
            current_group = unit.group_id
        port_keys = {
            name: expression_keys.get(expression, expression)
            for name, expression in zip(
                SLOT_ORDER, unit.slots, strict=True
            )
        }
        ports = {
            name: expression_cache.get(
                expression,
                cache_key=port_keys[name],
            )
            for name, expression in zip(
                SLOT_ORDER, unit.slots, strict=True
            )
        }
        values = _evaluate_effective_ports(
            unit.template_id,
            ports,
            port_keys,
            step=step,
            counters=counters,
            convolution_engine=convolution_engine,
            convolution_result_slots=(
                (
                    convolution_cache_schedule.slots_for_unit(unit_index)
                    if isinstance(
                        convolution_cache_schedule,
                        _CompactConvolutionSchedule,
                    )
                    else convolution_cache_schedule[unit.unit_id]
                )
                if convolution_cache_schedule is not None
                else _NO_CONVOLUTION_CACHE_SLOTS[
                    "overlap"
                    if unit.template_id == "overlap-uv-vw-v"
                    else "strict"
                ]
            ),
            convolution_result_keys=_unit_convolution_result_keys(
                unit, expression_keys=dependency_expression_keys
            ),
            derived_forward_slots=(
                derived_forward_schedule[unit.unit_id]
                if derived_forward_schedule is not None
                else _NO_DERIVED_FORWARD_SLOTS[
                    "overlap"
                    if unit.template_id == "overlap-uv-vw-v"
                    else "strict"
                ]
            ),
            overlap_result_slot=(
                overlap_cache_schedule[unit.unit_id]
                if overlap_cache_schedule is not None
                else None
            ),
            precomputed_overlap=precomputed_overlap_values.get(unit.unit_id),
            face_convolution_batch_size=face_convolution_batch_size,
        )
        counters["temporal_units_evaluated"] += 1
        counters[
            "overlap_units_evaluated"
            if unit.template_id == "overlap-uv-vw-v"
            else "strict_units_evaluated"
        ] += 1
        if unit_accumulation_batch_size == 1:
            for output_flat, input_flat, weight in unit.weights:
                for start in range(0, length, accumulation_chunk_size):
                    stop = min(start + accumulation_chunk_size, length)
                    output[start:stop, output_flat, input_flat] += (
                        weight * values[start:stop]
                    )
                counters["sparse_curve_accumulations"] += 1
        else:
            weight_row = np.zeros(superoperator_size, dtype=np.complex128)
            for output_flat, input_flat, weight in unit.weights:
                if not (
                    0 <= output_flat < dimension_squared
                    and 0 <= input_flat < dimension_squared
                ):
                    raise ExtendedRuntimeError(
                        "fused sparse weight lies outside the plan dimension"
                    )
                weight_row[
                    dimension_squared * output_flat + input_flat
                ] += weight
                counters["sparse_curve_accumulations"] += 1
            pending_values.append(values)
            pending_weight_rows.append(weight_row)
            if len(pending_values) >= unit_accumulation_batch_size:
                flush_accumulation_batch()
        if depth_output is not None:
            for depth, output_flat, input_flat, weight in unit.depth_weights:
                for start in range(0, length, accumulation_chunk_size):
                    stop = min(start + accumulation_chunk_size, length)
                    depth_output[depth][start:stop, output_flat, input_flat] += (
                        weight * values[start:stop]
                    )
                counters["history_depth_sparse_curve_accumulations"] += 1
        if depth_terminal_absolute is not None:
            absolute_curve = np.abs(values)
            for depth, output_flat, input_flat, weight in unit.depth_weights:
                depth_terminal_absolute[depth][
                    :, output_flat, input_flat
                ] += abs(weight) * absolute_curve
                counters["history_depth_terminal_activity_accumulations"] += 1
        if progress is not None and (
            unit_index == 0
            or (unit_index + 1) % 64 == 0
            or unit_index + 1 == total_unit_count
        ):
            progress(
                {
                    "phase": "fused-runtime",
                    "completed_units": unit_index + 1,
                    "total_units": total_unit_count,
                    "elapsed_seconds": perf_counter() - unit_evaluation_started,
                    "overlap_units_evaluated": counters[
                        "overlap_units_evaluated"
                    ],
                    "strict_units_evaluated": counters[
                        "strict_units_evaluated"
                    ],
                    "fft_api_calls": counters["fft_api_calls"],
                }
            )

    flush_accumulation_batch()
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
        "hilbert_dimension": dimension,
        "nt": length,
        "dtype": "complex128",
        "phase_cache_capacity": phase_cache_size,
        "phase_cache_entries": len(phase_cache.values),
        "gamma_semantic_identity_count": len(gamma),
        "gamma_host_buffer_count": len({id(values) for values in gamma.values()}),
        "gamma_finite_scan_count": len({id(values) for values in gamma.values()}),
        "expression_cache_scope": (
            "evaluation-global"
            if effective_persistent_expression_cache
            else "fusion-group"
        ),
        "persistent_expression_cache": effective_persistent_expression_cache,
        "persistent_monomial_cache": persistent_monomial_cache,
        "persistent_monomial_cache_entries": len(
            expression_cache.monomial_values
        ),
        "shared_forward_fft_cache": shared_forward_fft_cache,
        "shared_forward_fft_cache_scope": (
            "physical-expression-global+ephemeral-unit"
            if physical_cse_profile["selected"]
            else "semantic-expression-global+ephemeral-unit"
        ),
        "physical_gamma_cse": physical_cse_profile,
        "physical_dependency_dag": physical_dependency_profile,
        "semantic_forward_fft_cache_entries": len(
            convolution_engine.semantic_forward
        ),
        "semantic_forward_fft_cache_capacity": (
            semantic_forward_fft_cache_size
        ),
        "product_forward_fft_cache_capacity": product_forward_fft_cache_size,
        "product_forward_fft_cache_entries": len(
            convolution_engine.product_forward
        ),
        "product_forward_fft_cache_profile": product_cache_profile,
        "convolution_result_cache_requested_capacity": (
            convolution_result_cache_size
        ),
        "convolution_result_cache_capacity": (
            effective_convolution_result_cache_size
        ),
        "convolution_result_cache_policy": convolution_result_cache_policy,
        "convolution_result_cache_budget_mib": (
            convolution_result_cache_budget_mib
        ),
        "convolution_result_cache_curve_bytes": result_curve_bytes,
        "convolution_result_cache_peak_live_bytes": (
            int(
                convolution_cache_profile.get(
                    "convolution_result_peak_live_slots",
                    convolution_engine.convolution_result_cache_entries,
                )
            )
            * result_curve_bytes
        ),
        "convolution_result_cache_entries": (
            convolution_engine.convolution_result_cache_entries
        ),
        "convolution_result_cache_profile": convolution_cache_profile,
        "overlap_result_cache_capacity": overlap_result_cache_size,
        "overlap_result_cache_entries": (
            convolution_engine.overlap_result_cache_entries
        ),
        "overlap_result_cache_profile": overlap_cache_profile,
        "overlap_unit_batch_size": overlap_unit_batch_size,
        "overlap_staged_fft": overlap_staged_fft,
        "overlap_leaf_size": overlap_leaf_size,
        "fft_backend": fft_backend,
        "fft_workers": fft_workers,
        "fft_overwrite_inverse": fft_overwrite_inverse,
        "fft_overwrite_product_forward": fft_overwrite_product_forward,
        "face_convolution_batch_size": face_convolution_batch_size,
        "derived_forward_fft_cache_capacity": derived_forward_fft_cache_size,
        "derived_forward_fft_cache_entries": (
            convolution_engine.derived_forward_cache_entries
        ),
        "derived_forward_fft_cache_profile": derived_forward_profile,
        "accumulation_chunk_size": accumulation_chunk_size,
        "unit_accumulation_batch_size": unit_accumulation_batch_size,
        "no_quadratic_arrays": True,
        "execution_counts": dict(sorted(counters.items())),
        "output_semantics": "pre-outer-wrapper column-major superoperator",
    }
    if depth_output is not None:
        report["history_depth_accumulation"] = {
            "enabled": True,
            "depth_manifest_sha256": getattr(
                plan, "history_depth_manifest_sha256", None
            ),
            "temporal_curves_evaluated_once": True,
            "resident_sector_array_count": len(depth_output),
            "per_word_dense_arrays": 0,
            "terminal_absolute_activity_enabled": (
                depth_terminal_absolute is not None
            ),
            "terminal_absolute_activity_semantics": (
                "sum of absolute exact-fused effective-unit contributions "
                "before the outer wrapper"
            ),
        }
    return StreamedEvaluationResult(
        values=output,
        report=report,
        depth_values=depth_output,
        depth_terminal_absolute_values=depth_terminal_absolute,
    )

