"""Compile TCL6 contractions for finite-dimensional systems."""

from __future__ import annotations

from collections import Counter, defaultdict

from dataclasses import dataclass

from itertools import product

from time import perf_counter

from types import MappingProxyType

from typing import Any, Callable, Literal, Mapping, Sequence, NamedTuple

from .contraction_scalarization import enumerate_contraction_paths

from .dimension_compiler import build_dimension_join_plan

from .extended_runtime import SpecializedExecutionPlan, StreamedExecutionPlan, TemporalCore, WeightedTemporalCore, _classify_template, _parse_monomial

from .physical_specialization import exact_bohr_gap_partition, exact_coupling_pattern

from .spin_boson import stationary_scalar_bath_monomial_key, _swap_vw_support

from .symbolic_compiler import aggregate_formal_words, build_symbolic_index_plan, indexed_family_key

class GenericDimensionRuntimeError(ValueError):
    """Raised when finite-dimensional streamed compilation is invalid."""


ProgressCallback = Callable[[Mapping[str, Any]], None]


SparseLowering = Literal["path-enumeration", "symbolic-assignment-bank", "direct-temporal"]


@dataclass(frozen=True)
class GenericSpecializedPlanResult:
    """A coupling-specialized execution plan and compiler diagnostics."""

    plan: SpecializedExecutionPlan
    report: Mapping[str, Any]


def _validate_model(
    coupling_operator: Sequence[Sequence[complex]],
    energies: Sequence[float] | None,
    dimension: int | None,
) -> tuple[Any, Any | None, int]:
    import numpy as np

    coupling = np.asarray(coupling_operator, dtype=np.complex128)
    if coupling.ndim != 2 or coupling.shape[0] != coupling.shape[1]:
        raise GenericDimensionRuntimeError("coupling_operator must be square")
    inferred = int(coupling.shape[0])
    if dimension is None:
        dimension = inferred
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
        or dimension != inferred
    ):
        raise GenericDimensionRuntimeError(
            "dimension must match the positive square coupling dimension"
        )
    if not np.isfinite(coupling.real).all() or not np.isfinite(
        coupling.imag
    ).all():
        raise GenericDimensionRuntimeError("coupling_operator must be finite")
    energy = None
    if energies is not None:
        energy = np.asarray(energies, dtype=np.float64)
        if energy.shape != (dimension,) or not np.isfinite(energy).all():
            raise GenericDimensionRuntimeError(
                "energies must contain one finite value per level"
            )
    return coupling, energy, dimension


def _symbolic_assignment_topology(
    key: tuple[Any, ...],
) -> tuple[bool, int, tuple[tuple[int, int], ...]]:
    """Return the exact coupling-constraint graph for one symbolic family."""

    if len(key) != 6:
        raise GenericDimensionRuntimeError(
            "symbolic family key must contain factors and three phases"
        )
    output_diagonal, dummy_count, factors, _phase_u, _phase_v, _phase_w = key
    if (
        not isinstance(output_diagonal, bool)
        or isinstance(dummy_count, bool)
        or not isinstance(dummy_count, int)
        or dummy_count < 0
        or not isinstance(factors, tuple)
    ):
        raise GenericDimensionRuntimeError(
            "symbolic family assignment metadata is malformed"
        )
    valid_tokens = {-2, -1, *range(dummy_count)}
    coupling_edges: list[tuple[int, int]] = []
    for factor in factors:
        kind = getattr(factor, "kind", None)
        row = getattr(factor, "row", None)
        column = getattr(factor, "column", None)
        if (
            kind not in {"coupling", "density", "kernel"}
            or row not in valid_tokens
            or column not in valid_tokens
        ):
            raise GenericDimensionRuntimeError(
                "symbolic family contains an invalid indexed factor"
            )
        if kind == "coupling":
            coupling_edges.append((row, column))
    return output_diagonal, dummy_count, tuple(coupling_edges)


def _build_symbolic_assignment_bank(
    topology: tuple[bool, int, tuple[tuple[int, int], ...]],
    *,
    dimension: int,
    coupling_support: frozenset[tuple[int, int]],
    memory_limit_bytes: int = 64 * 1024**2,
) -> tuple[tuple[tuple[int, ...], ...], int]:
    """Materialize each support-admissible level assignment exactly once.

    The bank always stores assignments as ``(output_row, output_column,
    dummy_0, ...)``. Adjacency intersections prune partial assignments while
    preserving the original lexicographic order. Each topology is retained
    once, subject to a byte admission bound rather than a dimension whitelist.
    """

    output_diagonal, dummy_count, coupling_edges = topology
    free_count = dummy_count + (1 if output_diagonal else 2)
    candidate_count = dimension**free_count

    if type(memory_limit_bytes) is not int or memory_limit_bytes <= 0:
        raise GenericDimensionRuntimeError("assignment bank memory_limit_bytes must be positive")
    row_bytes = 64 + 40 * (dummy_count + 2)
    row_limit = memory_limit_bytes // row_bytes

    # Keep Cartesian lexicographic order, but intersect already-bound adjacency
    # constraints before visiting the next variable. No tolerance defines support.
    def variable(token):
        if token == -2: return 0
        if token == -1: return 0 if output_diagonal else 1
        return token + (1 if output_diagonal else 2)
    edges = tuple((variable(r), variable(c)) for r,c in coupling_edges)
    outgoing = {i: set() for i in range(dimension)}
    incoming = {i: set() for i in range(dimension)}
    for r,c in coupling_support:
        outgoing[r].add(c); incoming[c].add(r)
    compact = []
    admitted = []
    def visit(index):
        if index == free_count:
            if len(admitted) >= row_limit:
                raise GenericDimensionRuntimeError(
                    f"support assignment bank requires at least {(len(admitted)+1)*row_bytes} estimated bytes; "
                    f"memory_limit_bytes={memory_limit_bytes}")
            admitted.append((compact[0], compact[0], *compact[1:]) if output_diagonal else tuple(compact))
            return
        choices = set(range(dimension))
        for r,c in edges:
            if r == c == index:
                choices.intersection_update(i for i in range(dimension) if (i,i) in coupling_support)
            elif r == index and c < index:
                choices.intersection_update(incoming[compact[c]])
            elif c == index and r < index:
                choices.intersection_update(outgoing[compact[r]])
        for value in sorted(choices):
            compact.append(value); visit(index+1); compact.pop()
    visit(0)
    return tuple(admitted), candidate_count


def _monomial_from_symbolic_assignment(
    key: tuple[Any, ...],
    assignment: tuple[int, ...],
    *,
    dimension: int,
) -> tuple[Any, ...]:
    """Render one scalar monomial without rebuilding formal-AST provenance."""

    _output_diagonal, dummy_count, factors, phase_u, phase_v, phase_w = key
    if len(assignment) != dummy_count + 2:
        raise GenericDimensionRuntimeError(
            "symbolic assignment has the wrong number of level indices"
        )

    def level(token: int) -> int:
        if token == -2:
            return assignment[0]
        if token == -1:
            return assignment[1]
        return assignment[token + 2]

    concrete_factors: list[tuple[Any, ...]] = []
    for factor in factors:
        row = level(factor.row)
        column = level(factor.column)
        if factor.kind == "coupling":
            concrete_factors.append(("coupling_operator", row, column))
        elif factor.kind == "density":
            if factor.label != "rho":
                raise GenericDimensionRuntimeError(
                    "symbolic family contains an unknown density operator"
                )
            concrete_factors.append(
                ("density_operator", factor.label, row, column)
            )
        elif factor.kind == "kernel":
            concrete_factors.append(
                ("kernel", "Gamma", factor.label, row, column)
            )
        else:  # pragma: no cover - guarded by _symbolic_assignment_topology.
            raise GenericDimensionRuntimeError(
                "symbolic family contains an unsupported factor"
            )

    def concrete_phase(
        sparse_phase: tuple[tuple[int, int], ...],
    ) -> tuple[int, ...]:
        values = [0] * dimension
        for token, coefficient in sparse_phase:
            values[level(token)] += coefficient
        return tuple(values)

    return (
        assignment[0],
        assignment[1],
        tuple(sorted(concrete_factors)),
        concrete_phase(phase_u),
        concrete_phase(phase_v),
        concrete_phase(phase_w),
    )


def _finalize_temporal_groups(grouped, coupling, *, dimension, path_count, coefficient_l1, cancellation_tolerance):
    cores: list[WeightedTemporalCore] = []
    retained_weight_count = 0
    pruned_weight_count = 0
    maximum_pruned_relative_residual = 0.0
    for temporal_key in sorted(grouped):
        kernel_identities, phase_u, phase_v, phase_w = temporal_key
        retained_weights: list[tuple[int, int, complex]] = []
        for (output_flat, input_flat), (weight, weight_l1) in sorted(
            grouped[temporal_key].items()
        ):
            relative = abs(weight) / weight_l1 if weight_l1 else 0.0
            should_prune = weight == 0 or (
                cancellation_tolerance > 0.0
                and weight_l1
                and relative <= cancellation_tolerance
            )
            if should_prune:
                pruned_weight_count += 1
                maximum_pruned_relative_residual = max(
                    maximum_pruned_relative_residual, relative
                )
                continue
            retained_weights.append(
                (output_flat, input_flat, complex(weight))
            )
        if not retained_weights:
            continue
        core = TemporalCore(
            kernel_identities=kernel_identities,
            phase_u=phase_u,
            phase_v=phase_v,
            phase_w=phase_w,
            template_id=_classify_template(kernel_identities),
            contributions=(),
        )
        cores.append(
            WeightedTemporalCore(
                core=core, weights=tuple(retained_weights)
            )
        )
        retained_weight_count += len(retained_weights)

    source_plan = StreamedExecutionPlan(
        input_monomial_count=path_count,
        input_coefficient_l1=coefficient_l1,
        cores=tuple(weighted.core for weighted in cores),
        dimension=dimension,
    )
    frozen_coupling = coupling.copy()
    frozen_coupling.setflags(write=False)
    plan = SpecializedExecutionPlan(
        source_plan=source_plan,
        coupling_operator=frozen_coupling,
        cores=tuple(cores),
    )
    return plan, retained_weight_count, pruned_weight_count, maximum_pruned_relative_residual


class _DirectTemporalContribution(NamedTuple):
    temporal_key: tuple[Any, ...]
    output_flat: int
    input_flat: int
    coupling_entries: tuple[tuple[int, int], ...]


def _direct_temporal_assignment(key, assignment, *, dimension):
    """Lower one certified assignment without scalar factor objects/parsing.

    The static factors are unchanged by stationary reduction, so comparing
    the kernel/phase tuple gives exactly the scalar oracle's v/w representative.
    """
    _diagonal, dummy_count, factors, *phases = key
    if len(assignment) != dummy_count + 2:
        raise GenericDimensionRuntimeError("symbolic assignment length mismatch")
    def level(token):
        return assignment[token + 2]
    couplings, kernels, densities = [], [], []
    for factor in factors:
        row, column = level(factor.row), level(factor.column)
        if factor.kind == "coupling":
            couplings.append((row, column))
        elif factor.kind == "density" and factor.label == "rho":
            densities.append((row, column))
        elif factor.kind == "kernel":
            kernels.append((factor.label, 0, 0) if row == column else (factor.label, row, column))
        else:
            raise GenericDimensionRuntimeError("unsupported direct temporal factor")
    if len(kernels) != 3 or len(densities) != 1:
        raise GenericDimensionRuntimeError("invalid TCL6 factor counts")
    concrete_phases = []
    for phase in phases:
        vector = [0] * dimension
        for token, coefficient in phase:
            vector[level(token)] += coefficient
        concrete_phases.append(tuple(vector))
    temporal = (tuple(sorted(kernels)), *concrete_phases)
    if not any(support.lstrip("-") == "UV" for support, _, _ in kernels):
        image = (tuple(sorted((_swap_vw_support(support), row, col) for support, row, col in kernels)),
                 concrete_phases[0], concrete_phases[2], concrete_phases[1])
        temporal = min(temporal, image)
    density_row, density_column = densities[0]
    return _DirectTemporalContribution(temporal, assignment[0]+dimension*assignment[1],
        density_row+dimension*density_column, tuple(sorted(couplings)))


def compile_stationary_specialized_plan(
    records: Sequence[Mapping[str, Any]],
    *,
    coupling_operator: Sequence[Sequence[complex]],
    energies: Sequence[float] | None = None,
    dimension: int | None = None,
    early_model_specialization: bool = True,
    sparse_lowering: SparseLowering = "path-enumeration",
    cancellation_tolerance: float = 1.0e-13,
    progress: ProgressCallback | None = None,
    _structural_sink: list | None = None,
) -> GenericSpecializedPlanResult:
    """Stream ``data/hr-v1.jsonl`` into one exact model-bound temporal plan.

    Formal and stationary-bath cancellations are exact integers.  Exact
    coupling zeros are used only when ``early_model_specialization`` is true
    and the coupling support is sparse.  The default cancellation
    tolerance matches the previously validated finite-dimensional compiler and
    removes only cancellation residuals relative to their accumulated l1
    scale.  The opt-in ``symbolic-assignment-bank`` lowerer reuses each exact
    coupling-constraint assignment topology across symbolic families in
    sparse positive dimensions. ``direct-temporal`` additionally bypasses
    scalar-monomial construction while retaining that traversal and accumulation
    order. The path enumerator remains the default and the equivalence control.
    """

    import numpy as np

    coupling, energy, dimension = _validate_model(
        coupling_operator, energies, dimension
    )
    if not isinstance(early_model_specialization, bool):
        raise GenericDimensionRuntimeError(
            "early_model_specialization must be boolean"
        )
    if sparse_lowering not in {
        "path-enumeration",
        "symbolic-assignment-bank",
        "direct-temporal",
    }:
        raise GenericDimensionRuntimeError(
            "sparse_lowering must be 'path-enumeration' or "
            "'symbolic-assignment-bank' or 'direct-temporal'"
        )
    if (
        isinstance(cancellation_tolerance, bool)
        or not isinstance(cancellation_tolerance, (int, float))
        or not np.isfinite(cancellation_tolerance)
        or cancellation_tolerance < 0.0
    ):
        raise GenericDimensionRuntimeError(
            "cancellation_tolerance must be finite and nonnegative"
        )

    total_started = perf_counter()
    timings: dict[str, float] = {}
    started = perf_counter()
    coupling_pattern = exact_coupling_pattern(coupling)
    gap_partition = (
        None if energy is None else exact_bohr_gap_partition(energy)
    )
    early_selected = bool(
        early_model_specialization and not coupling_pattern.is_dense
    )
    assignment_bank_selected = sparse_lowering in {"symbolic-assignment-bank", "direct-temporal"}
    direct_temporal_selected = sparse_lowering == "direct-temporal"
    if assignment_bank_selected and (
        not early_selected
    ):
        raise GenericDimensionRuntimeError(
            f"{sparse_lowering} lowering requires exact sparse "
            "early specialization"
        )
    coupling_support = (
        coupling_pattern.nonzero_entries if early_selected else None
    )
    timings["exact_model_structure"] = perf_counter() - started

    started = perf_counter()
    symbolic = build_symbolic_index_plan(
        records, canonicalization="traversal"
    )
    # Keep the canonical formal-word order even for the assignment bank.  The
    # bank removes repeated support traversal, but changing the order in which
    # complex static weights are accumulated would introduce avoidable
    # last-bit differences from the path-enumeration oracle.
    formal_words = aggregate_formal_words(records)
    timings["symbolic_and_formal_collection"] = perf_counter() - started

    join_plan = None
    join_report: Mapping[str, Any] | None = None
    if early_selected:
        started = perf_counter()
        join_plan = build_dimension_join_plan(
            symbolic,
            dimension=dimension,
            coupling_support=coupling_pattern.nonzero_entries,
        )
        join_report = join_plan.report
        timings["dimension_support_join_plan"] = perf_counter() - started
    else:
        timings["dimension_support_join_plan"] = 0.0

    # Each terminal static entry retains [complex weight, positive l1 scale].
    grouped: dict[
        tuple[Any, ...], dict[tuple[int, int], list[Any]]
    ] = defaultdict(dict)
    product_cache: dict[tuple[tuple[int, int], ...], complex] = {}
    path_count = 0
    stationary_word_monomial_count = 0
    coefficient_l1 = 0
    assignment_banks: dict[
        tuple[bool, int, tuple[tuple[int, int], ...]],
        tuple[tuple[int, ...], ...],
    ] = {}
    assignment_bank_candidate_count = 0
    assignment_bank_admitted_count = 0
    enumeration_started = perf_counter()

    indexed_formal_rows = (
        tuple(
            (
                indexed_family_key(word, canonicalization="traversal"),
                coefficient,
            )
            for word, coefficient in formal_words
        )
        if assignment_bank_selected
        else ()
    )
    indexed_family_coefficients: Counter[tuple[Any, ...]] = Counter()
    for family_key, coefficient in indexed_formal_rows:
        indexed_family_coefficients[family_key] += coefficient
    # A traversal family that cancels to exact integer zero is absent from the
    # independent symbolic join certificate.  Remove only those whole
    # zero-families here; retain the canonical word order for every nonzero
    # family so complex accumulation stays bitwise aligned with the oracle.
    lowering_rows = (
        tuple(
            (family_key, coefficient)
            for family_key, coefficient in indexed_formal_rows
            if indexed_family_coefficients[family_key]
        )
        if assignment_bank_selected
        else formal_words
    )
    prelowering_canceled_formal_word_count = (
        len(indexed_formal_rows) - len(lowering_rows)
        if assignment_bank_selected
        else 0
    )
    certified_family_path_counts = (
        {
            family.key: count
            for family, count in zip(
                symbolic.families,
                join_plan.family_path_counts,
                strict=True,
            )
        }
        if assignment_bank_selected and join_plan is not None
        else {}
    )
    for ordinal, (lowering_source, word_coefficient) in enumerate(
        lowering_rows, start=1
    ):
        word_monomials: Counter[tuple[Any, ...]] = Counter()
        word_paths = 0
        if assignment_bank_selected:
            topology = _symbolic_assignment_topology(lowering_source)
            assignments = assignment_banks.get(topology)
            if assignments is None:
                assert coupling_support is not None
                assignments, candidate_count = _build_symbolic_assignment_bank(
                    topology,
                    dimension=dimension,
                    coupling_support=coupling_support,
                )
                assignment_banks[topology] = assignments
                assignment_bank_candidate_count += candidate_count
                assignment_bank_admitted_count += len(assignments)
            if len(assignments) != certified_family_path_counts.get(
                lowering_source
            ):
                raise GenericDimensionRuntimeError(
                    "symbolic assignment bank disagrees with the exact "
                    "support-join certificate"
                )
            for assignment in assignments:
                representative = (_direct_temporal_assignment(
                    lowering_source, assignment, dimension=dimension,
                ) if direct_temporal_selected else stationary_scalar_bath_monomial_key(
                    _monomial_from_symbolic_assignment(
                        lowering_source,
                        assignment,
                        dimension=dimension,
                    )
                ))
                word_monomials[representative] += 1
                word_paths += 1
        else:
            for path in enumerate_contraction_paths(
                lowering_source,
                dimension=dimension,
                coupling_support=coupling_support,
                support_join_pruning=early_selected,
            ):
                representative = stationary_scalar_bath_monomial_key(
                    path.monomial_key
                )
                word_monomials[representative] += 1
                word_paths += 1
        path_count += word_paths
        stationary_word_monomial_count += len(word_monomials)

        for key, multiplicity in word_monomials.items():
            coefficient = int(word_coefficient) * int(multiplicity)
            if coefficient == 0:
                continue
            if direct_temporal_selected:
                temporal_key, contribution = key.temporal_key, key
            else:
                temporal_key, contribution = _parse_monomial(
                    key, coefficient, dimension=dimension
                )
            coupling_entries = contribution.coupling_entries
            if _structural_sink is not None:
                _structural_sink.append((coefficient, temporal_key, coupling_entries,
                    contribution.output_flat, contribution.input_flat))
            product = product_cache.get(coupling_entries)
            if product is None:
                product = 1.0 + 0.0j
                for row, column in coupling_entries:
                    product *= coupling[row, column]
                product_cache[coupling_entries] = complex(product)
            delta = complex(coefficient * product)
            static_index = (
                contribution.output_flat,
                contribution.input_flat,
            )
            weights = grouped[temporal_key]
            state = weights.get(static_index)
            if state is None:
                weights[static_index] = [delta, abs(delta)]
            else:
                state[0] += delta
                state[1] += abs(delta)
            coefficient_l1 += abs(coefficient)

        if progress is not None:
            progress(
                {
                    "stage": "streamed-model-bound-lowering",
                    "completed_formal_words": ordinal,
                    "formal_word_count": len(lowering_rows),
                    "contraction_paths": path_count,
                    "temporal_keys": len(grouped),
                    "elapsed_seconds": perf_counter() - enumeration_started,
                }
            )

    lowering_seconds = perf_counter() - enumeration_started
    # Retain the established timing key for report consumers while giving the
    # assignment-bank path an accurately named stage.
    timings["streamed_path_enumeration_and_specialization"] = lowering_seconds
    timings["streamed_lowering_and_specialization"] = lowering_seconds
    finalization_started = perf_counter()
    plan, retained_weight_count, pruned_weight_count, maximum_pruned_relative_residual = _finalize_temporal_groups(
        grouped, coupling, dimension=dimension, path_count=path_count,
        coefficient_l1=coefficient_l1, cancellation_tolerance=cancellation_tolerance,
    )
    cores = plan.cores
    timings["temporal_plan_finalization"] = (
        perf_counter() - finalization_started
    )
    timings["total"] = perf_counter() - total_started

    report = {
        "profile": "hr-dimension-streamed-model-bound-plan-v1",
        "dataset": "hr-v1.jsonl",
        "hilbert_dimension": dimension,
        "source_record_count": len(records),
        # Preserve the established report meaning: this is the number of
        # canonical formal rows collected from the input, before the optional
        # assignment-bank lowerer removes whole exact-zero traversal families.
        "collected_formal_word_count": len(formal_words),
        "symbolic_plan": dict(symbolic.report),
        "contraction_path_count": path_count,
        "stationary_word_monomial_count": stationary_word_monomial_count,
        "temporal_core_count": len(cores),
        "sparse_weight_count": retained_weight_count,
        "pruned_numeric_cancellation_count": pruned_weight_count,
        "maximum_pruned_relative_residual": (
            maximum_pruned_relative_residual
        ),
        "coupling_product_cache_size": len(product_cache),
        "sparse_lowering": {
            "requested": sparse_lowering,
            "selected": (
                sparse_lowering
                if assignment_bank_selected
                else "path-enumeration"
            ),
            "exact_structural_support_only": assignment_bank_selected,
            "unique_constraint_topology_count": len(assignment_banks),
            "unique_cartesian_candidate_count": (
                assignment_bank_candidate_count
            ),
            "unique_admitted_assignment_count": (
                assignment_bank_admitted_count
            ),
            "logical_family_assignment_count": (
                path_count if assignment_bank_selected else 0
            ),
            "admitted_assignment_reuse_factor": (
                path_count / assignment_bank_admitted_count
                if assignment_bank_admitted_count
                else 0.0
            ),
            "support_join_certificate_verified": assignment_bank_selected,
            "path_provenance_materialized": not assignment_bank_selected,
            "formal_word_accumulation_order_preserved": True,
            "scalar_monomials_materialized": not direct_temporal_selected,
            "exact_zero_family_formal_words_removed": (
                prelowering_canceled_formal_word_count
            ),
        },
        "cancellation_tolerance": float(cancellation_tolerance),
        "exact_coupling_pattern": coupling_pattern.report,
        "exact_bohr_gap_partition": (
            None if gap_partition is None else gap_partition.report
        ),
        "model_bound_early_specialization": {
            "requested": early_model_specialization,
            "selected": early_selected,
            "reason": (
                "sparse exact coupling support lowered before scalarization"
                if early_selected
                else (
                    "coupling matrix has no exact structural zeros"
                    if early_model_specialization
                    else "explicitly disabled"
                )
            ),
        },
        "exact_support_join_lowering": {
            "requested_when_applicable": True,
            "selected": early_selected,
            "reason": (
                "exact sparse coupling relation used"
                if early_selected
                else "dense coupling uses the certified generic fallback"
            ),
            "certificate": (
                None if join_report is None else dict(join_report)
            ),
        },
        "streamed_compilation": True,
        "execution_plan": plan.report,
        "timing_seconds": timings,
    }
    return GenericSpecializedPlanResult(
        plan=plan, report=MappingProxyType(report)
    )

