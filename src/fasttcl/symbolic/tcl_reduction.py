"""Exact ordered-cumulant to Hadamard compiler for arbitrary even TCL order.

The compiler produces the pre-wrapper S, with K = -[A(t), S] + H.c.
Its domain certificate is an exact partial-order decision proof, independent
of the integration planner. No fast numerical contraction is assumed here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from math import factorial
from typing import Any, Iterable, Iterator

from .hadamard_reduction import PhysicalBound, ReducedBranch, physical_correlations, reduce_planned_branch
from .integration import LOWER_ZERO, IntegrationState, correlation, effective_bounds, initial_state, plan_hadamardable
from .linearize import CollectionResult, collect_contributions, linearize_expression
from .region_identity import RegionIdentityCertificate, certify_region_identity
from .region_semantics import ChamberSpace, RegionBound, signed_region_residual
from .tcl_cumulant import CumulativeGroup, SuperoperatorMonomial, cumulative_groups, ordered_cumulant, reconstruct_outer_wrapper, undirected_pairing
from .tcl_order import TCLOrder


class TCLReductionError(ValueError):
    """The requested source, domain proof, or structural reduction is invalid."""


@dataclass(frozen=True)
class PairingPlan:
    order: int
    matching: tuple[tuple[int, int], ...]
    leaves: tuple[IntegrationState, ...]
    source_bounds: tuple[RegionBound, ...]
    leaf_bounds: tuple[tuple[RegionBound, ...], ...]
    certificate: RegionIdentityCertificate
    exhaustive_verified: bool = False


@dataclass(frozen=True)
class ReducedTCLBranch:
    order: int
    source_id: str
    branch_id: str
    correlations: tuple[tuple[int, int], ...]
    reduced: ReducedBranch


@dataclass(frozen=True)
class TCLReductionResult:
    order: int
    source_groups: tuple[CumulativeGroup, ...]
    pairing_plans: tuple[PairingPlan, ...]
    collection: CollectionResult
    reduced_branch_count: int
    full_monomial_count: int
    prewrapper_monomial_count: int
    outer_wrapper_verified: bool

    @property
    def counts(self) -> dict[str, int]:
        return {
            "source_groups": len(self.source_groups),
            "connected_pairings": len(self.pairing_plans),
            "pairing_plan_leaves": sum(len(plan.leaves) for plan in self.pairing_plans),
            "reduced_branches": self.reduced_branch_count,
            "density_linear_contributions": self.collection.input_contribution_count,
            "collected_groups": len(self.collection.groups),
            "cancellation_groups": len(self.collection.canceled_groups),
            "surviving_groups": len(self.collection.survivors),
        }


def _physical_bounds(state: IntegrationState) -> tuple[PhysicalBound, ...]:
    return tuple(sorted(PhysicalBound(variable.value, 0 if bound.lower == LOWER_ZERO else bound.lower.value, bound.upper.value)
                        for variable, bound in effective_bounds(state).items()))


def _dataset_pairs(pairs: Iterable[tuple[int, int]]) -> list[dict[str, str]]:
    return [{"left": str(left), "right": str(right)} for left, right in pairs]


def _validate_matching(pairs: Iterable[tuple[int, int]], order: int) -> tuple[tuple[int, int], ...]:
    result = tuple(tuple(pair) for pair in pairs)
    if any(len(pair) != 2 or any(type(label) is not int for label in pair) for pair in result):
        raise TCLReductionError("matching endpoints must be pairs of integer labels")
    if sorted(label for pair in result for label in pair) != list(range(order)):
        raise TCLReductionError(f"matching must contain each label 0..{order - 1} exactly once")
    return result


def certify_pairing_plan(
    matching: tuple[tuple[int, int], ...], *, order: int,
    max_certificate_nodes: int | None = None, verify_exhaustive: bool = False,
) -> PairingPlan:
    """Plan and independently prove one matching at the requested order.

    ``verify_exhaustive`` also invokes the older factorial chamber oracle.
    It is useful for independent low-order regression, not large orders.
    """
    spec = TCLOrder(order)
    canonical = undirected_pairing(_validate_matching(matching, order))
    physical = physical_correlations(_dataset_pairs(canonical), order=order)
    seed = initial_state(spec.pair_grade, tuple(correlation(left, right) for left, right in physical))
    leaves = plan_hadamardable(seed)
    if not leaves:
        raise TCLReductionError("the integration planner produced no leaves")
    source = tuple(RegionBound(variable, 0, variable + 1) for variable in range(1, order))
    regions = tuple(tuple(RegionBound(bound.variable, bound.lower, bound.upper)
                          for bound in _physical_bounds(leaf)) for leaf in leaves)
    signed = tuple((leaf.coefficient, bounds) for leaf, bounds in zip(leaves, regions, strict=True))
    certificate = certify_region_identity(spec.internal_count, source, signed, max_nodes=max_certificate_nodes)
    if not certificate.is_zero:
        raise TCLReductionError(f"TCL{order} signed domain identity failed for {canonical!r}: {certificate!r}")
    if verify_exhaustive:
        space = ChamberSpace(spec.internal_count)
        residual = signed_region_residual(space.region(source), [(coefficient, space.region(bounds)) for coefficient, bounds in signed])
        if not residual.is_zero:
            raise TCLReductionError("partial-order proof and exhaustive chamber oracle disagree")
    return PairingPlan(order, canonical, leaves, source, regions, certificate, verify_exhaustive)


def build_pairing_plans(groups: Iterable[CumulativeGroup], *, order: int,
                        max_certificate_nodes: int | None = None,
                        verify_exhaustive: bool = False) -> tuple[PairingPlan, ...]:
    TCLOrder(order)
    matchings = sorted({undirected_pairing(group.correlations) for group in groups})
    return tuple(certify_pairing_plan(matching, order=order,
                 max_certificate_nodes=max_certificate_nodes, verify_exhaustive=verify_exhaustive)
                 for matching in matchings)


def iter_reduced_branches(groups: Iterable[CumulativeGroup], pairing_plans: Iterable[PairingPlan],
                          *, order: int) -> Iterator[ReducedTCLBranch]:
    spec = TCLOrder(order)
    plan_map: dict[tuple[tuple[int, int], ...], PairingPlan] = {}
    for plan in pairing_plans:
        TCLOrder(plan.order)
        if plan.order != order or not plan.certificate.is_zero:
            raise TCLReductionError("pairing plan has the wrong order or lacks a valid certificate")
        _validate_matching(plan.matching, order)
        if plan.matching in plan_map:
            raise TCLReductionError("duplicate pairing plan")
        # Public plans may have been loaded or copied with changed leaves.
        # Recheck the actual bound lists and signs, not a stale success flag.
        expected_source = tuple(RegionBound(variable, 0, variable + 1) for variable in range(1, order))
        actual_bounds = tuple(tuple(RegionBound(bound.variable, bound.lower, bound.upper)
                                   for bound in _physical_bounds(leaf)) for leaf in plan.leaves)
        physical_matching = undirected_pairing(physical_correlations(_dataset_pairs(plan.matching), order=order))
        if (not plan.leaves or plan.source_bounds != expected_source or plan.leaf_bounds != actual_bounds
                or plan.matching != undirected_pairing(plan.matching)):
            raise TCLReductionError("pairing plan metadata disagrees with its actual domains")
        for leaf in plan.leaves:
            leaf_matching = undirected_pairing((pair.first.value, pair.second.value) for pair in leaf.correlations)
            if leaf.n != spec.pair_grade or leaf.bar_index != spec.bar_index or leaf_matching != physical_matching:
                raise TCLReductionError("pairing plan leaf has the wrong order or matching")
        checked = certify_region_identity(spec.internal_count, expected_source,
            ((leaf.coefficient, bounds) for leaf, bounds in zip(plan.leaves, actual_bounds, strict=True)))
        if not checked.is_zero:
            raise TCLReductionError("pairing plan leaves no longer satisfy their signed domain identity")
        plan_map[plan.matching] = plan
    for index, group in enumerate(groups, 1):
        _validate_matching(group.correlations, order)
        source_id = f"CC{order}-{index:04d}"
        try:
            plan = plan_map[undirected_pairing(group.correlations)]
        except KeyError as exc:
            raise TCLReductionError(f"{source_id}: no certified pairing plan") from exc
        expression = group.expression
        for branch_index, leaf in enumerate(plan.leaves, 1):
            reduced = reduce_planned_branch(expression, _dataset_pairs(group.correlations),
                physical_order=tuple(0 if label == LOWER_ZERO else label.value for label in leaf.order),
                physical_bounds=_physical_bounds(leaf), coefficient=leaf.coefficient,
                bar_index=spec.bar_index)
            if len(reduced.steps) != spec.pair_grade:
                raise TCLReductionError(f"{source_id}: expected {spec.pair_grade} Hadamard eliminations")
            yield ReducedTCLBranch(order, source_id, f"{source_id}-B{branch_index:02d}", group.correlations, reduced)


def derive_tcl_reduction(order: int, *, max_certificate_nodes: int | None = None,
                         verify_exhaustive: bool = False) -> TCLReductionResult:
    """Compile the complete requested order from its ordered cumulant.

    There is no order ceiling. Term counts grow combinatorially; resource
    exhaustion or a failed certificate raises rather than returns a partial
    result as a complete derivation.
    """
    TCLOrder(order)
    groups = cumulative_groups(order)
    prewrapper: Counter[SuperoperatorMonomial] = Counter()
    for group in groups:
        for word, coefficient in group.polynomial.items():
            density = word.index(("density_operator", "rho"))
            prewrapper[SuperoperatorMonomial(group.correlations,
                tuple(int(atom[1]) for atom in word[:density]),
                tuple(int(atom[1]) for atom in word[density + 1:]))] += coefficient
    full = ordered_cumulant(order)
    if reconstruct_outer_wrapper(prewrapper) != full:
        raise TCLReductionError(f"the TCL{order} outer wrapper does not reconstruct the cumulant")
    full_count, prewrapper_count = len(full), len(prewrapper)
    del full, prewrapper
    plans = build_pairing_plans(groups, order=order, max_certificate_nodes=max_certificate_nodes,
                                verify_exhaustive=verify_exhaustive)
    leaf_counts = {plan.matching: len(plan.leaves) for plan in plans}
    branch_count = sum(leaf_counts[undirected_pairing(group.correlations)] for group in groups)
    contributions = (contribution for branch in iter_reduced_branches(groups, plans, order=order)
        for contribution in linearize_expression(branch.reduced.expression,
            coefficient=branch.reduced.coefficient, source_id=branch.branch_id))
    collection = collect_contributions(contributions)
    return TCLReductionResult(order, groups, plans, collection, branch_count,
                              full_count, prewrapper_count, True)


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def source_records(result: TCLReductionResult) -> Iterator[dict[str, Any]]:
    for index, group in enumerate(result.source_groups, 1):
        expression = group.expression
        yield {"schema_version": "tcl-cc-v1", "id": f"CC{result.order}-{index:04d}",
               "order": result.order, "coefficient": 1, "correlation_pairs": _dataset_pairs(group.correlations),
               "free_monomial_count": len(group.polynomial), "expression": expression,
               "expression_sha256": _digest(expression), "source": "exact-ordered-cumulant"}


def reduced_records(result: TCLReductionResult) -> Iterator[dict[str, Any]]:
    for index, group in enumerate(result.collection.survivors, 1):
        yield {"schema_version": "tcl-hr-term-v1", "id": f"TCL{result.order}-HR-{index:06d}",
               "order": result.order, "coefficient": group.coefficient, "expression": group.expression,
               "expression_sha256": _digest(group.expression), "lineage": [
                   {"source_id": item.source_id, "branch_path": list(item.branch_path), "coefficient": item.coefficient}
                   for item in group.contributors]}


def plan_records(result: TCLReductionResult) -> Iterator[dict[str, Any]]:
    def bound_record(bound: RegionBound) -> dict[str, int]:
        return {"variable": bound.variable, "lower": bound.lower, "upper": bound.upper}
    for index, plan in enumerate(result.pairing_plans, 1):
        yield {"schema_version": "tcl-plan-v1", "id": f"P{result.order}-{index:03d}",
               "order": result.order, "matching": [list(pair) for pair in plan.matching],
               "strict_chamber_count": factorial(result.order - 1),
               "source_bounds": [bound_record(bound) for bound in plan.source_bounds],
               "signed_residual_is_zero": plan.certificate.is_zero,
               "certificate_method": "exact-partial-order-decision",
               "certificate_nodes": plan.certificate.explored_nodes,
               "exhaustive_verified": plan.exhaustive_verified,
               "leaves": [{"coefficient": leaf.coefficient,
                   "physical_order": [0 if label == LOWER_ZERO else label.value for label in leaf.order],
                   "bounds": [bound_record(bound) for bound in bounds]}
                   for leaf, bounds in zip(plan.leaves, plan.leaf_bounds, strict=True)]}


__all__ = ["TCLReductionError", "PairingPlan", "ReducedTCLBranch", "TCLReductionResult",
           "certify_pairing_plan", "build_pairing_plans", "iter_reduced_branches", "derive_tcl_reduction",
           "source_records", "reduced_records", "plan_records"]
