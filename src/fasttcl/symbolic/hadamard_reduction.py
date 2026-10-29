"""Structural application of the Appendix-C/Hadamard reduction.

This module deliberately accepts an already planned integration branch.  The
integration planner is responsible for rearranging and splitting the original
simplex; this module is responsible only for applying Eq. (68) to the inner
variables and normalizing the surviving dummy variables at any positive even
order.  TCL6 remains the default public label convention.

All transformations operate on typed AST nodes.  No operator or kernel is
identified by substring replacement.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, cast

from .tcl_order import TCLOrder, is_time_symbol, validate_even_order

from .ast import (
    ASTNode,
    CommutatorNode,
    HadamardNode,
    OrderedProductNode,
    SumNode,
    TimeExpressionNode,
    TimeTerm,
    apply_integer_coefficient,
    commutator,
    hadamard,
    kernel,
    make_ordered_product,
    make_sum,
    operator,
    scalar,
    time_expression,
    validate_ast,
)


class HadamardReductionError(ValueError):
    """Raised when a planned branch does not satisfy the reduction premises."""


@dataclass(frozen=True, order=True)
class PhysicalBound:
    """Effective lower and upper physical-time labels for one integration."""

    variable: int
    lower: int
    upper: int

    def __post_init__(self) -> None:
        for name in ("variable", "lower", "upper"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"physical bound {name} must be an integer")


@dataclass(frozen=True)
class HadamardStep:
    """One replayable application of Eq. (68)."""

    integrated_variable: int
    source_operator_label: str
    partner: int
    partner_operator_label: str
    lower: int
    upper: int
    transpose: bool
    correlation_orientation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule": "H2",
            "integrated_variable": self.integrated_variable,
            "source_operator_label": self.source_operator_label,
            "partner": self.partner,
            "partner_operator_label": self.partner_operator_label,
            "lower": self.lower,
            "upper": self.upper,
            "transpose": self.transpose,
            "correlation_orientation": self.correlation_orientation,
        }


@dataclass(frozen=True)
class ReducedBranch:
    """A TCL branch after eliminating one integration per bath pair."""

    coefficient: int
    expression: ASTNode
    physical_order: tuple[int, ...]
    physical_bounds: tuple[PhysicalBound, ...]
    outer_variable_map: tuple[tuple[int, str], ...]
    steps: tuple[HadamardStep, ...]


def _time_symbol(label: str) -> str:
    """Accept explicit time symbols and the legacy one-letter shorthand.

    Internally retained variables always use ``t_<dummy>``.  Thus dummy
    operator ``t`` uses ``t_t`` without colliding with current time ``t``.
    """

    if is_time_symbol(label):
        return label
    if isinstance(label, str) and is_time_symbol(f"t_{label}"):
        return f"t_{label}"
    raise HadamardReductionError(f"unknown canonical time label {label!r}")


def dataset_label_to_physical(label: str, *, order: int = 6) -> int:
    """Map dataset labels ``0..order-1`` to physical ``order..1`` labels."""

    validate_even_order(order)
    if not isinstance(label, str) or not label.isascii() or not label.isdigit():
        raise HadamardReductionError(
            f"expected a numeric dataset operator label, got {label!r}"
        )
    dataset = int(label)
    if not 0 <= dataset < order:
        raise HadamardReductionError(
            f"expected a TCL{order} dataset label 0..{order - 1}, "
            f"got {label!r}"
        )
    return order - dataset


def physical_to_dataset_label(value: int, *, order: int = 6) -> str:
    """Inverse of :func:`dataset_label_to_physical` for operator time labels."""

    validate_even_order(order)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("physical operator time must be an integer")
    if value < 1 or value > order:
        raise HadamardReductionError(
            f"physical operator time must lie in 1..{order}, got {value}"
        )
    return str(order - value)


def physical_correlations(
    correlation_pairs: Sequence[Mapping[str, str]],
    *,
    order: int = 6,
) -> tuple[tuple[int, int], ...]:
    """Convert the record correlation list while retaining pair orientation."""

    validate_even_order(order)
    result = tuple(
        (
            dataset_label_to_physical(pair["left"], order=order),
            dataset_label_to_physical(pair["right"], order=order),
        )
        for pair in correlation_pairs
    )
    flattened = [value for pair in result for value in pair]
    if sorted(flattened) != list(range(1, order + 1)):
        raise HadamardReductionError(
            f"TCL{order} correlations must form an oriented perfect "
            f"matching of 1..{order}"
        )
    return result


def _time_difference(left: str, right: str) -> TimeExpressionNode:
    left_symbol = _time_symbol(left)
    right_symbol = _time_symbol(right)
    terms: list[TimeTerm] = [
        {"symbol": left_symbol, "coefficient": 1},  # type: ignore[typeddict-item]
        {"symbol": right_symbol, "coefficient": -1},  # type: ignore[typeddict-item]
    ]
    return time_expression(terms)


def _gamma_difference(
    first: tuple[str, str],
    second: tuple[str, str],
    *,
    transpose: bool,
) -> ASTNode:
    """Return ``Gamma(first)-Gamma(second)`` with ``Gamma(0)=0``."""

    first = (_time_symbol(first[0]), _time_symbol(first[1]))
    second = (_time_symbol(second[0]), _time_symbol(second[1]))
    first_zero = first[0] == first[1]
    second_zero = second[0] == second[1]
    if first_zero and second_zero:
        return scalar(0)
    if first_zero:
        return apply_integer_coefficient(
            kernel(
                [_time_difference(*second)],
                transpose=transpose,
            ),
            -1,
        )
    if second_zero:
        return kernel(
            [_time_difference(*first)],
            transpose=transpose,
        )
    return kernel(
        [_time_difference(*first), _time_difference(*second)],
        transpose=transpose,
        family="DeltaGamma",
    )


def build_hadamard_replacement(
    *,
    integrated_is_left: bool,
    partner_operator_label: str,
    partner_time: str,
    lower_time: str,
    upper_time: str,
) -> ASTNode:
    """Build the exact Eq. (68) replacement for one integrated operator.

    ``integrated_is_left`` describes the ordered correlation.  For
    ``<integrated, partner>`` the first line of Eq. (68) applies; for
    ``<partner, integrated>`` the transposed second line applies.  Time inputs
    may use explicit symbols ``t_a``, ``t_b``, etc.; legacy ``a``, ``b`` etc.
    remain accepted.  Use ``t_t`` for dummy ``t`` and ``t`` for current time.
    """

    if integrated_is_left:
        gamma_value = _gamma_difference(
            (upper_time, partner_time),
            (lower_time, partner_time),
            transpose=False,
        )
    else:
        gamma_value = _gamma_difference(
            (partner_time, lower_time),
            (partner_time, upper_time),
            transpose=True,
        )
    return hadamard(operator(partner_operator_label), gamma_value)


def _transform_operators(
    node: ASTNode,
    replacements: Mapping[str, ASTNode],
) -> ASTNode:
    node_type = node["type"]
    if node_type == "operator":
        label = cast(dict[str, str], node)["label"]
        return deepcopy(replacements.get(label, node))
    if node_type == "sum":
        return make_sum(
            [
                _transform_operators(term, replacements)
                for term in cast(SumNode, node)["terms"]
            ]
        )
    if node_type == "ordered_product":
        return make_ordered_product(
            [
                _transform_operators(factor, replacements)
                for factor in cast(OrderedProductNode, node)["factors"]
            ]
        )
    if node_type == "commutator":
        current = cast(CommutatorNode, node)
        return commutator(
            _transform_operators(current["left"], replacements),
            _transform_operators(current["right"], replacements),
        )
    if node_type == "hadamard":
        current = cast(HadamardNode, node)
        return hadamard(
            _transform_operators(current["left"], replacements),
            _transform_operators(current["right"], replacements),
        )
    return deepcopy(node)


def _operator_occurrences(node: ASTNode, label: str) -> int:
    node_type = node["type"]
    if node_type == "operator":
        return int(cast(dict[str, str], node)["label"] == label)
    if node_type == "sum":
        return sum(
            _operator_occurrences(term, label)
            for term in cast(SumNode, node)["terms"]
        )
    if node_type == "ordered_product":
        return sum(
            _operator_occurrences(factor, label)
            for factor in cast(OrderedProductNode, node)["factors"]
        )
    if node_type in {"commutator", "hadamard"}:
        current = cast(dict[str, ASTNode], node)
        return _operator_occurrences(
            current["left"], label
        ) + _operator_occurrences(current["right"], label)
    return 0


def _operator_occurrences_per_additive_path(
    node: ASTNode, label: str
) -> tuple[int, ...]:
    """Count ``label`` in every monomial induced by structural sums.

    H1 permits a linear superoperator around the integrated operator, so one
    occurrence in each additive branch is valid even when a sum contains the
    operator several times in total.  Multiple occurrences within one
    monomial are not valid.
    """

    node_type = node["type"]
    if node_type == "operator":
        return (int(cast(dict[str, str], node)["label"] == label),)
    if node_type == "sum":
        return tuple(
            count
            for term in cast(SumNode, node)["terms"]
            for count in _operator_occurrences_per_additive_path(term, label)
        )
    if node_type == "ordered_product":
        counts = (0,)
        for factor in cast(OrderedProductNode, node)["factors"]:
            factor_counts = _operator_occurrences_per_additive_path(
                factor, label
            )
            counts = tuple(
                left + right
                for left in counts
                for right in factor_counts
            )
        return counts
    if node_type in {"commutator", "hadamard"}:
        current = cast(dict[str, ASTNode], node)
        left_counts = _operator_occurrences_per_additive_path(
            current["left"], label
        )
        right_counts = _operator_occurrences_per_additive_path(
            current["right"], label
        )
        return tuple(
            left + right
            for left in left_counts
            for right in right_counts
        )
    return (0,)


def _bound_map(
    bounds: Iterable[PhysicalBound],
    *,
    order: int,
) -> dict[int, PhysicalBound]:
    result: dict[int, PhysicalBound] = {}
    for bound in bounds:
        if bound.variable in result:
            raise HadamardReductionError(
                f"duplicate bounds for physical variable {bound.variable}"
            )
        if not (0 <= bound.lower < bound.upper <= order):
            raise HadamardReductionError(
                f"invalid physical bounds {bound!r}"
            )
        if bound.variable in {bound.lower, bound.upper}:
            raise HadamardReductionError(
                f"an integration variable cannot be its own bound: {bound!r}"
            )
        result[bound.variable] = bound
    if set(result) != set(range(1, order)):
        raise HadamardReductionError(
            f"a TCL{order} branch must provide bounds for variables "
            f"1..{order - 1}"
        )
    return result


def _canonical_outer_map(
    order: tuple[int, ...],
    bounds: Mapping[int, PhysicalBound],
    *,
    bar_index: int,
    current_time: int,
) -> dict[int, str]:
    outer = order[2:bar_index]
    coordinates = TCLOrder(current_time)
    retained_count = coordinates.pair_grade - 1
    if len(outer) != retained_count:
        raise HadamardReductionError(
            f"TCL{current_time} Hadamard reduction must leave exactly "
            f"{retained_count} outer variables"
        )
    # A DAG has exactly one topological ordering iff each step has exactly
    # one available vertex.  This certifies the retained simplex without
    # enumerating (N-1)! permutations.
    vertices = {0, current_time, *outer}
    successors: dict[int, set[int]] = {vertex: set() for vertex in vertices}
    predecessors: dict[int, set[int]] = {vertex: set() for vertex in vertices}
    edges = {(0, current_time)}
    for variable in outer:
        bound = bounds[variable]
        if bound.lower not in vertices or bound.upper not in vertices:
            raise HadamardReductionError("outer bounds must depend only on outer times")
        edges.update(((0, variable), (variable, current_time),
                      (bound.lower, variable), (variable, bound.upper)))
    for lower, upper in edges:
        successors[lower].add(upper)
        predecessors[upper].add(lower)
    remaining = set(vertices)
    topological: list[int] = []
    while remaining:
        available = [vertex for vertex in remaining if not predecessors[vertex]]
        if len(available) != 1:
            reason = "a cycle" if not available else "multiple chambers"
            raise HadamardReductionError(
                "surviving outer bounds must select exactly one strict simplex "
                f"chamber, got {reason}"
            )
        vertex = available[0]
        topological.append(vertex)
        remaining.remove(vertex)
        for successor in successors[vertex]:
            predecessors[successor].remove(vertex)
    earliest_to_latest = topological[1:-1]
    return {
        physical: label
        for physical, label in zip(reversed(earliest_to_latest),
                                   coordinates.dummy_labels, strict=True)
    }


def reduce_planned_branch(
    expression: ASTNode,
    correlation_pairs: Sequence[Mapping[str, str]],
    *,
    physical_order: Sequence[int],
    physical_bounds: Iterable[PhysicalBound],
    coefficient: int = 1,
    bar_index: int | None = None,
) -> ReducedBranch:
    """Apply Eq. (68) to a valid planned even-order TCL branch.

    The planner supplies a complete physical permutation and effective bounds.
    The expression and correlations use the public Appendix-F labels.
    """

    validate_ast(expression)
    if isinstance(coefficient, bool) or not isinstance(coefficient, int):
        raise TypeError("branch coefficient must be an integer")
    if coefficient == 0:
        raise HadamardReductionError("branch coefficient must be nonzero")
    order = tuple(physical_order)
    if len(order) < 3:
        raise HadamardReductionError("physical order is too short")
    if any(isinstance(label, bool) or not isinstance(label, int) for label in order):
        raise TypeError("physical order labels must be integers")
    tcl_order = order[1]
    if tcl_order < 2 or tcl_order % 2:
        raise HadamardReductionError("current-time label must be positive and even")
    if sorted(order) != list(range(tcl_order + 1)) or order[:2] != (0, tcl_order):
        raise HadamardReductionError(
            "physical order must be a complete permutation beginning with "
            f"0,{tcl_order}"
        )
    expected_bar_index = tcl_order // 2 + 1
    if bar_index is None:
        bar_index = expected_bar_index
    if bar_index != expected_bar_index:
        raise HadamardReductionError(
            f"the TCL{tcl_order} vertical bar must have index "
            f"{expected_bar_index}"
        )

    bounds = _bound_map(physical_bounds, order=tcl_order)
    outer_map = _canonical_outer_map(
        order,
        bounds,
        bar_index=bar_index,
        current_time=tcl_order,
    )
    inner = order[bar_index:]
    outer_side = set(order[:bar_index])
    correlations = physical_correlations(correlation_pairs, order=tcl_order)

    for left, right in correlations:
        if (left in outer_side) == (right in outer_side):
            raise HadamardReductionError(
                "Condition 1 failed: a correlation does not cross the bar"
            )

    point_time: dict[int, str] = {0: "0", tcl_order: "t"}
    point_time.update({physical: f"t_{dummy}" for physical, dummy in outer_map.items()})
    point_operator: dict[int, str] = {tcl_order: "0"}
    point_operator.update(outer_map)

    current = deepcopy(expression)
    steps: list[HadamardStep] = []
    for integrated in reversed(inner):
        bound = bounds[integrated]
        if bound.lower not in outer_side or bound.upper not in outer_side:
            raise HadamardReductionError(
                "Condition 2 failed: an inner effective bound is not outer"
            )
        pair = next(
            (
                candidate
                for candidate in correlations
                if integrated in candidate
            ),
            None,
        )
        if pair is None:
            raise HadamardReductionError(
                f"no correlation contains physical variable {integrated}"
            )
        left, right = pair
        integrated_is_left = left == integrated
        partner = right if integrated_is_left else left
        if partner not in point_time or partner not in point_operator:
            raise HadamardReductionError(
                "Condition 1 failed: integrated variable partner is not outer"
            )

        source_label = physical_to_dataset_label(integrated, order=tcl_order)
        occurrence_counts = _operator_occurrences_per_additive_path(
            current, source_label
        )
        if not occurrence_counts or any(
            count != 1 for count in occurrence_counts
        ):
            raise HadamardReductionError(
                "H1 failed: integrated operator "
                f"{source_label} must occur exactly once in every "
                f"additive monomial, got {occurrence_counts!r}"
            )
        replacement = build_hadamard_replacement(
            integrated_is_left=integrated_is_left,
            partner_operator_label=point_operator[partner],
            partner_time=point_time[partner],
            lower_time=point_time[bound.lower],
            upper_time=point_time[bound.upper],
        )
        current = _transform_operators(
            current,
            {source_label: replacement},
        )
        steps.append(
            HadamardStep(
                integrated_variable=integrated,
                source_operator_label=source_label,
                partner=partner,
                partner_operator_label=point_operator[partner],
                lower=bound.lower,
                upper=bound.upper,
                transpose=not integrated_is_left,
                correlation_orientation=(
                    "integrated-left"
                    if integrated_is_left
                    else "integrated-right"
                ),
            )
        )

    rename_replacements = {
        physical_to_dataset_label(physical, order=tcl_order): operator(dummy)
        for physical, dummy in outer_map.items()
    }
    current = _transform_operators(current, rename_replacements)
    validate_ast(current, order=tcl_order)

    leftover = [
        physical_to_dataset_label(value, order=tcl_order)
        for value in inner
        if _operator_occurrences(
            current, physical_to_dataset_label(value, order=tcl_order)
        )
    ]
    if leftover:
        raise HadamardReductionError(
            f"integrated operator labels remain after reduction: {leftover}"
        )

    return ReducedBranch(
        coefficient=coefficient,
        expression=current,
        physical_order=order,
        physical_bounds=tuple(sorted(bounds.values())),
        outer_variable_map=tuple(sorted(outer_map.items())),
        steps=tuple(steps),
    )


__all__ = [
    "HadamardReductionError",
    "HadamardStep",
    "PhysicalBound",
    "ReducedBranch",
    "build_hadamard_replacement",
    "dataset_label_to_physical",
    "physical_correlations",
    "physical_to_dataset_label",
    "reduce_planned_branch",
]
