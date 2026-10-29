"""Exact signed-domain certificates without constructing permutation chambers.

The fixed endpoints are ``0`` and ``internal_count + 1``.  Each input region
is a conjunction of strict bounds on all internal variables, as in
``region_semantics``.  The identity checked is ``sum(sign * leaf) - source``.
Its equality is away from coincident-time boundaries; quadrature conventions
for those boundaries are deliberately outside this certificate.

The decision procedure maintains a partial order.  In each branch it closes
the conjunction of that order with every surviving region, discards
inconsistent conjunctions, removes implied edges, and collects identical
regions.  An unresolved comparison splits the remaining chambers into two
disjoint exhaustive cases.  A zero residual closes a branch without ordering
unrelated variables.  Thus this algorithm is exact at any order and never
constructs the factorial chamber universe, although difficult identities can
still require a large decision tree.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, TypeAlias

from .region_semantics import RegionBound, RegionSemanticsError


_Order: TypeAlias = tuple[int, ...]
_Terms: TypeAlias = tuple[tuple[int, _Order], ...]


@dataclass(frozen=True, slots=True)
class RegionIdentityCertificate:
    """A completed exact proof, or one strict-order counterexample.

    ``residual`` is the integer value of ``sum(sign * leaf) - source`` at
    ``witness_ordering``.  Successful certificates have neither a witness nor
    a residual.  ``explored_nodes`` counts partial-order search states visited.
    """

    is_zero: bool
    explored_nodes: int
    witness_ordering: tuple[int, ...] | None = None
    residual: int | None = None


class RegionIdentityLimitError(RuntimeError):
    """The requested search budget was exhausted; no proof was obtained."""

    def __init__(self, max_nodes: int, explored_nodes: int) -> None:
        self.max_nodes = max_nodes
        self.explored_nodes = explored_nodes
        super().__init__(
            f"region identity search reached max_nodes={max_nodes}; "
            "the identity has not been certified"
        )


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RegionSemanticsError(f"{name} must be a positive integer")
    return value


def _closure(rows: _Order) -> _Order | None:
    """Bitset transitive closure; ``None`` denotes an inconsistent order."""
    reach = list(rows)
    for middle, successors in enumerate(reach):
        bit = 1 << middle
        for lower in range(len(reach)):
            if reach[lower] & bit:
                reach[lower] |= successors
    if any(successors & (1 << label) for label, successors in enumerate(reach)):
        return None
    return tuple(reach)


def _region_order(internal_count: int, bounds: Iterable[RegionBound]) -> _Order:
    """Validate the complete bound grammar without constructing a region."""
    parsed = tuple(RegionBound.parse(bound) for bound in bounds)
    if len(parsed) != internal_count:
        raise RegionSemanticsError(
            f"expected {internal_count} bounds, got {len(parsed)}"
        )
    current_time = internal_count + 1
    seen: set[int] = set()
    rows = [0] * internal_count
    for bound in parsed:
        if not 1 <= bound.variable <= internal_count:
            raise RegionSemanticsError(
                "a bound variable must be an internal time label"
            )
        if not (0 <= bound.lower <= current_time and 0 <= bound.upper <= current_time):
            raise RegionSemanticsError(
                f"time label must lie in 0..{current_time}"
            )
        if len({bound.variable, bound.lower, bound.upper}) != 3:
            raise RegionSemanticsError(
                "a bound variable and its endpoints must be distinct"
            )
        if bound.variable in seen:
            raise RegionSemanticsError(
                f"duplicate bound for variable {bound.variable}"
            )
        seen.add(bound.variable)
        # The physical inequalities 0 < every internal time < current_time
        # are implicit.  Bounds opposing either endpoint are inconsistent.
        if bound.lower == current_time or bound.upper == 0:
            raise RegionSemanticsError("bound inequalities contain a cycle")
        if bound.lower:
            rows[bound.lower - 1] |= 1 << (bound.variable - 1)
        if bound.upper != current_time:
            rows[bound.variable - 1] |= 1 << (bound.upper - 1)
    closed = _closure(tuple(rows))
    if closed is None:
        raise RegionSemanticsError("bound inequalities contain a cycle")
    return closed


def _relative_cover(order: _Order, context: _Order) -> _Order:
    """Remove transitive edges and comparisons already implied by context.

    The transitive reduction of a finite DAG is unique.  Together with the
    context, the retained edges reconstruct ``order`` exactly, giving a
    canonical key for collecting equal indicator functions within a branch.
    """
    result = []
    for lower, successors in enumerate(order):
        indirect = 0
        remaining = successors
        while remaining:
            bit = remaining & -remaining
            indirect |= order[bit.bit_length() - 1]
            remaining ^= bit
        result.append(successors & ~indirect & ~context[lower])
    return tuple(result)


def _restrict(context: _Order, terms: _Terms) -> _Terms:
    collected: Counter[_Order] = Counter()
    for coefficient, order in terms:
        combined = _closure(tuple(a | b for a, b in zip(context, order)))
        if combined is not None:
            collected[_relative_cover(combined, context)] += coefficient
    return tuple(
        (coefficient, order)
        for order, coefficient in sorted(collected.items())
        if coefficient
    )


def _choose_comparison(terms: _Terms) -> tuple[int, int] | None:
    """Prefer a comparison occurring in many surviving region constraints."""
    frequency: Counter[tuple[int, int]] = Counter()
    for _, order in terms:
        for lower, successors in enumerate(order):
            while successors:
                bit = successors & -successors
                upper = bit.bit_length() - 1
                frequency[tuple(sorted((lower, upper)))] += 1
                successors ^= bit
    if not frequency:
        return None
    # Stable ties keep witnesses and diagnostic node counts reproducible.
    return min(frequency, key=lambda pair: (-frequency[pair], pair))


def _extend(context: _Order, lower: int, upper: int) -> _Order:
    rows = list(context)
    rows[lower] |= 1 << upper
    result = _closure(tuple(rows))
    if result is None:  # A selected comparison must be unresolved in context.
        raise AssertionError("attempted an inconsistent comparison branch")
    return result


def _witness(context: _Order) -> tuple[int, ...]:
    """One deterministic topological order, not a chamber enumeration."""
    remaining = set(range(len(context)))
    result: list[int] = []
    while remaining:
        available = min(
            label
            for label in remaining
            if not any(context[other] & (1 << label) for other in remaining)
        )
        result.append(available + 1)
        remaining.remove(available)
    return tuple(result)


def certify_region_identity(
    internal_count: int,
    source_bounds: Iterable[RegionBound],
    signed_bounds: Iterable[tuple[int, Iterable[RegionBound]]],
    *,
    max_nodes: int | None = None,
) -> RegionIdentityCertificate:
    """Prove an exact signed-domain identity or return a counterexample.

    All regions must specify one valid bound for each internal label, including
    regions with coefficient zero.  Coefficients must be Python integers
    (booleans and inexact values are rejected); they need not be just +/-1.
    Original cyclic regions are rejected, matching ``region_semantics``.
    A positive ``max_nodes`` caps search states and raises
    :class:`RegionIdentityLimitError` if exhausted.  An exception or resource
    failure is never a successful certificate.
    """
    count = _positive_integer(internal_count, "internal_count")
    if max_nodes is not None:
        _positive_integer(max_nodes, "max_nodes")
    terms: list[tuple[int, _Order]] = [(-1, _region_order(count, source_bounds))]
    for coefficient, bounds in signed_bounds:
        if isinstance(coefficient, bool) or not isinstance(coefficient, int):
            raise RegionSemanticsError("a region coefficient must be an integer")
        order = _region_order(count, bounds)
        if coefficient:
            terms.append((coefficient, order))

    # An explicit stack avoids a Python recursion-depth bound on order.
    pending: list[tuple[_Order, _Terms]] = [((0,) * count, tuple(terms))]
    explored_nodes = 0
    while pending:
        if max_nodes is not None and explored_nodes >= max_nodes:
            raise RegionIdentityLimitError(max_nodes, explored_nodes)
        context, current_terms = pending.pop()
        explored_nodes += 1
        current_terms = _restrict(context, current_terms)
        if not current_terms:
            continue
        comparison = _choose_comparison(current_terms)
        if comparison is None:
            residual = sum(coefficient for coefficient, _ in current_terms)
            if not residual:
                raise AssertionError("constant zero terms were not collected")
            return RegionIdentityCertificate(
                False, explored_nodes, _witness(context), residual
            )
        lower, upper = comparison
        # Every strict total order extending context takes exactly one branch.
        pending.append((_extend(context, upper, lower), current_terms))
        pending.append((_extend(context, lower, upper), current_terms))
    return RegionIdentityCertificate(True, explored_nodes)


__all__ = [
    "RegionIdentityCertificate",
    "RegionIdentityLimitError",
    "certify_region_identity",
]
