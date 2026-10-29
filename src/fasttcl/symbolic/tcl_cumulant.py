"""Exact ordered-cumulant source generation for even TCL orders.

The publication TCL6 pipeline starts from a compact, manually reviewed
cumulative-correlation inventory.  This module supplies the missing level-3
boundary: it constructs the ordered cumulant directly from projected moments,
expands every Liouvillian branch, Wick contracts each Gaussian bath moment,
and collects the result in an exact noncommutative superoperator basis.

No time integration or Hadamard identity is used here.  Coefficients are
ordinary Python integers and a monomial keeps all information that may not be
commuted away: oriented bath pairs, the ordered system word to the left of
``rho``, and the ordered system word to its right.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
from math import factorial
from typing import Iterable, Iterator, TypeAlias

from .ast import ASTNode
from .free_algebra import Polynomial, polynomial_to_ast


Pair: TypeAlias = tuple[int, int]
Pairing: TypeAlias = tuple[Pair, ...]
OperatorWord: TypeAlias = tuple[int, ...]


@dataclass(frozen=True, order=True)
class SuperoperatorMonomial:
    """One exact term ``correlations * left * rho * right``."""

    correlations: Pairing
    left: OperatorWord
    right: OperatorWord


SuperoperatorPolynomial: TypeAlias = Counter[SuperoperatorMonomial]


@dataclass(frozen=True)
class CumulantCensus:
    """Combinatorial counts before Liouvillian-branch cancellation."""

    order: int
    composition_count: int
    projected_structure_count: int
    wick_instance_count: int


@dataclass(frozen=True)
class CumulativeGroup:
    """One oriented-correlation group in the outer-wrapper source."""

    correlations: Pairing
    polynomial: Polynomial

    @property
    def expression(self) -> ASTNode:
        return polynomial_to_ast(self.polynomial)


def _validate_even_order(order: int) -> None:
    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError("order must be an integer")
    if order < 2 or order % 2:
        raise ValueError("order must be a positive even integer")


@lru_cache(maxsize=None)
def even_compositions(order: int) -> tuple[tuple[int, ...], ...]:
    """Return all ordered compositions of ``order`` into even blocks."""

    _validate_even_order(order)
    if order == 0:
        return ((),)
    result: list[tuple[int, ...]] = []
    for first in range(2, order + 1, 2):
        if first == order:
            result.append((first,))
        else:
            result.extend(
                (first,) + tail
                for tail in even_compositions(order - first)
            )
    return tuple(result)


@lru_cache(maxsize=None)
def pairings(labels: tuple[int, ...]) -> tuple[Pairing, ...]:
    """Return every perfect matching, retaining pair orientation by position."""

    if not labels:
        return ((),)
    if len(labels) % 2:
        return ()
    first = labels[0]
    result: list[Pairing] = []
    for partner_index in range(1, len(labels)):
        partner = labels[partner_index]
        remainder = labels[1:partner_index] + labels[partner_index + 1 :]
        for tail in pairings(remainder):
            result.append(((first, partner),) + tail)
    return tuple(result)


def _block_assignments(
    order: int, composition: tuple[int, ...]
) -> Iterator[tuple[tuple[int, ...], ...]]:
    """Shuffle moment blocks onto the common ordered simplex.

    Label zero is the differentiated/current-time Liouvillian and therefore
    belongs to the first block.  Labels inside every block retain their global
    time order.
    """

    remaining = tuple(range(1, order))

    def assign(
        block_index: int,
        available: tuple[int, ...],
        prefix: tuple[tuple[int, ...], ...],
    ) -> Iterator[tuple[tuple[int, ...], ...]]:
        if block_index == len(composition):
            if not available:
                yield prefix
            return
        requested = composition[block_index] - (1 if block_index == 0 else 0)
        for chosen in combinations(available, requested):
            chosen_set = frozenset(chosen)
            rest = tuple(value for value in available if value not in chosen_set)
            block = ((0,) + chosen) if block_index == 0 else chosen
            yield from assign(block_index + 1, rest, prefix + (block,))

    yield from assign(0, remaining, ())


def _structure_count(order: int, composition: tuple[int, ...]) -> int:
    numerator = factorial(order - 1)
    denominator = factorial(composition[0] - 1)
    for block_size in composition[1:]:
        denominator *= factorial(block_size)
    return numerator // denominator


def _pairing_count(size: int) -> int:
    if size == 0:
        return 1
    result = 1
    for value in range(1, size, 2):
        result *= value
    return result


def cumulant_census(order: int) -> CumulantCensus:
    """Return independent projected-structure and Wick-instance counts."""

    _validate_even_order(order)
    structures = 0
    wick_instances = 0
    compositions = even_compositions(order)
    for composition in compositions:
        count = _structure_count(order, composition)
        structures += count
        wick_per_structure = 1
        for block_size in composition:
            wick_per_structure *= _pairing_count(block_size)
        wick_instances += count * wick_per_structure
    return CumulantCensus(
        order=order,
        composition_count=len(compositions),
        projected_structure_count=structures,
        wick_instance_count=wick_instances,
    )


@dataclass(frozen=True)
class _BlockTerm:
    coefficient: int
    left: OperatorWord
    right: OperatorWord
    correlations: Pairing


@lru_cache(maxsize=None)
def _block_terms(block: tuple[int, ...]) -> tuple[_BlockTerm, ...]:
    result: list[_BlockTerm] = []
    for branches in product(("L", "R"), repeat=len(block)):
        left = tuple(
            label for label, branch in zip(block, branches, strict=True)
            if branch == "L"
        )
        right = tuple(
            label for label, branch in reversed(tuple(zip(block, branches, strict=True)))
            if branch == "R"
        )
        # Tr_B(B_left rho_B B_right) = <B_right B_left>.
        contour_word = right + left
        branch_coefficient = -1 if branches.count("R") % 2 else 1
        for position_pairing in pairings(tuple(range(len(block)))):
            correlations = tuple(
                sorted(
                    (contour_word[first], contour_word[second])
                    for first, second in position_pairing
                )
            )
            result.append(
                _BlockTerm(
                    coefficient=branch_coefficient,
                    left=left,
                    right=right,
                    correlations=correlations,
                )
            )
    return tuple(result)


def ordered_cumulant(order: int) -> SuperoperatorPolynomial:
    """Construct the full physical TCL ``order`` generator integrand.

    The expansion uses ``L X = -i[V,X]``.  At even order the common phase is
    ``(-i)**order = (-1)**(order/2)``; right actions supply the remaining
    branch signs.  Products of projected moments are composed in their exact
    noncommutative order.
    """

    _validate_even_order(order)
    result: SuperoperatorPolynomial = Counter()
    physical_phase = -1 if (order // 2) % 2 else 1

    for composition in even_compositions(order):
        cumulant_sign = -1 if (len(composition) - 1) % 2 else 1
        common_coefficient = physical_phase * cumulant_sign
        for blocks in _block_assignments(order, composition):
            block_expansions = tuple(_block_terms(block) for block in blocks)
            for selected in product(*block_expansions):
                coefficient = common_coefficient
                left: OperatorWord = ()
                right: OperatorWord = ()
                correlations: Pairing = ()
                for term in selected:
                    coefficient *= term.coefficient
                    left += term.left
                    # S1(S2(X)) has right word R2 R1.
                    right = term.right + right
                    correlations = tuple(sorted(correlations + term.correlations))
                monomial = SuperoperatorMonomial(correlations, left, right)
                result[monomial] += coefficient

    return Counter({term: value for term, value in result.items() if value})


def undirected_pairing(correlations: Iterable[Pair]) -> Pairing:
    """Canonicalize pair orientation without losing the matching."""

    return tuple(sorted(tuple(sorted(pair)) for pair in correlations))


def is_interval_connected(correlations: Iterable[Pair], order: int) -> bool:
    """Whether the matching crosses every strict cut of ``0..order-1``."""

    matching = undirected_pairing(correlations)
    endpoints = sorted(value for pair in matching for value in pair)
    if endpoints != list(range(order)):
        raise ValueError("correlations must be a perfect matching of the labels")
    return all(
        any(left <= cut < right for left, right in matching)
        for cut in range(order - 1)
    )


def prewrapper_polynomial(order: int) -> SuperoperatorPolynomial:
    """Extract ``S`` from ``K=-[A_0,S]+H.c.`` exactly.

    The selected half has an oriented bath pair beginning at zero and a left
    system word beginning with ``A_0``.  Removing that operator and reversing
    the outer minus yields the source consumed by the TCL6 HR wrapper.
    """

    full = ordered_cumulant(order)
    result: SuperoperatorPolynomial = Counter()
    for monomial, coefficient in full.items():
        zero_pairs = tuple(
            pair for pair in monomial.correlations if 0 in pair
        )
        if len(zero_pairs) != 1 or zero_pairs[0][0] != 0:
            continue
        if not monomial.left or monomial.left[0] != 0:
            continue
        if 0 in monomial.left[1:] or 0 in monomial.right:
            raise AssertionError("current-time operator occurs more than once")
        source = SuperoperatorMonomial(
            correlations=monomial.correlations,
            left=monomial.left[1:],
            right=monomial.right,
        )
        result[source] -= coefficient
    return Counter({term: value for term, value in result.items() if value})


def realigned_adjoint_polynomial(
    polynomial: SuperoperatorPolynomial,
) -> SuperoperatorPolynomial:
    """Apply the algebraic ``sharp`` operation used for the final H.c."""

    result: SuperoperatorPolynomial = Counter()
    for monomial, coefficient in polynomial.items():
        adjoint = SuperoperatorMonomial(
            correlations=tuple(
                sorted((right, left) for left, right in monomial.correlations)
            ),
            left=tuple(reversed(monomial.right)),
            right=tuple(reversed(monomial.left)),
        )
        result[adjoint] += coefficient
    return Counter({term: value for term, value in result.items() if value})


def reconstruct_outer_wrapper(
    source: SuperoperatorPolynomial,
) -> SuperoperatorPolynomial:
    """Return the exact polynomial ``-[A_0,S]+H.c.`` for ``source=S``."""

    raw: SuperoperatorPolynomial = Counter()
    for monomial, coefficient in source.items():
        raw[
            SuperoperatorMonomial(
                monomial.correlations,
                (0,) + monomial.left,
                monomial.right,
            )
        ] -= coefficient
        raw[
            SuperoperatorMonomial(
                monomial.correlations,
                monomial.left,
                monomial.right + (0,),
            )
        ] += coefficient
    raw = Counter({term: value for term, value in raw.items() if value})
    result = Counter(raw)
    result.update(realigned_adjoint_polynomial(raw))
    return Counter({term: value for term, value in result.items() if value})


def cumulative_groups(order: int) -> tuple[CumulativeGroup, ...]:
    """Collect the pre-wrapper source by oriented correlation product."""

    grouped: dict[Pairing, Polynomial] = {}
    for monomial, coefficient in prewrapper_polynomial(order).items():
        word = (
            tuple(("operator", str(label)) for label in monomial.left)
            + (("density_operator", "rho"),)
            + tuple(("operator", str(label)) for label in monomial.right)
        )
        polynomial = grouped.setdefault(monomial.correlations, Counter())
        polynomial[word] += coefficient

    result: list[CumulativeGroup] = []
    for correlations, polynomial in sorted(grouped.items()):
        cleaned = Counter(
            {word: value for word, value in polynomial.items() if value}
        )
        if cleaned:
            result.append(CumulativeGroup(correlations, cleaned))
    return tuple(result)


__all__ = [
    "CumulantCensus",
    "CumulativeGroup",
    "OperatorWord",
    "Pair",
    "Pairing",
    "SuperoperatorMonomial",
    "SuperoperatorPolynomial",
    "cumulant_census",
    "cumulative_groups",
    "even_compositions",
    "is_interval_connected",
    "ordered_cumulant",
    "pairings",
    "prewrapper_polynomial",
    "realigned_adjoint_polynomial",
    "reconstruct_outer_wrapper",
    "undirected_pairing",
]
