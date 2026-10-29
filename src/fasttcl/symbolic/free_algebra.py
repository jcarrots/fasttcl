"""Exact expansion in the free noncommutative operator algebra.

This module is deliberately small and independent of the publication
simplifier.  It is used at the source boundary to check workbook aggregation
without assuming commutativity, cyclicity, Hermiticity, or any bath-kernel
identity.  A polynomial is represented as an integer coefficient for each
ordered word of operator/density atoms.

For example, ``[1,2]p`` expands to the two words ``12p`` and ``21p`` with
coefficients ``+1`` and ``-1``.  The word order is never normalized.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, TypeAlias, cast

from .ast import (
    ASTNode,
    OrderedProductNode,
    ScalarNode,
    SumNode,
    apply_integer_coefficient,
    density_operator,
    make_ordered_product,
    make_sum,
    operator,
    scalar,
    validate_ast,
)


class FreeAlgebraError(ValueError):
    """Raised when an AST is outside the supported source algebra."""


Atom: TypeAlias = tuple[str, str]
Word: TypeAlias = tuple[Atom, ...]
Polynomial: TypeAlias = Counter[Word]


def _clean(polynomial: Polynomial) -> Polynomial:
    """Return a new polynomial with all zero coefficients removed."""

    return Counter(
        {
            word: coefficient
            for word, coefficient in polynomial.items()
            if coefficient
        }
    )


def add_polynomials(*polynomials: Polynomial) -> Polynomial:
    """Add exact integer polynomials without changing their word keys."""

    result: Polynomial = Counter()
    for polynomial in polynomials:
        result.update(polynomial)
    return _clean(result)


def scale_polynomial(
    polynomial: Polynomial, coefficient: int
) -> Polynomial:
    """Multiply every coefficient by an exact integer."""

    if isinstance(coefficient, bool) or not isinstance(coefficient, int):
        raise TypeError("coefficient must be an integer")
    return _clean(
        Counter(
            {
                word: coefficient * value
                for word, value in polynomial.items()
            }
        )
    )


def multiply_polynomials(
    left: Polynomial, right: Polynomial
) -> Polynomial:
    """Distribute an ordered product over two free polynomials."""

    result: Polynomial = Counter()
    for left_word, left_coefficient in left.items():
        for right_word, right_coefficient in right.items():
            result[left_word + right_word] += (
                left_coefficient * right_coefficient
            )
    return _clean(result)


def _product(polynomials: Iterable[Polynomial]) -> Polynomial:
    result: Polynomial = Counter({(): 1})
    for polynomial in polynomials:
        result = multiply_polynomials(result, polynomial)
    return result


def _expand(node: ASTNode, path: str) -> Polynomial:
    node_type = node["type"]

    if node_type == "scalar":
        return _clean(
            Counter({(): cast(ScalarNode, node)["value"]})
        )
    if node_type == "operator":
        return Counter(
            {
                (
                    ("operator", cast(dict[str, str], node)["label"]),
                ): 1
            }
        )
    if node_type == "density_operator":
        return Counter({(("density_operator", "rho"),): 1})
    if node_type == "sum":
        return add_polynomials(
            *(
                _expand(term, f"{path}.terms[{index}]")
                for index, term in enumerate(
                    cast(SumNode, node)["terms"]
                )
            )
        )
    if node_type == "ordered_product":
        return _product(
            _expand(factor, f"{path}.factors[{index}]")
            for index, factor in enumerate(
                cast(OrderedProductNode, node)["factors"]
            )
        )
    if node_type == "commutator":
        binary = cast(dict[str, ASTNode], node)
        left = _expand(binary["left"], f"{path}.left")
        right = _expand(binary["right"], f"{path}.right")
        return add_polynomials(
            multiply_polynomials(left, right),
            scale_polynomial(
                multiply_polynomials(right, left), -1
            ),
        )

    raise FreeAlgebraError(
        f"{path}: {node_type!r} is not part of the cumulative "
        "free-operator algebra"
    )


def expand_free_polynomial(expression: ASTNode) -> Polynomial:
    """Expand a cumulative AST into exact ordered words.

    Only integer scalars, sums, ordered products, commutators, system
    operators, and the density operator are accepted.  Rejecting Hadamard and
    kernel nodes here prevents a source-aggregation check from silently using
    downstream identities.
    """

    return _clean(_expand(validate_ast(expression), "$"))


def polynomial_rows(polynomial: Polynomial) -> list[dict[str, object]]:
    """Return a deterministic JSON-compatible representation."""

    rows: list[dict[str, object]] = []
    for word, coefficient in sorted(polynomial.items()):
        rows.append(
            {
                "coefficient": coefficient,
                "word": [
                    {"type": atom_type, "label": label}
                    for atom_type, label in word
                ],
            }
        )
    return rows


def polynomial_difference(
    left: Polynomial, right: Polynomial
) -> Polynomial:
    """Return ``left - right`` with exact zero removal."""

    return add_polynomials(left, scale_polynomial(right, -1))


def polynomial_to_ast(polynomial: Polynomial) -> ASTNode:
    """Encode a free polynomial as a deterministic sum of ordered words."""

    terms: list[ASTNode] = []
    for word, coefficient in sorted(_clean(polynomial).items()):
        factors: list[ASTNode] = []
        for atom_type, label in word:
            if atom_type == "operator":
                factors.append(operator(label))
            elif atom_type == "density_operator" and label == "rho":
                factors.append(density_operator())
            else:
                raise FreeAlgebraError(
                    f"unknown free-algebra atom {(atom_type, label)!r}"
                )
        monomial = make_ordered_product(factors)
        terms.append(
            apply_integer_coefficient(monomial, coefficient)
        )
    return make_sum(terms) if terms else scalar(0)


__all__ = [
    "Atom",
    "FreeAlgebraError",
    "Polynomial",
    "Word",
    "add_polynomials",
    "expand_free_polynomial",
    "multiply_polynomials",
    "polynomial_difference",
    "polynomial_rows",
    "polynomial_to_ast",
    "scale_polynomial",
]
