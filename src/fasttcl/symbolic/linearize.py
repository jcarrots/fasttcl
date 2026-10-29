"""Density-linear decomposition and exact collection of TCL expressions.

The Hadamard-reduced form is linear in the density operator.  This module
turns that mathematical property into a structural algorithm:

* additions are decomposed only along the unique density-bearing path;
* products, commutators, and Hadamard products reject density dependence in
  more than one child;
* integer scalars are hoisted without reordering noncommutative factors;
* monic ASTs are collected globally while retaining every contributor.

All transformations are pure.  Input AST dictionaries are validated but are
never modified in place.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Iterable, cast

from .ast import (
    ASTNode,
    OrderedProductNode,
    ScalarNode,
    SumNode,
    apply_integer_coefficient,
    commutator,
    hadamard,
    make_ordered_product,
    make_sum,
    scalar,
    validate_ast,
)


class DensityLinearityError(ValueError):
    """Raised when an expression is not structurally linear in ``rho``."""


@dataclass(frozen=True)
class DensityPiece:
    """One additive density-linear piece and its deterministic AST path."""

    expression: ASTNode
    path: tuple[str, ...]


@dataclass(frozen=True, order=True)
class ContributorLineage:
    """The signed contribution made by one source branch to a monic AST."""

    source_id: str
    branch_path: tuple[str, ...]
    coefficient: int


@dataclass(frozen=True)
class LinearContribution:
    """A monic expression with a signed coefficient and complete lineage."""

    coefficient: int
    expression: ASTNode
    lineage: tuple[ContributorLineage, ...]


@dataclass(frozen=True)
class CollectedGroup:
    """All contributions sharing one canonical monic expression."""

    key: str
    coefficient: int
    expression: ASTNode
    contributors: tuple[ContributorLineage, ...]

    @property
    def is_zero(self) -> bool:
        return self.coefficient == 0


@dataclass(frozen=True)
class CollectionResult:
    """Deterministic global collection result."""

    groups: tuple[CollectedGroup, ...]
    input_contribution_count: int

    @property
    def survivors(self) -> tuple[CollectedGroup, ...]:
        return tuple(group for group in self.groups if not group.is_zero)

    @property
    def canceled_groups(self) -> tuple[CollectedGroup, ...]:
        return tuple(group for group in self.groups if group.is_zero)

    @property
    def canceled_contribution_count(self) -> int:
        return sum(
            len(group.contributors) for group in self.canceled_groups
        )


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _density_count(node: ASTNode) -> int:
    node_type = node["type"]
    if node_type == "density_operator":
        return 1
    if node_type == "sum":
        return sum(
            _density_count(term)
            for term in cast(SumNode, node)["terms"]
        )
    if node_type == "ordered_product":
        return sum(
            _density_count(factor)
            for factor in cast(OrderedProductNode, node)["factors"]
        )
    if node_type in {"commutator", "hadamard"}:
        binary = cast(dict[str, ASTNode], node)
        return _density_count(binary["left"]) + _density_count(
            binary["right"]
        )
    return 0


def density_count(expression: ASTNode) -> int:
    """Return the number of structural density-operator occurrences."""

    return _density_count(validate_ast(expression))


def _linearity_failure(
    path: tuple[str, ...], message: str
) -> DensityLinearityError:
    location = "$" if not path else "$." + ".".join(path)
    return DensityLinearityError(f"{location}: {message}")


def _decompose(
    node: ASTNode, path: tuple[str, ...]
) -> list[DensityPiece]:
    node_type = node["type"]

    if node_type == "density_operator":
        return [DensityPiece(node, path)]

    if node_type == "sum":
        pieces: list[DensityPiece] = []
        for index, term in enumerate(cast(SumNode, node)["terms"]):
            term_path = path + (f"terms[{index}]",)
            if _density_count(term) == 0:
                raise _linearity_failure(
                    term_path,
                    "additive term does not contain the density operator",
                )
            pieces.extend(_decompose(term, term_path))
        return pieces

    if node_type == "ordered_product":
        product = cast(OrderedProductNode, node)
        dependent = [
            index
            for index, factor in enumerate(product["factors"])
            if _density_count(factor) != 0
        ]
        if not dependent:
            raise _linearity_failure(
                path, "expression does not contain the density operator"
            )
        if len(dependent) != 1:
            raise _linearity_failure(
                path,
                "ordered product has multiple density-bearing factors",
            )
        dependent_index = dependent[0]
        child_path = path + (f"factors[{dependent_index}]",)
        pieces = _decompose(
            product["factors"][dependent_index], child_path
        )
        return [
            DensityPiece(
                make_ordered_product(
                    product["factors"][:dependent_index]
                    + [piece.expression]
                    + product["factors"][dependent_index + 1 :]
                ),
                piece.path,
            )
            for piece in pieces
        ]

    if node_type in {"commutator", "hadamard"}:
        binary = cast(dict[str, ASTNode], node)
        dependent = [
            side
            for side in ("left", "right")
            if _density_count(binary[side]) != 0
        ]
        if not dependent:
            raise _linearity_failure(
                path, "expression does not contain the density operator"
            )
        if len(dependent) != 1:
            raise _linearity_failure(
                path,
                f"{node_type} has density dependence on both sides",
            )
        side = dependent[0]
        pieces = _decompose(binary[side], path + (side,))
        constructor = commutator if node_type == "commutator" else hadamard
        if side == "left":
            return [
                DensityPiece(
                    constructor(piece.expression, binary["right"]),
                    piece.path,
                )
                for piece in pieces
            ]
        return [
            DensityPiece(
                constructor(binary["left"], piece.expression),
                piece.path,
            )
            for piece in pieces
        ]

    raise _linearity_failure(
        path, "expression does not contain the density operator"
    )


def decompose_density(expression: ASTNode) -> tuple[DensityPiece, ...]:
    """Split an expression into additive pieces linear in ``rho``.

    A product-like node is linear only when exactly one of its children
    depends on the density operator.  Sums may contain multiple
    density-bearing terms because they are returned as separate pieces.
    Every returned piece contains exactly one density operator.
    """

    validated = validate_ast(expression)
    pieces = tuple(_decompose(validated, ()))
    for piece in pieces:
        count = _density_count(piece.expression)
        if count != 1:  # Defensive assertion around the recursive invariant.
            raise DensityLinearityError(
                "internal density decomposition error: "
                f"piece at {piece.path!r} contains {count} rho nodes"
            )
        validate_ast(piece.expression)
    return pieces


def _is_scalar_one(node: ASTNode) -> bool:
    return (
        node["type"] == "scalar"
        and cast(ScalarNode, node)["value"] == 1
    )


def _hoist_scalar(node: ASTNode) -> tuple[int, ASTNode]:
    node_type = node["type"]

    if node_type == "scalar":
        return cast(ScalarNode, node)["value"], scalar(1)

    if node_type == "ordered_product":
        coefficient = 1
        monic_factors: list[ASTNode] = []
        for factor in cast(OrderedProductNode, node)["factors"]:
            factor_coefficient, monic_factor = _hoist_scalar(factor)
            coefficient *= factor_coefficient
            if not _is_scalar_one(monic_factor):
                monic_factors.append(monic_factor)
        return coefficient, make_ordered_product(monic_factors)

    if node_type in {"commutator", "hadamard"}:
        binary = cast(dict[str, ASTNode], node)
        left_coefficient, monic_left = _hoist_scalar(binary["left"])
        right_coefficient, monic_right = _hoist_scalar(binary["right"])
        constructor = commutator if node_type == "commutator" else hadamard
        return (
            left_coefficient * right_coefficient,
            constructor(monic_left, monic_right),
        )

    if node_type == "sum":
        normalized_terms = [
            _hoist_scalar(term)
            for term in cast(SumNode, node)["terms"]
        ]
        nonzero_coefficients = [
            coefficient
            for coefficient, _ in normalized_terms
            if coefficient != 0
        ]
        if not nonzero_coefficients:
            return 0, scalar(1)

        common_magnitude = 0
        for coefficient in nonzero_coefficients:
            common_magnitude = math.gcd(
                common_magnitude, abs(coefficient)
            )
        first_sign = 1 if nonzero_coefficients[0] > 0 else -1
        common = first_sign * common_magnitude
        residual_terms = [
            apply_integer_coefficient(monic, coefficient // common)
            for coefficient, monic in normalized_terms
        ]
        return common, make_sum(residual_terms)

    # Kernel argument coefficients belong to affine time expressions, not to
    # the surrounding operator algebra, so leaf-like nodes stay untouched.
    return 1, node


def hoist_integer_scalar(expression: ASTNode) -> tuple[int, ASTNode]:
    """Return ``(coefficient, monic_expression)``.

    Scalars are moved through ordered products and through the bilinear
    commutator and Hadamard constructors.  Factor order, operand order, and
    kernel argument order are never changed.  A common integer factor is also
    removed from sums without reordering their terms.
    """

    validated = validate_ast(expression)
    coefficient, monic = _hoist_scalar(validated)
    validate_ast(monic)
    return coefficient, monic


def canonical_monic_key(expression: ASTNode) -> str:
    """Return canonical JSON for an expression with its overall scalar removed."""

    _, monic = hoist_integer_scalar(expression)
    return json.dumps(
        monic,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def linearize_expression(
    expression: ASTNode,
    *,
    coefficient: int = 1,
    source_id: str,
) -> tuple[LinearContribution, ...]:
    """Decompose and normalize one signed source expression."""

    if not _is_integer(coefficient):
        raise TypeError("coefficient must be an integer")
    if not isinstance(source_id, str) or not source_id:
        raise TypeError("source_id must be a non-empty string")

    contributions: list[LinearContribution] = []
    for piece in decompose_density(expression):
        local_coefficient, monic = hoist_integer_scalar(piece.expression)
        if _density_count(monic) != 1:
            raise DensityLinearityError(
                "scalar normalization did not preserve exactly one rho"
            )
        signed_coefficient = coefficient * local_coefficient
        lineage = ContributorLineage(
            source_id=source_id,
            branch_path=piece.path,
            coefficient=signed_coefficient,
        )
        contributions.append(
            LinearContribution(
                coefficient=signed_coefficient,
                expression=monic,
                lineage=(lineage,),
            )
        )
    return tuple(contributions)


def _normalized_contribution(
    contribution: LinearContribution,
) -> LinearContribution:
    if not _is_integer(contribution.coefficient):
        raise TypeError("contribution coefficient must be an integer")
    if not contribution.lineage:
        raise ValueError("contribution lineage must not be empty")
    if any(
        not _is_integer(lineage.coefficient)
        for lineage in contribution.lineage
    ):
        raise TypeError("lineage coefficients must be integers")
    if (
        sum(lineage.coefficient for lineage in contribution.lineage)
        != contribution.coefficient
    ):
        raise ValueError(
            "contribution coefficient must equal the sum of lineage "
            "coefficients"
        )

    embedded_coefficient, monic = hoist_integer_scalar(
        contribution.expression
    )
    if _density_count(monic) != 1:
        raise DensityLinearityError(
            "collected contribution must contain exactly one rho"
        )
    scaled_lineage = tuple(
        ContributorLineage(
            source_id=lineage.source_id,
            branch_path=lineage.branch_path,
            coefficient=lineage.coefficient * embedded_coefficient,
        )
        for lineage in contribution.lineage
    )
    return LinearContribution(
        coefficient=contribution.coefficient * embedded_coefficient,
        expression=monic,
        lineage=scaled_lineage,
    )


def collect_contributions(
    contributions: Iterable[LinearContribution],
) -> CollectionResult:
    """Collect identical monic ASTs globally and retain complete lineage."""

    grouped: dict[str, dict[str, object]] = {}
    input_count = 0
    for contribution in contributions:
        normalized = _normalized_contribution(contribution)
        input_count += 1
        key = canonical_monic_key(normalized.expression)
        if key not in grouped:
            grouped[key] = {
                "coefficient": 0,
                "expression": normalized.expression,
                "contributors": [],
            }
        group = grouped[key]
        group["coefficient"] = cast(int, group["coefficient"]) + (
            normalized.coefficient
        )
        cast(
            list[ContributorLineage], group["contributors"]
        ).extend(normalized.lineage)

    groups: list[CollectedGroup] = []
    for key in sorted(grouped):
        group = grouped[key]
        contributors = tuple(
            sorted(
                cast(
                    list[ContributorLineage], group["contributors"]
                )
            )
        )
        coefficient = cast(int, group["coefficient"])
        if coefficient != sum(
            contributor.coefficient for contributor in contributors
        ):
            raise AssertionError("collected lineage coefficient mismatch")
        groups.append(
            CollectedGroup(
                key=key,
                coefficient=coefficient,
                expression=cast(ASTNode, group["expression"]),
                contributors=contributors,
            )
        )

    return CollectionResult(
        groups=tuple(groups),
        input_contribution_count=input_count,
    )


__all__ = [
    "CollectedGroup",
    "CollectionResult",
    "ContributorLineage",
    "DensityLinearityError",
    "DensityPiece",
    "LinearContribution",
    "canonical_monic_key",
    "collect_contributions",
    "decompose_density",
    "density_count",
    "hoist_integer_scalar",
    "linearize_expression",
]
