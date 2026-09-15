"""CPU runtime extracted from the canonical TCL6 Numerical source; see data/source-provenance.json."""

from __future__ import annotations

from collections import Counter

from typing import Any, TypeAlias, cast

from .ast import ASTNode, validate_ast

class FormalBilinearError(ValueError):
    """Raised when a post-H2 AST cannot be represented exactly."""


FormalAtom: TypeAlias = tuple[Any, ...]


FormalWord: TypeAlias = tuple[FormalAtom, ...]


FormalPolynomial: TypeAlias = Counter[FormalWord]


def _clean(polynomial: FormalPolynomial) -> FormalPolynomial:
    return Counter(
        {
            word: coefficient
            for word, coefficient in polynomial.items()
            if coefficient
        }
    )


def formal_add(
    *polynomials: FormalPolynomial,
) -> FormalPolynomial:
    result: FormalPolynomial = Counter()
    for polynomial in polynomials:
        result.update(polynomial)
    return _clean(result)


def formal_scale(
    polynomial: FormalPolynomial, coefficient: int
) -> FormalPolynomial:
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


def formal_multiply(
    left: FormalPolynomial,
    right: FormalPolynomial,
) -> FormalPolynomial:
    result: FormalPolynomial = Counter()
    for left_word, left_value in left.items():
        for right_word, right_value in right.items():
            result[left_word + right_word] += left_value * right_value
    return _clean(result)


def _time_form(node: ASTNode) -> tuple[tuple[str, int], ...]:
    if node["type"] != "time_expression":
        raise FormalBilinearError(
            "a kernel argument must be a time_expression"
        )
    coefficients: dict[str, int] = {}
    for term in cast(dict[str, Any], node)["terms"]:
        symbol = cast(str, term["symbol"])
        coefficient = cast(int, term["coefficient"])
        if symbol == "0":
            continue
        coefficients[symbol] = (
            coefficients.get(symbol, 0) + coefficient
        )
    return tuple(
        (symbol, coefficient)
        for symbol, coefficient in sorted(coefficients.items())
        if coefficient
    )


def _gamma_atom(
    *, transpose: bool, argument: tuple[tuple[str, int], ...]
) -> FormalPolynomial:
    # The timed spectral density is an integral from zero, so Gamma(0)=0.
    if not argument:
        return Counter()
    return Counter(
        {(("kernel", "Gamma", transpose, argument),): 1}
    )


def _expand(node: ASTNode, path: str) -> FormalPolynomial:
    node_type = node["type"]
    current = cast(dict[str, Any], node)

    if node_type == "scalar":
        return _clean(Counter({(): cast(int, current["value"])}))
    if node_type == "operator":
        return Counter(
            {(("operator", cast(str, current["label"])),): 1}
        )
    if node_type == "density_operator":
        return Counter({(("density_operator", "rho"),): 1})
    if node_type == "time_expression":
        return Counter({(("time_expression", _time_form(node)),): 1})
    if node_type == "kernel":
        transpose = cast(bool, current["transpose"])
        arguments = tuple(
            _time_form(cast(ASTNode, argument))
            for argument in current["arguments"]
        )
        if current["family"] == "Gamma":
            if len(arguments) != 1:
                raise FormalBilinearError(
                    f"{path}: Gamma must have one argument"
                )
            return _gamma_atom(
                transpose=transpose, argument=arguments[0]
            )
        if current["family"] == "DeltaGamma":
            if len(arguments) != 2:
                raise FormalBilinearError(
                    f"{path}: DeltaGamma must have two arguments"
                )
            return formal_add(
                _gamma_atom(
                    transpose=transpose, argument=arguments[0]
                ),
                formal_scale(
                    _gamma_atom(
                        transpose=transpose,
                        argument=arguments[1],
                    ),
                    -1,
                ),
            )
        raise FormalBilinearError(
            f"{path}: unknown kernel family {current['family']!r}"
        )
    if node_type == "sum":
        return formal_add(
            *(
                _expand(cast(ASTNode, term), f"{path}.terms[{index}]")
                for index, term in enumerate(current["terms"])
            )
        )
    if node_type == "ordered_product":
        result: FormalPolynomial = Counter({(): 1})
        for index, factor in enumerate(current["factors"]):
            result = formal_multiply(
                result,
                _expand(
                    cast(ASTNode, factor),
                    f"{path}.factors[{index}]",
                ),
            )
        return result
    if node_type == "commutator":
        left = _expand(cast(ASTNode, current["left"]), f"{path}.left")
        right = _expand(
            cast(ASTNode, current["right"]), f"{path}.right"
        )
        return formal_add(
            formal_multiply(left, right),
            formal_scale(formal_multiply(right, left), -1),
        )
    if node_type == "hadamard":
        left = _expand(cast(ASTNode, current["left"]), f"{path}.left")
        right = _expand(
            cast(ASTNode, current["right"]), f"{path}.right"
        )
        result: FormalPolynomial = Counter()
        for left_word, left_value in left.items():
            for right_word, right_value in right.items():
                result[(("hadamard", left_word, right_word),)] += (
                    left_value * right_value
                )
        return _clean(result)
    raise FormalBilinearError(
        f"{path}: unsupported AST node {node_type!r}"
    )


def expand_formal_bilinear(
    expression: ASTNode,
    *,
    coefficient: int = 1,
) -> FormalPolynomial:
    """Return the exact formal polynomial for one post-H2 expression."""

    return formal_scale(
        _expand(validate_ast(expression), "$"), coefficient
    )

