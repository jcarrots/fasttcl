"""JSON-compatible AST nodes for the TCL6 CC and HR shorthand.

The public constructors in this module always return ordinary ``dict`` and
``list`` values.  The ``TypedDict`` declarations are for static type checkers;
they do not introduce custom runtime objects that would complicate JSON
serialization.
"""

from __future__ import annotations

from typing import Any, Literal, TypeAlias, TypedDict, cast

from .tcl_order import TCLOrder, is_time_symbol


class ScalarNode(TypedDict):
    type: Literal["scalar"]
    value: int


class OperatorNode(TypedDict):
    type: Literal["operator"]
    label: str


class DensityOperatorNode(TypedDict):
    type: Literal["density_operator"]
    symbol: Literal["rho"]


class TimeTerm(TypedDict):
    symbol: str
    coefficient: int


class TimeExpressionNode(TypedDict):
    type: Literal["time_expression"]
    terms: list[TimeTerm]


class KernelNode(TypedDict):
    type: Literal["kernel"]
    family: Literal["Gamma", "DeltaGamma"]
    transpose: bool
    arguments: list[TimeExpressionNode]


class SumNode(TypedDict):
    type: Literal["sum"]
    terms: list["ASTNode"]


class OrderedProductNode(TypedDict):
    type: Literal["ordered_product"]
    factors: list["ASTNode"]


class CommutatorNode(TypedDict):
    type: Literal["commutator"]
    left: "ASTNode"
    right: "ASTNode"


class HadamardNode(TypedDict):
    type: Literal["hadamard"]
    left: "ASTNode"
    right: "ASTNode"


ASTNode: TypeAlias = (
    SumNode
    | OrderedProductNode
    | CommutatorNode
    | OperatorNode
    | DensityOperatorNode
    | HadamardNode
    | KernelNode
    | TimeExpressionNode
    | ScalarNode
)

class ASTValidationError(ValueError):
    """Raised when a value is not a valid even-order TCL AST."""


def scalar(value: int) -> ScalarNode:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("scalar value must be an integer")
    return {"type": "scalar", "value": value}


def operator(label: str) -> OperatorNode:
    if not isinstance(label, str) or not label:
        raise TypeError("operator label must be a non-empty string")
    return {"type": "operator", "label": label}


def density_operator() -> DensityOperatorNode:
    return {"type": "density_operator", "symbol": "rho"}


def time_expression(terms: list[TimeTerm]) -> TimeExpressionNode:
    return {
        "type": "time_expression",
        "terms": [
            {"symbol": term["symbol"], "coefficient": term["coefficient"]}
            for term in terms
        ],
    }


def kernel(
    arguments: list[TimeExpressionNode],
    *,
    transpose: bool = False,
    family: Literal["Gamma", "DeltaGamma"] = "Gamma",
) -> KernelNode:
    return {
        "type": "kernel",
        "family": family,
        "transpose": transpose,
        "arguments": list(arguments),
    }


def commutator(left: ASTNode, right: ASTNode) -> CommutatorNode:
    return {"type": "commutator", "left": left, "right": right}


def hadamard(left: ASTNode, right: ASTNode) -> HadamardNode:
    return {"type": "hadamard", "left": left, "right": right}


def make_sum(nodes: list[ASTNode]) -> ASTNode:
    """Build a normalized sum, flattening sums and recognizing DeltaGamma."""

    terms: list[ASTNode] = []
    for node in nodes:
        if node["type"] == "sum":
            terms.extend(cast(SumNode, node)["terms"])
        else:
            terms.append(node)
    if not terms:
        return scalar(0)
    if len(terms) == 1:
        return terms[0]
    if len(terms) == 2 and terms[0]["type"] == "kernel":
        first = cast(KernelNode, terms[0])
        second = terms[1]
        if second["type"] == "ordered_product":
            second_factors = cast(OrderedProductNode, second)["factors"]
            if (
                len(second_factors) == 2
                and second_factors[0] == scalar(-1)
                and second_factors[1]["type"] == "kernel"
            ):
                second_kernel = cast(KernelNode, second_factors[1])
                if (
                    first["family"] == "Gamma"
                    and second_kernel["family"] == "Gamma"
                    and first["transpose"] == second_kernel["transpose"]
                    and len(first["arguments"]) == 1
                    and len(second_kernel["arguments"]) == 1
                ):
                    return kernel(
                        [first["arguments"][0], second_kernel["arguments"][0]],
                        transpose=first["transpose"],
                        family="DeltaGamma",
                    )
    return {"type": "sum", "terms": terms}


def make_ordered_product(nodes: list[ASTNode]) -> ASTNode:
    """Build a product while retaining the exact noncommutative factor order."""

    factors: list[ASTNode] = []
    for node in nodes:
        if node["type"] == "ordered_product":
            factors.extend(cast(OrderedProductNode, node)["factors"])
        else:
            factors.append(node)
    if not factors:
        return scalar(1)
    if len(factors) == 1:
        return factors[0]
    return {"type": "ordered_product", "factors": factors}


def apply_integer_coefficient(node: ASTNode, coefficient: int) -> ASTNode:
    """Prefix ``node`` with an integer scalar without reordering its factors."""

    if coefficient == 1:
        return node
    if node["type"] == "scalar":
        return scalar(cast(ScalarNode, node)["value"] * coefficient)
    if node["type"] == "ordered_product":
        factors = list(cast(OrderedProductNode, node)["factors"])
        if factors and factors[0]["type"] == "scalar":
            first = cast(ScalarNode, factors[0])
            factors[0] = scalar(first["value"] * coefficient)
            return make_ordered_product(factors)
    return make_ordered_product([scalar(coefficient), node])


_EXPECTED_KEYS: dict[str, frozenset[str]] = {
    "sum": frozenset({"type", "terms"}),
    "ordered_product": frozenset({"type", "factors"}),
    "commutator": frozenset({"type", "left", "right"}),
    "operator": frozenset({"type", "label"}),
    "density_operator": frozenset({"type", "symbol"}),
    "hadamard": frozenset({"type", "left", "right"}),
    "kernel": frozenset(
        {"type", "family", "transpose", "arguments"}
    ),
    "time_expression": frozenset({"type", "terms"}),
    "scalar": frozenset({"type", "value"}),
}


def validate_ast(value: Any, *, order: int | TCLOrder | None = None) -> ASTNode:
    """Validate a plain JSON value and return it with an ``ASTNode`` type.

    Validation is deliberately strict: unknown fields, empty variadic nodes,
    booleans used as integers, and object cycles are all rejected.  Supplying
    ``order`` additionally requires canonical reduced operator and time labels
    belonging to that order.  Omit it when validating unreduced numeric labels.
    """

    coordinates = order if isinstance(order, TCLOrder) else (
        TCLOrder(order) if order is not None else None
    )
    operator_labels = None if coordinates is None else coordinates.operator_slots
    time_symbols = None if coordinates is None else coordinates.time_to_gap
    active: set[int] = set()

    def fail(path: str, message: str) -> None:
        raise ASTValidationError(f"{path}: {message}")

    def visit(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            fail(path, "node must be an object")
        node_id = id(node)
        if node_id in active:
            fail(path, "AST contains an object cycle")
        active.add(node_id)
        try:
            node_type = node.get("type")
            if node_type not in _EXPECTED_KEYS:
                fail(path, f"unknown node type {node_type!r}")
            if frozenset(node) != _EXPECTED_KEYS[node_type]:
                missing = sorted(_EXPECTED_KEYS[node_type] - frozenset(node))
                extra = sorted(frozenset(node) - _EXPECTED_KEYS[node_type])
                fail(path, f"wrong fields (missing={missing}, extra={extra})")

            if node_type in {"sum", "ordered_product"}:
                field = "terms" if node_type == "sum" else "factors"
                children = node[field]
                if not isinstance(children, list) or not children:
                    fail(f"{path}.{field}", "must be a non-empty array")
                for index, child in enumerate(children):
                    visit(child, f"{path}.{field}[{index}]")
            elif node_type in {"commutator", "hadamard"}:
                visit(node["left"], f"{path}.left")
                visit(node["right"], f"{path}.right")
            elif node_type == "operator":
                if not isinstance(node["label"], str) or not node["label"]:
                    fail(f"{path}.label", "must be a non-empty string")
                if operator_labels is not None and node["label"] not in operator_labels:
                    fail(f"{path}.label", "is not a canonical operator label for this order")
            elif node_type == "density_operator":
                if node["symbol"] != "rho":
                    fail(f"{path}.symbol", "must equal 'rho'")
            elif node_type == "kernel":
                if node["family"] not in {"Gamma", "DeltaGamma"}:
                    fail(
                        f"{path}.family",
                        "must equal 'Gamma' or 'DeltaGamma'",
                    )
                if not isinstance(node["transpose"], bool):
                    fail(f"{path}.transpose", "must be a boolean")
                arguments = node["arguments"]
                expected_arguments = (
                    1 if node["family"] == "Gamma" else 2
                )
                if (
                    not isinstance(arguments, list)
                    or len(arguments) != expected_arguments
                ):
                    fail(
                        f"{path}.arguments",
                        f"must contain exactly {expected_arguments} item(s)",
                    )
                for index, argument in enumerate(arguments):
                    visit(argument, f"{path}.arguments[{index}]")
                    if argument.get("type") != "time_expression":
                        fail(
                            f"{path}.arguments[{index}]",
                            "kernel arguments must be time_expression nodes",
                        )
            elif node_type == "time_expression":
                terms = node["terms"]
                if not isinstance(terms, list) or not terms:
                    fail(f"{path}.terms", "must be a non-empty array")
                for index, term in enumerate(terms):
                    term_path = f"{path}.terms[{index}]"
                    if not isinstance(term, dict):
                        fail(term_path, "must be an object")
                    if frozenset(term) != frozenset(
                        {"symbol", "coefficient"}
                    ):
                        fail(
                            term_path,
                            "fields must be exactly symbol and coefficient",
                        )
                    if not is_time_symbol(term["symbol"]):
                        fail(
                            f"{term_path}.symbol",
                            "must equal '0', 't', or 't_' followed by lowercase letters",
                        )
                    if time_symbols is not None and term["symbol"] not in time_symbols:
                        fail(f"{term_path}.symbol", "is not a time coordinate for this order")
                    coefficient = term["coefficient"]
                    if isinstance(coefficient, bool) or not isinstance(
                        coefficient, int
                    ):
                        fail(
                            f"{term_path}.coefficient",
                            "must be an integer",
                        )
            elif node_type == "scalar":
                scalar_value = node["value"]
                if isinstance(scalar_value, bool) or not isinstance(
                    scalar_value, int
                ):
                    fail(f"{path}.value", "must be an integer")
        finally:
            active.remove(node_id)

    visit(value, "$")
    return cast(ASTNode, value)


__all__ = [
    "ASTNode",
    "ASTValidationError",
    "CommutatorNode",
    "DensityOperatorNode",
    "HadamardNode",
    "KernelNode",
    "OperatorNode",
    "OrderedProductNode",
    "ScalarNode",
    "SumNode",
    "TimeExpressionNode",
    "TimeTerm",
    "apply_integer_coefficient",
    "commutator",
    "density_operator",
    "hadamard",
    "kernel",
    "make_ordered_product",
    "make_sum",
    "operator",
    "scalar",
    "time_expression",
    "validate_ast",
]
