"""Expression nodes and validation for the TCL6 generator records."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias, TypedDict, cast

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
    symbol: Literal["0", "t", "t_a", "t_b"]
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


TIME_SYMBOLS = frozenset({"0", "t", "t_a", "t_b"})


class ASTValidationError(ValueError):
    """Raised when a value is not a valid TCL6 AST."""


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


def validate_ast(value: Any) -> ASTNode:
    """Validate a plain JSON value and return it with an ``ASTNode`` type.

    Validation is deliberately strict: unknown fields, empty variadic nodes,
    booleans used as integers, and object cycles are all rejected.
    """

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
                    if term["symbol"] not in TIME_SYMBOLS:
                        fail(
                            f"{term_path}.symbol",
                            f"must be one of {sorted(TIME_SYMBOLS)}",
                        )
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

