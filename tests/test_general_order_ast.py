from __future__ import annotations

import pytest

from fasttcl.symbolic.ast import (
    ASTValidationError, density_operator, kernel, make_ordered_product,
    operator, time_expression, validate_ast,
)
from fasttcl.symbolic.hadamard_reduction import (
    HadamardReductionError, PhysicalBound, dataset_label_to_physical,
    physical_to_dataset_label, reduce_planned_branch,
)
from fasttcl.symbolic.integration import correlation, effective_bounds, initial_state, plan_hadamardable
from fasttcl.symbolic.tcl_cumulant import (
    cumulative_groups, ordered_cumulant, prewrapper_polynomial,
    reconstruct_outer_wrapper,
)
from fasttcl.symbolic.tcl_order import TCLOrder, canonical_dummy_label, is_time_symbol


def _walk(node):
    yield node
    for value in node.values():
        if isinstance(value, dict) and "type" in value:
            yield from _walk(value)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, dict) and "type" in child:
                    yield from _walk(child)


def _adjacent_pair_branch(order: int, *, coefficient: int = 1):
    """An analytically separable simplex: retain even and integrate odd times."""
    physical_order = (0, order, *range(order - 2, 0, -2), *range(order - 1, 0, -2))
    bounds = tuple(
        PhysicalBound(variable, variable - 1 if variable % 2 else 0,
                      variable + 1 if variable % 2 else variable + 2)
        for variable in range(1, order)
    )
    pairs = [
        {"left": str(order - even), "right": str(order - even + 1)}
        for even in range(2, order + 1, 2)
    ]
    expression = make_ordered_product(
        [operator(str(label)) for label in range(1, order)] + [density_operator()]
    )
    return reduce_planned_branch(
        expression, pairs, physical_order=physical_order,
        physical_bounds=bounds, coefficient=coefficient,
    )


def test_order_coordinates_have_unbounded_unambiguous_labels():
    assert [canonical_dummy_label(i) for i in (0, 2, 25, 26, 27, 51, 52, 701, 702)] == [
        "a", "c", "z", "aa", "ab", "az", "ba", "zz", "aaa",
    ]
    small = TCLOrder(2)
    assert (small.pair_grade, small.internal_count, small.bar_index) == (1, 1, 2)
    assert small.dummy_labels == ()
    assert small.operator_slots == {"0": 0}
    assert small.time_to_gap == {"0": (0,), "t": (1,)}
    assert TCLOrder(8).time_to_gap == {
        "0": (0, 0, 0, 0), "t": (1, 1, 1, 1), "t_a": (0, 1, 1, 1),
        "t_b": (0, 0, 1, 1), "t_c": (0, 0, 0, 1),
    }
    large = TCLOrder(56)
    assert large.dummy_labels[-1] == "aa"
    assert large.operator_slots["t"] == 20
    assert large.time_to_gap["t_t"] == (0,) * 20 + (1,) * 8
    assert large.time_to_gap["t"] == (1,) * 28


@pytest.mark.parametrize("order", [True, False, 2.0, "6", None])
def test_order_rejects_nonintegers(order):
    with pytest.raises(TypeError, match="integer"):
        TCLOrder(order)


@pytest.mark.parametrize("order", [-4, 0, 1, 3, 7])
def test_order_rejects_nonpositive_and_odd(order):
    with pytest.raises(ValueError, match="positive even"):
        TCLOrder(order)


@pytest.mark.parametrize("symbol", ["t_", "t_A", "t_1", "ta", "t_á", "t_a-b", [], None])
def test_ast_rejects_invalid_time_symbol_grammar(symbol):
    assert not is_time_symbol(symbol)
    with pytest.raises(ASTValidationError, match="lowercase"):
        validate_ast(time_expression([{"symbol": symbol, "coefficient": 1}]))


def test_ast_optionally_checks_coordinates_against_order():
    node = kernel([time_expression([{"symbol": "t_aa", "coefficient": 1}])])
    assert validate_ast(node) is node
    assert validate_ast(node, order=56) is node
    with pytest.raises(ASTValidationError, match="coordinate"):
        validate_ast(node, order=54)
    with pytest.raises(ASTValidationError, match="operator label"):
        validate_ast(operator("d"), order=8)
    with pytest.raises(ASTValidationError, match="operator label"):
        validate_ast(operator("10"), order=12)


@pytest.mark.parametrize("order", [2, 4])
def test_small_orders_reconstruct_and_reduce_complete_cumulant(order):
    assert reconstruct_outer_wrapper(prewrapper_polynomial(order)) == ordered_cumulant(order)
    for group in cumulative_groups(order):
        seed = initial_state(order // 2, tuple(
            correlation(order - left, order - right) for left, right in group.correlations
        ))
        for leaf in plan_hadamardable(seed):
            reduced = reduce_planned_branch(
                group.expression,
                [{"left": str(a), "right": str(b)} for a, b in group.correlations],
                physical_order=tuple(getattr(label, "value", 0) for label in leaf.order),
                physical_bounds=tuple(
                    PhysicalBound(variable.value, getattr(bound.lower, "value", 0), bound.upper.value)
                    for variable, bound in effective_bounds(leaf).items()
                ), coefficient=leaf.coefficient,
            )
            assert len(reduced.steps) == order // 2
            validate_ast(reduced.expression, order=order)


@pytest.mark.parametrize("order", [12, 56])
def test_reduction_handles_multidigit_sources_and_more_than_26_retained_times(order):
    reduced = _adjacent_pair_branch(order)
    assert len(reduced.steps) == order // 2
    assert len(reduced.outer_variable_map) == order // 2 - 1
    validate_ast(reduced.expression, order=order)
    labels = {node["label"] for node in _walk(reduced.expression) if node["type"] == "operator"}
    assert labels == set(TCLOrder(order).operator_slots)
    assert not (labels - {"0"}) & {str(index) for index in range(1, order)}
    if order == 56:
        symbols = {term["symbol"] for node in _walk(reduced.expression)
                   if node["type"] == "time_expression" for term in node["terms"]}
        assert {"t", "t_t", "t_aa"} <= symbols


def test_reduction_preserves_integer_branch_multiplicity():
    ordinary = _adjacent_pair_branch(4)
    weighted = _adjacent_pair_branch(4, coefficient=-3)
    assert weighted.coefficient == -3
    assert ordinary.expression == weighted.expression
    for value in (True, 1.0):
        with pytest.raises(TypeError, match="integer"):
            _adjacent_pair_branch(4, coefficient=value)
    with pytest.raises(HadamardReductionError, match="nonzero"):
        _adjacent_pair_branch(4, coefficient=0)


def test_reduction_rejects_outer_region_with_multiple_chambers():
    with pytest.raises(HadamardReductionError, match="multiple chambers"):
        reduce_planned_branch(
            make_ordered_product([operator(str(i)) for i in range(1, 6)] + [density_operator()]),
            [{"left": "0", "right": "1"}, {"left": "2", "right": "3"},
             {"left": "4", "right": "5"}],
            physical_order=(0, 6, 4, 2, 5, 3, 1),
            physical_bounds=(PhysicalBound(4, 0, 6), PhysicalBound(2, 0, 6),
                             PhysicalBound(5, 4, 6), PhysicalBound(3, 2, 4),
                             PhysicalBound(1, 0, 2)),
        )


def test_label_conversion_validates_order_and_plain_integer_types():
    assert dataset_label_to_physical("10", order=12) == 2
    assert physical_to_dataset_label(2, order=12) == "10"
    for order in (False, 4.0, "6"):
        with pytest.raises(TypeError):
            dataset_label_to_physical("0", order=order)
    for value in (True, 2.0, "2"):
        with pytest.raises(TypeError):
            physical_to_dataset_label(value)
    with pytest.raises(TypeError):
        PhysicalBound(True, 0, 2)
