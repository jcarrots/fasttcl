from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from hashlib import sha256
import json

import numpy as np
import pytest

from fasttcl.symbolic.ast import (
    density_operator, hadamard, kernel, make_ordered_product, operator,
    time_expression,
)
from fasttcl.symbolic.tcl_order import TCLOrder
from fasttcl.symbolic.tcl_scalarization import (
    TCLScalarizationError, compile_scalar_terms, evaluate_scalar_terms,
)


def _argument(*terms):
    return time_expression([
        {"symbol": symbol, "coefficient": coefficient}
        for symbol, coefficient in terms
    ])


def _synthetic(order: int):
    """An independent valid-degree word using every retained time label."""

    specification = TCLOrder(order)
    labels = ("0",) + specification.dummy_labels
    symbols = ("t",) + tuple("t_" + label for label in specification.dummy_labels)
    factors = []
    for index, (label, symbol) in enumerate(zip(labels, symbols, strict=True)):
        following = symbols[index + 1] if index + 1 < len(symbols) else "0"
        sign = 1 if index % 2 == 0 else -1
        argument = _argument((symbol, sign), (following, -sign))
        gamma = kernel([argument], transpose=bool(index % 2))
        if index == 0 and order >= 4:
            gamma = kernel([
                argument, _argument(("t", -1)),
            ], family="DeltaGamma", transpose=True)
        factors.append(hadamard(operator(label), gamma))
    # N reduced H2 factors plus N-1 unintegrated system operators.
    factors[1:1] = [density_operator()]
    factors.extend(operator(label) for label in specification.dummy_labels)
    return make_ordered_product(factors)


def _gamma(argument, row, column):
    # Deliberately complex and asymmetric: transpose must not conjugate.
    return argument * (1.0 + 0.3 * row + 0.7j * (column + 1)) + argument**2 * (0.2 + 0.1j * row)


def _matrix_ast(node, *, density, energies, coupling, time_values):
    """Literal matrix evaluation, independent of formal/scalar compilation."""

    kind = node["type"]
    dimension = len(energies)
    if kind == "density_operator":
        return density
    if kind == "scalar":
        return node["value"] * np.eye(dimension, dtype=complex)
    if kind == "operator":
        label = node["label"]
        symbol = "t" if label == "0" else "t_" + label
        omega = energies[:, None] - energies[None, :]
        return coupling * np.exp(1j * omega * time_values[symbol])
    if kind == "kernel":
        values = []
        for argument in node["arguments"]:
            t = sum(term["coefficient"] * time_values[term["symbol"]]
                    for term in argument["terms"])
            matrix = np.array([[_gamma(t, row, column) for column in range(dimension)]
                               for row in range(dimension)])
            values.append(matrix.T if node["transpose"] else matrix)
        return values[0] if node["family"] == "Gamma" else values[0] - values[1]
    arguments = dict(density=density, energies=energies, coupling=coupling, time_values=time_values)
    if kind == "ordered_product":
        result = np.eye(dimension, dtype=complex)
        for factor in node["factors"]:
            result = result @ _matrix_ast(factor, **arguments)
        return result
    if kind == "hadamard":
        return _matrix_ast(node["left"], **arguments) * _matrix_ast(node["right"], **arguments)
    if kind == "sum":
        return sum(_matrix_ast(term, **arguments) for term in node["terms"])
    if kind == "commutator":
        left = _matrix_ast(node["left"], **arguments)
        right = _matrix_ast(node["right"], **arguments)
        return left @ right - right @ left
    raise AssertionError(f"unexpected AST kind {kind}")


@pytest.mark.parametrize("order,dimension", [(2, 1), (2, 3), (4, 2), (4, 3), (6, 2)])
def test_scalar_compilation_matches_direct_complex_matrix_ast(order, dimension):
    rng = np.random.default_rng(812 + order + dimension)
    energies = np.sort(rng.normal(size=dimension))
    # General complex A checks that conjugation is retained explicitly.
    coupling = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
    gaps = np.arange(1, order // 2 + 1, dtype=float) / (order + 1)
    specification = TCLOrder(order)
    times = {symbol: float(np.dot(vector, gaps))
             for symbol, vector in specification.time_to_gap.items()}
    expression = _synthetic(order)
    source = compile_scalar_terms([(2, expression), (-1, expression)], order=order, dimension=dimension)
    raw = source.outer_raw()
    completed = raw.hermitian_completion()
    numerical = [evaluate_scalar_terms(result, energies=energies, coupling=coupling,
                                      gaps=gaps, gamma=_gamma)
                 for result in (source, raw, completed)]
    at = coupling * np.exp(1j * (energies[:, None] - energies[None, :]) * sum(gaps))
    literal_source = np.zeros_like(numerical[0])
    literal_raw = np.zeros_like(numerical[0])
    for row in range(dimension):
        for column in range(dimension):
            matrix = np.zeros((dimension, dimension), dtype=complex)
            matrix[row, column] = 1
            value = _matrix_ast(expression, density=matrix, energies=energies,
                                coupling=coupling, time_values=times)
            literal_source[:, row + dimension * column] = value.reshape(-1, order="F")
            literal_raw[:, row + dimension * column] = (-at @ value + value @ at).reshape(-1, order="F")
    exchange = np.arange(dimension**2).reshape(dimension, dimension, order="F").T.reshape(-1, order="F")
    literal_complete = literal_raw + literal_raw.conj()[exchange][:, exchange]
    for actual, expected in zip(numerical, (literal_source, literal_raw, literal_complete), strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-12)
    assert all(len(term.signature.phases) == order // 2 for term in source.terms)
    assert all(len(term.signature.kernels) == order // 2 for term in source.terms)
    assert all(len(monomial) == order - 1 for term in source.terms for monomial, _ in term.coefficients)
    assert all(len(monomial) == order for term in raw.terms for monomial, _ in term.coefficients)


def _legacy_projection(compilation):
    result = Counter()
    for term in compilation.terms:
        s = term.signature
        phases = tuple((phase[1] - phase[0]) // 2 for phase in s.phases)
        kernels = []
        for factor in s.kernels:
            interval = factor.support
            support = "K" if (interval.start, interval.stop) == (0, 4) else "UVWX"[interval.start:interval.stop]
            if interval.sign < 0:
                support = "-" + support
            kernels.append((support, factor.row, factor.column))
        key = (s.output_row, s.output_column, s.input_row, s.input_column,
               *phases, tuple(sorted(kernels)))
        result[key] += sum(value for _, value in term.coefficients)
    return json.loads(json.dumps([[key, value] for key, value in sorted(result.items()) if value]))


def test_generic_tcl8_matches_frozen_spin_boson_scalarization():
    from fasttcl.symbolic.tcl_reduction import certify_pairing_plan, iter_reduced_branches
    from fasttcl.symbolic.tcl_cumulant import cumulative_groups, undirected_pairing

    reference = json.loads((Path(__file__).parent / "fixtures/tcl8-scalar-reference.json").read_text(encoding="utf-8"))
    group = cumulative_groups(8)[0]
    assert [list(pair) for pair in group.correlations] == reference["correlations"]
    plan = certify_pairing_plan(undirected_pairing(group.correlations), order=8)
    branches = tuple(iter_reduced_branches((group,), (plan,), order=8))
    expressions = tuple((branch.reduced.coefficient, branch.reduced.expression) for branch in branches)
    generic = compile_scalar_terms(expressions, order=8, dimension=2,
                                   coupling_support=frozenset({(0, 1), (1, 0)}))
    for kind, stage in (("source", generic), ("outer_raw", generic.outer_raw())):
        records = _legacy_projection(stage)
        canonical = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        assert len(records) == reference[kind]["records"]
        assert sha256(canonical.encode("utf-8")).hexdigest() == reference[kind]["sha256"]



def test_order_ten_uses_five_gaps_and_fourth_retained_time():
    expression = _synthetic(10)
    result = compile_scalar_terms([(1, expression)], order=10, dimension=2,
                                 coupling_support=frozenset({(0, 1), (1, 0)}))
    assert result.terms
    assert all(len(term.signature.phases) == 5 for term in result.terms)
    assert any(kernel.support.start == 4 for term in result.terms for kernel in term.signature.kernels)
    energies = np.array([-0.7, 0.8])
    coupling = np.array([[0, 0.4 + 0.2j], [0.1 - 0.3j, 0]])
    gaps = np.array([0.15, 0.13, 0.08, 0.11, 0.19])
    times = {symbol: float(np.dot(vector, gaps)) for symbol, vector in TCLOrder(10).time_to_gap.items()}
    value = evaluate_scalar_terms(result, energies=energies, coupling=coupling, gaps=gaps, gamma=_gamma)
    for row in range(2):
        for column in range(2):
            density = np.zeros((2, 2), dtype=complex)
            density[row, column] = 1
            reference = _matrix_ast(expression, density=density, energies=energies,
                                    coupling=coupling, time_values=times)
            np.testing.assert_allclose(value[:, row + 2 * column], reference.reshape(-1, order="F"),
                                       atol=1e-18, rtol=2e-12)


def test_complex_linear_completion_on_nonhermitian_input_and_hermitian_coupling():
    rng = np.random.default_rng(1245)
    energies = np.array([-0.5, 0.3, 0.9])
    draw = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    coupling = (draw + draw.conj().T) / 2
    x = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    assert not np.allclose(x, x.conj().T)
    gaps = np.array([0.17, 0.29])
    times = {symbol: float(np.dot(vector, gaps)) for symbol, vector in TCLOrder(4).time_to_gap.items()}
    expression = _synthetic(4)
    completed = compile_scalar_terms([(1, expression)], order=4, dimension=3).outer_raw().hermitian_completion()
    generator = evaluate_scalar_terms(completed, energies=energies, coupling=coupling, gaps=gaps, gamma=_gamma)
    at = coupling * np.exp(1j * (energies[:, None] - energies[None, :]) * sum(gaps))

    def literal_raw(matrix):
        value = _matrix_ast(expression, density=matrix, energies=energies,
                            coupling=coupling, time_values=times)
        return -at @ value + value @ at

    reference = literal_raw(x) + literal_raw(x.conj().T).conj().T
    actual = (generator @ x.reshape(-1, order="F")).reshape(3, 3, order="F")
    np.testing.assert_allclose(actual, reference, atol=2e-12, rtol=2e-12)
    # The incorrect state-adjoint completion fails this intentionally complex input.
    incorrect = literal_raw(x) + literal_raw(x).conj().T
    assert np.max(np.abs(incorrect - reference)) > 1e-3
    np.testing.assert_allclose(np.trace(actual), 0, atol=2e-12)


def test_scalar_coordinates_extend_past_single_letter_dummy_labels():
    result = compile_scalar_terms([(1, _synthetic(56))], order=56, dimension=1)
    assert result.terms
    assert all(len(term.signature.phases) == 28 for term in result.terms)
    assert all(len(term.signature.kernels) == 28 for term in result.terms)
    assert any(k.support.start == 27 for term in result.terms for k in term.signature.kernels)
    # A scalar system has an exactly vanishing outer commutator at every order.
    assert result.outer_raw().terms == ()


def test_serialization_is_deterministic_and_content_binds_order():
    expression = _synthetic(4)
    one = compile_scalar_terms([(1, expression)], order=4, dimension=2)
    split = compile_scalar_terms([(-2, expression), (3, expression)], order=4, dimension=2)
    assert one.payload() == split.payload()
    payload = one.payload()
    digest = payload.pop("sha256")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert digest == sha256(canonical.encode("utf-8")).hexdigest()
    assert json.loads(json.dumps(payload)) == payload
    empty4 = compile_scalar_terms([], order=4, dimension=2)
    empty6 = compile_scalar_terms([], order=6, dimension=2)
    assert empty4.payload()["sha256"] != empty6.payload()["sha256"]


def test_invalid_degrees_dimensions_and_stages_fail_explicitly():
    with pytest.raises(TCLScalarizationError, match="degree"):
        compile_scalar_terms([(1, _synthetic(4))], order=6, dimension=2)
    with pytest.raises(TCLScalarizationError, match="dimension"):
        compile_scalar_terms([], order=2, dimension=True)
    with pytest.raises(TCLScalarizationError, match="coefficients"):
        compile_scalar_terms([(True, _synthetic(2))], order=2, dimension=2)
    with pytest.raises(TCLScalarizationError, match="coupling_support"):
        compile_scalar_terms([], order=2, dimension=2, coupling_support=frozenset({(0, 2)}))
    source = compile_scalar_terms([(1, _synthetic(2))], order=2, dimension=2)
    with pytest.raises(TCLScalarizationError, match="outer_raw"):
        source.hermitian_completion()
    with pytest.raises(TCLScalarizationError, match="prewrapper"):
        source.outer_raw().outer_raw()


def test_noncontiguous_or_repeated_gap_kernel_support_is_rejected():
    expression = make_ordered_product([
        hadamard(operator("0"), kernel([_argument(("t", 2))])), density_operator(),
    ])
    with pytest.raises(TCLScalarizationError, match="contiguous"):
        compile_scalar_terms([(1, expression)], order=2, dimension=2)


@pytest.mark.parametrize("zero_mode", ["canceled", "coefficient", "support"])
@pytest.mark.parametrize("invalid_field", ["operator", "time"])
def test_order_specific_symbols_are_checked_before_cancellation_or_path_pruning(zero_mode, invalid_field):
    expression = deepcopy(_synthetic(2))
    first = expression["factors"][0]
    if invalid_field == "operator":
        first["left"]["label"] = "a"  # No retained internal time at TCL2.
    else:
        first["right"]["arguments"][0]["terms"][0]["symbol"] = "t_a"
    expressions = [(1, expression)]
    support = None
    if zero_mode == "canceled":
        expressions.append((-1, expression))
    elif zero_mode == "coefficient":
        expressions = [(0, expression)]
    else:
        support = frozenset()
    with pytest.raises(TCLScalarizationError, match="this order"):
        compile_scalar_terms(expressions, order=2, dimension=2, coupling_support=support)


@pytest.mark.parametrize("stage", ["prewrapper", "outer_raw", "completed"])
def test_reference_evaluator_enforces_declared_exact_coupling_support(stage):
    compilation = compile_scalar_terms([(1, _synthetic(2))], order=2, dimension=2,
                                       coupling_support=frozenset({(0, 1), (1, 0)}))
    if stage != "prewrapper":
        compilation = compilation.outer_raw()
    if stage == "completed":
        compilation = compilation.hermitian_completion()
    options = dict(energies=[0.0, 1.0], gaps=[0.2], gamma=_gamma)
    coupling = np.array([[0.0, 0.3j], [0.2, 0.0]])
    evaluate_scalar_terms(compilation, coupling=coupling, **options)
    coupling[0, 0] = 1e-300  # No approximate-zero interpretation is permitted.
    with pytest.raises(TCLScalarizationError, match="exact coupling_support"):
        evaluate_scalar_terms(compilation, coupling=coupling, **options)


@pytest.mark.parametrize("replacement,message", [
    ({"energies": [0, float("nan")]}, "energies must be finite and real"),
    ({"energies": [0, float("inf")]}, "energies must be finite and real"),
    ({"energies": [0, 1 + 1e-300j]}, "energies must be finite and real"),
    ({"gaps": [float("nan")]}, "gaps must be finite and real"),
    ({"gaps": [float("inf")]}, "gaps must be finite and real"),
    ({"gaps": [0.2 + 1e-300j]}, "gaps must be finite and real"),
    ({"gaps": [-1e-300]}, "gaps must be nonnegative"),
    ({"coupling": [[0, 1], [1, float("nan")]]}, "coupling matrix must be finite"),
    ({"coupling": [[0, 1], [1, float("inf")]]}, "coupling matrix must be finite"),
    ({"coupling": [0, 1]}, "coupling matrix has the wrong dimension"),
    ({"energies": [0]}, "wrong dimension"),
    ({"gaps": [0.1, 0.2]}, "wrong dimension"),
    ({"gamma": lambda *_: float("nan")}, "Gamma callback must return finite"),
])
def test_reference_evaluator_rejects_invalid_numeric_inputs(replacement, message):
    compilation = compile_scalar_terms([(1, _synthetic(2))], order=2, dimension=2)
    options = dict(energies=[0.0, 1.0], coupling=np.eye(2), gaps=[0.2], gamma=_gamma)
    options.update(replacement)
    with pytest.raises(TCLScalarizationError, match=message):
        evaluate_scalar_terms(compilation, **options)
