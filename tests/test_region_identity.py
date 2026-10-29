from __future__ import annotations

from fractions import Fraction
from itertools import permutations, product

import pytest

from fasttcl.symbolic.region_identity import (
    RegionIdentityLimitError,
    certify_region_identity,
)
from fasttcl.symbolic.region_semantics import (
    ChamberSpace,
    RegionBound,
    RegionSemanticsError,
    signed_region_residual,
)


def _full(count: int) -> tuple[RegionBound, ...]:
    return tuple(RegionBound(variable, 0, count + 1) for variable in range(1, count + 1))


def _chain(count: int, ordering: tuple[int, ...]) -> tuple[RegionBound, ...]:
    predecessors = dict(zip(ordering[1:], ordering[:-1]))
    return tuple(
        RegionBound(variable, predecessors.get(variable, 0), count + 1)
        for variable in range(1, count + 1)
    )


def _assert_matches_oracle(count, source, terms):
    space = ChamberSpace(count)
    expected = signed_region_residual(
        space.region(source),
        [(coefficient, space.region(bounds)) for coefficient, bounds in terms],
    )
    actual = certify_region_identity(count, source, terms)
    assert actual.is_zero == expected.is_zero
    assert actual.explored_nodes > 0
    if actual.is_zero:
        assert actual.witness_ordering is None
        assert actual.residual is None
    else:
        witness = actual.witness_ordering
        assert sorted(witness) == list(range(1, count + 1))
        index = space.chambers.index(witness)
        # The legacy residual uses source - sum(leaves), while the decision
        # certificate exposes sum(leaves) - source.
        assert actual.residual == -expected.values[index]
        assert actual.residual != 0
    return actual


def test_exact_signed_tcl6_domain_identity_and_wrong_sign() -> None:
    source = _chain(5, (1, 2, 3, 4, 5))
    positive = (
        RegionBound(3, 0, 6),
        RegionBound(5, 3, 6),
        RegionBound(4, 3, 5),
        RegionBound(1, 0, 3),
        RegionBound(2, 0, 3),
    )
    negative = (
        RegionBound(1, 0, 6),
        RegionBound(4, 1, 6),
        RegionBound(5, 4, 6),
        RegionBound(3, 1, 4),
        RegionBound(2, 0, 1),
    )
    assert _assert_matches_oracle(5, source, [(1, positive), (-1, negative)]).is_zero
    assert not _assert_matches_oracle(5, source, [(1, positive), (1, negative)]).is_zero


@pytest.mark.parametrize("count", [1, 2, 3])
def test_every_low_order_bound_graph_against_exhaustive_oracle(count: int) -> None:
    """Exhaust every grammatical bound assignment through three variables."""
    choices = [
        tuple(
            RegionBound(variable, lower, upper)
            for lower in range(count + 2)
            for upper in range(count + 2)
            if len({variable, lower, upper}) == 3
        )
        for variable in range(1, count + 1)
    ]
    space = ChamberSpace(count)
    distinct_regions = {}
    for bounds in product(*choices):
        try:
            region = space.region(bounds)
        except RegionSemanticsError:
            with pytest.raises(RegionSemanticsError):
                certify_region_identity(count, bounds, [(1, bounds)])
        else:
            _assert_matches_oracle(count, _full(count), [(1, bounds)])
            assert certify_region_identity(count, bounds, [(3, bounds), (-2, bounds)]).is_zero
            distinct_regions.setdefault(region.incidence_vector, bounds)

    # Compare all different semantic regions, not just exact input equality.
    for source, leaf in product(distinct_regions.values(), repeat=2):
        _assert_matches_oracle(count, source, [(1, leaf)])


@pytest.mark.parametrize("count", [2, 3, 4, 5])
def test_all_chambers_partition_full_region_and_missing_leaf_is_detected(count: int) -> None:
    terms = [(1, _chain(count, order)) for order in permutations(range(1, count + 1))]
    assert _assert_matches_oracle(count, _full(count), terms).is_zero
    assert not _assert_matches_oracle(count, _full(count), terms[:-1]).is_zero


def test_transitively_equal_bounds_collect_without_searching() -> None:
    # Both conjunctions impose 1 < 2 < 3, using different redundant edges.
    source = [(1, 0, 2), (2, 0, 3), (3, 0, 4)]
    alternate = [(1, 0, 3), (2, 1, 3), (3, 1, 4)]
    result = certify_region_identity(3, source, [(1, alternate)], max_nodes=1)
    assert result.is_zero
    assert result.explored_nodes == 1


def test_higher_order_partition_does_not_materialize_unrelated_orderings(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("factorial chamber enumeration was called")

    monkeypatch.setattr("fasttcl.symbolic.region_semantics._chambers", forbidden)
    monkeypatch.setattr("fasttcl.symbolic.region_semantics.IntegrationRegion", forbidden)
    count = 19
    # Only four disjoint pair comparisons matter.  Eleven further variables
    # remain free, so there are vastly more chambers than decision nodes.
    leaves = []
    for directions in product((False, True), repeat=4):
        predecessors = {}
        for pair, reversed_pair in enumerate(directions):
            lower, upper = 2 * pair + 1, 2 * pair + 2
            if reversed_pair:
                lower, upper = upper, lower
            predecessors[upper] = lower
        bounds = tuple(
            RegionBound(variable, predecessors.get(variable, 0), count + 1)
            for variable in range(1, count + 1)
        )
        leaves.append((1, bounds))
    proof = certify_region_identity(count, _full(count), leaves, max_nodes=31)
    assert proof.is_zero
    assert proof.explored_nodes == 31
    wrong = certify_region_identity(count, _full(count), leaves[:-1])
    assert not wrong.is_zero
    assert wrong.residual == -1
    assert len(wrong.witness_ordering) == count


def test_general_order_signed_cancellation_and_iterable_inputs() -> None:
    count = 11
    source = _chain(count, (1, 2, 3))
    broad = _chain(count, (1, 3))
    # Within 1 < 3, the other two positions of label 2 are subtracted.
    leaves = [(1, broad), (-1, _chain(count, (2, 1, 3))), (-1, _chain(count, (1, 3, 2)))]
    proof = certify_region_identity(
        count,
        (bound for bound in source),
        ((coefficient, (bound for bound in bounds)) for coefficient, bounds in leaves),
    )
    assert proof.is_zero
    assert proof.explored_nodes < 20


def test_search_limit_fails_without_claiming_a_certificate() -> None:
    source = _full(2)
    leaves = [(1, _chain(2, (1, 2))), (1, _chain(2, (2, 1)))]
    with pytest.raises(RegionIdentityLimitError, match="not been certified") as captured:
        certify_region_identity(2, source, leaves, max_nodes=1)
    assert captured.value.explored_nodes == 1
    assert captured.value.max_nodes == 1
    assert certify_region_identity(2, source, leaves, max_nodes=3).is_zero


@pytest.mark.parametrize("bad", [True, False, 1.0, Fraction(1), "1", None])
def test_noninteger_coefficients_are_rejected(bad) -> None:
    with pytest.raises(RegionSemanticsError, match="coefficient"):
        certify_region_identity(2, _full(2), [(bad, _full(2))])


@pytest.mark.parametrize("bad", [True, False, 0, -1, 2.0, "2", None])
def test_invalid_count_is_rejected(bad) -> None:
    with pytest.raises(RegionSemanticsError, match="positive integer"):
        certify_region_identity(bad, [], [])


@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.0, "1"])
def test_invalid_search_budget_is_rejected(bad) -> None:
    with pytest.raises(RegionSemanticsError, match="max_nodes"):
        certify_region_identity(2, _full(2), [], max_nodes=bad)


@pytest.mark.parametrize(
    "bad_bounds",
    [
        [(1, 0, 3)],
        [(1, 0, 3), (1, 0, 3)],
        [(1, 0, 3), (3, 0, 2)],
        [(1, -1, 3), (2, 0, 3)],
        [(1, 0, 4), (2, 0, 3)],
        [(1, 0, 1), (2, 0, 3)],
        [(1, 2, 3), (2, 1, 3)],
        [(1, 3, 2), (2, 0, 3)],
        [(1, 2, 0), (2, 0, 3)],
        [(True, 0, 3), (2, 0, 3)],
        [(1, 0, "3"), (2, 0, 3)],
        [{"variable": 1, "lower": 0, "upper": 3, "extra": 1}, (2, 0, 3)],
    ],
)
def test_invalid_bounds_rejected_even_with_zero_coefficient(bad_bounds) -> None:
    with pytest.raises(RegionSemanticsError):
        certify_region_identity(2, _full(2), [(0, bad_bounds)])


def test_nonzero_constant_residual_and_empty_leaf_list() -> None:
    source = _full(12)
    result = certify_region_identity(12, source, [(4, source)])
    assert not result.is_zero
    assert result.residual == 3
    assert result.witness_ordering == tuple(range(1, 13))
    empty = certify_region_identity(12, source, [])
    assert not empty.is_zero
    assert empty.residual == -1
    assert empty.explored_nodes == 1
