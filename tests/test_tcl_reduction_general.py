from __future__ import annotations

from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from fasttcl.symbolic.tcl_cumulant import CumulativeGroup, cumulative_groups, pairings
from fasttcl.symbolic.tcl_reduction import (
    TCLReductionError, certify_pairing_plan, derive_tcl_reduction,
    iter_reduced_branches, reduced_records,
)
from fasttcl.symbolic.tcl_symbolic_cli import compile_to_directory, main


@pytest.mark.parametrize("order", [2, 4, 6])
def test_complete_derivation_and_exhaustive_certification(order):
    result = derive_tcl_reduction(order, verify_exhaustive=True)
    assert result.outer_wrapper_verified
    assert result.collection.survivors
    assert all(plan.certificate.is_zero and plan.exhaustive_verified for plan in result.pairing_plans)
    assert all(len(branch.reduced.steps) == order // 2
        for branch in iter_reduced_branches(result.source_groups, result.pairing_plans, order=order))
    assert all(row["order"] == order for row in reduced_records(result))


def test_all_tcl10_connected_matchings_have_exact_domain_proofs():
    from fasttcl.symbolic.tcl_cumulant import is_interval_connected
    count = 0
    for matching in pairings(tuple(range(10))):
        if is_interval_connected(matching, 10):
            plan = certify_pairing_plan(matching, order=10)
            assert plan.certificate.is_zero
            count += 1
    assert count == 706


@pytest.mark.parametrize("order", [10, 12])
def test_higher_order_branch_reaches_new_time_symbols(order):
    # A single exact free-algebra source exercises the complete planning,
    # Hadamard reduction and AST collection interface beyond TCL8.
    matching = ((0, order - 1),) + tuple((i, i + 1) for i in range(1, order - 1, 2))
    word = tuple(("operator", str(i)) for i in range(1, order)) + (("density_operator", "rho"),)
    group = CumulativeGroup(matching, Counter({word: 1}))
    plan = certify_pairing_plan(matching, order=order)
    branches = tuple(iter_reduced_branches((group,), (plan,), order=order))
    assert branches
    assert all(len(branch.reduced.steps) == order // 2 for branch in branches)
    assert any('t_d' in json.dumps(branch.reduced.expression) for branch in branches)


def test_reject_mismatched_plan():
    plan = certify_pairing_plan(((0, 1),), order=2)
    with pytest.raises(TCLReductionError, match="wrong order"):
        tuple(iter_reduced_branches(cumulative_groups(4), (plan,), order=4))


def test_changed_plan_leaf_cannot_reuse_old_certificate():
    plan = certify_pairing_plan(((0, 1),), order=2)
    changed = replace(plan, leaves=(replace(plan.leaves[0], coefficient=2),))
    with pytest.raises(TCLReductionError, match="signed domain"):
        tuple(iter_reduced_branches(cumulative_groups(2), (changed,), order=2))


@pytest.mark.parametrize("order", [True, 0, 1, 3, -2, 4.0, "4"])
def test_invalid_order_fails_before_derivation(order):
    with pytest.raises((TypeError, ValueError)):
        derive_tcl_reduction(order)


def test_cli_artifacts_deterministic_and_hashes_cover_records(tmp_path):
    one = compile_to_directory(4, tmp_path / "one", dimension=2)
    two = compile_to_directory(4, tmp_path / "two", dimension=2)
    assert one["status"] == two["status"] == "passed"
    assert one["artifacts"] == two["artifacts"]
    for item in one["artifacts"].values():
        data = (tmp_path / "one" / item["path"]).read_bytes()
        assert sha256(data).hexdigest() == item["sha256"]
        assert len(data.splitlines()) == item["records"]
        assert all(json.loads(line)["order"] == 4 for line in data.splitlines())
    with pytest.raises(ValueError, match="empty"):
        compile_to_directory(4, tmp_path / "one")


def test_cli_estimate_and_invalid_input(capsys):
    assert main(["--order", "10", "--estimate"]) == 0
    estimate = json.loads(capsys.readouterr().out)
    assert estimate["order"] == 10 and estimate["retained_integrations"] == 4
    with pytest.raises(SystemExit) as exc:
        main(["--order", "3", "--estimate"])
    assert exc.value.code == 2
