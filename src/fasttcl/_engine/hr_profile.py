"""CPU runtime extracted from the canonical TCL6 Numerical source; see data/source-provenance.json."""

from __future__ import annotations

from collections import Counter

from dataclasses import dataclass

import hashlib

from typing import Any, Iterable, Mapping, Sequence

from .extended_evaluator import ScalarMonomial, inventory_counts

from .io_utils import canonical_json

from .simplifier import HR_EXPECTED_COUNTS

from .symbolic_compiler import build_symbolic_index_plan, lower_symbolic_index_plan

HR_EVALUATOR_PROFILE = "hadamard-reduced-simplex-v1"


EXPECTED_HR_SURVIVOR_COEFFICIENT_COUNTS = {
    -2: 1,
    -1: 63,
    1: 60,
}


EXPECTED_HR_SCALAR_INVENTORY = {
    "surviving": 33_884,
    "strict": 33_052,
    "overlap": 832,
    "coefficient_l1": 42_168,
}


class HadamardReducedProfileError(ValueError):
    """Raised when the production HR inventory drifts."""


@dataclass(frozen=True)
class HadamardReducedInventory:
    """Exact HR rows, scalar monomials, and provenance."""

    records: tuple[Mapping[str, Any], ...]
    monomials: tuple[ScalarMonomial, ...]
    report: Mapping[str, Any]


def _rows_sha256(rows: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def assert_hr_scalar_inventory(
    monomials: Sequence[ScalarMonomial],
) -> dict[str, int]:
    """Assert the exact HR d=2 scalar inventory and return its counts."""

    counts = inventory_counts(monomials)
    if counts != EXPECTED_HR_SCALAR_INVENTORY:
        raise HadamardReducedProfileError(
            "HR scalar inventory mismatch: "
            f"expected {EXPECTED_HR_SCALAR_INVENTORY!r}, got {counts!r}"
        )
    return counts


def _validated_hr_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    rows = tuple(records)
    if len(rows) != 124:
        raise HadamardReducedProfileError(
            f"HR inventory must contain 124 records, got {len(rows)}"
        )
    for ordinal, record in enumerate(rows, start=1):
        expected_id = f"hr-{ordinal:03d}"
        expected_label = f"HR{ordinal:03d}"
        if record.get("id") != expected_id:
            raise HadamardReducedProfileError(
                f"HR record {ordinal} must have id {expected_id!r}"
            )
        if record.get("equation_label") != expected_label:
            raise HadamardReducedProfileError(
                f"HR record {ordinal} must have label {expected_label!r}"
            )
        if record.get("representation") != "hadamard_reduced":
            raise HadamardReducedProfileError(
                f"{expected_label} must declare representation "
                "'hadamard_reduced'"
            )
    return rows


def load_hr_inventory(
    records: Sequence[Mapping[str, Any]],
    *,
    coupling_support: frozenset[tuple[int, int]] | None = None,
) -> HadamardReducedInventory:
    """Validate and scalarize the canonical serialized HR dataset.

    ``coupling_support`` is an internal model-bound compiler input.  When it
    is supplied, entries excluded by an exact structural-zero certificate are
    pruned while index paths are generated; the generic pinned inventory
    remains the default and the compatibility oracle.
    """

    rows = _validated_hr_records(records)
    coefficient_counts = Counter(
        int(record["coefficient"]) for record in rows
    )
    if coefficient_counts != Counter(
        EXPECTED_HR_SURVIVOR_COEFFICIENT_COUNTS
    ):
        raise HadamardReducedProfileError(
            "HR survivor coefficient distribution mismatch: "
            f"{dict(sorted(coefficient_counts.items()))!r}"
        )

    symbolic_plan = build_symbolic_index_plan(
        rows, canonicalization="traversal"
    )
    monomials = lower_symbolic_index_plan(
        symbolic_plan,
        dimension=2,
        coupling_support=coupling_support,
    )
    scalar_counts = (
        assert_hr_scalar_inventory(monomials)
        if coupling_support is None
        else inventory_counts(monomials)
    )
    semantic_records = (
        {
            "id": record["id"],
            "equation_label": record["equation_label"],
            "coefficient": record["coefficient"],
            "expression": record["expression"],
        }
        for record in rows
    )
    algebraic_records = (
        {
            "coefficient": record["coefficient"],
            "expression": record["expression"],
        }
        for record in rows
    )
    report = {
        "profile": HR_EVALUATOR_PROFILE,
        "integration_plan_profile": "hr",
        "source_representation": "hr-v1.jsonl",
        "source_hr_record_count": len(rows),
        "simplification_counts": dict(HR_EXPECTED_COUNTS),
        "historical_appendix_f_checkpoint_used": False,
        "historical_appendix_f_profile_preserved_separately": True,
        "survivor_coefficient_counts": {
            str(coefficient): count
            for coefficient, count in sorted(coefficient_counts.items())
        },
        "scalar_inventory": scalar_counts,
        "model_bound_early_specialization": {
            "enabled": coupling_support is not None,
            "coupling_support": (
                None
                if coupling_support is None
                else [list(value) for value in sorted(coupling_support)]
            ),
            "zero_policy": (
                "not applicable"
                if coupling_support is None
                else "caller-certified exact structural zeros; no tolerance"
            ),
        },
        "symbolic_compiler": dict(symbolic_plan.report),
        "hr_records_sha256": _rows_sha256(rows),
        "hr_semantics_sha256": _rows_sha256(semantic_records),
        "label_independent_algebra_sha256": _rows_sha256(algebraic_records),
        "scalar_inventory_sha256": _rows_sha256(
            [key, coefficient] for key, coefficient in monomials
        ),
    }
    return HadamardReducedInventory(
        records=rows,
        monomials=monomials,
        report=report,
    )

