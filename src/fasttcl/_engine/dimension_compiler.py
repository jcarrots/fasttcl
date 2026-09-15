"""CPU runtime extracted from the canonical TCL6 Numerical source; see data/source-provenance.json."""

from __future__ import annotations

from dataclasses import dataclass

from functools import lru_cache

import hashlib

from types import MappingProxyType

from typing import Any, Mapping, cast

from .formal_bilinear import FormalAtom, FormalWord

from .io_utils import canonical_json

from .symbolic_compiler import SymbolicIndexPlan

class DimensionCompilerError(ValueError):
    """Raised when an exact dimension/support join plan is invalid."""


SparseCounts = tuple[tuple[int, int, int], ...]


def _validate_support(
    dimension: int,
    coupling_support: frozenset[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise DimensionCompilerError("dimension must be an integer")
    if dimension < 1:
        raise DimensionCompilerError("dimension must be positive")
    if not isinstance(coupling_support, frozenset):
        raise DimensionCompilerError("coupling_support must be a frozenset")
    for edge in coupling_support:
        if (
            not isinstance(edge, tuple)
            or len(edge) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in edge)
            or not all(0 <= value < dimension for value in edge)
        ):
            raise DimensionCompilerError(
                "coupling_support contains an out-of-range matrix index"
            )
    return tuple(sorted(coupling_support))


def _as_map(rows: SparseCounts) -> dict[tuple[int, int], int]:
    return {(row, column): count for row, column, count in rows}


@lru_cache(maxsize=None)
def _atom_counts(
    atom: FormalAtom,
    dimension: int,
    support: tuple[tuple[int, int], ...],
) -> SparseCounts:
    kind = atom[0]
    if kind == "operator":
        return tuple((row, column, 1) for row, column in support)
    if kind in {"density_operator", "kernel"}:
        return tuple(
            (row, column, 1)
            for row in range(dimension)
            for column in range(dimension)
        )
    if kind == "hadamard":
        left = _as_map(
            _word_counts(cast(FormalWord, atom[1]), dimension, support)
        )
        right = _as_map(
            _word_counts(cast(FormalWord, atom[2]), dimension, support)
        )
        return tuple(
            (row, column, left[(row, column)] * right[(row, column)])
            for row, column in sorted(left.keys() & right.keys())
            if left[(row, column)] and right[(row, column)]
        )
    raise DimensionCompilerError(f"unsupported formal atom {atom!r}")


@lru_cache(maxsize=None)
def _word_counts(
    word: FormalWord,
    dimension: int,
    support: tuple[tuple[int, int], ...],
) -> SparseCounts:
    relation: dict[tuple[int, int], int] = {
        (index, index): 1 for index in range(dimension)
    }
    for atom in word:
        atom_rows = _atom_counts(atom, dimension, support)
        outgoing: dict[int, list[tuple[int, int]]] = {}
        for row, column, count in atom_rows:
            outgoing.setdefault(row, []).append((column, count))
        composed: dict[tuple[int, int], int] = {}
        for (start, middle), prefix_count in relation.items():
            for stop, atom_count in outgoing.get(middle, ()):
                key = (start, stop)
                composed[key] = (
                    composed.get(key, 0) + prefix_count * atom_count
                )
        relation = composed
        if not relation:
            break
    return tuple(
        (row, column, count)
        for (row, column), count in sorted(relation.items())
        if count
    )


@dataclass(frozen=True)
class DimensionJoinPlan:
    """Compressed exact path-count certificate for one ``(plan,d,support)``."""

    symbolic_plan_digest: str
    dimension: int
    coupling_support: tuple[tuple[int, int], ...]
    family_path_counts: tuple[int, ...]
    digest: str
    report: Mapping[str, Any]


def build_dimension_join_plan(
    plan: SymbolicIndexPlan,
    *,
    dimension: int,
    coupling_support: frozenset[tuple[int, int]],
) -> DimensionJoinPlan:
    """Compile a bath/grid-independent exact support-join certificate."""

    if not isinstance(plan, SymbolicIndexPlan):
        raise DimensionCompilerError("plan must be a SymbolicIndexPlan")
    support = _validate_support(dimension, coupling_support)
    family_path_counts = tuple(
        sum(
            count
            for _row, _column, count in _word_counts(
                family.representative,
                dimension,
                support,
            )
        )
        for family in plan.families
    )
    family_relation_counts = tuple(
        len(_word_counts(family.representative, dimension, support))
        for family in plan.families
    )
    cartesian_assignments = sum(
        dimension ** (int(family.key[1]) + 2)
        for family in plan.families
    )
    joined_paths = sum(family_path_counts)
    logical_source_paths = sum(
        count * family.source_word_count
        for count, family in zip(
            family_path_counts, plan.families, strict=True
        )
    )
    payload = {
        "profile": "hr-dimension-support-join-plan-v1",
        "symbolic_plan_digest": plan.digest,
        "dimension": dimension,
        "coupling_support": [list(edge) for edge in support],
        "family_path_counts": list(family_path_counts),
        "family_relation_counts": list(family_relation_counts),
    }
    digest = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    report = MappingProxyType(
        {
            "profile": payload["profile"],
            "digest": digest,
            "symbolic_plan_digest": plan.digest,
            "dimension": dimension,
            "dimension_hard_coded": False,
            "coupling_support_count": len(support),
            "coupling_support_density": (
                len(support) / dimension**2 if dimension else 0.0
            ),
            "symbolic_family_count": len(plan.families),
            "nonempty_family_count": sum(
                count > 0 for count in family_path_counts
            ),
            "cartesian_assignment_count": cartesian_assignments,
            "joined_representative_path_count": joined_paths,
            "joined_logical_source_path_count": logical_source_paths,
            "path_pruning_fraction": (
                1.0 - joined_paths / cartesian_assignments
                if cartesian_assignments
                else 0.0
            ),
            "maximum_family_path_count": max(
                family_path_counts, default=0
            ),
            "maximum_family_relation_count": max(
                family_relation_counts, default=0
            ),
            "materializes_scalar_monomials": False,
            "bath_independent": True,
            "time_grid_independent": True,
            "exact_structural_zero_policy": True,
            "floating_tolerance_used": False,
            "generic_dense_fallback_preserved": True,
            "physical_facade_supported": dimension == 2,
        }
    )
    return DimensionJoinPlan(
        symbolic_plan_digest=plan.digest,
        dimension=dimension,
        coupling_support=support,
        family_path_counts=family_path_counts,
        digest=digest,
        report=report,
    )

