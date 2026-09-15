"""Group equivalent scalar contractions before assigning system indices."""

from __future__ import annotations

from collections import Counter

from dataclasses import dataclass

import hashlib

from itertools import permutations

from types import MappingProxyType

from typing import Any, Iterable, Mapping, Sequence, cast

from .contraction_scalarization import _summarize_word_scalarization, _summarize_word_scalarization_with_coupling, _summarize_word_scalarization_with_coupling_product, _support_for_time_form, monomial_key_to_json

from .formal_bilinear import FormalAtom, FormalWord, expand_formal_bilinear

from .io_utils import canonical_json

class SymbolicCompilerError(ValueError):
    """Raised when HR records cannot form a certified symbolic plan."""


_OUTPUT_ROW = -2


_OUTPUT_COLUMN = -1


_OPERATOR_LAGS = {
    "0": ("u", "v", "w"),
    "a": ("v", "w"),
    "b": ("w",),
}


_LAG_ORDER = ("u", "v", "w")


@dataclass(frozen=True, order=True)
class IndexedFactor:
    """One scalar factor whose matrix entries use symbolic index tokens."""

    kind: str
    label: str
    row: int
    column: int


@dataclass(frozen=True)
class SymbolicIndexFamily:
    """One alpha-canonical, dimension-independent scalar family."""

    key: tuple[Any, ...]
    coefficient: int
    representative: FormalWord
    source_word_count: int


@dataclass(frozen=True)
class SymbolicIndexPlan:
    """Exact symbolic families collected before fixed-dimensional lowering."""

    families: tuple[SymbolicIndexFamily, ...]
    digest: str
    report: Mapping[str, Any]


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def add(self, value: int) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: int) -> int:
        self.add(value)
        parent = self.parent[value]
        if parent != value:
            parent = self.find(parent)
            self.parent[value] = parent
        return parent

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        # Stable roots make the non-exhaustive parts of the artifact
        # deterministic.  Alpha canonicalization below does not depend on the
        # chosen dummy roots.
        if left_root > right_root:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root


@dataclass
class _IndexedBuilder:
    next_variable: int
    factors: list[tuple[str, str, int, int]]
    operator_entries: list[tuple[str, int, int]]
    equalities: _UnionFind

    @classmethod
    def create(cls) -> "_IndexedBuilder":
        equalities = _UnionFind()
        equalities.add(_OUTPUT_ROW)
        equalities.add(_OUTPUT_COLUMN)
        return cls(0, [], [], equalities)

    def variable(self) -> int:
        value = self.next_variable
        self.next_variable += 1
        self.equalities.add(value)
        return value

    def emit_word(self, word: FormalWord, row: int, column: int) -> None:
        if not word:
            self.equalities.union(row, column)
            return
        chain = [row]
        chain.extend(self.variable() for _ in range(len(word) - 1))
        chain.append(column)
        for position, atom in enumerate(word):
            self.emit_atom(atom, chain[position], chain[position + 1])

    def emit_atom(self, atom: FormalAtom, row: int, column: int) -> None:
        kind = atom[0]
        if kind == "operator":
            label = cast(str, atom[1])
            if label not in _OPERATOR_LAGS:
                raise SymbolicCompilerError(
                    f"unknown operator time label {label!r}"
                )
            self.factors.append(("coupling", "A", row, column))
            self.operator_entries.append((label, row, column))
            return
        if kind == "density_operator":
            if atom[1] != "rho":
                raise SymbolicCompilerError(
                    f"unknown density operator {atom[1]!r}"
                )
            self.factors.append(("density", "rho", row, column))
            return
        if kind == "kernel":
            if len(atom) != 4 or atom[1] != "Gamma":
                raise SymbolicCompilerError(f"invalid Gamma atom {atom!r}")
            support = _support_for_time_form(
                cast(tuple[tuple[str, int], ...], atom[3])
            )
            gamma_row, gamma_column = (
                (column, row) if bool(atom[2]) else (row, column)
            )
            self.factors.append(
                ("kernel", support, gamma_row, gamma_column)
            )
            return
        if kind == "hadamard":
            if len(atom) != 3:
                raise SymbolicCompilerError(f"invalid Hadamard atom {atom!r}")
            self.emit_word(cast(FormalWord, atom[1]), row, column)
            self.emit_word(cast(FormalWord, atom[2]), row, column)
            return
        raise SymbolicCompilerError(f"unsupported formal atom {atom!r}")


def aggregate_formal_words(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[FormalWord, int], ...]:
    """Collect equal formal words across all records before scalarization."""

    combined: Counter[FormalWord] = Counter()
    for record in records:
        coefficient = record.get("coefficient")
        expression = record.get("expression")
        if isinstance(coefficient, bool) or not isinstance(coefficient, int):
            raise SymbolicCompilerError("HR coefficients must be integers")
        combined.update(
            expand_formal_bilinear(expression, coefficient=coefficient)
        )
    return tuple(
        sorted(
            (
                (word, coefficient)
                for word, coefficient in combined.items()
                if coefficient
            ),
            key=lambda item: canonical_json(item[0]),
        )
    )


def _mapped_phase(
    operator_entries: Iterable[tuple[str, int, int]],
    mapping: Mapping[int, int],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    phase: dict[str, Counter[int]] = {
        lag: Counter() for lag in _LAG_ORDER
    }
    for label, row, column in operator_entries:
        for lag in _OPERATOR_LAGS[label]:
            phase[lag][mapping[row]] += 1
            phase[lag][mapping[column]] -= 1
    return tuple(
        tuple(sorted((index, value) for index, value in phase[lag].items() if value))
        for lag in _LAG_ORDER
    )


def _render_family(
    builder: _IndexedBuilder,
    dummy_mapping: Mapping[int, int],
    *,
    output_diagonal: bool,
) -> tuple[Any, ...]:
    roots = builder.equalities
    row_root = roots.find(_OUTPUT_ROW)
    column_root = roots.find(_OUTPUT_COLUMN)

    def mapped(value: int) -> int:
        root = roots.find(value)
        if root == row_root:
            return _OUTPUT_ROW
        if root == column_root:
            return _OUTPUT_COLUMN
        return dummy_mapping[root]

    mapping = {value: mapped(value) for value in roots.parent}
    factors = tuple(
        sorted(
            IndexedFactor(kind, label, mapping[row], mapping[column])
            for kind, label, row, column in builder.factors
        )
    )
    phase = _mapped_phase(builder.operator_entries, mapping)
    return output_diagonal, len(dummy_mapping), factors, *phase


def indexed_family_key(
    word: FormalWord, *, canonicalization: str = "alpha"
) -> tuple[Any, ...]:
    """Return a dimension-symbolic scalar-family key for ``word``.

    ``traversal`` preserves the word's deterministic dummy-index creation
    order and is linear in the IR size. ``alpha`` additionally minimizes over
    all dummy-index names; it is a proof-oriented graph-isomorphism check and
    can be factorial in the seven-index TCL6 case.
    """

    if canonicalization not in {"traversal", "alpha"}:
        raise SymbolicCompilerError(
            "canonicalization must be 'traversal' or 'alpha'"
        )

    builder = _IndexedBuilder.create()
    builder.emit_word(word, _OUTPUT_ROW, _OUTPUT_COLUMN)
    roots = builder.equalities
    row_root = roots.find(_OUTPUT_ROW)
    column_root = roots.find(_OUTPUT_COLUMN)
    output_diagonal = row_root == column_root
    external_roots = {row_root, column_root}
    dummy_roots = sorted(
        {
            roots.find(value)
            for value in roots.parent
            if roots.find(value) not in external_roots
        }
    )
    if not dummy_roots:
        return _render_family(builder, {}, output_diagonal=output_diagonal)

    if canonicalization == "traversal":
        return _render_family(
            builder,
            {root: index for index, root in enumerate(dummy_roots)},
            output_diagonal=output_diagonal,
        )

    best: tuple[Any, ...] | None = None
    for ordering in permutations(dummy_roots):
        mapping = {root: index for index, root in enumerate(ordering)}
        candidate = _render_family(
            builder, mapping, output_diagonal=output_diagonal
        )
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best


def _jsonable_symbolic(value: Any) -> Any:
    if isinstance(value, IndexedFactor):
        return [value.kind, value.label, value.row, value.column]
    if isinstance(value, tuple):
        return [_jsonable_symbolic(item) for item in value]
    return value


def _symbolic_digest(families: Sequence[SymbolicIndexFamily]) -> str:
    payload = tuple(
        (
            _jsonable_symbolic(family.key),
            family.coefficient,
            family.source_word_count,
        )
        for family in families
    )
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def build_symbolic_index_plan(
    records: Sequence[Mapping[str, Any]],
    *,
    canonicalization: str = "traversal",
) -> SymbolicIndexPlan:
    """Compile HR records into dimension-independent indexed families."""

    term_local_occurrences = 0
    distinct_words: set[FormalWord] = set()
    for record in records:
        coefficient = record.get("coefficient")
        if isinstance(coefficient, bool) or not isinstance(coefficient, int):
            raise SymbolicCompilerError("HR coefficients must be integers")
        polynomial = expand_formal_bilinear(
            record.get("expression"), coefficient=coefficient
        )
        term_local_occurrences += len(polynomial)
        distinct_words.update(polynomial)

    words = aggregate_formal_words(records)
    grouped: dict[tuple[Any, ...], list[Any]] = {}
    for word, coefficient in words:
        if canonicalization == "formal-word":
            key = ("formal-word", word)
        elif canonicalization in {"traversal", "alpha"}:
            key = indexed_family_key(
                word, canonicalization=canonicalization
            )
        else:
            raise SymbolicCompilerError(
                "canonicalization must be formal-word, traversal, or alpha"
            )
        state = grouped.setdefault(key, [0, word, 0])
        state[0] += coefficient
        state[2] += 1
        if canonical_json(word) < canonical_json(state[1]):
            state[1] = word

    families = tuple(
        SymbolicIndexFamily(
            key=key,
            coefficient=int(state[0]),
            representative=cast(FormalWord, state[1]),
            source_word_count=int(state[2]),
        )
        for key, state in sorted(grouped.items(), key=lambda item: repr(item[0]))
        if state[0]
    )
    digest = _symbolic_digest(families)
    report = MappingProxyType(
        {
            "profile": "hr-symbolic-index-plan-v1",
            "canonicalization": canonicalization,
            "dimension_independent": True,
            "source_record_count": len(records),
            "term_local_formal_word_occurrences": term_local_occurrences,
            "distinct_formal_word_count": len(distinct_words),
            "nonzero_formal_word_count": len(words),
            "formal_words_canceled_before_scalarization": (
                len(distinct_words) - len(words)
            ),
            "indexed_family_count": len(families),
            "alpha_equivalent_words_collected": len(words) - len(families),
            "coefficient_l1": sum(abs(family.coefficient) for family in families),
            "symbolic_plan_sha256": digest,
        }
    )
    return SymbolicIndexPlan(families=families, digest=digest, report=report)


def lower_symbolic_index_plan(
    plan: SymbolicIndexPlan,
    *,
    dimension: int,
    coupling_support: frozenset[tuple[int, int]] | None = None,
    coupling_join_pruning: bool = True,
) -> tuple[tuple[tuple[Any, ...], int], ...]:
    """Lower a symbolic plan with the established fixed-d scalar oracle."""

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise SymbolicCompilerError("dimension must be a positive integer")
    if not isinstance(coupling_join_pruning, bool):
        raise SymbolicCompilerError("coupling_join_pruning must be boolean")
    combined: Counter[tuple[Any, ...]] = Counter()
    support_tuple = (
        None
        if coupling_support is None
        else tuple(sorted(coupling_support))
    )
    for family in plan.families:
        summary = (
            _summarize_word_scalarization(
                family.representative, dimension
            )
            if support_tuple is None
            else (
                _summarize_word_scalarization_with_coupling(
                    family.representative,
                    dimension,
                    support_tuple,
                )
                if coupling_join_pruning
                else _summarize_word_scalarization_with_coupling_product(
                    family.representative,
                    dimension,
                    support_tuple,
                )
            )
        )
        for key, multiplicity in summary.monomials:
            combined[key] += family.coefficient * multiplicity
    return tuple(
        sorted(
            (
                (key, coefficient)
                for key, coefficient in combined.items()
                if coefficient
            ),
            key=lambda item: canonical_json(monomial_key_to_json(item[0])),
        )
    )

