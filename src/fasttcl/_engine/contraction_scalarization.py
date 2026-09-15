"""Expand operator contractions into scalar kernel products."""

from __future__ import annotations

from collections import Counter

from dataclasses import dataclass

from functools import lru_cache

import hashlib

from itertools import product

from typing import Any, Iterable, Iterator, Sequence, cast

from .formal_bilinear import FormalAtom, FormalWord

from .io_utils import canonical_json

_TIME_TO_LAG: dict[str, tuple[int, int, int]] = {
    "0": (0, 0, 0),
    "t": (1, 1, 1),
    "t_a": (0, 1, 1),
    "t_b": (0, 0, 1),
}


_POSITIVE_SUPPORTS: dict[tuple[int, int, int], str] = {
    (1, 1, 1): "K",
    (1, 0, 0): "U",
    (0, 1, 0): "V",
    (0, 0, 1): "W",
    (1, 1, 0): "UV",
    (0, 1, 1): "VW",
}


_OPERATOR_TIME = {"0": "k", "a": "p", "b": "q"}


class ContractionScalarizationError(ValueError):
    """Raised when a formal word cannot be scalarized without guessing."""


def _validate_dimension(dimension: int) -> int:
    """Return a valid finite Hilbert-space dimension.

    ``bool`` is rejected explicitly even though it is an ``int`` subclass:
    accepting ``True`` as a one-dimensional Hilbert space would make public
    API mistakes unnecessarily difficult to diagnose.
    """

    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension < 1
    ):
        raise ContractionScalarizationError(
            "dimension must be a positive finite integer"
        )
    return dimension


def _validate_matrix_index(
    row: int,
    column: int,
    dimension: int,
    *,
    context: str,
) -> None:
    if (
        isinstance(row, bool)
        or isinstance(column, bool)
        or not isinstance(row, int)
        or not isinstance(column, int)
        or not (0 <= row < dimension)
        or not (0 <= column < dimension)
    ):
        raise ContractionScalarizationError(
            f"{context} matrix index {(row, column)!r} is outside "
            f"dimension {dimension}"
        )


def _origin_json(origin: tuple[str | int, ...]) -> list[str | int]:
    return list(origin)


def _normalized_time_form(
    terms: Iterable[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    coefficients: Counter[str] = Counter()
    for symbol, coefficient in terms:
        if symbol not in _TIME_TO_LAG:
            raise ContractionScalarizationError(
                f"unknown time symbol {symbol!r}"
            )
        if isinstance(coefficient, bool) or not isinstance(coefficient, int):
            raise ContractionScalarizationError(
                "time coefficients must be integers"
            )
        if symbol != "0":
            coefficients[symbol] += coefficient
    return tuple(
        (symbol, coefficient)
        for symbol, coefficient in sorted(coefficients.items())
        if coefficient
    )


def _support_for_time_form(
    time_form: Iterable[tuple[str, int]],
) -> str:
    vector = [0, 0, 0]
    for symbol, coefficient in _normalized_time_form(time_form):
        basis = _TIME_TO_LAG[symbol]
        for index in range(3):
            vector[index] += coefficient * basis[index]
    key = cast(tuple[int, int, int], tuple(vector))
    if key == (0, 0, 0):
        raise ContractionScalarizationError(
            "Gamma(0) must be eliminated before scalarization"
        )
    support = _POSITIVE_SUPPORTS.get(key)
    if support is not None:
        return support
    opposite = cast(tuple[int, int, int], tuple(-value for value in key))
    support = _POSITIVE_SUPPORTS.get(opposite)
    if support is not None:
        return f"-{support}"
    raise ContractionScalarizationError(
        f"unsupported Gamma lag vector {key!r}"
    )


@dataclass(frozen=True)
class _Branch:
    factors: tuple[tuple[Any, ...], ...]
    contractions: tuple[
        tuple[tuple[str | int, ...], tuple[int, ...]], ...
    ]


def _factor_to_json(factor: tuple[Any, ...]) -> dict[str, Any]:
    kind = factor[0]
    if kind == "operator":
        _, label, row, column, origin = factor
        return {
            "kind": "operator",
            "label": label,
            "time_index": _OPERATOR_TIME[label],
            "matrix_index": [row, column],
            "canonical_amplitude": {
                "kind": "coupling_operator",
                "matrix_index": [row, column],
            },
            "origin": _origin_json(origin),
        }
    if kind == "density_operator":
        _, symbol, row, column, origin = factor
        return {
            "kind": "density_operator",
            "symbol": symbol,
            "matrix_index": [row, column],
            "origin": _origin_json(origin),
        }
    if kind == "kernel":
        (
            _,
            family,
            transpose,
            support,
            row,
            column,
            gamma_row,
            gamma_column,
            origin,
        ) = factor
        return {
            "kind": "kernel",
            "family": family,
            "transpose": transpose,
            "support": support,
            "display_index": [row, column],
            "gamma_index": [gamma_row, gamma_column],
            "transpose_mapping": (
                "GammaT[r,c]=Gamma[c,r]"
                if transpose
                else "Gamma[r,c]=Gamma[r,c]"
            ),
            "origin": _origin_json(origin),
        }
    raise ContractionScalarizationError(
        f"unknown primitive scalar factor {factor!r}"
    )


def _canonical_factor(factor: tuple[Any, ...]) -> tuple[Any, ...]:
    kind = factor[0]
    if kind == "operator":
        # Labels 0/a/b denote the same energy-basis coupling matrix.  Their
        # time labels are retained exactly in the separate phase key.
        return ("coupling_operator", factor[2], factor[3])
    if kind == "density_operator":
        return ("density_operator", factor[1], factor[2], factor[3])
    if kind == "kernel":
        # The source transpose flag and display indices remain in path
        # provenance; the algebraic scalar key uses the exact mapped Gamma
        # entry so GammaT[r,c] can equal Gamma[c,r] and never Gamma*[r,c].
        return ("kernel", factor[1], factor[3], factor[6], factor[7])
    raise ContractionScalarizationError(
        f"unknown primitive scalar factor {factor!r}"
    )


def _add_vectors(*vectors: tuple[int, ...]) -> tuple[int, ...]:
    if not vectors:
        return ()
    length = len(vectors[0])
    if any(len(vector) != length for vector in vectors):
        raise ContractionScalarizationError("phase-vector length mismatch")
    return tuple(sum(vector[index] for vector in vectors) for index in range(length))


def _subtract_vectors(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _phase_vectors(
    factors: Sequence[tuple[Any, ...]], dimension: int
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    dimension = _validate_dimension(dimension)
    original = {
        "k": [0] * dimension,
        "p": [0] * dimension,
        "q": [0] * dimension,
    }
    for factor in factors:
        if factor[0] != "operator":
            continue
        _, label, row, column, _ = factor
        _validate_matrix_index(
            row,
            column,
            dimension,
            context="operator phase",
        )
        bucket = original[_OPERATOR_TIME[label]]
        # omega_rc = epsilon_r - epsilon_c.
        bucket[row] += 1
        bucket[column] -= 1
    k_phase = tuple(original["k"])
    p_phase = tuple(original["p"])
    q_phase = tuple(original["q"])
    u_phase = k_phase
    v_phase = _add_vectors(k_phase, p_phase)
    w_phase = _add_vectors(k_phase, p_phase, q_phase)

    # Invert u=k-p, v=p-q, w=q.  This is an exact integer check for every
    # energy coefficient, not a floating-point spot check.
    if k_phase != u_phase:
        raise ContractionScalarizationError("failed k/u phase identity")
    if p_phase != _subtract_vectors(v_phase, u_phase):
        raise ContractionScalarizationError("failed p/(v-u) phase identity")
    if q_phase != _subtract_vectors(w_phase, v_phase):
        raise ContractionScalarizationError("failed q/(w-v) phase identity")
    return k_phase, p_phase, q_phase, u_phase, v_phase, w_phase


def _phase_json(
    phase: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ]
) -> dict[str, Any]:
    k_phase, p_phase, q_phase, u_phase, v_phase, w_phase = phase
    return {
        "bohr_frequency_definition": "omega_rc=epsilon_r-epsilon_c",
        "original": {
            "k": list(k_phase),
            "p": list(p_phase),
            "q": list(q_phase),
        },
        "lag_factorized": {
            "u": list(u_phase),
            "v": list(v_phase),
            "w": list(w_phase),
        },
        "substitution": {
            "k": "u+v+w",
            "p": "v+w",
            "q": "w",
        },
        "identity": (
            "Omega_k*k+Omega_p*p+Omega_q*q = "
            "Omega_k*u+(Omega_k+Omega_p)*v+"
            "(Omega_k+Omega_p+Omega_q)*w"
        ),
        "verified": True,
    }


def _enumerate_atom_element(
    atom: FormalAtom,
    row: int,
    column: int,
    dimension: int,
    origin: tuple[str | int, ...],
    coupling_support: frozenset[tuple[int, int]] | None = None,
    support_join_pruning: bool = True,
) -> Iterator[_Branch]:
    kind = atom[0]
    if kind == "operator":
        label = atom[1]
        if label not in _OPERATOR_TIME:
            raise ContractionScalarizationError(
                f"unknown operator label {label!r}"
            )
        if (
            coupling_support is not None
            and (row, column) not in coupling_support
        ):
            return
        yield _Branch(
            (("operator", label, row, column, origin),), ()
        )
        return
    if kind == "density_operator":
        if atom[1] != "rho":
            raise ContractionScalarizationError(
                f"unknown density symbol {atom[1]!r}"
            )
        yield _Branch(
            (("density_operator", "rho", row, column, origin),), ()
        )
        return
    if kind == "kernel":
        if len(atom) != 4 or atom[1] != "Gamma":
            raise ContractionScalarizationError(
                f"invalid Gamma atom {atom!r}"
            )
        transpose = atom[2]
        if not isinstance(transpose, bool):
            raise ContractionScalarizationError(
                "Gamma transpose state must be boolean"
            )
        support = _support_for_time_form(
            cast(tuple[tuple[str, int], ...], atom[3])
        )
        gamma_row, gamma_column = (
            (column, row) if transpose else (row, column)
        )
        yield _Branch(
            (
                (
                    "kernel",
                    "Gamma",
                    transpose,
                    support,
                    row,
                    column,
                    gamma_row,
                    gamma_column,
                    origin,
                ),
            ),
            (),
        )
        return
    if kind == "hadamard":
        if len(atom) != 3:
            raise ContractionScalarizationError(
                f"invalid Hadamard atom {atom!r}"
            )
        left_word = cast(FormalWord, atom[1])
        right_word = cast(FormalWord, atom[2])
        for left in _enumerate_word_element(
            left_word,
            row,
            column,
            dimension,
            (*origin, "hadamard", "left"),
            coupling_support,
            support_join_pruning,
        ):
            for right in _enumerate_word_element(
                right_word,
                row,
                column,
                dimension,
                (*origin, "hadamard", "right"),
                coupling_support,
                support_join_pruning,
            ):
                yield _Branch(
                    left.factors + right.factors,
                    left.contractions + right.contractions,
                )
        return
    raise ContractionScalarizationError(
        f"unsupported formal atom {atom!r}"
    )


def _enumerate_word_element(
    word: FormalWord,
    row: int,
    column: int,
    dimension: int,
    origin: tuple[str | int, ...],
    coupling_support: frozenset[tuple[int, int]] | None = None,
    support_join_pruning: bool = True,
) -> Iterator[_Branch]:
    if not word:
        if row == column:
            yield _Branch((), ((origin, (row, column)),))
        return

    if coupling_support is not None and support_join_pruning:
        yield from _enumerate_word_element_support_join(
            word,
            row,
            column,
            dimension,
            origin,
            coupling_support,
        )
        return

    for internal in product(range(dimension), repeat=len(word) - 1):
        indices = (row, *internal, column)
        partial: list[_Branch] = [_Branch((), ())]
        for position, atom in enumerate(word):
            atom_branches = tuple(
                _enumerate_atom_element(
                    atom,
                    indices[position],
                    indices[position + 1],
                    dimension,
                    (*origin, "factor", position),
                    coupling_support,
                    support_join_pruning,
                )
            )
            next_partial: list[_Branch] = []
            for prefix in partial:
                for branch in atom_branches:
                    next_partial.append(
                        _Branch(
                            prefix.factors + branch.factors,
                            prefix.contractions + branch.contractions,
                        )
                    )
            partial = next_partial
        for branch in partial:
            yield _Branch(
                branch.factors,
                ((origin, indices),) + branch.contractions,
            )


@lru_cache(maxsize=None)
def _support_atom_pairs(
    atom: FormalAtom,
    dimension: int,
    coupling_support: tuple[tuple[int, int], ...],
) -> frozenset[tuple[int, int]]:
    """Return exact matrix positions where one atom can have a branch."""

    kind = atom[0]
    if kind == "operator":
        return frozenset(coupling_support)
    if kind in {"density_operator", "kernel"}:
        return frozenset(
            (row, column)
            for row in range(dimension)
            for column in range(dimension)
        )
    if kind == "hadamard":
        left = _support_word_pairs(
            cast(FormalWord, atom[1]), dimension, coupling_support
        )
        right = _support_word_pairs(
            cast(FormalWord, atom[2]), dimension, coupling_support
        )
        return left & right
    raise ContractionScalarizationError(
        f"unsupported formal atom {atom!r}"
    )


@lru_cache(maxsize=None)
def _support_word_pairs(
    word: FormalWord,
    dimension: int,
    coupling_support: tuple[tuple[int, int], ...],
) -> frozenset[tuple[int, int]]:
    """Relationally compose atom supports for one formal matrix word."""

    relation = frozenset((index, index) for index in range(dimension))
    for atom in word:
        atom_relation = _support_atom_pairs(
            atom, dimension, coupling_support
        )
        outgoing: dict[int, tuple[int, ...]] = {}
        for left, right in atom_relation:
            outgoing.setdefault(left, ())
            outgoing[left] = (*outgoing[left], right)
        relation = frozenset(
            (start, stop)
            for start, middle in relation
            for stop in outgoing.get(middle, ())
        )
        if not relation:
            break
    return relation


def _enumerate_word_element_support_join(
    word: FormalWord,
    row: int,
    column: int,
    dimension: int,
    origin: tuple[str | int, ...],
    coupling_support: frozenset[tuple[int, int]],
) -> Iterator[_Branch]:
    """Enumerate only index chains admitted by exact relational joins."""

    support_tuple = tuple(sorted(coupling_support))
    atom_pairs = tuple(
        _support_atom_pairs(atom, dimension, support_tuple)
        for atom in word
    )
    suffix_pairs = tuple(
        _support_word_pairs(word[position:], dimension, support_tuple)
        for position in range(len(word) + 1)
    )
    outgoing = tuple(
        {
            source: tuple(
                sorted(
                    target
                    for candidate_source, target in relation
                    if candidate_source == source
                )
            )
            for source in range(dimension)
        }
        for relation in atom_pairs
    )

    def visit(
        position: int,
        current: int,
        chain: tuple[int, ...],
        branch: _Branch,
    ) -> Iterator[_Branch]:
        if position == len(word):
            if current == column:
                yield _Branch(
                    branch.factors,
                    ((origin, chain),) + branch.contractions,
                )
            return
        atom = word[position]
        for next_index in outgoing[position].get(current, ()):
            if (next_index, column) not in suffix_pairs[position + 1]:
                continue
            for atom_branch in _enumerate_atom_element(
                atom,
                current,
                next_index,
                dimension,
                (*origin, "factor", position),
                coupling_support,
                True,
            ):
                yield from visit(
                    position + 1,
                    next_index,
                    (*chain, next_index),
                    _Branch(
                        branch.factors + atom_branch.factors,
                        branch.contractions + atom_branch.contractions,
                    ),
                )

    yield from visit(0, row, (row,), _Branch((), ()))


def _canonical_monomial_key(
    output_row: int,
    output_column: int,
    factors: Sequence[tuple[Any, ...]],
    dimension: int,
) -> tuple[Any, ...]:
    dimension = _validate_dimension(dimension)
    _validate_matrix_index(
        output_row,
        output_column,
        dimension,
        context="output",
    )
    phase = _phase_vectors(factors, dimension)
    canonical_factors = tuple(sorted(_canonical_factor(factor) for factor in factors))
    return (
        output_row,
        output_column,
        canonical_factors,
        phase[3],
        phase[4],
        phase[5],
    )


def monomial_key_to_json(key: tuple[Any, ...]) -> dict[str, Any]:
    """Return a deterministic, JSON-compatible scalar monomial key."""

    output_row, output_column, factors, phase_u, phase_v, phase_w = key
    dimension = _validate_dimension(len(phase_u))
    if len(phase_v) != dimension or len(phase_w) != dimension:
        raise ContractionScalarizationError(
            "monomial phase-vector dimensions do not match"
        )
    _validate_matrix_index(
        output_row,
        output_column,
        dimension,
        context="output",
    )
    factor_rows: list[dict[str, Any]] = []
    density_indices: list[tuple[int, int]] = []
    for factor in factors:
        if factor[0] == "coupling_operator":
            _validate_matrix_index(
                factor[1],
                factor[2],
                dimension,
                context="coupling operator",
            )
            factor_rows.append(
                {"kind": "coupling_operator", "matrix_index": list(factor[1:])}
            )
        elif factor[0] == "density_operator":
            _validate_matrix_index(
                factor[2],
                factor[3],
                dimension,
                context="density operator",
            )
            density_indices.append((factor[2], factor[3]))
            factor_rows.append(
                {
                    "kind": "density_operator",
                    "symbol": factor[1],
                    "matrix_index": list(factor[2:]),
                }
            )
        elif factor[0] == "kernel":
            _validate_matrix_index(
                factor[3],
                factor[4],
                dimension,
                context="Gamma",
            )
            factor_rows.append(
                {
                    "kind": "kernel",
                    "family": factor[1],
                    "support": factor[2],
                    "gamma_index": list(factor[3:]),
                }
            )
        else:
            raise ContractionScalarizationError(
                f"unknown canonical scalar factor {factor!r}"
            )
    if len(density_indices) != 1:
        raise ContractionScalarizationError(
            "every scalar monomial must contain exactly one density entry"
        )
    density_row, density_column = density_indices[0]
    return {
        "output_index": [output_row, output_column],
        "density_input_index": [density_row, density_column],
        "column_major_mapping": {
            "dimension": dimension,
            "output_flat": output_row + dimension * output_column,
            "input_flat": density_row + dimension * density_column,
            "formula": {
                "output_flat": "output_row+d*output_column",
                "input_flat": "rho_row+d*rho_column",
            },
        },
        "commutative_scalar_factors": factor_rows,
        "operator_phase_by_lag": {
            "u": list(phase_u),
            "v": list(phase_v),
            "w": list(phase_w),
        },
    }


@dataclass(frozen=True)
class ScalarContractionPath:
    """One explicit fixed-index scalar path through a matrix formal word."""

    output_row: int
    output_column: int
    factors: tuple[tuple[Any, ...], ...]
    contractions: tuple[
        tuple[tuple[str | int, ...], tuple[int, ...]], ...
    ]
    dimension: int

    @property
    def monomial_key(self) -> tuple[Any, ...]:
        return _canonical_monomial_key(
            self.output_row,
            self.output_column,
            self.factors,
            self.dimension,
        )

    def to_json(self) -> dict[str, Any]:
        phase = _phase_vectors(self.factors, self.dimension)
        return {
            "output_index": [self.output_row, self.output_column],
            "contractions": [
                {
                    "word_path": _origin_json(origin),
                    "index_chain": list(indices),
                }
                for origin, indices in self.contractions
            ],
            "ordered_scalar_factors": [
                _factor_to_json(factor) for factor in self.factors
            ],
            "canonical_monomial": monomial_key_to_json(self.monomial_key),
            "operator_phase_factorization": _phase_json(phase),
        }


def enumerate_contraction_paths(
    word: FormalWord,
    *,
    dimension: int = 2,
    coupling_support: frozenset[tuple[int, int]] | None = None,
    support_join_pruning: bool = True,
) -> Iterator[ScalarContractionPath]:
    """Enumerate all output entries and all contraction paths for ``word``."""

    dimension = _validate_dimension(dimension)
    if coupling_support is not None:
        for row, column in coupling_support:
            _validate_matrix_index(
                row,
                column,
                dimension,
                context="coupling support",
            )
    for output_row in range(dimension):
        for output_column in range(dimension):
            for branch in _enumerate_word_element(
                word,
                output_row,
                output_column,
                dimension,
                ("top_level_word",),
                coupling_support,
                support_join_pruning,
            ):
                yield ScalarContractionPath(
                    output_row=output_row,
                    output_column=output_column,
                    factors=branch.factors,
                    contractions=branch.contractions,
                    dimension=dimension,
                )


@dataclass(frozen=True)
class _WordScalarSummary:
    path_count: int
    monomials: tuple[tuple[tuple[Any, ...], int], ...]
    paths_sha256: str


@lru_cache(maxsize=None)
def _summarize_word_scalarization(
    word: FormalWord, dimension: int
) -> _WordScalarSummary:
    """Enumerate one distinct formal word once and cache its exact summary."""

    monomials: Counter[tuple[Any, ...]] = Counter()
    digest = hashlib.sha256()
    path_count = 0
    for ordinal, path in enumerate(
        enumerate_contraction_paths(word, dimension=dimension)
    ):
        key = path.monomial_key
        monomials[key] += 1
        digest.update(
            canonical_json([ordinal, path.contractions, key]).encode("utf-8")
        )
        digest.update(b"\n")
        path_count += 1
    return _WordScalarSummary(
        path_count=path_count,
        monomials=tuple(
            sorted(
                monomials.items(),
                key=lambda item: canonical_json(item[0]),
            )
        ),
        paths_sha256=digest.hexdigest(),
    )


@lru_cache(maxsize=None)
def _summarize_word_scalarization_with_coupling(
    word: FormalWord,
    dimension: int,
    coupling_support: tuple[tuple[int, int], ...],
) -> _WordScalarSummary:
    """Summarize only paths allowed by an exact structural-zero pattern."""

    support = frozenset(coupling_support)
    monomials: Counter[tuple[Any, ...]] = Counter()
    digest = hashlib.sha256()
    path_count = 0
    for ordinal, path in enumerate(
        enumerate_contraction_paths(
            word,
            dimension=dimension,
            coupling_support=support,
        )
    ):
        key = path.monomial_key
        monomials[key] += 1
        digest.update(
            canonical_json([ordinal, path.contractions, key]).encode("utf-8")
        )
        digest.update(b"\n")
        path_count += 1
    return _WordScalarSummary(
        path_count=path_count,
        monomials=tuple(
            sorted(
                monomials.items(),
                key=lambda item: canonical_json(item[0]),
            )
        ),
        paths_sha256=digest.hexdigest(),
    )


@lru_cache(maxsize=None)
def _summarize_word_scalarization_with_coupling_product(
    word: FormalWord,
    dimension: int,
    coupling_support: tuple[tuple[int, int], ...],
) -> _WordScalarSummary:
    """Compatibility control using the old d^k Cartesian support filter."""

    support = frozenset(coupling_support)
    monomials: Counter[tuple[Any, ...]] = Counter()
    digest = hashlib.sha256()
    path_count = 0
    for ordinal, path in enumerate(
        enumerate_contraction_paths(
            word,
            dimension=dimension,
            coupling_support=support,
            support_join_pruning=False,
        )
    ):
        key = path.monomial_key
        monomials[key] += 1
        digest.update(
            canonical_json([ordinal, path.contractions, key]).encode("utf-8")
        )
        digest.update(b"\n")
        path_count += 1
    return _WordScalarSummary(
        path_count=path_count,
        monomials=tuple(
            sorted(
                monomials.items(),
                key=lambda item: canonical_json(item[0]),
            )
        ),
        paths_sha256=digest.hexdigest(),
    )

