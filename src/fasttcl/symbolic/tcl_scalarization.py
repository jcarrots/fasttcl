"""Exact scalar compilation of Hadamard-reduced coefficients at any even order.

Matrix products enumerate their contraction indices, while Hadamard products
share the output indices of their operands.  Scalar histories retain integer
energy coefficients for every gap and a polynomial in the entries of the
coupling matrix.  No bath symmetry, numerical tolerance, or specialized
convolution algorithm is used.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
from typing import Any, Callable, Iterable, Sequence, TypeAlias, cast

from .ast import ASTNode, ASTValidationError, validate_ast
from .formal_bilinear import FormalAtom, FormalWord, expand_formal_bilinear
from .tcl_order import TCLOrder


class TCLScalarizationError(ValueError):
    """A reduced expression or scalar inventory violates its declared order."""


@dataclass(frozen=True, order=True)
class IntervalSupport:
    """The signed sum of gaps ``start <= j < stop`` (zero-based)."""

    sign: int
    start: int
    stop: int

    def __post_init__(self) -> None:
        if (
            type(self.sign) is not int
            or self.sign not in (-1, 1)
            or type(self.start) is not int
            or type(self.stop) is not int
            or not 0 <= self.start < self.stop
        ):
            raise TCLScalarizationError("invalid signed contiguous gap interval")


@dataclass(frozen=True, order=True)
class GammaFactor:
    """A Gamma matrix entry; transpose is represented by exchanged indices."""

    support: IntervalSupport
    row: int
    column: int
    conjugated: bool = False


@dataclass(frozen=True, order=True)
class CouplingFactor:
    """One entry of A, with explicit conjugation for the sharp operation."""

    row: int
    column: int
    conjugated: bool = False


CouplingMonomial: TypeAlias = tuple[CouplingFactor, ...]
CouplingPolynomial: TypeAlias = tuple[tuple[CouplingMonomial, int], ...]
GapPhases: TypeAlias = tuple[tuple[int, ...], ...]


@dataclass(frozen=True, order=True)
class ScalarSignature:
    """A temporal history and its input/output matrix channel."""

    output_row: int
    output_column: int
    input_row: int
    input_column: int
    phases: GapPhases
    kernels: tuple[GammaFactor, ...]


@dataclass(frozen=True)
class ScalarTerm:
    signature: ScalarSignature
    coefficients: CouplingPolynomial


@dataclass(frozen=True)
class ScalarCompilation:
    """Collected scalar histories with exact polynomial system coefficients."""

    order: int
    dimension: int
    stage: str
    terms: tuple[ScalarTerm, ...]
    coupling_support: frozenset[tuple[int, int]] | None = None

    @property
    def signature_count(self) -> int:
        return len(self.terms)

    @property
    def monomial_count(self) -> int:
        return sum(len(term.coefficients) for term in self.terms)

    def outer_raw(self) -> ScalarCompilation:
        """Expand ``-[A(t),S]`` with an exact additional coupling factor."""

        return outer_raw_scalar_terms(self)

    def hermitian_completion(self) -> ScalarCompilation:
        """Return ``T + sharp(T)`` as a complex-linear superoperator."""

        return complete_scalar_terms(self)

    def records(self) -> tuple[dict[str, Any], ...]:
        return scalar_records(self)

    def payload(self) -> dict[str, Any]:
        return scalar_payload(self)


@dataclass(frozen=True, order=True)
class _Path:
    density: tuple[int, int] | None
    phases: GapPhases
    kernels: tuple[GammaFactor, ...]
    coupling: CouplingMonomial


_Collected: TypeAlias = dict[ScalarSignature, Counter[CouplingMonomial]]


def _positive_dimension(dimension: int) -> int:
    if type(dimension) is not int or dimension < 1:
        raise TCLScalarizationError("dimension must be a positive integer")
    return dimension


def _combine(left: _Path, right: _Path) -> _Path:
    if left.density is not None and right.density is not None:
        raise TCLScalarizationError("a scalar path contains two density entries")
    return _Path(
        density=left.density if left.density is not None else right.density,
        phases=tuple(
            tuple(a + b for a, b in zip(x, y, strict=True))
            for x, y in zip(left.phases, right.phases, strict=True)
        ),
        kernels=tuple(sorted(left.kernels + right.kernels)),
        coupling=tuple(sorted(left.coupling + right.coupling)),
    )


def _degrees(word: FormalWord) -> tuple[int, int, int]:
    operators = kernels = densities = 0
    for atom in word:
        kind = atom[0]
        if kind == "operator":
            operators += 1
        elif kind == "kernel":
            kernels += 1
        elif kind == "density_operator":
            densities += 1
        elif kind == "hadamard":
            for child in atom[1:]:
                a, b, c = _degrees(cast(FormalWord, child))
                operators += a
                kernels += b
                densities += c
        else:
            raise TCLScalarizationError(f"unsupported formal atom {atom!r}")
    return operators, kernels, densities


class _Compiler:
    def __init__(
        self,
        specification: TCLOrder,
        dimension: int,
        coupling_support: frozenset[tuple[int, int]] | None,
    ) -> None:
        self.specification = specification
        self.dimension = dimension
        self.coupling_support = coupling_support
        self.zero_phases = tuple(
            (0,) * dimension for _ in range(specification.pair_grade)
        )
        self.empty = _Path(None, self.zero_phases, (), ())
        # Caches are scoped to one compilation, not retained across orders/models.
        self.atom_paths = lru_cache(maxsize=None)(self._atom_paths)
        self.word_paths = lru_cache(maxsize=None)(self._word_paths)

    def support(self, time_form: tuple[tuple[str, int], ...]) -> IntervalSupport:
        vector = [0] * self.specification.pair_grade
        for symbol, coefficient in time_form:
            try:
                basis = self.specification.time_to_gap[symbol]
            except KeyError as exc:
                raise TCLScalarizationError(
                    f"unknown TCL{self.specification.order} time symbol {symbol!r}"
                ) from exc
            for index, value in enumerate(basis):
                vector[index] += coefficient * value
        nonzero = [index for index, value in enumerate(vector) if value]
        if not nonzero:
            raise TCLScalarizationError("Gamma(0) must vanish before scalarization")
        start, stop = nonzero[0], nonzero[-1] + 1
        sign = vector[start]
        if sign not in (-1, 1) or any(
            vector[index] != sign for index in range(start, stop)
        ):
            raise TCLScalarizationError(
                f"Gamma argument is not a signed contiguous gap interval: {vector}"
            )
        return IntervalSupport(sign, start, stop)

    def _atom_paths(
        self, atom: FormalAtom, row: int, column: int
    ) -> tuple[tuple[_Path, int], ...]:
        kind = atom[0]
        if kind == "operator":
            label = cast(str, atom[1])
            try:
                slot = self.specification.operator_slots[label]
            except KeyError as exc:
                raise TCLScalarizationError(
                    f"unknown TCL{self.specification.order} operator label {label!r}"
                ) from exc
            if self.coupling_support is not None and (row, column) not in self.coupling_support:
                return ()
            phases = [list(gap) for gap in self.zero_phases]
            for gap in range(slot, self.specification.pair_grade):
                phases[gap][row] += 1
                phases[gap][column] -= 1
            return ((
                _Path(None, tuple(tuple(gap) for gap in phases), (),
                      (CouplingFactor(row, column),)),
                1,
            ),)
        if kind == "density_operator":
            if atom[1] != "rho":
                raise TCLScalarizationError(f"unknown density symbol {atom[1]!r}")
            return ((_Path((row, column), self.zero_phases, (), ()), 1),)
        if kind == "kernel":
            if len(atom) != 4 or atom[1] != "Gamma":
                raise TCLScalarizationError(f"invalid Gamma atom {atom!r}")
            gamma_row, gamma_column = (column, row) if atom[2] else (row, column)
            support = self.support(cast(tuple[tuple[str, int], ...], atom[3]))
            return ((
                _Path(None, self.zero_phases,
                      (GammaFactor(support, gamma_row, gamma_column),), ()),
                1,
            ),)
        if kind == "hadamard":
            collected: Counter[_Path] = Counter()
            for left, a in self.word_paths(cast(FormalWord, atom[1]), row, column):
                for right, b in self.word_paths(cast(FormalWord, atom[2]), row, column):
                    collected[_combine(left, right)] += a * b
            return tuple(collected.items())
        raise TCLScalarizationError(f"unsupported formal atom {atom!r}")

    def _word_paths(
        self, word: FormalWord, row: int, column: int
    ) -> tuple[tuple[_Path, int], ...]:
        if not word:
            return ((self.empty, 1),) if row == column else ()
        # Recurse by matrix-product suffix so structural zeros prune before
        # enumerating an entire Cartesian product of contraction indices.
        if len(word) == 1:
            return self.atom_paths(word[0], row, column)
        collected: Counter[_Path] = Counter()
        for intermediate in range(self.dimension):
            left_paths = self.atom_paths(word[0], row, intermediate)
            if not left_paths:
                continue
            right_paths = self.word_paths(word[1:], intermediate, column)
            for left, a in left_paths:
                for right, b in right_paths:
                    collected[_combine(left, right)] += a * b
        return tuple(collected.items())


def _result(
    collected: _Collected, *, order: int, dimension: int, stage: str,
    coupling_support: frozenset[tuple[int, int]] | None = None,
) -> ScalarCompilation:
    terms: list[ScalarTerm] = []
    for signature, coefficients in sorted(collected.items()):
        nonzero = tuple(sorted((monomial, value) for monomial, value in coefficients.items() if value))
        if nonzero:
            terms.append(ScalarTerm(signature, nonzero))
    return ScalarCompilation(order, dimension, stage, tuple(terms), coupling_support)


def _add(
    collected: _Collected,
    signature: ScalarSignature,
    monomial: CouplingMonomial,
    coefficient: int,
) -> None:
    collected.setdefault(signature, Counter())[monomial] += coefficient


def compile_scalar_terms(
    expressions: Iterable[tuple[int, ASTNode]],
    *,
    order: int,
    dimension: int,
    coupling_support: frozenset[tuple[int, int]] | None = None,
) -> ScalarCompilation:
    """Compile reduced pre-wrapper ASTs at any even order and finite dimension.

    Each nonzero formal word must contain ``order-1`` coupling operators,
    ``order//2`` Gamma factors and exactly one density operator.  Coupling
    entries remain symbolic.  ``coupling_support`` optionally declares exact
    structural zeros; no numerical threshold is used.
    """

    specification = TCLOrder(order)
    _positive_dimension(dimension)
    if coupling_support is not None:
        if not isinstance(coupling_support, frozenset) or any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(type(index) is not int or not 0 <= index < dimension for index in pair)
            for pair in coupling_support
        ):
            raise TCLScalarizationError("coupling_support must be a frozenset of valid matrix-index pairs")
    compiler = _Compiler(specification, dimension, coupling_support)
    words: Counter[FormalWord] = Counter()
    for coefficient, expression in expressions:
        if type(coefficient) is not int:
            raise TCLScalarizationError("source coefficients must be integers")
        try:
            validate_ast(expression, order=specification)
        except ASTValidationError as exc:
            raise TCLScalarizationError(str(exc)) from exc
        words.update(expand_formal_bilinear(expression, coefficient=coefficient))
    collected: _Collected = {}
    expected = (order - 1, specification.pair_grade, 1)
    for word, coefficient in words.items():
        if not coefficient:
            continue
        actual = _degrees(word)
        if actual != expected:
            raise TCLScalarizationError(
                f"TCL{order} pre-wrapper degree must be {expected} "
                f"(operators, kernels, density), got {actual}"
            )
        for row, column in product(range(dimension), repeat=2):
            for path, multiplicity in compiler.word_paths(word, row, column):
                if path.density is None:
                    raise TCLScalarizationError("word path has no density entry")
                signature = ScalarSignature(
                    row, column, *path.density, path.phases, path.kernels
                )
                _add(collected, signature, path.coupling, coefficient * multiplicity)
    return _result(collected, order=order, dimension=dimension, stage="prewrapper",
                   coupling_support=coupling_support)


def _shift_phases(phases: GapPhases, row: int, column: int) -> GapPhases:
    result = [list(gap) for gap in phases]
    for gap in result:
        gap[row] += 1
        gap[column] -= 1
    return tuple(tuple(gap) for gap in result)


def outer_raw_scalar_terms(source: ScalarCompilation) -> ScalarCompilation:
    """Apply the raw outer commutator without assuming a special coupling."""

    if source.stage != "prewrapper":
        raise TCLScalarizationError("outer commutator requires a prewrapper compilation")
    collected: _Collected = {}
    for term in source.terms:
        s = term.signature
        for index in range(source.dimension):
            left = ScalarSignature(
                index, s.output_column, s.input_row, s.input_column,
                _shift_phases(s.phases, index, s.output_row), s.kernels,
            )
            right = ScalarSignature(
                s.output_row, index, s.input_row, s.input_column,
                _shift_phases(s.phases, s.output_column, index), s.kernels,
            )
            for monomial, coefficient in term.coefficients:
                if source.coupling_support is None or (index, s.output_row) in source.coupling_support:
                    _add(collected, left,
                         tuple(sorted(monomial + (CouplingFactor(index, s.output_row),))),
                         -coefficient)
                if source.coupling_support is None or (s.output_column, index) in source.coupling_support:
                    _add(collected, right,
                         tuple(sorted(monomial + (CouplingFactor(s.output_column, index),))),
                         coefficient)
    return _result(collected, order=source.order, dimension=source.dimension, stage="outer_raw",
                   coupling_support=source.coupling_support)


def complete_scalar_terms(raw: ScalarCompilation) -> ScalarCompilation:
    """Apply ``T + J*conj(T)*J`` to matrix-unit coefficients.

    Both the input and output matrix indices are exchanged in the sharp
    term.  This is complex-linear in a general input matrix; taking the
    adjoint of ``T(rho)`` would only be valid for Hermitian inputs.
    """

    if raw.stage != "outer_raw":
        raise TCLScalarizationError("Hermitian completion requires outer_raw terms")
    collected: _Collected = {}
    for term in raw.terms:
        s = term.signature
        sharp = ScalarSignature(
            s.output_column, s.output_row, s.input_column, s.input_row,
            tuple(tuple(-value for value in gap) for gap in s.phases),
            tuple(sorted(GammaFactor(k.support, k.row, k.column, not k.conjugated)
                         for k in s.kernels)),
        )
        for monomial, coefficient in term.coefficients:
            _add(collected, s, monomial, coefficient)
            conjugate = tuple(sorted(
                CouplingFactor(f.row, f.column, not f.conjugated) for f in monomial
            ))
            _add(collected, sharp, conjugate, coefficient)
    return _result(collected, order=raw.order, dimension=raw.dimension, stage="completed",
                   coupling_support=raw.coupling_support)


def scalar_records(compilation: ScalarCompilation) -> tuple[dict[str, Any], ...]:
    """Return deterministic JSON-compatible records with explicit conventions."""

    records: list[dict[str, Any]] = []
    for index, term in enumerate(compilation.terms, 1):
        s = term.signature
        records.append({
            "schema_version": "tcl-scalar-term-v1",
            "id": f"TCL{compilation.order}-{compilation.stage}-{index:06d}",
            "order": compilation.order,
            "dimension": compilation.dimension,
            "stage": compilation.stage,
            "output_index": [s.output_row, s.output_column],
            "input_index": [s.input_row, s.input_column],
            "gap_energy_coefficients": [list(gap) for gap in s.phases],
            "kernels": [{
                "support": {"sign": k.support.sign, "start": k.support.start, "stop": k.support.stop},
                "gamma_index": [k.row, k.column],
                "conjugated": k.conjugated,
            } for k in s.kernels],
            "coupling_polynomial": [{
                "coefficient": coefficient,
                "factors": [{"index": [f.row, f.column], "conjugated": f.conjugated}
                            for f in monomial],
            } for monomial, coefficient in term.coefficients],
        })
    return tuple(records)


def scalar_payload(compilation: ScalarCompilation) -> dict[str, Any]:
    """Return a content-bound JSON artifact, including empty compilations."""

    payload: dict[str, Any] = {
        "schema_version": "tcl-scalar-compilation-v1",
        "order": compilation.order,
        "dimension": compilation.dimension,
        "stage": compilation.stage,
        "coupling_support": None if compilation.coupling_support is None else [
            list(pair) for pair in sorted(compilation.coupling_support)
        ],
        "conventions": {
            "phase": "exp(i*sum_j gap[j]*sum_r energy_coefficients[j][r]*E[r]))",
            "support": "sign*sum(gaps[start:stop]); zero-based, stop-exclusive",
            "vectorization": "column-major; matrix_index=row+dimension*column",
            "coupling": "symbolic entries of A; repeated factors are powers",
        },
        "records": list(scalar_records(compilation)),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    payload["sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def evaluate_scalar_terms(
    compilation: ScalarCompilation,
    *,
    energies: Sequence[float],
    coupling: Any,
    gaps: Sequence[float],
    gamma: Callable[[float, int, int], complex],
) -> Any:
    """Slow literal evaluation at one gap point, for validation and inspection.

    ``gamma(argument, row, column)`` supplies finite signed-time Gamma entries.
    The result is a column-major ``d**2`` by ``d**2`` superoperator.  This
    evaluates the reduced integrand and does not integrate over the simplex.
    Gaps must be nonnegative, and any declared coupling structural zeros must
    be exactly zero in the supplied matrix.
    """

    import numpy as np

    dimension = compilation.dimension
    try:
        energy_values = np.asarray(energies, dtype=complex)
        gap_values = np.asarray(gaps, dtype=complex)
        coupling_array = np.asarray(coupling, dtype=complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TCLScalarizationError("energies, gaps and coupling must be numeric arrays") from exc
    if not np.isfinite(energy_values).all() or np.any(energy_values.imag != 0):
        raise TCLScalarizationError("energies must be finite and real")
    if not np.isfinite(gap_values).all() or np.any(gap_values.imag != 0):
        raise TCLScalarizationError("gaps must be finite and real")
    energy_array = energy_values.real
    gap_array = gap_values.real
    if energy_array.shape != (dimension,) or gap_array.shape != (compilation.order // 2,):
        raise TCLScalarizationError("energy or gap vector has the wrong dimension")
    if np.any(gap_array < 0):
        raise TCLScalarizationError("gaps must be nonnegative")
    if coupling_array.shape != (dimension, dimension):
        raise TCLScalarizationError("coupling matrix has the wrong dimension")
    if not np.isfinite(coupling_array).all():
        raise TCLScalarizationError("coupling matrix must be finite")
    if compilation.coupling_support is not None and any(
        coupling_array[row, column] != 0
        for row, column in product(range(dimension), repeat=2)
        if (row, column) not in compilation.coupling_support
    ):
        raise TCLScalarizationError("coupling matrix violates the declared exact coupling_support")
    result = np.zeros((dimension * dimension, dimension * dimension), dtype=complex)
    for term in compilation.terms:
        s = term.signature
        static = 0j
        for monomial, coefficient in term.coefficients:
            value = complex(coefficient)
            for factor in monomial:
                entry = coupling_array[factor.row, factor.column]
                value *= entry.conjugate() if factor.conjugated else entry
            static += value
        temporal = np.exp(1j * float(gap_array @ np.asarray(s.phases) @ energy_array))
        for factor in s.kernels:
            argument = factor.support.sign * float(gap_array[factor.support.start:factor.support.stop].sum())
            value = complex(gamma(argument, factor.row, factor.column))
            if not np.isfinite(value):
                raise TCLScalarizationError("Gamma callback must return finite values")
            temporal *= value.conjugate() if factor.conjugated else value
        result[s.output_row + dimension * s.output_column,
               s.input_row + dimension * s.input_column] += static * temporal
    return result


__all__ = [
    "CouplingFactor", "CouplingMonomial", "CouplingPolynomial", "GammaFactor",
    "GapPhases", "IntervalSupport", "ScalarCompilation", "ScalarSignature",
    "ScalarTerm", "TCLScalarizationError", "compile_scalar_terms",
    "complete_scalar_terms", "evaluate_scalar_terms", "outer_raw_scalar_terms",
    "scalar_payload", "scalar_records",
]
