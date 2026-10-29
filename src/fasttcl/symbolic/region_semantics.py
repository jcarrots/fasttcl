"""Exact finite semantics for time-integration regions.

The internal variables ``1..m`` lie between the fixed lower endpoint ``0``
and current-time endpoint ``q = m + 1``.  A strict chamber is a permutation
``pi`` interpreted as

``0 < t_pi[0] < ... < t_pi[m - 1] < t_q``.

Supported bounds have the deliberately small grammar
``lower < variable < upper`` where all three entries are declared labels.
Every internal variable must have exactly one bound.  The resulting region is
checked in two independent ways:

* enumerate linear extensions of the bound DAG;
* evaluate the original bound predicates at an exact rational representative
  of every chamber.

The two incidence vectors must agree.  Equality is equality away from
time-coincidence boundaries; callers are responsible for the regularity
assumption described in ``docs/independent-derivation-plan.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from itertools import permutations
from math import factorial
from typing import Iterable, Mapping, Sequence, TypeAlias


class RegionSemanticsError(ValueError):
    """Raised when a region or chamber violates the exact finite model."""


Chamber: TypeAlias = tuple[int, ...]
Edge: TypeAlias = tuple[int, int]
ExactScalar: TypeAlias = int | Fraction
BoundInput: TypeAlias = (
    "RegionBound" | Mapping[str, object] | tuple[object, object, object]
)


def _require_plain_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RegionSemanticsError(f"{field_name} must be an integer label")
    return value


def _exact_fraction(value: object, *, field_name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise RegionSemanticsError(
            f"{field_name} must be an exact integer or Fraction"
        )
    return Fraction(value)


@lru_cache(maxsize=None)
def _chambers(internal_count: int) -> tuple[Chamber, ...]:
    return tuple(permutations(range(1, internal_count + 1)))


@dataclass(frozen=True, slots=True, order=True)
class RegionBound:
    """One restricted inequality ``lower < variable < upper``."""

    variable: int
    lower: int
    upper: int

    def __post_init__(self) -> None:
        for name in ("variable", "lower", "upper"):
            _require_plain_int(getattr(self, name), field_name=name)

    @classmethod
    def parse(cls, value: object) -> "RegionBound":
        """Parse a strict bound tuple or JSON-shaped mapping.

        Strings, symbolic expressions, ``min``/``max`` objects, and mappings
        with extra fields are intentionally unsupported.
        """

        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            required = {"variable", "lower", "upper"}
            if set(value) != required:
                raise RegionSemanticsError(
                    "a bound mapping must contain exactly "
                    "'variable', 'lower', and 'upper'"
                )
            return cls(
                variable=_require_plain_int(
                    value["variable"], field_name="variable"
                ),
                lower=_require_plain_int(value["lower"], field_name="lower"),
                upper=_require_plain_int(value["upper"], field_name="upper"),
            )
        if isinstance(value, (tuple, list)) and len(value) == 3:
            variable, lower, upper = value
            return cls(
                variable=_require_plain_int(
                    variable, field_name="variable"
                ),
                lower=_require_plain_int(lower, field_name="lower"),
                upper=_require_plain_int(upper, field_name="upper"),
            )
        raise RegionSemanticsError(
            "a bound must be RegionBound, a three-integer sequence, "
            "or an exact three-field mapping"
        )


@dataclass(frozen=True, slots=True)
class ChamberSpace:
    """The finite chamber universe for ``m`` internal time variables."""

    internal_count: int

    def __post_init__(self) -> None:
        count = _require_plain_int(
            self.internal_count, field_name="internal_count"
        )
        if count < 1:
            raise RegionSemanticsError("internal_count must be positive")

    @property
    def current_time(self) -> int:
        return self.internal_count + 1

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(range(self.current_time + 1))

    @property
    def internal_labels(self) -> tuple[int, ...]:
        return tuple(range(1, self.current_time))

    @property
    def chambers(self) -> tuple[Chamber, ...]:
        """All chambers in stable lexicographic order."""

        return _chambers(self.internal_count)

    @property
    def chamber_count(self) -> int:
        return factorial(self.internal_count)

    def validate_label(self, label: object) -> int:
        result = _require_plain_int(label, field_name="time label")
        if result not in range(self.current_time + 1):
            raise RegionSemanticsError(
                f"time label must lie in 0..{self.current_time}"
            )
        return result

    def validate_chamber(self, chamber: Sequence[object]) -> Chamber:
        if isinstance(chamber, (str, bytes)):
            raise RegionSemanticsError("a chamber must be a label permutation")
        result = tuple(
            _require_plain_int(value, field_name="chamber label")
            for value in chamber
        )
        if sorted(result) != list(self.internal_labels):
            raise RegionSemanticsError(
                "a chamber must contain every internal label exactly once"
            )
        return result

    def rational_representative(
        self, chamber: Sequence[object]
    ) -> tuple[Fraction, ...]:
        """Return exact coordinates indexed by physical time label.

        The fixed endpoints are 0 and 1.  The variable in chamber position
        ``k`` receives ``k / (m + 1)``.
        """

        valid = self.validate_chamber(chamber)
        values = [Fraction(0) for _ in self.labels]
        values[self.current_time] = Fraction(1)
        denominator = self.internal_count + 1
        for position, label in enumerate(valid, start=1):
            values[label] = Fraction(position, denominator)
        return tuple(values)

    def region(self, bounds: Iterable[object]) -> "IntegrationRegion":
        return IntegrationRegion(self, tuple(RegionBound.parse(x) for x in bounds))

    def pullback(
        self, chamber: Sequence[object]
    ) -> "PermutationPullback":
        return PermutationPullback(self, self.validate_chamber(chamber))


def _normalize_bounds(
    space: ChamberSpace, bounds: Iterable[object]
) -> tuple[RegionBound, ...]:
    parsed = tuple(RegionBound.parse(bound) for bound in bounds)
    if len(parsed) != space.internal_count:
        raise RegionSemanticsError(
            f"expected {space.internal_count} bounds, got {len(parsed)}"
        )
    by_variable: dict[int, RegionBound] = {}
    for bound in parsed:
        if bound.variable not in space.internal_labels:
            raise RegionSemanticsError(
                "a bound variable must be an internal time label"
            )
        space.validate_label(bound.lower)
        space.validate_label(bound.upper)
        if len({bound.variable, bound.lower, bound.upper}) != 3:
            raise RegionSemanticsError(
                "a bound variable and its endpoints must be distinct"
            )
        if bound.variable in by_variable:
            raise RegionSemanticsError(
                f"duplicate bound for variable {bound.variable}"
            )
        by_variable[bound.variable] = bound
    if set(by_variable) != set(space.internal_labels):
        raise RegionSemanticsError(
            "every internal variable must have exactly one bound"
        )
    return tuple(by_variable[label] for label in space.internal_labels)


def _transitive_closure(
    label_count: int, edges: Iterable[Edge]
) -> tuple[Edge, ...]:
    reach = [[False] * label_count for _ in range(label_count)]
    for lower, upper in edges:
        if lower == upper:
            raise RegionSemanticsError("an inequality cannot be self-directed")
        reach[lower][upper] = True
    for middle in range(label_count):
        for lower in range(label_count):
            if not reach[lower][middle]:
                continue
            for upper in range(label_count):
                if reach[middle][upper]:
                    reach[lower][upper] = True
    if any(reach[label][label] for label in range(label_count)):
        raise RegionSemanticsError("bound inequalities contain a cycle")
    return tuple(
        (lower, upper)
        for lower in range(label_count)
        for upper in range(label_count)
        if reach[lower][upper]
    )


def _inequality_graph(
    space: ChamberSpace, bounds: tuple[RegionBound, ...]
) -> tuple[Edge, ...]:
    q = space.current_time
    edges: set[Edge] = {(0, q)}
    for variable in space.internal_labels:
        edges.add((0, variable))
        edges.add((variable, q))
    for bound in bounds:
        edges.add((bound.lower, bound.variable))
        edges.add((bound.variable, bound.upper))
    return _transitive_closure(q + 1, edges)


def _positions(space: ChamberSpace, chamber: Chamber) -> tuple[int, ...]:
    result = [0] * (space.current_time + 1)
    result[0] = 0
    result[space.current_time] = space.internal_count + 1
    for position, label in enumerate(chamber, start=1):
        result[label] = position
    return tuple(result)


@dataclass(frozen=True, slots=True)
class IntegrationRegion:
    """A complete restricted bound list with exact chamber semantics."""

    space: ChamberSpace
    bounds: tuple[RegionBound, ...]
    normalized_edges: tuple[Edge, ...] = field(init=False)
    compatible_chambers: tuple[Chamber, ...] = field(init=False)
    incidence_vector: tuple[int, ...] = field(init=False)
    predicate_incidence_vector: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.space, ChamberSpace):
            raise TypeError("space must be a ChamberSpace")
        bounds = _normalize_bounds(self.space, self.bounds)
        edges = _inequality_graph(self.space, bounds)

        linear_vector: list[int] = []
        predicate_vector: list[int] = []
        compatible: list[Chamber] = []
        for chamber in self.space.chambers:
            positions = _positions(self.space, chamber)
            is_linear_extension = all(
                positions[lower] < positions[upper]
                for lower, upper in edges
            )
            linear_vector.append(int(is_linear_extension))
            if is_linear_extension:
                compatible.append(chamber)

            values = self.space.rational_representative(chamber)
            predicate_vector.append(
                int(
                    all(
                        values[bound.lower]
                        < values[bound.variable]
                        < values[bound.upper]
                        for bound in bounds
                    )
                )
            )

        if linear_vector != predicate_vector:
            raise AssertionError(
                "DAG linear extensions and rational predicate evaluation "
                "disagree"
            )

        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "normalized_edges", edges)
        object.__setattr__(
            self, "compatible_chambers", tuple(compatible)
        )
        object.__setattr__(
            self, "incidence_vector", tuple(linear_vector)
        )
        object.__setattr__(
            self,
            "predicate_incidence_vector",
            tuple(predicate_vector),
        )

    def contains_chamber(self, chamber: Sequence[object]) -> bool:
        valid = self.space.validate_chamber(chamber)
        positions = _positions(self.space, valid)
        return all(
            positions[lower] < positions[upper]
            for lower, upper in self.normalized_edges
        )

    def predicate_contains_chamber(
        self, chamber: Sequence[object]
    ) -> bool:
        """Evaluate the original bounds at an exact chamber point."""

        values = self.space.rational_representative(chamber)
        return all(
            values[bound.lower]
            < values[bound.variable]
            < values[bound.upper]
            for bound in self.bounds
        )

    @property
    def independent_oracles_agree(self) -> bool:
        return self.incidence_vector == self.predicate_incidence_vector

    def as_signed_vector(
        self, coefficient: ExactScalar = 1
    ) -> "SignedRegionVector":
        return SignedRegionVector.from_region(self, coefficient)


@dataclass(frozen=True, slots=True)
class SignedRegionVector:
    """An exact signed or rational combination of chamber indicators."""

    space: ChamberSpace
    values: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.space, ChamberSpace):
            raise TypeError("space must be a ChamberSpace")
        values = tuple(
            _exact_fraction(value, field_name="region-vector coefficient")
            for value in self.values
        )
        if len(values) != self.space.chamber_count:
            raise RegionSemanticsError(
                "region-vector length does not match the chamber space"
            )
        object.__setattr__(self, "values", values)

    @classmethod
    def zero(cls, space: ChamberSpace) -> "SignedRegionVector":
        return cls(space, (Fraction(0),) * space.chamber_count)

    @classmethod
    def from_region(
        cls,
        region: IntegrationRegion,
        coefficient: ExactScalar = 1,
    ) -> "SignedRegionVector":
        scale = _exact_fraction(
            coefficient, field_name="region coefficient"
        )
        return cls(
            region.space,
            tuple(scale * value for value in region.incidence_vector),
        )

    def _require_same_space(
        self, other: "SignedRegionVector"
    ) -> None:
        if self.space != other.space:
            raise RegionSemanticsError(
                "signed region vectors belong to different chamber spaces"
            )

    def __add__(
        self, other: "SignedRegionVector"
    ) -> "SignedRegionVector":
        if not isinstance(other, SignedRegionVector):
            return NotImplemented
        self._require_same_space(other)
        return SignedRegionVector(
            self.space,
            tuple(
                left + right
                for left, right in zip(
                    self.values, other.values, strict=True
                )
            ),
        )

    def __sub__(
        self, other: "SignedRegionVector"
    ) -> "SignedRegionVector":
        if not isinstance(other, SignedRegionVector):
            return NotImplemented
        return self + (-other)

    def __neg__(self) -> "SignedRegionVector":
        return SignedRegionVector(
            self.space, tuple(-value for value in self.values)
        )

    def __mul__(self, scalar_value: ExactScalar) -> "SignedRegionVector":
        scale = _exact_fraction(
            scalar_value, field_name="region-vector scale"
        )
        return SignedRegionVector(
            self.space, tuple(scale * value for value in self.values)
        )

    def __rmul__(self, scalar_value: ExactScalar) -> "SignedRegionVector":
        return self * scalar_value

    @property
    def is_zero(self) -> bool:
        return all(value == 0 for value in self.values)

    @property
    def support(self) -> tuple[tuple[Chamber, Fraction], ...]:
        return tuple(
            (chamber, coefficient)
            for chamber, coefficient in zip(
                self.space.chambers, self.values, strict=True
            )
            if coefficient != 0
        )

    def as_integer_tuple(self) -> tuple[int, ...]:
        if any(value.denominator != 1 for value in self.values):
            raise RegionSemanticsError(
                "region vector contains non-integer coefficients"
            )
        return tuple(value.numerator for value in self.values)


def signed_region_sum(
    space: ChamberSpace,
    terms: Iterable[tuple[ExactScalar, IntegrationRegion]],
) -> SignedRegionVector:
    """Return the exact vector sum of signed regions."""

    result = SignedRegionVector.zero(space)
    for coefficient, region in terms:
        if region.space != space:
            raise RegionSemanticsError(
                "a signed term belongs to a different chamber space"
            )
        result = result + region.as_signed_vector(coefficient)
    return result


def signed_region_residual(
    parent: IntegrationRegion,
    children: Iterable[tuple[ExactScalar, IntegrationRegion]],
) -> SignedRegionVector:
    """Return ``parent - sum(coefficient * child)`` exactly."""

    return parent.as_signed_vector() - signed_region_sum(
        parent.space, children
    )


def _permutation_sign(chamber: Chamber) -> int:
    inversions = sum(
        left > right
        for index, left in enumerate(chamber)
        for right in chamber[index + 1 :]
    )
    return -1 if inversions % 2 else 1


@dataclass(frozen=True, slots=True)
class PermutationPullback:
    """The exact chamber-to-canonical label and Jacobian witness.

    For chamber ``pi``, ``physical_to_canonical[pi[k - 1]] == k``.
    The fixed endpoints map to themselves.
    """

    space: ChamberSpace
    chamber: Chamber
    physical_to_canonical: tuple[int, ...] = field(init=False)
    canonical_to_physical: tuple[int, ...] = field(init=False)
    jacobian_matrix: tuple[tuple[int, ...], ...] = field(init=False)
    jacobian_determinant: int = field(init=False)
    jacobian_absolute_value: int = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.space, ChamberSpace):
            raise TypeError("space must be a ChamberSpace")
        chamber = self.space.validate_chamber(self.chamber)
        q = self.space.current_time

        physical_to_canonical = [0] * (q + 1)
        physical_to_canonical[0] = 0
        physical_to_canonical[q] = q
        canonical_to_physical = [0, *chamber, q]
        for canonical, physical in enumerate(chamber, start=1):
            physical_to_canonical[physical] = canonical

        matrix = tuple(
            tuple(
                int(chamber[canonical - 1] == physical)
                for canonical in self.space.internal_labels
            )
            for physical in self.space.internal_labels
        )
        determinant = _permutation_sign(chamber)

        if sorted(physical_to_canonical) != list(range(q + 1)):
            raise AssertionError("pullback label map is not bijective")
        if (
            physical_to_canonical[0] != 0
            or physical_to_canonical[q] != q
        ):
            raise AssertionError("pullback does not fix both endpoints")
        if abs(determinant) != 1:
            raise AssertionError("permutation Jacobian is not unimodular")

        object.__setattr__(self, "chamber", chamber)
        object.__setattr__(
            self,
            "physical_to_canonical",
            tuple(physical_to_canonical),
        )
        object.__setattr__(
            self,
            "canonical_to_physical",
            tuple(canonical_to_physical),
        )
        object.__setattr__(self, "jacobian_matrix", matrix)
        object.__setattr__(
            self, "jacobian_determinant", determinant
        )
        object.__setattr__(self, "jacobian_absolute_value", 1)

    @property
    def endpoint_map(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (
            (0, self.physical_to_canonical[0]),
            (
                self.space.current_time,
                self.physical_to_canonical[self.space.current_time],
            ),
        )

    @property
    def is_bijective(self) -> bool:
        return sorted(self.physical_to_canonical) == list(
            self.space.labels
        )

    def apply_label(self, label: object) -> int:
        valid = self.space.validate_label(label)
        return self.physical_to_canonical[valid]

    def invert_label(self, label: object) -> int:
        valid = self.space.validate_label(label)
        return self.canonical_to_physical[valid]

    def apply_chamber(self, chamber: Sequence[object]) -> Chamber:
        valid = self.space.validate_chamber(chamber)
        return tuple(self.apply_label(label) for label in valid)

    def apply_bound(self, bound: object) -> RegionBound:
        parsed = RegionBound.parse(bound)
        return RegionBound(
            variable=self.apply_label(parsed.variable),
            lower=self.apply_label(parsed.lower),
            upper=self.apply_label(parsed.upper),
        )

    def apply_bounds(
        self, bounds: Iterable[object]
    ) -> tuple[RegionBound, ...]:
        return tuple(self.apply_bound(bound) for bound in bounds)

    def apply_correlations(
        self,
        correlations: Iterable[Sequence[object]],
        *,
        canonical_sort: bool = True,
    ) -> tuple[tuple[int, int], ...]:
        """Relabel oriented pairs without reversing either endpoint order."""

        result: list[tuple[int, int]] = []
        for pair in correlations:
            if isinstance(pair, (str, bytes)) or len(pair) != 2:
                raise RegionSemanticsError(
                    "a correlation must have exactly two oriented endpoints"
                )
            left, right = pair
            result.append(
                (self.apply_label(left), self.apply_label(right))
            )
        if canonical_sort:
            result.sort()
        return tuple(result)

    def apply_region(self, region: IntegrationRegion) -> IntegrationRegion:
        if region.space != self.space:
            raise RegionSemanticsError(
                "region and pullback belong to different chamber spaces"
            )
        return self.space.region(self.apply_bounds(region.bounds))


__all__ = [
    "BoundInput",
    "Chamber",
    "ChamberSpace",
    "Edge",
    "ExactScalar",
    "IntegrationRegion",
    "PermutationPullback",
    "RegionBound",
    "RegionSemanticsError",
    "SignedRegionVector",
    "signed_region_residual",
    "signed_region_sum",
]
