"""CPU runtime extracted from the canonical TCL6 Numerical source; see data/source-provenance.json."""

from __future__ import annotations

from dataclasses import dataclass

import hashlib

from types import MappingProxyType

from typing import Any, Mapping, Sequence

from .convolution_runtime import _numpy

from .io_utils import canonical_json

class PhysicalSpecializationError(ValueError):
    """Raised when exact model structure cannot be certified."""


@dataclass(frozen=True)
class ExactCouplingPattern:
    """Bit-exact nonzero support of one square coupling matrix."""

    dimension: int
    nonzero_entries: frozenset[tuple[int, int]]
    zero_entries: frozenset[tuple[int, int]]
    digest: str

    @property
    def is_dense(self) -> bool:
        return len(self.nonzero_entries) == self.dimension**2

    @property
    def report(self) -> dict[str, Any]:
        return {
            "profile": "exact-coupling-zero-pattern-v1",
            "dimension": self.dimension,
            "nonzero_entry_count": len(self.nonzero_entries),
            "zero_entry_count": len(self.zero_entries),
            "density": len(self.nonzero_entries) / self.dimension**2,
            "nonzero_entries": [list(value) for value in sorted(self.nonzero_entries)],
            "zero_entries": [list(value) for value in sorted(self.zero_entries)],
            "digest": self.digest,
            "zero_policy": "complex value equals exactly 0+0j; no tolerance",
        }


@dataclass(frozen=True)
class ExactBohrGapPartition:
    """Bit-exact partition of oriented matrix entries by energy difference."""

    dimension: int
    pair_to_class: Mapping[tuple[int, int], int]
    class_tokens: tuple[str, ...]
    digest: str

    @property
    def report(self) -> dict[str, Any]:
        members: dict[int, list[list[int]]] = {
            index: [] for index in range(len(self.class_tokens))
        }
        for pair, class_index in sorted(self.pair_to_class.items()):
            members[class_index].append(list(pair))
        return {
            "profile": "exact-bohr-gap-partition-v1",
            "dimension": self.dimension,
            "oriented_pair_count": self.dimension**2,
            "unique_gap_count": len(self.class_tokens),
            "gap_classes": [
                {
                    "class_index": index,
                    "float_hex": token,
                    "members": members[index],
                }
                for index, token in enumerate(self.class_tokens)
            ],
            "digest": self.digest,
            "equality_policy": "bit-identical float64 energy differences; no tolerance",
        }


def exact_coupling_pattern(
    matrix: Sequence[Sequence[complex]],
) -> ExactCouplingPattern:
    """Return the exact structural-zero certificate for ``matrix``."""

    np = _numpy()
    values = np.asarray(matrix, dtype=np.complex128)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise PhysicalSpecializationError("coupling matrix must be square")
    if values.shape[0] < 1:
        raise PhysicalSpecializationError("coupling matrix cannot be empty")
    if not np.isfinite(values.real).all() or not np.isfinite(values.imag).all():
        raise PhysicalSpecializationError("coupling matrix contains a nonfinite value")
    dimension = int(values.shape[0])
    all_entries = frozenset(
        (row, column)
        for row in range(dimension)
        for column in range(dimension)
    )
    nonzero = frozenset(
        pair for pair in all_entries if complex(values[pair]) != 0j
    )
    zero = all_entries - nonzero
    payload = {
        "dimension": dimension,
        "nonzero_entries": sorted(nonzero),
        "zero_entries": sorted(zero),
    }
    digest = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return ExactCouplingPattern(
        dimension=dimension,
        nonzero_entries=nonzero,
        zero_entries=zero,
        digest=digest,
    )


def exact_bohr_gap_partition(
    energies: Sequence[float],
) -> ExactBohrGapPartition:
    """Partition ``(row,column)`` pairs by exact float64 ``E_row-E_column``."""

    np = _numpy()
    values = np.asarray(energies, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise PhysicalSpecializationError("energies must be a nonempty vector")
    if not np.isfinite(values).all():
        raise PhysicalSpecializationError("energies contain a nonfinite value")
    dimension = int(values.size)
    pair_tokens: dict[tuple[int, int], str] = {}
    for row in range(dimension):
        for column in range(dimension):
            difference = float(values[row] - values[column])
            if difference == 0.0:
                difference = 0.0
            pair_tokens[(row, column)] = difference.hex()
    class_tokens = tuple(sorted(set(pair_tokens.values())))
    token_to_class = {
        token: index for index, token in enumerate(class_tokens)
    }
    pair_to_class = {
        pair: token_to_class[token]
        for pair, token in pair_tokens.items()
    }
    payload = {
        "dimension": dimension,
        "class_tokens": class_tokens,
        "pair_to_class": sorted(pair_to_class.items()),
    }
    digest = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return ExactBohrGapPartition(
        dimension=dimension,
        pair_to_class=MappingProxyType(pair_to_class),
        class_tokens=class_tokens,
        digest=digest,
    )

