"""Classify scalar kernel products and count overlapping time dependencies."""

from __future__ import annotations

from collections import Counter

from typing import Any, Sequence

EXPECTED_SURVIVING_MONOMIALS = 36_928


EXPECTED_STRICT_MONOMIALS = 36_096


EXPECTED_OVERLAP_MONOMIALS = 832


MonomialKey = tuple[Any, ...]


ScalarMonomial = tuple[MonomialKey, int]


ScalarMonomialInventory = Sequence[ScalarMonomial]


class ExtendedEvaluatorError(ValueError):
    """Raised when a complete scalar polynomial cannot be evaluated exactly."""


def _kernel_supports(key: MonomialKey) -> tuple[str, ...]:
    return tuple(
        factor[2] for factor in key[2] if factor[0] == "kernel"
    )


def is_overlap_monomial(key: MonomialKey) -> bool:
    """Return whether ``key`` has exactly the residual UV/VW/V signature."""

    bases = [
        support[1:] if support.startswith("-") else support
        for support in _kernel_supports(key)
    ]
    return len(bases) == 3 and Counter(bases) == Counter(
        {"UV": 1, "VW": 1, "V": 1}
    )


def inventory_counts(
    monomials: ScalarMonomialInventory,
) -> dict[str, int]:
    """Return exact strict/overlap coverage counts for an inventory."""

    overlap = sum(is_overlap_monomial(key) for key, _ in monomials)
    return {
        "surviving": len(monomials),
        "strict": len(monomials) - overlap,
        "overlap": overlap,
        "coefficient_l1": sum(abs(coefficient) for _, coefficient in monomials),
    }


def assert_publication_appendix_f_inventory(
    monomials: ScalarMonomialInventory,
) -> None:
    """Require the complete frozen-publication Appendix-F inventory."""

    counts = inventory_counts(monomials)
    expected = {
        "surviving": EXPECTED_SURVIVING_MONOMIALS,
        "strict": EXPECTED_STRICT_MONOMIALS,
        "overlap": EXPECTED_OVERLAP_MONOMIALS,
        "coefficient_l1": 45_508,
    }
    if counts != expected:
        raise ExtendedEvaluatorError(
            f"incomplete Appendix-F scalar inventory: {counts!r}, "
            f"expected {expected!r}"
        )

