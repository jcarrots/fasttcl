"""Order and coordinate conventions shared by even-order TCL compilers."""

from __future__ import annotations

from dataclasses import dataclass
import re


def validate_even_order(order: int) -> int:
    """Return a positive even order, rejecting booleans and non-integers."""

    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError("order must be an integer")
    if order < 2 or order % 2:
        raise ValueError("order must be a positive even integer")
    return order


def canonical_dummy_label(index: int) -> str:
    """Return zero-based labels ``a,...,z,aa,...`` without an order cap."""

    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("dummy index must be an integer")
    if index < 0:
        raise ValueError("dummy index must be nonnegative")
    letters: list[str] = []
    value = index + 1
    while value:
        value, digit = divmod(value - 1, 26)
        letters.append(chr(ord("a") + digit))
    return "".join(reversed(letters))


def is_time_symbol(symbol: object) -> bool:
    """Whether a value is an endpoint or a canonical retained-time symbol."""

    return isinstance(symbol, str) and (
        symbol in {"0", "t"} or re.fullmatch(r"t_[a-z]+", symbol) is not None
    )


@dataclass(frozen=True, slots=True)
class TCLOrder:
    """Coordinates for order ``2N`` with ``N`` gaps and ``N-1`` retained times.

    Slots are current time, then retained times from latest to earliest.
    The gap vector is ``(t-t_a, t_a-t_b, ..., t_last)``.  A dummy named
    ``t`` has time symbol ``t_t`` and remains distinct from current time ``t``.
    """

    order: int

    def __post_init__(self) -> None:
        validate_even_order(self.order)

    @property
    def pair_grade(self) -> int:
        return self.order // 2

    @property
    def internal_count(self) -> int:
        return self.order - 1

    @property
    def bar_index(self) -> int:
        return self.pair_grade + 1

    @property
    def dummy_labels(self) -> tuple[str, ...]:
        return tuple(canonical_dummy_label(i) for i in range(self.pair_grade - 1))

    @property
    def operator_slots(self) -> dict[str, int]:
        return {"0": 0, **{label: i for i, label in enumerate(self.dummy_labels, 1)}}

    @property
    def time_to_gap(self) -> dict[str, tuple[int, ...]]:
        n = self.pair_grade
        result = {"0": (0,) * n, "t": (1,) * n}
        result.update(
            {f"t_{label}": (0,) * slot + (1,) * (n - slot)
             for slot, label in enumerate(self.dummy_labels, 1)}
        )
        return result


__all__ = [
    "TCLOrder", "canonical_dummy_label", "is_time_symbol", "validate_even_order",
]
