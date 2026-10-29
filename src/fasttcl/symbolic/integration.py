"""Deterministic Appendix B/C integration planning.

The planner in this module operates on the *physical* Appendix B labels:
``LOWER_ZERO`` is the lower endpoint, while ``TimeIndex(2*n)`` is the
current time.  This deliberately avoids the Appendix F shorthand collision
where operator label ``"0"`` denotes the current-time operator.

All public state is immutable.  A planning step fixes one label on the right
of the active block and one label on the left, so the active-label count drops
by exactly two.  Split branches store their effective bounds explicitly:

``s^*``
    ``+ integral_0^g_s`` (only the greater child remains relevant).

``s_*``
    ``- integral_0^l_s`` (only the lesser child remains relevant).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache
from typing import Iterable, Literal, TypeAlias


class IntegrationPlanError(ValueError):
    """Raised when an integration state violates the Appendix B/C model."""


@dataclass(frozen=True, slots=True)
class LowerZero:
    """The lower endpoint ``t_0 = 0`` from Appendix B."""

    def __str__(self) -> str:
        return "0"


LOWER_ZERO = LowerZero()


@dataclass(frozen=True, slots=True, order=True)
class TimeIndex:
    """A physical system time ``t_i`` with ``i >= 1``."""

    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError("time index must be an integer")
        if self.value < 1:
            raise ValueError("time index must be positive")

    def __str__(self) -> str:
        return str(self.value)


IntegrationLabel: TypeAlias = LowerZero | TimeIndex


def _label_value(label: IntegrationLabel) -> int:
    return 0 if isinstance(label, LowerZero) else label.value


def _order_key(order: tuple[IntegrationLabel, ...]) -> tuple[int, ...]:
    return tuple(_label_value(label) for label in order)


def physical_order(*indices: int) -> tuple[IntegrationLabel, ...]:
    """Build a typed physical order from integer Appendix B labels."""

    labels: list[IntegrationLabel] = []
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("physical-order labels must be integers")
        labels.append(LOWER_ZERO if index == 0 else TimeIndex(index))
    return tuple(labels)


@dataclass(frozen=True, slots=True)
class OrientedCorrelation:
    """An ordered BCF pair representing ``C(t_first - t_second)``."""

    first: TimeIndex
    second: TimeIndex

    def __post_init__(self) -> None:
        if self.first == self.second:
            raise ValueError("a correlation cannot pair a time with itself")

    @property
    def endpoints(self) -> frozenset[TimeIndex]:
        return frozenset((self.first, self.second))


def correlation(first: int, second: int) -> OrientedCorrelation:
    """Convenience constructor for an ordered physical correlation."""

    return OrientedCorrelation(TimeIndex(first), TimeIndex(second))


@dataclass(frozen=True, slots=True)
class IntegralBound:
    """Effective lower and upper bounds for one integration variable."""

    lower: IntegrationLabel
    upper: TimeIndex

    def __post_init__(self) -> None:
        if _label_value(self.lower) >= self.upper.value:
            raise ValueError("an integral lower bound must precede its upper")


@dataclass(frozen=True, slots=True)
class Children:
    """The Appendix B lesser and greater children of one variable."""

    lesser: IntegrationLabel
    greater: TimeIndex

    @property
    def bound(self) -> IntegralBound:
        return IntegralBound(self.lesser, self.greater)


class SplitAnnotation(str, Enum):
    """Which original child remains relevant after the exact C4 split."""

    UPPER_ONLY = "superscript_star"
    LOWER_ONLY = "subscript_star"


@dataclass(frozen=True, slots=True)
class BoundOverride:
    """A split variable's original children and new effective bound."""

    variable: TimeIndex
    annotation: SplitAnnotation
    original_children: Children
    effective_bound: IntegralBound


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """The replayable, correlation-independent portion of planner state."""

    order: tuple[IntegrationLabel, ...]
    left_fixed_count: int
    right_fixed_count: int
    coefficient: int
    overrides: tuple[BoundOverride, ...] = ()


EventRule: TypeAlias = Literal[
    "move_partner",
    "split_upper",
    "split_lower",
    "retain_bounds",
    "choose_child",
    "move_child",
]


@dataclass(frozen=True, slots=True)
class RewriteEvent:
    """One auditable state transition from the Appendix C algorithm."""

    rule: EventRule
    before: StateSnapshot
    after: StateSnapshot
    focus: TimeIndex
    related: IntegrationLabel | None = None
    children: Children | None = None
    annotation: SplitAnnotation | None = None
    effective_bound: IntegralBound | None = None
    sign_multiplier: int = 1


@dataclass(frozen=True, slots=True)
class IntegrationState:
    """Immutable state for one signed Appendix C integration branch."""

    order: tuple[IntegrationLabel, ...]
    correlations: tuple[OrientedCorrelation, ...]
    bar_index: int
    left_fixed_count: int = 2
    right_fixed_count: int = 0
    coefficient: int = 1
    overrides: tuple[BoundOverride, ...] = ()
    events: tuple[RewriteEvent, ...] = ()

    def __post_init__(self) -> None:
        _validate_complete_state(self)

    @property
    def n(self) -> int:
        return (len(self.order) - 1) // 2

    @property
    def active_labels(self) -> tuple[TimeIndex, ...]:
        end = len(self.order) - self.right_fixed_count
        return tuple(
            label
            for label in self.order[self.left_fixed_count : end]
            if isinstance(label, TimeIndex)
        )

    @property
    def outer_labels(self) -> tuple[TimeIndex, ...]:
        return tuple(
            label
            for label in self.order[2 : self.bar_index]
            if isinstance(label, TimeIndex)
        )

    @property
    def inner_labels(self) -> tuple[TimeIndex, ...]:
        return tuple(
            label
            for label in self.order[self.bar_index :]
            if isinstance(label, TimeIndex)
        )

    @property
    def current_time(self) -> TimeIndex:
        current = self.order[1]
        if not isinstance(current, TimeIndex):  # guarded by validation
            raise AssertionError("current time is not a TimeIndex")
        return current

    def snapshot(self) -> StateSnapshot:
        return StateSnapshot(
            order=self.order,
            left_fixed_count=self.left_fixed_count,
            right_fixed_count=self.right_fixed_count,
            coefficient=self.coefficient,
            overrides=self.overrides,
        )


def _validate_matching(
    correlations: tuple[OrientedCorrelation, ...], n: int
) -> None:
    if len(correlations) != n:
        raise IntegrationPlanError(
            f"TCL_{2 * n} requires {n} correlation pairs"
        )
    endpoints = [
        endpoint
        for pair in correlations
        for endpoint in (pair.first, pair.second)
    ]
    expected = [TimeIndex(index) for index in range(1, 2 * n + 1)]
    if sorted(endpoints) != expected:
        raise IntegrationPlanError(
            "correlations must be a perfect matching of physical "
            f"indices 1 through {2 * n}"
        )


def _validate_complete_state(state: IntegrationState) -> None:
    order = state.order
    if len(order) < 3 or len(order) % 2 != 1:
        raise IntegrationPlanError(
            "a TCL_2n order must contain 2n+1 labels"
        )
    n = (len(order) - 1) // 2
    expected = physical_order(0, *range(1, 2 * n + 1))
    if sorted(order, key=_label_value) != list(expected):
        raise IntegrationPlanError(
            "order must contain lower zero and each physical index exactly once"
        )
    if order[0] != LOWER_ZERO or order[1] != TimeIndex(2 * n):
        raise IntegrationPlanError(
            "order must begin with lower zero and current time 2n"
        )
    if state.bar_index != n + 1:
        raise IntegrationPlanError(
            f"bar index must be n+1 ({n + 1})"
        )
    if not (2 <= state.left_fixed_count <= state.bar_index):
        raise IntegrationPlanError("invalid left fixed-prefix length")
    if not (0 <= state.right_fixed_count <= n - 1):
        raise IntegrationPlanError("invalid right fixed-suffix length")
    if state.left_fixed_count != state.right_fixed_count + 2:
        raise IntegrationPlanError(
            "complete recursion states must fix one left and one right label"
        )
    if (
        state.left_fixed_count + state.right_fixed_count
        >= len(state.order)
    ):
        raise IntegrationPlanError("a state must retain an active label")
    if isinstance(state.coefficient, bool) or not isinstance(
        state.coefficient, int
    ):
        raise TypeError("branch coefficient must be an integer")
    if state.coefficient == 0:
        raise IntegrationPlanError("branch coefficient cannot be zero")

    _validate_matching(state.correlations, n)

    override_variables = [override.variable for override in state.overrides]
    if len(set(override_variables)) != len(override_variables):
        raise IntegrationPlanError("a variable has multiple bound overrides")
    if tuple(sorted(override_variables)) != tuple(override_variables):
        raise IntegrationPlanError(
            "bound overrides must be ordered by physical index"
        )
    order_set = set(order)
    if any(variable not in order_set for variable in override_variables):
        raise IntegrationPlanError(
            "bound override refers to a variable outside the order"
        )


def make_state(
    order: Iterable[IntegrationLabel],
    correlations: Iterable[OrientedCorrelation],
    *,
    bar_index: int | None = None,
    coefficient: int = 1,
) -> IntegrationState:
    """Construct an initial, unfixed state for a supplied Appendix B order."""

    order_tuple = tuple(order)
    if len(order_tuple) < 3 or len(order_tuple) % 2 != 1:
        raise IntegrationPlanError(
            "a TCL_2n order must contain 2n+1 labels"
        )
    n = (len(order_tuple) - 1) // 2
    return IntegrationState(
        order=order_tuple,
        correlations=tuple(correlations),
        bar_index=n + 1 if bar_index is None else bar_index,
        coefficient=coefficient,
    )


def initial_state(
    n: int,
    correlations: Iterable[OrientedCorrelation],
    *,
    coefficient: int = 1,
) -> IntegrationState:
    """Return the standard time-ordered TCL_2n seed state."""

    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 1:
        raise ValueError("n must be positive")
    order = physical_order(0, 2 * n, *range(2 * n - 1, 0, -1))
    return IntegrationState(
        order=order,
        correlations=tuple(correlations),
        bar_index=n + 1,
        coefficient=coefficient,
    )


def lesser_greater(
    order: tuple[IntegrationLabel, ...], variable: TimeIndex
) -> Children:
    """Compute Appendix B lesser/greater children from earlier labels."""

    try:
        position = order.index(variable)
    except ValueError as exc:
        raise IntegrationPlanError(
            f"time {variable.value} is not in the integration order"
        ) from exc
    if position < 2:
        raise IntegrationPlanError(
            "children are defined only for integration variables"
        )

    earlier = order[:position]
    lower_candidates = [
        label
        for label in earlier
        if _label_value(label) < variable.value
    ]
    upper_candidates = [
        label
        for label in earlier
        if _label_value(label) > variable.value
    ]
    if not lower_candidates or not upper_candidates:
        raise IntegrationPlanError(
            f"time {variable.value} lacks a valid lesser or greater child"
        )
    lesser = max(lower_candidates, key=_label_value)
    greater_label = min(upper_candidates, key=_label_value)
    if not isinstance(greater_label, TimeIndex):
        raise AssertionError("greater child cannot be lower zero")
    return Children(lesser=lesser, greater=greater_label)


def appendix_b_bounds(
    order: tuple[IntegrationLabel, ...],
) -> dict[TimeIndex, IntegralBound]:
    """Return the Appendix B bounds for every integration variable."""

    return {
        variable: lesser_greater(order, variable).bound
        for variable in order[2:]
        if isinstance(variable, TimeIndex)
    }


def effective_bounds(
    state: IntegrationState,
) -> dict[TimeIndex, IntegralBound]:
    """Return ordinary bounds with exact C4 split bounds substituted."""

    bounds = appendix_b_bounds(state.order)
    for override in state.overrides:
        bounds[override.variable] = override.effective_bound
    return bounds


def condition_1(state: IntegrationState) -> bool:
    """Whether every ordered correlation crosses the fixed vertical bar."""

    positions = {label: index for index, label in enumerate(state.order)}
    for pair in state.correlations:
        first_left = positions[pair.first] < state.bar_index
        second_left = positions[pair.second] < state.bar_index
        if first_left == second_left:
            return False
    return True


def condition_2(state: IntegrationState) -> bool:
    """Whether every *effective* inner bound depends only on outer times."""

    positions = {label: index for index, label in enumerate(state.order)}
    bounds = effective_bounds(state)

    def is_outer(label: IntegrationLabel) -> bool:
        if isinstance(label, LowerZero):
            return True
        return positions[label] < state.bar_index

    return all(
        is_outer(bounds[variable].lower)
        and is_outer(bounds[variable].upper)
        for variable in state.inner_labels
    )


def _partner_map(
    correlations: tuple[OrientedCorrelation, ...],
) -> dict[TimeIndex, TimeIndex]:
    partners: dict[TimeIndex, TimeIndex] = {}
    for pair in correlations:
        partners[pair.first] = pair.second
        partners[pair.second] = pair.first
    return partners


def _sorted_overrides(
    overrides: tuple[BoundOverride, ...],
    added: BoundOverride,
) -> tuple[BoundOverride, ...]:
    return tuple(
        sorted((*overrides, added), key=lambda item: item.variable.value)
    )


def _move_within_active(
    snapshot: StateSnapshot,
    variable: TimeIndex,
    *,
    to_left: bool,
) -> StateSnapshot:
    order = list(snapshot.order)
    active_end = len(order) - snapshot.right_fixed_count
    try:
        position = order.index(variable)
    except ValueError as exc:
        raise IntegrationPlanError(
            f"time {variable.value} is absent from the order"
        ) from exc
    if not snapshot.left_fixed_count <= position < active_end:
        raise IntegrationPlanError(
            f"time {variable.value} is fixed and cannot move"
        )
    order.pop(position)
    insertion = (
        snapshot.left_fixed_count
        if to_left
        else active_end - 1
    )
    order.insert(insertion, variable)
    return replace(snapshot, order=tuple(order))


@dataclass(frozen=True, slots=True)
class _PlannedLeaf:
    snapshot: StateSnapshot
    events: tuple[RewriteEvent, ...]


def plan_hadamardable(
    state: IntegrationState,
) -> tuple[IntegrationState, ...]:
    """Apply the deterministic Appendix C recursion to one seed state.

    If both children of ``s`` are already fixed, every nonfixed child choice
    is semantically valid.  The planner recursively evaluates each choice,
    selects the one producing the fewest leaves, and breaks equal-count ties
    by the candidate's current left-to-right position and physical label.
    The lookahead solver is memoized by immutable state snapshot.
    """

    partners = _partner_map(state.correlations)
    bar_index = state.bar_index
    seed_snapshot = state.snapshot()

    def event(
        rule: EventRule,
        before: StateSnapshot,
        after: StateSnapshot,
        focus: TimeIndex,
        *,
        related: IntegrationLabel | None = None,
        children: Children | None = None,
        annotation: SplitAnnotation | None = None,
        effective_bound: IntegralBound | None = None,
        sign_multiplier: int = 1,
    ) -> RewriteEvent:
        return RewriteEvent(
            rule=rule,
            before=before,
            after=after,
            focus=focus,
            related=related,
            children=children,
            annotation=annotation,
            effective_bound=effective_bound,
            sign_multiplier=sign_multiplier,
        )

    def move_child(
        snapshot: StateSnapshot,
        child: TimeIndex,
        s: TimeIndex,
    ) -> tuple[StateSnapshot, RewriteEvent]:
        moved = _move_within_active(snapshot, child, to_left=True)
        after = replace(
            moved,
            left_fixed_count=moved.left_fixed_count + 1,
        )
        return after, event(
            "move_child",
            snapshot,
            after,
            child,
            related=s,
        )

    @lru_cache(maxsize=None)
    def solve(snapshot: StateSnapshot) -> tuple[_PlannedLeaf, ...]:
        active_end = len(snapshot.order) - snapshot.right_fixed_count
        active = snapshot.order[snapshot.left_fixed_count : active_end]
        if len(active) == 1:
            return (_PlannedLeaf(snapshot=snapshot, events=()),)
        if len(active) < 1 or len(active) % 2 == 0:
            raise IntegrationPlanError(
                "Appendix C recursion requires a positive odd active count"
            )

        r_label = snapshot.order[snapshot.left_fixed_count - 1]
        if not isinstance(r_label, TimeIndex):
            raise IntegrationPlanError(
                "the rightmost fixed left label must be a system time"
            )
        try:
            s = partners[r_label]
        except KeyError as exc:
            raise IntegrationPlanError(
                f"fixed time {r_label.value} has no correlation partner"
            ) from exc

        moved_s = _move_within_active(snapshot, s, to_left=False)
        after_partner = replace(
            moved_s,
            right_fixed_count=moved_s.right_fixed_count + 1,
        )
        partner_event = event(
            "move_partner",
            snapshot,
            after_partner,
            s,
            related=r_label,
        )

        children = lesser_greater(after_partner.order, s)
        fixed_left = set(
            after_partner.order[: after_partner.left_fixed_count]
        )
        lesser_fixed = children.lesser in fixed_left
        greater_fixed = children.greater in fixed_left

        branch_inputs: list[
            tuple[
                StateSnapshot,
                TimeIndex,
                RewriteEvent,
            ]
        ] = []

        if not lesser_fixed and not greater_fixed:
            if not isinstance(children.lesser, TimeIndex):
                raise AssertionError("unfixed lesser child is not a time")

            upper_bound = IntegralBound(LOWER_ZERO, children.greater)
            upper_override = BoundOverride(
                variable=s,
                annotation=SplitAnnotation.UPPER_ONLY,
                original_children=children,
                effective_bound=upper_bound,
            )
            upper_snapshot = replace(
                after_partner,
                overrides=_sorted_overrides(
                    after_partner.overrides, upper_override
                ),
            )
            upper_event = event(
                "split_upper",
                after_partner,
                upper_snapshot,
                s,
                related=children.greater,
                children=children,
                annotation=SplitAnnotation.UPPER_ONLY,
                effective_bound=upper_bound,
            )
            branch_inputs.append(
                (upper_snapshot, children.greater, upper_event)
            )

            lower_bound = IntegralBound(LOWER_ZERO, children.lesser)
            lower_override = BoundOverride(
                variable=s,
                annotation=SplitAnnotation.LOWER_ONLY,
                original_children=children,
                effective_bound=lower_bound,
            )
            lower_snapshot = replace(
                after_partner,
                coefficient=-after_partner.coefficient,
                overrides=_sorted_overrides(
                    after_partner.overrides, lower_override
                ),
            )
            lower_event = event(
                "split_lower",
                after_partner,
                lower_snapshot,
                s,
                related=children.lesser,
                children=children,
                annotation=SplitAnnotation.LOWER_ONLY,
                effective_bound=lower_bound,
                sign_multiplier=-1,
            )
            branch_inputs.append(
                (lower_snapshot, children.lesser, lower_event)
            )
        elif lesser_fixed != greater_fixed:
            relevant = (
                children.greater
                if lesser_fixed
                else children.lesser
            )
            if not isinstance(relevant, TimeIndex):
                raise AssertionError("relevant child is not a time")
            retain_event = event(
                "retain_bounds",
                after_partner,
                after_partner,
                s,
                related=relevant,
                children=children,
                effective_bound=children.bound,
            )
            branch_inputs.append(
                (after_partner, relevant, retain_event)
            )
        else:
            # Appendix C permits any nonfixed digit here.  Use memoized
            # lookahead to minimize the number of final branches.
            candidate_labels = tuple(
                label
                for label in after_partner.order[
                    after_partner.left_fixed_count :
                    len(after_partner.order)
                    - after_partner.right_fixed_count
                ]
                if isinstance(label, TimeIndex)
            )
            if not candidate_labels:
                raise IntegrationPlanError(
                    "both children are fixed but no active child remains"
                )

            evaluated: list[
                tuple[
                    tuple[int, int, int],
                    StateSnapshot,
                    RewriteEvent,
                    tuple[_PlannedLeaf, ...],
                ]
            ] = []
            for position, candidate in enumerate(candidate_labels):
                child_snapshot, child_event = move_child(
                    after_partner, candidate, s
                )
                leaves = solve(child_snapshot)
                # The position-first tie-break preserves stable left-to-right
                # order (and the paper's C3--C6 worked rearrangement).
                score = (len(leaves), position, candidate.value)
                choose_event = event(
                    "choose_child",
                    after_partner,
                    after_partner,
                    s,
                    related=candidate,
                    children=children,
                    effective_bound=children.bound,
                )
                evaluated.append(
                    (
                        score,
                        child_snapshot,
                        choose_event,
                        tuple(
                            _PlannedLeaf(
                                snapshot=leaf.snapshot,
                                events=(child_event, *leaf.events),
                            )
                            for leaf in leaves
                        ),
                    )
                )
            _, _, choose_event, chosen_leaves = min(
                evaluated, key=lambda item: item[0]
            )
            return tuple(
                _PlannedLeaf(
                    snapshot=leaf.snapshot,
                    events=(
                        partner_event,
                        choose_event,
                        *leaf.events,
                    ),
                )
                for leaf in chosen_leaves
            )

        planned: list[_PlannedLeaf] = []
        for branch_snapshot, relevant_child, branch_event in branch_inputs:
            child_snapshot, child_event = move_child(
                branch_snapshot, relevant_child, s
            )
            for leaf in solve(child_snapshot):
                planned.append(
                    _PlannedLeaf(
                        snapshot=leaf.snapshot,
                        events=(
                            partner_event,
                            branch_event,
                            child_event,
                            *leaf.events,
                        ),
                    )
                )
        return tuple(planned)

    planned_leaves = solve(seed_snapshot)
    leaves: list[IntegrationState] = []
    for planned in planned_leaves:
        leaf = IntegrationState(
            order=planned.snapshot.order,
            correlations=state.correlations,
            bar_index=bar_index,
            left_fixed_count=planned.snapshot.left_fixed_count,
            right_fixed_count=planned.snapshot.right_fixed_count,
            coefficient=planned.snapshot.coefficient,
            overrides=planned.snapshot.overrides,
            events=(*state.events, *planned.events),
        )
        if len(leaf.active_labels) != 1:
            raise AssertionError("planner did not terminate at TCL2")
        if not condition_1(leaf):
            raise IntegrationPlanError(
                "planned leaf does not satisfy correlation Condition 1"
            )
        if not condition_2(leaf):
            raise IntegrationPlanError(
                "planned leaf does not satisfy effective-bound Condition 2"
            )
        leaves.append(leaf)
    return tuple(leaves)


def replay_events(
    initial: IntegrationState,
    events: Iterable[RewriteEvent],
) -> IntegrationState:
    """Replay and verify an event sequence from ``initial``."""

    event_tuple = tuple(events)
    snapshot = initial.snapshot()
    for index, rewrite in enumerate(event_tuple):
        if rewrite.before != snapshot:
            raise IntegrationPlanError(
                f"event {index} does not continue from the prior state"
            )
        snapshot = rewrite.after
    return IntegrationState(
        order=snapshot.order,
        correlations=initial.correlations,
        bar_index=initial.bar_index,
        left_fixed_count=snapshot.left_fixed_count,
        right_fixed_count=snapshot.right_fixed_count,
        coefficient=snapshot.coefficient,
        overrides=snapshot.overrides,
        events=(*initial.events, *event_tuple),
    )


__all__ = [
    "LOWER_ZERO",
    "BoundOverride",
    "Children",
    "IntegrationLabel",
    "IntegrationPlanError",
    "IntegrationState",
    "IntegralBound",
    "LowerZero",
    "OrientedCorrelation",
    "RewriteEvent",
    "SplitAnnotation",
    "StateSnapshot",
    "TimeIndex",
    "appendix_b_bounds",
    "condition_1",
    "condition_2",
    "correlation",
    "effective_bounds",
    "initial_state",
    "lesser_greater",
    "make_state",
    "physical_order",
    "plan_hadamardable",
    "replay_events",
]
