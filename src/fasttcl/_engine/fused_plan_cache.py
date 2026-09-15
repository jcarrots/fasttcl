"""Identify compiled plans for cache reuse."""

from __future__ import annotations

import hashlib

import math

from .extended_runtime import ExtendedRuntimeError, SpecializedExecutionPlan

from .io_utils import canonical_json

class FusedPlanCacheError(ExtendedRuntimeError):
    """Raised when a cached fusion certificate is malformed or mismatched."""


def _complex_token(value: complex) -> tuple[str, str]:
    number = complex(value)
    if not math.isfinite(number.real) or not math.isfinite(number.imag):
        raise FusedPlanCacheError("cached complex weights must be finite")
    return number.real.hex(), number.imag.hex()


def specialized_plan_digest(plan: SpecializedExecutionPlan) -> str:
    """Hash every ordered specialized core and contracted sparse weight."""

    core_rows = []
    instrumented = plan.source_plan.history_depth_manifest_sha256 is not None
    for weighted in plan.cores:
        row = {
            "kernel_identities": weighted.core.kernel_identities,
            "phase_u": weighted.core.phase_u,
            "phase_v": weighted.core.phase_v,
            "phase_w": weighted.core.phase_w,
            "template_id": weighted.core.template_id,
            "weights": [
                (output_flat, input_flat, _complex_token(weight))
                for output_flat, input_flat, weight in weighted.weights
            ],
        }
        if instrumented:
            row["depth_weights"] = [
                (depth, output_flat, input_flat, _complex_token(weight))
                for depth, output_flat, input_flat, weight in weighted.depth_weights
            ]
        core_rows.append(row)
    payload = {
        "source_plan": plan.source_plan.report,
        "coupling_operator": [
            [_complex_token(value) for value in row]
            for row in plan.coupling_operator
        ],
        "cores": core_rows,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

