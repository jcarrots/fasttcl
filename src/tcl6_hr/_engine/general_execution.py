"""CPU runtime extracted from the canonical TCL6 Numerical source; see data/source-provenance.json."""

from __future__ import annotations

from collections import Counter

from dataclasses import dataclass

import hashlib

from pathlib import Path

from typing import Any

@dataclass(frozen=True)
class GeneralTemporalPlan:
    """Lossless adapter: one shared temporal unit per source core, in source order."""
    source_plan: Any
    units: tuple
    source_core_ids: tuple
    digest: str

    @property
    def dimension(self):
        return self.source_plan.source_plan.dimension

    @property
    def required_gamma_identities(self):
        return frozenset(i for u in self.units for s in u.slots for t in s.terms for i in t.monomial.kernel_identities)

    @property
    def report(self):
        return {"profile": "general-exact-temporal-adapter-v1", "hilbert_dimension": self.dimension,
                "fused_temporal_unit_count": len(self.units), "source_active_temporal_core_count": len(self.units),
                "fused_strict_temporal_unit_count": sum(u.template_id != "overlap-uv-vw-v" for u in self.units),
                "fused_overlap_temporal_unit_count": sum(u.template_id == "overlap-uv-vw-v" for u in self.units),
                "fused_sparse_weight_count": sum(len(u.weights) for u in self.units),
                "template_counts": dict(Counter(u.template_id for u in self.units)),
                "fusion_semantics": "lossless source-core adapter with exact runtime reuse",
                "numeric_rank_factorization": False, "digest": self.digest}


def temporal_plan(source):
    from .fused_runtime import FusedTemporalUnit, _effective_monomials, _single_expression
    from .fused_plan_cache import specialized_plan_digest
    units = tuple(FusedTemporalUnit(
        unit_id=f"unit-{k:07d}", group_id=f"unit-{k:07d}", template_id=w.core.template_id,
        slots=tuple(_single_expression(m) for m in _effective_monomials(w)), weights=w.weights,
        source_core_ids=(f"core-{k:07d}",), basis_source_core_id=f"core-{k:07d}",
        varying_slot="U", depth_weights=w.depth_weights,
    ) for k,w in enumerate(source.cores))
    implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    digest = hashlib.sha256(("general-temporal-v1:" + implementation + ":" +
                             specialized_plan_digest(source)).encode()).hexdigest()
    return GeneralTemporalPlan(source, units, tuple(u.source_core_ids[0] for u in units), digest)

