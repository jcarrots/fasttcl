"""Compile and evaluate the Hadamard-reduced TCL6 generator on the CPU.

The bath supplies the correlation used in the coefficient calculation. No
additional coupling constant is applied here. Output is in the system energy
basis, interaction picture, with column-major operator vectorization.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .general_model import EnergyBasisModel, HRResourceBudget
from .spin_boson import EnergyBasisTwoLevelModel


@dataclass(frozen=True)
class HadamardReducedPlan:
    model: Any
    source: Any
    execution_plan: Any
    fusion: str
    required_gamma_identities: frozenset
    report: dict
    resources: HRResourceBudget


@dataclass(frozen=True)
class HadamardReducedResult:
    times: Any
    generator: Any
    report: dict


def compile_hr_plan(*, data_dir: Path, model, fusion="exact",
                    early_model_specialization=True, domain_completion="off",
                    resources=None):
    """Compile one model independently of the bath and time grid.

    Two-level models can combine equivalent kernel products before evaluation.
    Larger models first expand the system-index contractions. Both paths use
    the HR001--HR124 generator records.
    """
    from .io_utils import read_jsonl
    from .hr_profile import load_hr_inventory, _validated_hr_records
    from .physical_specialization import exact_coupling_pattern
    from .spin_boson import reduce_stationary_scalar_bath_inventory
    from .extended_runtime import compile_streamed_plan, specialize_streamed_plan
    from .fused_runtime import fuse_specialized_plan

    if fusion not in {"exact", "off"}:
        raise ValueError("fusion must be 'exact' or 'off'")
    if domain_completion != "off":
        raise ValueError("this CPU package supports domain_completion='off'")
    if type(early_model_specialization) is not bool:
        raise TypeError("early_model_specialization must be boolean")
    if not isinstance(model, (EnergyBasisModel, EnergyBasisTwoLevelModel)):
        raise TypeError("model must be an EnergyBasisModel or EnergyBasisTwoLevelModel")
    budget = HRResourceBudget() if resources is None else resources
    if not isinstance(budget, HRResourceBudget):
        raise TypeError("resources must be HRResourceBudget")
    started = perf_counter()
    data_path = Path(data_dir) / "hr-v1.jsonl"
    records = _validated_hr_records(read_jsonl(data_path))
    pattern = exact_coupling_pattern(model.coupling_energy)
    early = early_model_specialization and not pattern.is_dense
    if isinstance(model, EnergyBasisTwoLevelModel):
        inventory = load_hr_inventory(
            records, coupling_support=pattern.nonzero_entries if early else None)
        reduction = reduce_stationary_scalar_bath_inventory(inventory.monomials)
        streamed = compile_streamed_plan(reduction.monomials, require_complete=False)
        source = specialize_streamed_plan(streamed, coupling_operator=model.coupling_energy)
        execution_plan = fuse_specialized_plan(source, max_abs_coefficient=4) if fusion == "exact" else source
        compiler_report = {"inventory": dict(inventory.report),
                           "stationary_reduction": dict(reduction.report),
                           "path": "paper-two-level-exact-fusion"}
    else:
        from .symbolic_compiler import build_symbolic_index_plan
        from .dimension_compiler import build_dimension_join_plan
        from .generic_dimension_runtime import compile_stationary_specialized_plan
        from .general_execution import temporal_plan
        dimension = len(model.energies)
        symbolic = build_symbolic_index_plan(records)
        certificate = build_dimension_join_plan(symbolic, dimension=dimension,
                                               coupling_support=pattern.nonzero_entries)
        contributions = int(certificate.report["joined_logical_source_path_count"])
        budget.admit("compiler contributions", contributions, budget.max_contributions)
        budget.admit("compiler contributions/IR estimate (bytes)",
                     contributions * 1024 + dimension * dimension * 256, budget.compile_bytes)
        result = compile_stationary_specialized_plan(
            records, coupling_operator=model.coupling_energy, energies=model.energies,
            early_model_specialization=early_model_specialization,
            sparse_lowering="direct-temporal" if early else "path-enumeration")
        source = result.plan
        execution_plan = temporal_plan(source) if fusion == "exact" else source
        compiler_report = {**dict(result.report), "path": "general-dimension-source-reuse"}
    identities = (execution_plan.required_gamma_identities if fusion == "exact" else
                  {identity for weighted in source.cores for identity in weighted.core.kernel_identities})
    report = {"profile": "standalone-hr124-cpu-plan-v1", "source_record_count": 124,
              "model": model.report, "fusion_selected": fusion,
              "early_model_specialization": early, "domain_completion": "off",
              "compiler": compiler_report, "execution_plan": execution_plan.report,
              "compile_seconds": perf_counter() - started}
    return HadamardReducedPlan(model, source, execution_plan, fusion,
                               frozenset(identities), report, budget)


def evaluate_hr_plan(plan, *, nt: int, dt: float, bath, backend="numpy",
                     execution="auto"):
    """Evaluate the complete wrapped K6 coefficient on t[k] = k*dt."""
    from .spin_boson import _build_active_physical_gamma_sequences
    from .extended_runtime import evaluate_specialized_plan_numpy
    from .fused_runtime import evaluate_fused_plan_numpy
    from .postprocessing import apply_hr_outer_wrapper, wrapper_diagnostics

    if not isinstance(plan, HadamardReducedPlan):
        raise TypeError("plan must be returned by compile_hr_plan")
    if backend not in {"numpy", "auto"}:
        raise ValueError("only the NumPy CPU backend is included")
    if execution not in {"auto", "source", "reuse"}:
        raise ValueError("execution must be 'auto', 'source' or 'reuse'")
    if type(nt) is not int or nt <= 0:
        raise ValueError("nt must be a positive integer")
    if isinstance(dt, bool) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    selected = ("reuse" if plan.fusion == "exact" else "source") if execution == "auto" else execution
    if selected == "reuse" and plan.fusion != "exact":
        raise ValueError("reuse execution requires fusion='exact'")
    dimension = len(plan.model.energies)
    output_bytes = 16 * nt * dimension**4
    fft_length = 1 << max(0, (2 * nt - 2).bit_length())
    gamma_bytes = 16 * nt * len(plan.required_gamma_identities)
    host_required = 4 * output_bytes + 2 * gamma_bytes + 128 * 16 * fft_length
    plan.resources.admit("host output/Gamma/workspace estimate (bytes)",
                         host_required, plan.resources.host_bytes)
    started = perf_counter()
    grid, bound = _build_active_physical_gamma_sequences(
        plan.model, bath, plan.required_gamma_identities, nt=nt, dt=float(dt))
    ratios = plan.model.level_ratios(float(dt))
    kwargs = dict(gamma_sequences=bound.sequences, level_ratios=ratios, length=nt, step=float(dt))
    if selected == "source":
        raw = evaluate_specialized_plan_numpy(plan.source, **kwargs)
    elif isinstance(plan.model, EnergyBasisTwoLevelModel):
        raw = evaluate_fused_plan_numpy(plan.execution_plan, **kwargs)
    else:
        cache_bytes = min((plan.resources.host_bytes - host_required) // 4, 64 * 1024**2)
        slots = max(0, min(len(plan.execution_plan.units) * 5, cache_bytes // (16 * fft_length + 256)))
        raw = evaluate_fused_plan_numpy(
            plan.execution_plan, **kwargs, shared_forward_fft_cache=True,
            semantic_forward_fft_cache_size=slots, persistent_expression_cache=False,
            convolution_result_cache_size=slots, convolution_result_cache_policy="liveness",
            product_forward_fft_cache_size=min(slots, 64),
            convolution_result_cache_budget_mib=max(1, cache_bytes) / 1024**2,
            fft_backend="numpy", physical_gamma_cse=True, physical_dependency_dag=True)
    wrapped = apply_hr_outer_wrapper(raw.values, coupling_operator=plan.model.coupling_energy,
                                     level_ratios=ratios, xp=np)
    if not np.isfinite(wrapped.values).all():
        raise ValueError("K6 contains nonfinite values")
    report = {"profile": "standalone-hr124-cpu-evaluation-v1", "backend": "numpy",
              "execution": selected, "picture": "interaction", "basis": "energy",
              "vectorization": "column-major", "partial": False,
              "output_shape": list(wrapped.values.shape), "gamma": dict(grid.report),
              "runtime": dict(raw.report), "wrapper": dict(wrapped.report),
              "diagnostics": wrapper_diagnostics(wrapped.values, xp=np),
              "evaluation_seconds": perf_counter() - started}
    return HadamardReducedResult(float(dt) * np.arange(nt), wrapped.values, report)
