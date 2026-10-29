"""Compile an arbitrary even TCL order into certified symbolic artifacts."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from .tcl_cumulant import cumulant_census
from .tcl_order import TCLOrder
from .tcl_reduction import derive_tcl_reduction, plan_records, reduced_records, source_records


def _write_records(path: Path, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    count = 0
    digest = sha256()
    with path.open("wb") as handle:
        for record in records:
            line = (json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            handle.write(line)
            digest.update(line)
            count += 1
    return {"path": path.name, "records": count, "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def source_estimate(order: int) -> dict[str, int]:
    spec = TCLOrder(order)
    census = cumulant_census(order)
    return {"order": order, "pair_count": spec.pair_grade,
            "retained_integrations": spec.pair_grade - 1,
            "composition_count": census.composition_count,
            "projected_structure_count": census.projected_structure_count,
            "wick_instance_count": census.wick_instance_count,
            "raw_liouvillian_wick_contributions": census.wick_instance_count * 2**order}


def compile_to_directory(order: int, output: Path, *, dimension: int | None = None,
                         coupling_support: frozenset[tuple[int, int]] | None = None,
                         max_certificate_nodes: int | None = None,
                         verify_exhaustive: bool = False) -> dict[str, Any]:
    """Write a complete derivation; a manifest is written only after success.

    The destination must be absent or empty so an interrupted compilation
    cannot leave a previous successful manifest beside partial new artifacts.
    ``dimension`` requests exact symbolic matrix-index lowering as well.
    """
    spec = TCLOrder(order)
    if dimension is not None and (isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1):
        raise ValueError("dimension must be a positive integer")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output must be a new or empty directory")
    if coupling_support is not None and dimension is None:
        raise ValueError("coupling_support requires dimension")
    started = perf_counter()
    result = derive_tcl_reduction(order, max_certificate_nodes=max_certificate_nodes,
                                  verify_exhaustive=verify_exhaustive)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "source": _write_records(output / "cumulant-source.jsonl", source_records(result)),
        "pairing_plans": _write_records(output / "pairing-plans.jsonl", plan_records(result)),
        "hr_terms": _write_records(output / "hr-terms.jsonl", reduced_records(result)),
    }
    scalarization = None
    if dimension is not None:
        from .tcl_scalarization import compile_scalar_terms
        scalar = compile_scalar_terms(((group.coefficient, group.expression)
            for group in result.collection.survivors), order=order, dimension=dimension,
            coupling_support=coupling_support)
        outer_raw = scalar.outer_raw()
        complete = outer_raw.hermitian_completion()
        for name, stage in (("scalar_source", scalar), ("scalar_outer_raw", outer_raw),
                            ("scalar_generator", complete)):
            artifacts[name] = _write_records(output / f"{name.replace('_', '-')}.jsonl", stage.records())
        scalarization = {"dimension": dimension,
            "coupling_support": None if coupling_support is None else [list(pair) for pair in sorted(coupling_support)],
            "source_signature_count": scalar.signature_count,
            "outer_raw_signature_count": outer_raw.signature_count,
            "generator_signature_count": complete.signature_count,
            "coefficient_format": "exact integer polynomials in coupling-matrix entries",
            "phase_format": "integer energy coefficients for each ordered time gap",
            "hermitian_completion": "complex-linear realigned conjugation"}
    manifest = {"schema_version": "tcl-symbolic-derivation-v1", "status": "passed",
        "order": order, "source_estimate": source_estimate(order),
        "retained_labels": list(spec.dummy_labels), "gap_count": spec.pair_grade,
        "outer_wrapper_verified": result.outer_wrapper_verified,
        "full_monomial_count": result.full_monomial_count,
        "prewrapper_monomial_count": result.prewrapper_monomial_count,
        "reduction_counts": result.counts,
        "domain_certificate": {"method": "exact-partial-order-decision",
            "all_signed_residuals_zero": all(plan.certificate.is_zero for plan in result.pairing_plans),
            "explored_nodes": sum(plan.certificate.explored_nodes for plan in result.pairing_plans),
            "exhaustive_cross_check": verify_exhaustive},
        "scalarization": scalarization,
        "artifacts": artifacts, "elapsed_seconds": perf_counter() - started,
        "scope": "symbolic cumulant, certified Hadamard reduction, and optional scalar sums; no fast numerical contraction runtime"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, required=True, help="positive even perturbative order, e.g. 2, 4, 6, 8, 10")
    parser.add_argument("--output", type=Path, help="new or empty artifact directory")
    parser.add_argument("--dimension", type=int, help="also compile scalar sums for this system dimension")
    parser.add_argument("--coupling-support", type=Path, help="JSON list of allowed [row,column] coupling entries")
    parser.add_argument("--max-certificate-nodes", type=int, help="optional exact proof work limit; raises if exceeded")
    parser.add_argument("--verify-exhaustive", action="store_true", help="also check all time orderings (factorial cost)")
    parser.add_argument("--estimate", action="store_true", help="print source-size estimate without generating terms")
    args = parser.parse_args(argv)
    try:
        if args.estimate:
            payload = source_estimate(args.order)
        else:
            support = None
            if args.coupling_support is not None:
                values = json.loads(args.coupling_support.read_text(encoding="utf-8"))
                if not isinstance(values, list) or any(not isinstance(pair, list) or len(pair) != 2 for pair in values):
                    raise ValueError("coupling-support JSON must be a list of [row,column] pairs")
                support = frozenset(tuple(pair) for pair in values)
            payload = compile_to_directory(args.order,
                args.output or Path("generated") / f"tcl{args.order}-symbolic-v1",
                dimension=args.dimension, coupling_support=support,
                max_certificate_nodes=args.max_certificate_nodes, verify_exhaustive=args.verify_exhaustive)
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
