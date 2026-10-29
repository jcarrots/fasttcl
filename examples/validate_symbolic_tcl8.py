"""Compare the complete generic TCL8 inventory with frozen legacy digests.

Run from a repository checkout after installing FastTCL. This slower check is
separate from pytest; its reference digests cover every source/reduced record.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from fasttcl.symbolic.tcl_reduction import derive_tcl_reduction, reduced_records, source_records


def main() -> None:
    reference = json.loads((Path(__file__).resolve().parents[1] /
        "tests/fixtures/tcl8-inventory-reference.json").read_text(encoding="utf-8"))
    started = perf_counter()
    result = derive_tcl_reduction(8)
    checks = {}
    for kind, records in (("source", source_records(result)), ("reduced", reduced_records(result))):
        digest = sha256()
        count = 0
        for record in records:
            record.pop("schema_version")
            line = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
            digest.update(line.encode("utf-8"))
            count += 1
        checks[kind] = {"records": count, "sha256": digest.hexdigest()}
        if count != reference[kind]["records"] or digest.hexdigest() != reference[kind]["sha256"]:
            raise RuntimeError(f"TCL8 {kind} inventory differs from the frozen reference: {checks[kind]}")
    print(json.dumps({"status": "passed", "order": 8, "checks": checks,
        "counts": result.counts, "seconds": perf_counter() - started}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
