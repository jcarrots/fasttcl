"""Extract the approved plotted values from a canonical TCL6 Paper checkout.

This optional maintenance command is not needed to replot the bundled data.
It deliberately reads only accepted paper inputs, not simulation checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_record(value: np.ndarray) -> dict:
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "nan_count": int(np.count_nonzero(np.isnan(value))),
            "infinity_count": int(np.count_nonzero(np.isinf(value))),
            "sha256_c_order_bytes": hashlib.sha256(value.tobytes(order="C")).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", required=True, type=Path)
    args = parser.parse_args()
    paper = args.paper_root.resolve()
    out = HERE / "data"
    out.mkdir(parents=True, exist_ok=True)
    source_records = {}
    outputs = {}

    def source(name: str) -> Path:
        path = paper / name
        source_records[name] = {"sha256": sha(path), "bytes": path.stat().st_size}
        return path

    def read(name: str) -> dict:
        return json.loads(source(name).read_text(encoding="utf-8"))

    def save(number: int, arrays: dict, parameters: dict, inputs: list[str], transformation: str) -> None:
        stem = f"figure{number}"
        data_path = out / (stem + ".npz")
        params_path = out / (stem + ".json")
        np.savez_compressed(data_path, **arrays)
        params_path.write_text(json.dumps(parameters, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        outputs[stem] = {
            "data": "data/" + data_path.name, "sha256": sha(data_path),
            "bytes": data_path.stat().st_size,
            "parameters": "data/" + params_path.name, "parameters_sha256": sha(params_path),
            "source_inputs": inputs, "transformation": transformation,
            "arrays": {name: array_record(value) for name, value in arrays.items()},
        }

    # These three approved renderers define the selected arrays and transforms.
    for script in ("paper-data/figure-style/render_examples_1_3.py",
                   "paper-data/figure-style/render_example_2.py",
                   "paper-data/figure-style/render_all.py"):
        source(script)
    style = read("paper-data/figure-style/provenance.json")
    terms = read("figures/corrected-term-direct-vs-optimized-ohmic-manifest.json")
    term_source = "paper-data/figure-style/term-verification.npz"
    path = source(term_source)
    if sha(path) != terms["source"]["results_sha256"]:
        raise ValueError("Term arrays do not match the approved figure manifest")
    with np.load(path, allow_pickle=False) as z:
        times, ix = z["times"], z["checkpoints"]
        tj = times[ix]
        arrays = {"times": times, "direct_times": tj, "direct_indices": ix}
        for term in terms["terms"]:
            old = "term_" + term["machine_label"]
            new = term["paper_label"]
            direct = z[old + "_direct_gpu_wrapped"]
            reduced = z[old + "_optimized_cupy_wrapped"]
            dn = np.linalg.norm(direct, axis=(1, 2))
            rn = np.linalg.norm(reduced, axis=(1, 2))
            mask = (tj > 0) & (dn > 0)
            arrays[new + "_direct_norm"] = dn
            arrays[new + "_reduced_norm"] = rn
            arrays[new + "_error_times"] = tj[mask]
            arrays[new + "_relative_error"] = np.linalg.norm(direct - reduced[ix], axis=(1, 2))[mask] / dn[mask]
    save(3, arrays, {
        "title": "Direct and reduced TCL6 term evaluation",
        "hamiltonian_lab": terms["model"]["hamiltonian_lab"],
        "coupling_lab": terms["model"]["coupling_lab"],
        "coupling_operator": "A=sigma_z/2", "lambda_squared": 0.1,
        "bath": {"model": "zero-temperature Ohmic exponential cutoff", "omega_c": 10.0,
                 "unit_correlation": "C_B(t)=omega_c^2/[2*(1+i*omega_c*t)^2]",
                 "physical_correlation": "C_lambda=lambda^2 C_B", "temperature": 0.0},
        "grid": terms["grid"], "terms": [t["paper_label"] for t in terms["terms"]],
        "palette": terms["presentation"]["palette"],
        "array_semantics": style["coupling_notation"]["example1"],
        "quadrature": "cumulative trapezoid for signed Gamma; nested composite trapezoid for the two remaining integrals",
        "relative_error": "Frobenius norm(direct-reduced)/Frobenius norm(direct), omitting zero denominator and t=0",
        "archived_backend_selection": "direct_gpu_wrapped versus optimized_cupy_wrapped, as in the approved renderer",
        "rescaling": "None. The accepted term maps already contain lambda^6; do not multiply again.",
    }, [term_source, "figures/corrected-term-direct-vs-optimized-ohmic-manifest.json",
        "paper-data/figure-style/provenance.json"],
        "Identical Frobenius-norm and relative-error operations to the approved renderer. All plotted samples retained; unused generator matrices omitted.")

    csv_name = "paper-data/runtime-scaling/phoenix-h012-k6-scaling-through-nt25.csv"
    mad_name = "paper-data/runtime-scaling/kernel-summary.csv"
    setup_name = "paper-data/runtime-scaling/benchmark-setup.json"
    with source(csv_name).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with source(mad_name).open(newline="", encoding="utf-8") as f:
        deviations = {(r["scope"], int(r["nt"])): float(r["wall_mad_seconds"]) for r in csv.DictReader(f)}
    setup = read(setup_name)
    runtime_provenance = read("paper-data/runtime-scaling/provenance.json")
    arrays = {}
    for scope, key in [("H0", "H0"), ("H1", "H1"), ("H2", "H2"), ("K6 full", "K6")]:
        subset = sorted((r for r in rows if r["scope"] == scope), key=lambda r: int(r["nt"]))
        if len(subset) != 18 or any(r["validation"] != "validated" for r in subset):
            raise ValueError("Unexpected approved runtime series")
        n = np.array([int(r["nt"]) for r in subset], dtype=np.int64)
        arrays[key + "_nt"] = n
        arrays[key + "_seconds"] = np.array([float(r["seconds"]) for r in subset])
        arrays[key + "_guide_seconds"] = np.array([float(r["reference_seconds"]) for r in subset])
        if scope != "K6 full":
            arrays[key + "_mad_seconds"] = np.array([deviations[(scope, int(nt))] for nt in n])
    save(4, arrays, {
        "title": "Measured runtime scaling", "hardware": "One Intel Xeon Gold 6226 CPU core at 2.7 GHz",
        "fixture": setup["fixture"], "bath": setup["bath"], "full_generator_dt": setup["dt"],
        "dtype": setup["output_dtype"], "nt_powers": list(range(8, 26)),
        "scope": setup["scope"], "implementation_route": setup["implementation"]["route"],
        "implementation_options": setup["options"],
        "kernel_definitions": setup["implementation"]["kernel_definitions"],
        "sampling": {"kernels": "Median of five samples; MAD error bars",
                     "K6": "Median of three samples through 2^20; one sample thereafter; no error bars"},
        "included_time": "Signed-Gamma preparation and full K6 evaluation",
        "excluded_time": "Compilation, subsequent validation, K2/K4, and state propagation",
        "guide_formulas": {"H0": "N", "H1": "N*(1+log2(N))", "H2": "N*(1+log2(N))^2", "K6": "N*(1+log2(N))^2"},
        "guide_anchor_powers": {"H0": 25, "H1": 25, "H2": 25, "K6": 22},
        "caveats": ["Guides are anchored comparisons, not fitted complexity estimates.",
                    "Kernel and full-generator workloads differ; timings are not additive.",
                    "Fixed dt means increasing N extends duration, not time-step refinement.",
                    "Replotting preserves historical measurements; it does not benchmark the present machine.",
                    "The bath strength 0.05 is already present in the benchmark correlation; no additional coupling multiplier is applied."],
        "original_renderer": {"relative_path": "experiments/performance_campaign/plot_h012_k6_publication.py",
                              "sha256": runtime_provenance["source_scripts"][1]["sha256"], "palette": "distinct"},
    }, [csv_name, mad_name, setup_name, "paper-data/runtime-scaling/provenance.json"],
        "Copy all 72 displayed timing/guide values and the 54 kernel MADs. Omit operational paths, job identifiers, and unplotted larger grids.")

    reference_name = "paper-data/density-comparison/tcl-reference.npz"
    report_name = "paper-data/density-comparison/tcl-reference-report.json"
    tempo_name = "paper-data/density-comparison/tempo-provenance.json"
    ref_report, tempo_report = read(report_name), read(tempo_name)
    arrays, model_parameters = {}, {}
    density_inputs = [reference_name, report_name, tempo_name]
    ref_path = source(reference_name)
    if sha(ref_path) != tempo_report["tcl_reference_sha256"]:
        raise ValueError("Density reference does not match accepted TEMPO provenance")
    with np.load(ref_path, allow_pickle=False) as ref:
        for model in ["unbiased", "biased"]:
            name = f"paper-data/density-comparison/{model}/trajectories.npz"
            density_inputs.append(name)
            path = source(name)
            if sha(path) != tempo_report["models"][model]["trajectory_sha256"]:
                raise ValueError("TEMPO trajectory hash mismatch")
            with np.load(path, allow_pickle=False) as z:
                t, rho = z[model + "_times"], z[model + "_rho"]
            rt = ref[model + "_times"]
            ix = np.arange(len(t)) * 10
            if len(t) != 2001 or np.max(np.abs(rt[ix] - t)) >= 5e-12:
                raise ValueError("TEMPO/TCL grids are not aligned as in the approved plot")
            arrays[model + "_tcl_times"], arrays[model + "_tempo_times"] = rt, t
            arrays[model + "_tempo_rho00"] = rho[:, 0, 0].real
            arrays[model + "_tempo_rho01"] = rho[:, 0, 1].real
            for order in ["tcl2", "tcl4", "tcl6"]:
                r = ref[model + "_rho_" + order]
                arrays[model + "_" + order + "_rho00"] = r[:, 0, 0].real
                arrays[model + "_" + order + "_rho01"] = r[:, 0, 1].real
                arrays[model + "_" + order + "_tempo_distance"] = np.linalg.norm(r[ix] - rho, axis=(1, 2))
            arrays[model + "_tcl6_tcl4_distance"] = np.linalg.norm(
                ref[model + "_rho_tcl6"][ix] - ref[model + "_rho_tcl4"][ix], axis=(1, 2))
            m = ref_report["models"][model]["model"]
            model_parameters[model] = {k: m[k] for k in ["hamiltonian_lab", "coupling_lab", "energies", "energy_gap"]}
            model_parameters[model]["tempo"] = tempo_report["models"][model]["parameters"]
            model_parameters[model]["tempo_raw_diagnostics"] = tempo_report["models"][model]["raw_diagnostics"]
    protocol = dict(ref_report["shared_protocol"])
    protocol["lambda_squared"] = protocol.pop("alpha")
    save(5, arrays, {
        "title": "Cumulative TCL dynamics and finite-resolution TEMPO", "models": model_parameters,
        "protocol": protocol, "Omega": 1.0, "coupling_operator": "A=sigma_z/2",
        "unit_correlation": "C_B(t)=omega_c^2/[2*(1+i*omega_c*t)^2]",
        "initial_state_lab": [[1.0, 0.0], [0.0, 0.0]], "display_basis": "laboratory basis",
        "display_picture": "Schrodinger", "plot_tcl_dt": 0.0025, "plot_tcl_samples_per_model": 20001,
        "tempo_version": "OQuPy 0.5.0", "tempo_samples_per_model": 2001,
        "array_semantics": style["coupling_notation"]["example2"],
        "cumulative_orders": "TCL2/TCL4/TCL6 include sum_{m=1}^n lambda^(2m) K_(2m), n=1/2/3",
        "comparison": "Frobenius norm of raw full density-matrix differences on coincident time points; no interpolation or state repair",
        "caveat": "TEMPO is a finite-resolution comparison, not a certified continuum-converged reference. Plotted residuals are discrepancies, not certified TCL errors.",
    }, density_inputs, "Apply exactly the approved renderer's real-entry extraction, stride-10 coincident-grid comparison, and full-matrix Frobenius norms. Preserve all plotted samples. No state correction, interpolation, or coupling rescaling.")

    structure_name = "paper-data/figure-style/structured-band-three-boson.npz"
    structure_report_name = "paper-data/figure-style/structured-band-report.json"
    structure_provenance_name = "paper-data/structured-three-boson/provenance.json"
    report = read(structure_report_name)
    provenance = read(structure_provenance_name)
    path = source(structure_name)
    if sha(path) != provenance["accepted_input"]["npz_sha256"].lower():
        raise ValueError("Structured-bath arrays do not match the approved input")
    if source_records[structure_report_name]["sha256"] != provenance["accepted_input"]["report_sha256"].lower():
        raise ValueError("Structured-bath parameters do not match the approved input")
    with np.load(path, allow_pickle=False) as z:
        keys = [name for prefix in ["one", "two", "three"] for name in [prefix, prefix + "_energy"]]
        keys += ["density_times", "population_excited_tcl2", "population_excited_tcl4", "population_excited_tcl6", "golden_rule_population"]
        arrays = {key: z[key] for key in keys}
        anchor = report["population"]["post_slip_anchor_time"]
        if not np.array_equal(np.isnan(arrays["golden_rule_population"]), arrays["density_times"] < anchor):
            raise ValueError("Unexpected missing-value mask in the post-slip kinetic curve")
        late = z["rate_times"] >= 120
        for key in ["rate_times", "rate_down_tcl2", "rate_down_tcl4", "rate_down_tcl6"]:
            arrays[key] = z[key][late]
    runtime = report["runtime"]["reported_density"]
    save(6, arrays, {
        "title": "Three-boson relaxation in a structured bath", "Omega": 1.0, "lambda_squared": 0.05,
        "hamiltonian": "H_S=(Omega/2)*sigma_x", "coupling_operator": "A=sigma_z/2",
        "unit_spectrum": "J_B(omega)=2*Omega*j(omega/Omega)",
        "physical_spectrum": "J_lambda=lambda^2*J_B",
        "correlation_convention": "C_B(t)=(1/2)*integral_0^infinity J_B(omega)*exp(-i*omega*t) d omega",
        "band": {"shape": "j(u)=8/(3W)*sin(pi*(u-u_minus)/W)^4 within the band and zero outside",
                 "u_minus": 1/5, "u_plus": 7/15, "width": 4/15, "integral_j": 1.0},
        "temperature": 0.0, "initial_bath": "vacuum", "initial_system": "excited energy state |+x><+x|",
        "rotating_wave_approximation": False,
        "grid": {"rate_dt": report["settings"]["rate_dt"], "rate_t_end": 360.0,
                 "display_rate_start": 120.0, "density_generator_dt": runtime["sample_dt"],
                 "density_generator_samples": runtime["generator_samples"],
                 "density_dt": report["settings"]["density_output_dt"],
                 "density_samples": runtime["density_samples"], "density_t_end": 1400.0,
                 "phase_space_samples_per_unit": report["settings"]["phase_space_samples_per_unit"]},
        "quadrature": {"Gamma": "cumulative trapezoid", "TCL4": "composite trapezoid", "TCL6": "nested composite trapezoid", "golden_rule_Gauss_Legendre_nodes_per_variable": 512},
        "golden_rule": {"formula": "gamma_3b=c3*lambda^6*Omega", "c3": report["golden_rule"]["coefficient_gl512"],
                        "rate": report["golden_rule"]["main_rate"],
                        "population_anchor_time": report["population"]["post_slip_anchor_time"],
                        "population_anchor_value": report["population"]["post_slip_anchor_population"]},
        "phase_space_display": "one/two/three are j, j*j, j*j*j divided by their respective maxima; plot vertical offsets 2,1,0",
        "array_semantics": style["coupling_notation"]["example3"],
        "caveats": ["Black population curve uses the independent leading T-matrix rate anchored to the TCL6 population at Omega*t=240; it is not an independent full trajectory.",
                    "The kinetic population is NaN before the anchor time, intentionally suppressing that unmodeled interval in the plot.",
                    "Raw TCL4 can be transiently nonpositive. No population curve is projected or repaired.",
                    "Normalized phase-space curves identify energy support, not the complete transition rate."],
    }, [structure_name, structure_report_name, structure_provenance_name],
        "Copy the exact plotted phase-space and population arrays; retain precisely the rate samples with t>=120 used by the approved renderer. No interpolation, decimation, or rescaling.")

    manifest = {
        "schema": "tcl6-paper-figure-reproduction-v1", "source_root": "canonical TCL6 Paper repository",
        "scope": "Replot approved stored numerical results for Figures 3-6; no simulation or benchmark rerun",
        "source_path_base": "Paths under source_inputs are relative to the canonical paper repository, and are provenance only at replot time.",
        "font_policy": "Bundled Matplotlib DejaVu Sans, including mathtext; no external fonts or LaTeX required",
        "physical_values_recomputed": False, "plotted_observables_extracted": True,
        "source_inputs": source_records, "figures": outputs,
    }
    (HERE / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figures": {k: v["bytes"] for k, v in outputs.items()},
                      "npz_bytes_total": sum(v["bytes"] for v in outputs.values())}, indent=2))


if __name__ == "__main__":
    main()
