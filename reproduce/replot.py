"""Replot the approved Figures 3-6 from the small bundled data package.

Run from any directory: python /path/to/reproduce/replot.py --output-dir figures
No TCL solver, TEMPO installation, LaTeX, GPU, or network access is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

HERE = Path(__file__).resolve().parent
STYLES = {"TCL2": dict(color="#777777", ls="-.", lw=1.25),
          "TCL4": dict(color="#D55E00", ls="--", lw=1.35),
          "TCL6": dict(color="#0072B2", ls="-", lw=1.4)}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "mathtext.fontset": "dejavusans", "text.usetex": False,
        "axes.spines.top": True, "axes.spines.right": True, "axes.linewidth": .8,
        "xtick.major.width": .8, "ytick.major.width": .8, "savefig.facecolor": "white",
        "legend.frameon": True, "legend.framealpha": 1., "legend.facecolor": "white",
        "legend.edgecolor": "black", "legend.fancybox": False,
    })


def load(number: int, manifest: dict) -> tuple[dict, dict]:
    entry = manifest["figures"][f"figure{number}"]
    path, param_path = HERE / entry["data"], HERE / entry["parameters"]
    if sha(path) != entry["sha256"] or sha(param_path) != entry["parameters_sha256"]:
        raise ValueError(f"Figure {number}: bundled data or parameters failed SHA-256 verification")
    with np.load(path, allow_pickle=False) as f:
        arrays = {name: f[name] for name in f.files}
    if set(arrays) != set(entry["arrays"]):
        raise ValueError(f"Figure {number}: unexpected array inventory")
    for name, value in arrays.items():
        expected = entry["arrays"][name]
        if list(value.shape) != expected["shape"] or str(value.dtype) != expected["dtype"]:
            raise ValueError(f"Figure {number}: shape/dtype mismatch in {name}")
        if hashlib.sha256(value.tobytes(order="C")).hexdigest() != expected["sha256_c_order_bytes"]:
            raise ValueError(f"Figure {number}: array hash mismatch in {name}")
        if np.count_nonzero(np.isnan(value)) != expected["nan_count"] or np.count_nonzero(np.isinf(value)) != expected["infinity_count"]:
            raise ValueError(f"Figure {number}: missing-value mask mismatch in {name}")
    return arrays, json.loads(param_path.read_text(encoding="utf-8"))


def figure3(z: dict, p: dict):
    fig, (a, b) = plt.subplots(2, 1, figsize=(7, 4.25), sharex=True,
                             gridspec_kw={"height_ratios": (2.05, 1.)})
    fig.subplots_adjust(left=.115, right=.98, bottom=.12, top=.975, hspace=.10)
    t, tj = z["times"], z["direct_times"]
    handles, positive_norms, positive_errors = [], [], []
    for term, color in zip(p["terms"], p["palette"]):
        dn, rn = z[term + "_direct_norm"], z[term + "_reduced_norm"]
        mask = (tj > 0) & (dn > 0)
        error = z[term + "_relative_error"]
        positive_norms.extend(dn[dn > 0])
        positive_errors.extend(error[error > 0])
        a.semilogy(t[t > 0], rn[t > 0], color=color, lw=1.3)
        a.scatter(tj[mask], dn[mask], facecolors="none", edgecolors=color, s=15, linewidths=.8, zorder=3)
        b.semilogy(z[term + "_error_times"], error, color=color, lw=1.1, marker="o", ms=3, mfc="white", mew=.8)
        handles.append(Line2D([], [], color=color, lw=1.4, label=term))
    a.set_xlim(0, t[-1])
    b.set_xticks(np.arange(0, 20.01, 2.5))
    a.set_ylim(min(positive_norms) / 1.7, max(positive_norms) * 3.5)
    b.set_ylim(min(positive_errors) / 1.8, max(positive_errors) * 1.5)
    a.set_ylabel(r"$\|\lambda^6\mathcal{K}_{6,I}^{(\nu)}(t)\|_{\mathrm{F}}$")
    b.set_ylabel(r"$\epsilon_{\nu}(t_j)$")
    b.set_xlabel(r"$t$")
    for ax, label in [(a, "(a)"), (b, "(b)")]:
        ax.text(.012, .965, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
        ax.grid(axis="y", which="major", color="#dedede", lw=.5)
        ax.set_axisbelow(True)
    handles += [Line2D([], [], color="#222222", lw=1.3, label="Reduced"),
                Line2D([], [], color="#222222", ls="none", marker="o", ms=3.8, mfc="white", label="Direct")]
    a.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .995), ncol=4,
             columnspacing=1.1, handlelength=1.7, borderpad=.35, labelspacing=.3)
    return fig


def figure4(z: dict, p: dict):
    with plt.rc_context({"font.size": 10, "axes.labelsize": 11}):
        fig, ax = plt.subplots(figsize=(6.8, 4.5))
        fig.subplots_adjust(left=.13, right=.98, bottom=.15, top=.97)
        handles = []
        for key, color, label in [("H0", "#0072B2", r"$H_0$ kernel"),
                                  ("H1", "#D55E00", r"$H_1$ kernel"),
                                  ("H2", "#009E73", r"$H_2$ kernel"),
                                  ("K6", "#AA4499", r"Full $\mathcal{K}_6$")]:
            n, y = z[key + "_nt"], z[key + "_seconds"]
            ax.errorbar(n, y, yerr=z.get(key + "_mad_seconds"), color=color, marker="o", markersize=3.5,
                        linewidth=1.25, elinewidth=.8, capsize=1.8, zorder=3)
            ax.plot(n, z[key + "_guide_seconds"], color=color, ls="--", lw=1, alpha=.55, zorder=2)
            handles.append(Line2D([], [], color=color, marker="o", markersize=3.5, linewidth=1.25, label=label))
        handles.append(Line2D([], [], color=".55", ls="--", lw=1, label="Scaling"))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        powers = (8, 12, 16, 20, 25)
        ax.set_xticks([2**power for power in powers], [rf"$2^{{{power}}}$" for power in powers])
        ax.set_yticks([10.**power for power in (-4, -2, 0, 2, 4)])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set(xlim=(2**7.7, 2**25.4), ylim=(2e-5, 8e4),
               xlabel=r"Number of time steps, $N_t$", ylabel="Wall time (s)")
        ax.grid(which="major", color=".92", linewidth=.6, zorder=0)
        ax.legend(handles=handles, frameon=False, loc="upper left", handlelength=2.5)
    return fig


def figure5(z: dict, p: dict):
    error_styles = {name: {**style, "ls": "--"} for name, style in STYLES.items()}
    fig, axs = plt.subplots(2, 3, figsize=(7, 5.65))
    fig.subplots_adjust(left=.10, right=.985, bottom=.17, top=.82, wspace=.44, hspace=.78)
    for row, model in enumerate(["unbiased", "biased"]):
        rt, t = z[model + "_tcl_times"], z[model + "_tempo_times"]
        for col, obs in enumerate(["rho00", "rho01"]):
            for name, style in STYLES.items():
                axs[row, col].plot(rt, z[model + "_" + name.lower() + "_" + obs], **style)
            axs[row, col].plot(t, z[model + "_tempo_" + obs], color="#171717", lw=.8,
                              marker="o", markersize=3, markevery=100, mew=0, zorder=4)
        for name, style in error_styles.items():
            axs[row, 2].semilogy(t[1:], z[model + "_" + name.lower() + "_tempo_distance"][1:], **style)
        axs[row, 2].semilogy(t[1:], z[model + "_tcl6_tcl4_distance"][1:], color="#009E73", ls=":", lw=1.3)
        axs[row, 2].set_ylim(1e-6, 1e-1)
        axs[row, 2].set_yticks([1e-6, 1e-4, 1e-2])
        for col, ax in enumerate(axs[row]):
            ax.set_xlim(0, 50)
            ax.set_xticks([0, 10, 20, 30, 40, 50])
            ax.set_xlabel(r"$\Omega t$", labelpad=3)
            ax.grid(axis="y", which="major", color="#dedede", lw=.5)
            ax.text(0, 1.085, "(" + chr(97 + row * 3 + col) + ")", transform=ax.transAxes, fontsize=11, fontweight="bold")
        fig.text(.5425, axs[row, 0].get_position().y1 + .061, model.capitalize(), ha="center", va="bottom", fontsize=10, fontweight="bold")
        axs[row, 0].set_ylabel(r"$\mathrm{Re}\,\rho_{00}$", labelpad=3)
        axs[row, 1].set_ylabel(r"$\mathrm{Re}\,\rho_{01}$", labelpad=3)
        axs[row, 2].set_ylabel(r"$\|\Delta\rho\|_{F}$", labelpad=3)
    handles = [Line2D([], [], label=name, **style) for name, style in STYLES.items()]
    handles.append(Line2D([], [], color="#171717", lw=.8, marker="o", markersize=3, label="TEMPO"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .99), ncol=4, columnspacing=1.8, handlelength=2.5)
    residual = [Line2D([], [], label=name + " - TEMPO", **style) for name, style in error_styles.items()]
    residual.append(Line2D([], [], color="#009E73", ls=":", lw=1.3, label="TCL6 - TCL4"))
    fig.legend(handles=residual, loc="lower center", bbox_to_anchor=(.54, .005), ncol=2, columnspacing=2., handlelength=2.5)
    return fig


def figure6(z: dict, p: dict):
    fig = plt.figure(figsize=(7, 5.65))
    a, b, c = fig.add_axes([.12, .59, .34, .30]), fig.add_axes([.61, .59, .365, .30]), fig.add_axes([.12, .105, .855, .34])
    for name, key, baseline, label in [("TCL2", "one", 2, r"one boson: $j$"),
                                      ("TCL4", "two", 1, r"two bosons: $j^{*2}$"),
                                      ("TCL6", "three", 0, r"three bosons: $j^{*3}$")]:
        e, v = z[key + "_energy"], z[key]
        a.fill_between(e, baseline, v + baseline, color=STYLES[name]["color"], alpha=.13, lw=0)
        a.plot(e, v + baseline, **STYLES[name])
        if key == "one":
            a.text(.53, baseline + .65, label, fontsize=9)
        else:
            a.text(.19, baseline + .78, label.replace(": ", ":\n"), fontsize=9, va="top")
    a.axvline(1, color="#222222", lw=1, ls="--")
    a.text(1.025, 3., "qubit gap", rotation=90, va="top", fontsize=9)
    a.set(xlim=(.15, 1.45), ylim=(-.08, 3.12), yticks=[0, 1, 2, 3],
          xlabel=r"Emitted energy $E/\Omega$", ylabel="Spectral phase space\n(normalized)")
    rt = z["rate_times"]
    b.axhline(0, color=".65", lw=.6)
    for name, style in STYLES.items():
        b.plot(rt, z["rate_down_" + name.lower()], **style)
    b.axhline(p["golden_rule"]["rate"], color="#171717", lw=1, ls="--", label=r"Three-boson $T$ matrix")
    b.set(xlim=(120, 360), xlabel=r"$\Omega t$", ylabel=r"Downward rate $\gamma_\downarrow/\Omega$")
    b.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    b.legend(loc="upper right", fontsize=8.5, handlelength=1.6)
    inset = b.inset_axes([.52, .17, .43, .27])
    vals = []
    for name in ["TCL2", "TCL4"]:
        v = z["rate_down_" + name.lower()]
        vals.append(v)
        inset.plot(rt, v, **STYLES[name])
    lim = 1.15 * np.max(np.abs(np.concatenate(vals)))
    inset.set(xlim=(120, 360), ylim=(-lim, lim), xticks=[120, 360])
    inset.tick_params(labelsize=8, pad=1)
    inset.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    inset.yaxis.get_offset_text().set_fontsize(8)
    dt = z["density_times"]
    for name, style in STYLES.items():
        c.plot(dt, z["population_excited_" + name.lower()], **style)
    c.plot(dt, z["golden_rule_population"], color="#171717", lw=1, ls="--", label=r"$T$-matrix kinetic curve")
    c.axvline(p["golden_rule"]["population_anchor_time"], color=".65", lw=.7, ls="--")
    c.set(xlim=(0, 1400), ylim=(-.02, 1.04), xlabel=r"$\Omega t$", ylabel=r"Excited population $P_e(t)$")
    c.legend(loc="center right")
    for ax, label in [(a, "(a)"), (b, "(b)"), (c, "(c)")]:
        ax.text(.20 if ax is b else 0, 1.07, label, transform=ax.transAxes, fontsize=11, fontweight="bold")
        ax.grid(axis="y", which="major", color="#dedede", lw=.5)
        ax.tick_params(direction="out")
    fig.legend(handles=[Line2D([], [], label=name, **style) for name, style in STYLES.items()],
               loc="upper center", bbox_to_anchor=(.55, 1.), ncol=3, handlelength=2.5, columnspacing=2)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reproduced-figures"))
    parser.add_argument("--figure", choices=["all", "3", "4", "5", "6"], default="all")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--verify-only", action="store_true", help="Check all selected data and parameter hashes without writing plots")
    args = parser.parse_args()
    if args.dpi < 50:
        parser.error("--dpi must be at least 50")
    manifest = json.loads((HERE / "provenance.json").read_text(encoding="utf-8"))
    selected = [3, 4, 5, 6] if args.figure == "all" else [int(args.figure)]
    loaded = {number: load(number, manifest) for number in selected}
    if args.verify_only:
        print("Verified data and parameter hashes for figures: " + ", ".join(map(str, selected)))
        return
    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots = {3: figure3, 4: figure4, 5: figure5, 6: figure6}
    for number in selected:
        fig = plots[number](*loaded[number])
        path = args.output_dir / f"figure{number}.png"
        fig.savefig(path, dpi=args.dpi)
        plt.close(fig)
        print(f"Figure {number}: {path} (sha256={sha(path)})")
    print("Stored approved results replotted; no simulations or timing measurements rerun.")
    if 5 in selected:
        print("TEMPO remains a finite-resolution comparison, not a certified continuum-converged reference.")


if __name__ == "__main__":
    main()
