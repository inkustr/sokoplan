from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Row:
    level_id: str
    success: bool
    nodes: int
    runtime_s: float
    solution_len: int


def _read_csv(path: Path) -> Dict[str, Row]:
    out: Dict[str, Row] = {}
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            level_id = (row.get("level_id") or "").strip()
            if not level_id:
                continue
            out[level_id] = Row(
                level_id=level_id,
                success=str(row.get("success", "")).strip() == "True",
                nodes=int(float(row.get("nodes") or 0)),
                runtime_s=float(row.get("runtime") or 0.0),
                solution_len=int(float(row.get("solution_len") or -1)),
            )
    return out


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _ratio_guides(ax, xmin: float, xmax: float) -> None:
    xs = np.array([xmin, xmax], dtype=float)
    # y = x (center axis) + decade diagonals.
    ax.plot(xs, xs, "--", color="black", linewidth=1.1, label="x1")
    ax.plot(xs, 10.0 * xs, ":", color="gray", linewidth=1.0, label="x10 / x0.1")
    ax.plot(xs, 0.1 * xs, ":", color="gray", linewidth=1.0)
    ax.plot(xs, 100.0 * xs, ":", color="silver", linewidth=1.0, label="x100 / x0.01")
    ax.plot(xs, 0.01 * xs, ":", color="silver", linewidth=1.0)


def plot_01_box_violin(common: List[str], gnn: Dict[str, Row], hun: Dict[str, Row], out: Path) -> None:
    # Compact distribution comparison in one figure.
    g_nodes = [max(1, gnn[l].nodes) for l in common]
    h_nodes = [max(1, hun[l].nodes) for l in common]
    g_rt = [max(1e-9, gnn[l].runtime_s) for l in common]
    h_rt = [max(1e-9, hun[l].runtime_s) for l in common]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    parts_n = axes[0].violinplot([h_nodes, g_nodes], showmeans=True, showextrema=False)
    for b in parts_n["bodies"]:
        b.set_alpha(0.45)
    axes[0].boxplot([h_nodes, g_nodes], widths=0.22, patch_artist=True)
    axes[0].set_xticks([1, 2], ["Hungarian", "GNN"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("nodes (log)")
    axes[0].set_title("Nodes distribution")
    axes[0].grid(alpha=0.25)

    parts_t = axes[1].violinplot([h_rt, g_rt], showmeans=True, showextrema=False)
    for b in parts_t["bodies"]:
        b.set_alpha(0.45)
    axes[1].boxplot([h_rt, g_rt], widths=0.22, patch_artist=True)
    axes[1].set_xticks([1, 2], ["Hungarian", "GNN"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("runtime_s (log)")
    axes[1].set_title("Runtime distribution")
    axes[1].grid(alpha=0.25)

    _save(fig, out)


def plot_02_scatter(common: List[str], gnn: Dict[str, Row], hun: Dict[str, Row], out: Path) -> None:
    h_nodes = np.array([max(1, hun[l].nodes) for l in common], dtype=float)
    g_nodes = np.array([max(1, gnn[l].nodes) for l in common], dtype=float)
    h_rt = np.array([max(1e-9, hun[l].runtime_s) for l in common], dtype=float)
    g_rt = np.array([max(1e-9, gnn[l].runtime_s) for l in common], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    ax.scatter(h_nodes, g_nodes, s=11, alpha=0.45, color="#1f77b4", edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    nmin = min(np.min(h_nodes), np.min(g_nodes))
    nmax = max(np.max(h_nodes), np.max(g_nodes))
    _ratio_guides(ax, nmin, nmax)
    ax.set_xlim(nmin, nmax)
    ax.set_ylim(nmin, nmax)
    ax.set_xlabel("Hungarian nodes")
    ax.set_ylabel("GNN nodes")
    ax.set_title("Boxoban: nodes scatter (log-log)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True, fontsize=8, loc="upper left")

    ax = axes[1]
    ax.scatter(h_rt, g_rt, s=11, alpha=0.45, color="#ff7f0e", edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    tmin = min(np.min(h_rt), np.min(g_rt))
    tmax = max(np.max(h_rt), np.max(g_rt))
    _ratio_guides(ax, tmin, tmax)
    ax.set_xlim(tmin, tmax)
    ax.set_ylim(tmin, tmax)
    ax.set_xlabel("Hungarian runtime (s)")
    ax.set_ylabel("GNN runtime (s)")
    ax.set_title("Boxoban: runtime scatter (log-log)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True, fontsize=8, loc="upper left")

    _save(fig, out)


def plot_03_ecdf_nodes(common: List[str], gnn: Dict[str, Row], hun: Dict[str, Row], out: Path) -> None:
    h_nodes = np.sort(np.array([max(1, hun[l].nodes) for l in common], dtype=float))
    g_nodes = np.sort(np.array([max(1, gnn[l].nodes) for l in common], dtype=float))
    yh = np.linspace(0.0, 1.0, len(h_nodes), endpoint=True)
    yg = np.linspace(0.0, 1.0, len(g_nodes), endpoint=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(h_nodes, yh, label="Hungarian", color="#ff7f0e")
    ax.plot(g_nodes, yg, label="GNN", color="#1f77b4")
    ax.set_xscale("log")
    ax.set_xlabel("expanded nodes (log scale)")
    ax.set_ylabel("ECDF")
    ax.set_title("ECDF of expanded nodes")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    _save(fig, out)


def plot_04_solution_delta(common: List[str], gnn: Dict[str, Row], hun: Dict[str, Row], out: Path) -> None:
    # Delta in pushes vs reference (Hungarian), shown as percentages.
    deltas = []
    deltas_pct = []
    for lid in common:
        gr, hr = gnn[lid], hun[lid]
        if not (gr.success and hr.success):
            continue
        if gr.solution_len < 0 or hr.solution_len <= 0:
            continue
        d = int(gr.solution_len - hr.solution_len)
        deltas.append(d)
        deltas_pct.append(100.0 * float(d) / float(hr.solution_len))

    if not deltas:
        raise SystemExit("No comparable successful solution lengths found for histogram.")

    # Integer push deltas on x-axis, y-axis as percentage of comparable levels.
    vals, counts = np.unique(np.array(deltas, dtype=int), return_counts=True)
    perc = 100.0 * counts.astype(float) / float(len(deltas))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(vals, perc, width=0.9, color="#1f77b4", alpha=0.9)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Δ solution length (GNN pushes - Hungarian pushes)")
    ax.set_ylabel("levels (%)")
    ax.set_title("Histogram of solution-length delta (percentage of levels)")
    ax.grid(axis="y", alpha=0.25)

    # Helpful text with mean relative overhead.
    mean_rel = float(np.mean(deltas_pct))
    ax.text(
        0.99,
        0.98,
        f"mean relative delta: {mean_rel:.2f}%",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.7"),
    )
    _save(fig, out)


def plot_05_sorted_runtime(common: List[str], gnn: Dict[str, Row], hun: Dict[str, Row], out: Path) -> None:
    g_rt = sorted([max(1e-9, gnn[l].runtime_s) for l in common])
    h_rt = sorted([max(1e-9, hun[l].runtime_s) for l in common])
    x = np.arange(len(common))

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    ax.plot(x, h_rt, label="Hungarian", color="#ff7f0e")
    ax.plot(x, g_rt, label="GNN", color="#1f77b4")
    ax.set_yscale("log")
    ax.set_xlabel("levels (sorted independently per method)")
    ax.set_ylabel("runtime (s, log scale)")
    ax.set_title("Sorted runtime curves")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    _save(fig, out)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate 5 evaluation plots for Boxoban GNN vs Hungarian.")
    p.add_argument("--gnn_csv", default="results/eval/boxoban/gnn.csv")
    p.add_argument("--hun_csv", default="results/eval/boxoban/hungarian.csv")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    gnn = _read_csv(Path(args.gnn_csv))
    hun = _read_csv(Path(args.hun_csv))
    common = sorted(set(gnn.keys()) & set(hun.keys()))
    if not common:
        raise SystemExit("No common levels between gnn_csv and hun_csv.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_01_box_violin(common, gnn, hun, out_dir / "01_box_violin_nodes_runtime.png")
    plot_02_scatter(common, gnn, hun, out_dir / "02_scatter_nodes_runtime_loglog.png")
    plot_03_ecdf_nodes(common, gnn, hun, out_dir / "03_ecdf_nodes.png")
    plot_04_solution_delta(common, gnn, hun, out_dir / "04_solution_len_delta_hist_pct.png")
    plot_05_sorted_runtime(common, gnn, hun, out_dir / "05_sorted_runtime_curves.png")

    print(f"[DONE] wrote 5 plots to: {out_dir}")


if __name__ == "__main__":
    main()
